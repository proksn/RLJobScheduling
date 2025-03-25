import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# -------------------------------
# Hilfsfunktion: Berechnung der Überlappung zweier Zeitintervalle
def interval_overlap(start1, end1, start2, end2):
    """Berechnet die überlappende Dauer der Intervalle [start1, end1] und [start2, end2]."""
    return max(0, min(end1, end2) - max(start1, start2))

# -------------------------------
# Daten laden: schedule_log.csv oder Dummy-Daten, falls nicht vorhanden
@st.cache_data
def load_schedule_data():
    file_path = "schedule_log.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        # Dummy-Daten zur Demonstration
        data = [
            {"order_id": 1, "machine": "M1", "start_time": 10, "finish_time": 100},
            {"order_id": 2, "machine": "M2", "start_time": 50, "finish_time": 200},
            {"order_id": 3, "machine": "M3", "start_time": 150, "finish_time": 300},
            {"order_id": 4, "machine": "M1", "start_time": 250, "finish_time": 400},
            {"order_id": 5, "machine": "M2", "start_time": 350, "finish_time": 480},
            {"order_id": 6, "machine": "M3", "start_time": 400, "finish_time": 550},
            {"order_id": 7, "machine": "M1", "start_time": 500, "finish_time": 600},
        ]
        df = pd.DataFrame(data)
    return df

# -------------------------------
# Überschrift und Beschreibung
st.title("Dashboard: Produktionsplanung und Maschinen-KPIs")
st.markdown("""
Dieses Dashboard zeigt wichtige Kennzahlen (KPIs) der Produktionsplanung:
- **Maschinenauslastung:** Donut-Charts je Maschine (nebeneinander).
- **Auftragsliste pro Maschine:** Übersicht, welche Aufträge an welcher Maschine bearbeitet wurden.
- **Gantt-Diagramm:** Visualisiert den zeitlichen Ablauf der Aufträge am ausgewählten Tag.
- **Gesamtübersicht:** Alle vorhandenen Aufträge.
""")

# -------------------------------
# Tag-Auswahl über Sidebar
total_time_units = 480  # Zeiteinheiten pro Tag
df_schedule = load_schedule_data()
max_time = df_schedule['finish_time'].max() if not df_schedule.empty else total_time_units
num_days = int(max_time // total_time_units) + 1
selected_day = st.sidebar.selectbox("Wähle Tag", list(range(1, num_days + 1)))
day_start = (selected_day - 1) * total_time_units
day_end = selected_day * total_time_units
st.sidebar.markdown(f"**Zeitraum:** {day_start} bis {day_end} Zeiteinheiten")

# Filter: Aufträge, die zumindest teilweise in den ausgewählten Tag fallen
df_day = df_schedule[(df_schedule['start_time'] < day_end) & (df_schedule['finish_time'] > day_start)].copy()

# -------------------------------
# Section 1: Donut-Charts (Maschinenauslastung)
st.header(f"Maschinenauslastung am Tag {selected_day}")
machines = ["M1", "M2", "M3"]
cols = st.columns(3)

for i, machine in enumerate(machines):
    df_machine = df_day[df_day['machine'] == machine]
    total_util = 0
    for _, row in df_machine.iterrows():
        overlap = interval_overlap(row['start_time'], row['finish_time'], day_start, day_end)
        total_util += overlap
    total_util = min(total_util, total_time_units)
    free_time = total_time_units - total_util
    utilization_percentage = (total_util / total_time_units) * 100 if total_time_units > 0 else 0

    fig = go.Figure(data=[go.Pie(
        labels=["Ausgelastet", "Frei"],
        values=[total_util, free_time],
        hole=0.5,
        textinfo="label+percent",
        marker=dict(colors=['#EF553B', '#DDDDDD'])
    )])
    fig.update_layout(title=f"Maschine {machine}: {utilization_percentage:.1f}% Auslastung")
    cols[i].plotly_chart(fig, use_container_width=True)

# -------------------------------
# Section 2: Auftragsliste pro Maschine
st.header(f"Auftragsliste pro Maschine am Tag {selected_day}")
for machine in machines:
    st.subheader(f"Maschine {machine}")
    df_machine = df_day[df_day['machine'] == machine].copy()
    if not df_machine.empty:
        df_machine['Dauer'] = df_machine.apply(
            lambda row: interval_overlap(row['start_time'], row['finish_time'], day_start, day_end),
            axis=1
        )
        df_machine = df_machine[['order_id', 'start_time', 'finish_time', 'Dauer']]
        st.dataframe(df_machine.reset_index(drop=True))
    else:
        st.write("Keine Aufträge für diese Maschine.")

# -------------------------------
# Section 3: Gantt-Diagramm für den ausgewählten Tag
st.header(f"Gantt-Diagramm am Tag {selected_day}")
if not df_day.empty:
    # Berechne relative Start-/Endzeiten (bezogen auf den Tagesbeginn)
    df_day['start_rel'] = df_day['start_time'].apply(lambda x: max(x, day_start) - day_start)
    df_day['finish_rel'] = df_day['finish_time'].apply(lambda x: min(x, day_end) - day_start)
    df_day['Duration'] = df_day['finish_rel'] - df_day['start_rel']

    # Konvertiere order_id in String, damit sie als Kategorie genutzt wird
    df_day["order_id"] = df_day["order_id"].astype(str)

    # Debug-Anzeige: Zeige alle relevanten Spalten
    st.subheader("Gefilterte Daten (debug)")
    st.write(df_day[['order_id', 'machine', 'start_time', 'finish_time', 'start_rel', 'finish_rel', 'Duration']])

    # Nur Einträge mit positiver Dauer
    df_day_gantt = df_day[df_day['Duration'] > 0]
    if df_day_gantt.empty:
        st.write("Keine Aufträge mit positiver Dauer an diesem Tag.")
    else:
        # Fallback: Manueller Gantt-Ansatz mit go.Bar
        fig_manual = go.Figure()
        for _, row in df_day_gantt.iterrows():
            fig_manual.add_trace(
                go.Bar(
                    x=[row['Duration']],           # Breite des Balkens
                    y=[row['machine']],            # Maschine als Kategorie
                    base=row['start_rel'],         # Startpunkt (links)
                    orientation='h',               # Horizontaler Balken
                    name=f"Order {row['order_id']}",
                    text=f"Order {row['order_id']}",
                    hoverinfo="text",
                    marker_line_width=1,
                    marker_line_color="black",
                )
            )
        fig_manual.update_layout(
            barmode='stack',
            xaxis=dict(title="Zeit (relativ zum Tagesbeginn)", type='linear'),
            yaxis=dict(autorange='reversed', title="Maschine"),
            height=400,
            title=f"Gantt-Diagramm (manuell) am Tag {selected_day}"
        )
        st.subheader("Gantt-Diagramm (manueller Ansatz)")
        st.plotly_chart(fig_manual, use_container_width=True)
else:
    st.write("Keine Daten für das Gantt-Diagramm.")

# -------------------------------
# Section 4: Gesamtübersicht aller Aufträge
st.header("Gesamtübersicht aller Aufträge")
st.dataframe(df_schedule.reset_index(drop=True))


# =============================================================================
# Test-Abschnitt
# =============================================================================

def test_interval_overlap():
    """Testet verschiedene Szenarien für die Funktion interval_overlap."""
    results = []
    # 1) Keine Überlappung (Ende = Anfang)
    results.append(("Keine Overlap 1", interval_overlap(0, 10, 10, 20) == 0))
    # 2) Teil-Überlappung
    results.append(("Teil-Overlap", interval_overlap(0, 10, 5, 15) == 5))
    # 3) Komplette Überlappung
    results.append(("Komplette Overlap", interval_overlap(0, 10, 0, 10) == 10))
    # 4) Verschachtelte Überlappung
    results.append(("Verschachtelte Overlap", interval_overlap(5, 15, 0, 20) == 10))
    # 5) Kein Overlap (start2 > end1)
    results.append(("Keine Overlap 2", interval_overlap(0, 5, 6, 10) == 0))
    return results

def test_day_filter(df, day_start, day_end):
    """
    Testet, ob der DataFrame df_day nur Aufträge enthält,
    die wirklich in [day_start, day_end) überlappen.
    """
    if df.empty:
        return [("Tag-Filter: Keine Daten", True)]
    checks = []
    for idx, row in df.iterrows():
        c1 = row['start_time'] < day_end
        c2 = row['finish_time'] > day_start
        checks.append(c1 and c2)
    return [("Tag-Filter: Überlappung korrekt", all(checks))]

def test_gantt_duration(df):
    """
    Testet, ob für alle Einträge in df_day_gantt (Duration > 0) gilt,
    dass start_rel < finish_rel.
    """
    if df.empty:
        return [("Gantt-Duration: Keine Aufträge mit Dauer > 0", True)]
    checks = []
    for idx, row in df.iterrows():
        checks.append(row['start_rel'] < row['finish_rel'])
    return [("Gantt-Duration: Start < End", all(checks))]

def run_all_tests():
    """Führt alle Tests durch und gibt deren Ergebnisse als Liste von (Testname, Ergebnis) zurück."""
    results = []
    results.extend(test_interval_overlap())
    results.extend(test_day_filter(df_day, day_start, day_end))
    df_gantt_test = df_day[df_day['Duration'] > 0] if not df_day.empty else pd.DataFrame()
    results.extend(test_gantt_duration(df_gantt_test))
    return results

# Sidebar-Button zum Ausführen der Tests
if st.sidebar.button("Tests durchführen"):
    test_results = run_all_tests()
    st.subheader("Testergebnisse")
    for test_name, passed in test_results:
        color = "green" if passed else "red"
        st.markdown(f"- **{test_name}**: <span style='color:{color}'>{'OK' if passed else 'FEHLER'}</span>", unsafe_allow_html=True)
