import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches


def plot_gantt(schedule_log):
    """
    Plot eines Gantt-Diagramms für die Einträge in schedule_log.
    Jeder Eintrag ist ein Dict:
      { "order_id": int, "machine": str, "start_time": float, "finish_time": float }
    """
    # Maschinen in Reihen unterteilen
    machine_list = sorted(list(set(entry["machine"] for entry in schedule_log)))

    # Figure vergrößern
    fig, ax = plt.subplots(figsize=(12, 7))

    # Liste aller Order-IDs
    order_ids = sorted(list(set(entry["order_id"] for entry in schedule_log)))

    # Statt random-Farben => vordefinierte Farbmap (tab20 hat 20 Farben)
    # Falls es mehr als 20 Order-IDs gibt, kann man die Palette mehrfach durchlaufen
    cmap = plt.cm.get_cmap("tab20", len(order_ids))

    # Dictionary: Order-ID -> Farbe
    color_map = {}
    for i, oid in enumerate(order_ids):
        # Damit wir bei sehr vielen Order-IDs nicht out-of-range gehen, modulo benutzen
        color_map[oid] = cmap(i % 20)

    # Pro Maschine => y-Position
    machine_ypos = {m: i for i, m in enumerate(machine_list)}

    # Zum Sammeln der Legendeneinträge
    legend_patches = []

    # Balkenplot
    for entry in schedule_log:
        m = entry["machine"]
        y_pos = machine_ypos[m]
        start = entry["start_time"]
        finish = entry["finish_time"]
        duration = finish - start
        oid = entry["order_id"]

        ax.barh(
            y=y_pos,
            width=duration,
            left=start,
            height=0.5,
            color=color_map[oid],
            edgecolor="black"
        )

        # Nur beschriften, wenn der Balken lang genug ist
        if duration > 50:  # Schwellenwert anpassen
            ax.text(
                x=start + duration / 2,
                y=y_pos,
                s=f"{oid}",  # Kürzeres Label
                va='center',
                ha='center',
                color='white',
                fontsize=8
            )

    # Für die Legende: jedes Order-ID einmal aufnehmen
    for oid in order_ids:
        patch = mpatches.Patch(color=color_map[oid], label=f"Order {oid}")
        legend_patches.append(patch)

    # Legende anzeigen
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.set_yticks(list(machine_ypos.values()))
    ax.set_yticklabels(list(machine_ypos.keys()))
    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")
    ax.set_title("Gantt Chart of Orders")

    # M1 oben lassen, wenn man das so haben möchte:
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()