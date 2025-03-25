import time
from stable_baselines3.common.monitor import Monitor
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from gantplot import plot_gantt
from environment import ThreeMachineEnv
import os

output_file = 'C:\\Users\\wolfg\\PycharmProjects\\Prozessoptimierung\\utilization_output.xlsx'
model_save_dir = "./ppo_models/"

orderspath = "C:\\Users\\wolfg\\PycharmProjects\\Prozessoptimierung\\Otherorders\\"
dateien = [f for f in os.listdir(orderspath) if os.path.isfile(os.path.join(orderspath, f))]

best_model_path = os.path.join(model_save_dir, "best_model.zip")

# Öffne den ExcelWriter, um alle Ergebnisse in einer Datei zu speichern
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    current_row = 0

    for idx, datei in enumerate(dateien):
        # Lese die Auftragsdaten ein
        df_orders = pd.read_csv(os.path.join(orderspath, datei))
        # Erstelle die Umgebung mit den Auftragsdaten
        env = ThreeMachineEnv(df_orders, max_queue_size=5, time_step=1)

        # Lade das beste Modell
        best_model = PPO.load(best_model_path, env=env)
        obs = env.reset()
        done = False
        step_count = 0

        while not done:
            action, _states = best_model.predict(obs, deterministic=True)
            print(f"Step = {step_count}, Action = {action}")
            obs, reward, done, info = env.step(action)
            env.render()
            step_count += 1

        # Erstelle den Zeitplan-DataFrame aus dem Environment-Log
        schedule_data = env.schedule_log
        df_schedule = pd.DataFrame(schedule_data)

        # Berechne die Bearbeitungszeit für jeden Auftrag
        df_schedule["processing_time"] = df_schedule["finish_time"] - df_schedule["start_time"]

        # Berechne die Maschinenauslastung
        utilization = df_schedule.groupby("machine")["processing_time"].sum().reset_index()
        available_time = df_schedule.groupby("machine").apply(
            lambda x: x["finish_time"].max() - x["start_time"].min()
        ).reset_index(name="available_time")
        utilization = utilization.merge(available_time, on="machine")
        utilization["utilization_percentage"] = (utilization["processing_time"] / utilization["available_time"]) * 100

        # Bestimme die maximale Bearbeitungszeit als Information
        bearbeitzeit = "Max Bearbeitungszeit: " + str(max(df_schedule["finish_time"]))

        # Ermittlung der verspäteten Aufträge: Vergleich finish_time mit deadline
        if "deadline" in df_schedule.columns:
            verspätete_auftraege = df_schedule[df_schedule["finish_time"] > df_schedule["deadline"]]
            count_verspaetet = len(verspätete_auftraege)
            if "order_id" in verspätete_auftraege.columns:
                order_ids = verspätete_auftraege["order_id"].tolist()
            else:
                order_ids = []
        else:
            print("Keine Deadline-Information in df_schedule gefunden.")
            count_verspaetet = 0
            order_ids = []

        # Schreibe die Auslastungsdaten in das Excel-Dokument
        utilization.to_excel(writer, sheet_name='Sheet1', startrow=current_row, index=False)
        worksheet = writer.sheets['Sheet1']
        target_row = current_row + len(utilization) + 1

        # Schreibe die maximale Bearbeitungszeit
        worksheet.cell(row=target_row + 1, column=1, value=bearbeitzeit)

        # Schreibe die Anzahl der verspäteten Aufträge
        worksheet.cell(row=target_row + 3, column=1, value="Verspätete Aufträge Anzahl:")
        worksheet.cell(row=target_row + 3, column=2, value=count_verspaetet)

        # Schreibe die Order IDs der verspäteten Aufträge
        worksheet.cell(row=target_row + 4, column=1, value="Order IDs verspätet:")
        worksheet.cell(row=target_row + 4, column=2, value=str(order_ids))

        # Aktualisiere current_row für den nächsten Durchlauf
        current_row += len(utilization) + 6

        # Speichere den Zeitplan auch als CSV-Datei
        df_schedule.to_csv("scheduler_log" + str(idx) + ".csv", index=False)
