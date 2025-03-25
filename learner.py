import time
from stable_baselines3.common.monitor import Monitor
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from gantplot import plot_gantt
from environment import ThreeMachineEnv
import os


starttime = time.time()
# Pfad für Datensätze
orderspath = "C:\\Users\\wolfg\\PycharmProjects\\Prozessoptimierung\\orders\\"
dateien = [f for f in os.listdir(orderspath) if os.path.isfile(os.path.join(orderspath, f))]

# TensorBoard-Logging einrichten
log_dir = "./ppo_tensorboard/"
os.makedirs(log_dir, exist_ok=True)

# Modell speichern
model_save_dir = "./ppo_models/"
os.makedirs(model_save_dir, exist_ok=True)

# Variablen für Bestes Modell
best_reward = float('-inf')
best_model_path = os.path.join(model_save_dir, "best_model.zip")

# Trainingsschleife über alle Datensätze
for idx, datei in enumerate(dateien):
    print(f"\n🔄 Training mit Datensatz {idx + 1}/{len(dateien)}: {datei}")

    # 1) Daten einlesen
    df_orders = pd.read_csv(os.path.join(orderspath, datei))

    # 2) Environment erzeugen
    env = ThreeMachineEnv(df_orders, max_queue_size=5, time_step=1)
    #env = Monitor(env)

    # 3) RL-Modell (PPO) anlegen oder vorheriges Modell laden
    if idx == 0:
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=1e-3, n_steps=256, tensorboard_log=log_dir)
    else:
        model = PPO.load(os.path.join(model_save_dir, f"ppo_latest.zip"), env=env)

    # 4) Modell trainieren und nach 1000 Steps speichern
    model.learn(total_timesteps=1000, reset_num_timesteps=False)

    # 5) Modell speichern nach 1000 Steps
    model.save(os.path.join(model_save_dir, f"ppo_latest.zip"))

    # 6) Modell evaluieren (Reward berechnen)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
    print(f"📈 Durchschnittlicher Reward nach {1000 * (idx + 1)} Steps: {mean_reward:.2f} ± {std_reward:.2f}")

    # 7) Mean Reward in TensorBoard loggen
    model.logger.record("evaluation/mean_reward", mean_reward)
    model.logger.dump(model.num_timesteps)  # Sicherstellen, dass der Wert ins Log geschrieben wird

    # 8) Bestes Modell basierend auf Reward speichern
    if mean_reward > best_reward:
        best_reward = mean_reward
        model.save(best_model_path)
        print(f"🏆 Neues bestes Modell gespeichert mit Reward {best_reward:.2f}")

# 9) Finale Evaluation mit dem besten Modell
print("\n✅ Training abgeschlossen! Evaluierung des besten Modells...")
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


endtime=time.time()

timeused=endtime-starttime

print(str(timeused))
# 10) Gantt-Diagramm plotten und Log-Daten speichern
schedule_data = env.schedule_log

# Konvertiere die Log-Daten in einen DataFrame und speichere sie als CSV
import pandas as pd
df_schedule = pd.DataFrame(schedule_data)
df_schedule.to_csv("schedule_log.csv", index=False)

# Optional: Das Gantt-Diagramm plotten (siehe unten Anpassung in gantplot.py)
from gantplot import plot_gantt
plot_gantt(schedule_data)

print(f"\n🎉 Bestes Modell gespeichert unter: {best_model_path}")
print("📊 TensorBoard Logs unter:", log_dir)