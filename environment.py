import gym
from gym import spaces
import numpy as np
import pandas as pd
from collections import deque

class ThreeMachineEnv(gym.Env):
    """
    Einfaches Scheduling-Env mit 3 Maschinen (M1, M2, M3).
    Wir erfassen zusätzlich Start- und Endzeiten jedes Auftrags auf jeder Maschine,
    um später ein Gantt-Diagramm plotten zu können.
    """

    def __init__(self, orders_df, max_queue_size=10, time_step=1):
        super(ThreeMachineEnv, self).__init__()

        self.orders_df = orders_df.copy()
        self.orders_df.Deadline_days=self.orders_df.Deadline_days*480
        self.time_step = time_step

        # Für das Gantt-Diagramm: (order_id, machine, start_time, finish_time)
        self.schedule_log = []

        # max_queue_size: maximale Anzahl an Aufträgen pro Warteschlange (für das Action-Space-Design)
        self.max_queue_size = max_queue_size

        self.machines = {
            'M1': {'is_busy': False, 'current_order': None, 'time_to_finish': 0, 'start_time': None},
            'M2': {'is_busy': False, 'current_order': None, 'time_to_finish': 0, 'start_time': None},
            'M3': {'is_busy': False, 'current_order': None, 'time_to_finish': 0, 'start_time': None},
        }

        self.queue = {
            'M1': deque(),
            'M2': deque(),
            'M3': deque(),
        }

        self.current_time = 0
        self.completed_orders = []
        self.done = False

        # Observation: 7-dim [time_to_finish_M1, time_to_finish_M2, time_to_finish_M3,
        #                     queue_len_M1, queue_len_M2, queue_len_M3, current_time]
        self.observation_space = spaces.Box(
            low=0,
            high=1e6,
            shape=(7,),
            dtype=np.float32
        )

        # Action: MultiDiscrete([max_queue_size+1, max_queue_size+1, max_queue_size+1])
        # -> pro Maschine ein Wert in [0..max_queue_size], 0 = "Nichts tun"
        self.action_space = spaces.MultiDiscrete([self.max_queue_size + 1] * 3)

        self.reset()

    def reset(self):
        """
        Setzt das Env auf einen Startzustand zurück:
        - Maschinen sind frei
        - Warteschlangen werden geleert und 15 zufällige Aufträge zugewiesen
        - schedule_log wird geleert
        """
        self.current_time = 0
        self.done = False
        self.completed_orders.clear()
        self.schedule_log.clear()

        for m in self.machines:
            self.machines[m]['is_busy'] = False
            self.machines[m]['current_order'] = None
            self.machines[m]['time_to_finish'] = 0
            self.machines[m]['start_time'] = None

        for m in self.queue:
            self.queue[m].clear()

        # Beispiel: 15 zufällige Aufträge aus dem DataFrame
        #subset = self.orders_df.sample(n=6, replace=False)
        subset = self.orders_df #.sample(n=6, replace=False)

        for _, row in subset.iterrows():
            seq = row['OperationSequence'].split('->')
            first_machine = seq[0]
            self.queue[first_machine].append(row['OrderID'])

        return self._get_obs()

    def step(self, action):
        """
        Führt einen Zeitschritt aus:
          1) Update der laufenden Aufträge (Zeitfortschritt)
          2) Aktionen (wenn Maschinen idle)
          3) Belohnung & done-Bedingung
          4) Rückgabe (obs, reward, done, info)
        """
        # Setze Strafe zurück
        self.invalid_action_penalty = 0

        # 1) Update laufende Aufträge
        orders_finished_this_step = []
        for m in self.machines:
            if self.machines[m]['is_busy']:
                self.machines[m]['time_to_finish'] -= self.time_step
                if self.machines[m]['time_to_finish'] <= 0:
                    finished_order = self.machines[m]['current_order']
                    start_time = self.machines[m]['start_time']

                    # Maschine wieder frei
                    self.machines[m]['is_busy'] = False
                    self.machines[m]['current_order'] = None
                    self.machines[m]['start_time'] = None
                    self.machines[m]['time_to_finish'] = 0

                    # Gantt-Log
                    self.schedule_log.append({
                        "order_id": finished_order,
                        "machine": m,
                        "start_time": start_time,
                        "finish_time": self.current_time
                    })

                    # Auftrag weiterleiten oder fertig
                    self.move_to_next_machine(finished_order, m)
                    orders_finished_this_step.append(finished_order)

        # 2) Aktionen (wenn Maschinen idle)
        aM1, aM2, aM3 = action
        self._handle_action_for_machine('M1', aM1)
        self._handle_action_for_machine('M2', aM2)
        self._handle_action_for_machine('M3', aM3)

        # 3) Zeit +1
        self.current_time += self.time_step

        # Reward: -Anzahl offener Aufträge, +10 pro fertiggestellten Auftrag
        not_finished = len(self.orders_df) - len(self.completed_orders)
        reward = -not_finished
        reward += 10 * len(orders_finished_this_step)
        # Füge Strafen für ungültige Aktionen hinzu
        reward += self.invalid_action_penalty
        for mach in self.machines:
            if self.machines[m]['is_busy']:
                reward+=100
            else:
                reward-=100

        # done? Wenn alle Aufträge (aus df) abgearbeitet sind
        if len(self.completed_orders) == len(self.orders_df):
            self.done = True

        return self._get_obs(), reward, self.done, {}

    def _handle_action_for_machine(self, machine_name, action_val):
        if self.machines[machine_name]['is_busy']:
            return

        queue_size = len(self.queue[machine_name])

        # Falls Queue nicht leer, Aktion=0 => forced pick = 1 (den ersten Auftrag)
        if action_val == 0 and queue_size > 0:
            action_val = 1

        # Falls Queue leer, Aktion != 0 => man könnte strafen, ...
        if queue_size == 0:
            # out-of-range-Aktion oder "Pick Auftrag" macht keinen Sinn
            # Hier ggf. Strafe
            return

        # Jetzt behandeln wir out-of-range
        idx = action_val - 1
        if idx >= queue_size:
            # Clampen: wähle das letzte Element
            idx = queue_size - 1
        if idx < 0:
            # falls action_val == 0, aber wir haben das oben schon behandelt
            idx = 0

        chosen_order_id = list(self.queue[machine_name])[idx]
        self.queue[machine_name].remove(chosen_order_id)
        self.start_order(machine_name, chosen_order_id)

    def start_order(self, machine_name, order_id):
        """
        Maschine 'machine_name' beginnt den Auftrag 'order_id' (Rüstzeit + Prozesszeit).
        """
        row = self.orders_df[self.orders_df['OrderID'] == order_id].iloc[0]
        ruest = row[f"{machine_name}_Ruest"]
        proc = row[f"{machine_name}_Proc"]
        total_time = int(ruest + proc)

        self.machines[machine_name]['is_busy'] = True
        self.machines[machine_name]['current_order'] = order_id
        self.machines[machine_name]['time_to_finish'] = total_time
        self.machines[machine_name]['start_time'] = self.current_time

    def move_to_next_machine(self, order_id, current_machine):
        """
        Schiebt Auftrag in die Queue der nächsten Maschine laut OperationSequence.
        Wenn keine weitere Operation, gilt der Auftrag als vollständig.
        """
        row = self.orders_df[self.orders_df['OrderID'] == order_id].iloc[0]
        seq = row['OperationSequence'].split('->')
        idx_machine = seq.index(current_machine)

        if idx_machine < len(seq) - 1:
            next_machine = seq[idx_machine + 1]
            self.queue[next_machine].append(order_id)
        else:
            self.completed_orders.append(order_id)

    def _get_obs(self):
        """
        Beobachtung (State) in Form eines 7D-Vektors:
          - time_to_finish je Maschine,
          - Queue-Längen je Maschine,
          - current_time
        """
        obs = np.array([
            float(self.machines['M1']['time_to_finish']),
            float(self.machines['M2']['time_to_finish']),
            float(self.machines['M3']['time_to_finish']),
            float(len(self.queue['M1'])),
            float(len(self.queue['M2'])),
            float(len(self.queue['M3'])),
            float(self.current_time)
        ], dtype=np.float32)
        return obs

    def render(self, mode='human'):
        """
        Konsolenausgabe zur Übersicht.
        """
        print(f"Time={self.current_time}")
        for m in ['M1', 'M2', 'M3']:
            print(f"  {m}: busy={self.machines[m]['is_busy']}, "
                  f"order={self.machines[m]['current_order']}, "
                  f"time_remaining={self.machines[m]['time_to_finish']} min, "
                  f"queue={list(self.queue[m])}")
        print(f"  Completed orders: {len(self.completed_orders)}")
        #print("corders)"+str(len(self.completed_orders)))
        #print("dforders"+str(len(self.orders_df)))
        #print(self.orders_df)
