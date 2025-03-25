# RLJobScheduling
This project implements a custom reinforcement learning environment for job scheduling optimization on a production floor with three machines. The environment simulates the processing of orders, each requiring operations on up to three different machines (M1, M2, M3). It was designed for research in reinforcement learning with applications to scheduling problems and production planning. The repository also includes scripts for training an RL agent using PPO (via Stable Baselines 3), visualizing scheduling results with Gantt charts and machine utilization dashboards (using Streamlit and Plotly), and running scheduled jobs with evaluation and reporting.


## Three-Machine Scheduling Environment

This repository contains a custom reinforcement learning environment for optimizing job scheduling on a production floor with three machines. It was developed to simulate real-world scheduling challenges and to experiment with reinforcement learning approaches for process optimization.

### Overview

The project includes:
- **Custom Gym Environment:** Implements a scheduling simulation where each order follows a defined sequence of operations on three machines (M1, M2, M3). The environment logs start and finish times to enable detailed performance analysis and Gantt chart plotting.
- **Visualization Dashboard:** A Streamlit app (`visualizier.py`) that displays:
  - Machine utilization (donut charts).
  - Order lists per machine.
  - A Gantt chart for visualizing the schedule.
  - Overall production overview.
- **RL Training Scripts:** 
  - `learner.py` trains a PPO agent (using Stable Baselines 3) on multiple order datasets.
  - `scheduler.py` applies the trained model to new datasets, evaluates performance, and exports scheduling metrics and logs to Excel.
- **Gantt Plotting:** Utility functions (and an external module `gantplot.py`) to plot Gantt charts from schedule logs.


