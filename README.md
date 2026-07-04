# Multi-Agent Reinforcement Learning for Smart Traffic Control

A research project comparing state-of-the-art Deep Reinforcement Learning agents on a smart traffic signal control task. 

This repository contains a modular, production-ready implementation of a **Deep Q-Network (DQN)** agent incorporating advanced reinforcement learning features, alongside **Q-Learning** and **Proximal Policy Optimization (PPO)** baselines. All agents are trained on the exact same traffic intersection simulated in **SUMO (Simulation of Urban Mobility)** using the **TraCI** API and a custom **Gymnasium** environment wrapper.

---

## 🏗️ Project Structure

The project is organized in a modular structure:

```
project/
│
├── env/
│   ├── __init__.py
│   ├── traffic_env.py       # Core Gymnasium environment wrapper for SUMO
│   ├── state_extractor.py   # Extracts and normalizes observation features from TraCI
│   └── reward_calculator.py  # Calculates composite delta-based rewards
│
├── models/
│   ├── __init__.py
│   ├── network.py           # PyTorch deep neural networks (Standard & Dueling)
│   ├── replay_buffer.py     # Experience Replay Buffers (Uniform & Prioritized PER)
│   └── dqn.py               # Deep Q-Network Agent core logic
│
├── baselines/
│   ├── q_learning.py        # Tabular Q-learning agent baseline
│   ├── q_learning_trainer.py # Training coordinator for Q-learning
│   ├── ppo.py               # Actor-Critic PPO agent baseline
│   └── ppo_trainer.py       # Training coordinator for PPO
│
├── train_dqn.py             # Main entry point to train the DQN agent
├── evaluate_dqn.py          # Evaluates DQN, Q-Learning, and PPO to generate comparison charts
├── utils.py                 # Reproducibility seeds, logger settings, and plotting scripts
├── config.py                # Hyperparameters, reward weights, and environment configs
└── requirements.txt         # Python package dependencies
```

---

## 🚦 Environment Details

The intersection models a 4-way single-lane junction (North, South, East, West incoming lanes).

### State Representation (22 Dimensions)
All observation state values are normalized to `[0.0, 1.0]` to ensure neural network stability:
1. **Vehicle Counts** (4 values): Current vehicle count per incoming lane.
2. **Queue Lengths** (4 values): Halting vehicle counts per incoming lane (speed < 0.1 m/s).
3. **Waiting Times** (4 values): Accumulated waiting times of stopped vehicles per incoming lane.
4. **Current Phase One-Hot** (4 values): One-hot encoding of the active traffic light phase.
5. **Phase Elapsed Time** (1 value): Duration current phase has been green (normalized).
6. **Episode Elapsed Time** (1 value): Fraction of simulation episode steps completed (normalized).

### Action Space (4 Discrete Choices)
* **0**: Green for North-South direction.
* **1**: Green for East-West direction.
* **2**: Extend current green phase.
* **3**: Switch active traffic signal phase.

*Note: Transitioning between NS and EW triggers a mandatory 3-second yellow light transition internally to prevent collision warnings and emulate realistic traffic signal safety constraints.*

### Reward Function
Calculated at the step-level using a delta-based composite formulation:
$$R = -\alpha \cdot \Delta W - \beta \cdot \Delta Q + \gamma \cdot T$$
* **$\Delta W$**: Change in total vehicle waiting time since the last step.
* **$\Delta Q$**: Change in total halting queue length since the last step.
* **$T$**: Step-level throughput (number of vehicles that crossed the intersection).
* Configurable weights ($\alpha=0.4$, $\beta=0.3$, $\gamma=0.3$) are defined in `config.py`.

---

## ⚡ DQN Features Implemented

* **Double DQN**: Decouples action selection (online network) from evaluation (target network) to prevent Q-value overestimation.
* **Dueling DQN Architecture**: Splits the value estimation into separate State Value ($V(s)$) and Action Advantage ($A(s, a)$) streams, resulting in faster convergence.
* **Soft Target Update**: Slowly updates the target network weights using interpolation ($\theta^- \leftarrow \tau \theta + (1 - \tau) \theta^-$) for smoother learning.
* **Epsilon-Greedy Exploration**: Exponentially decays exploration rate ($\epsilon$) from 1.0 to 0.01.
* **Huber Loss**: Smooth $L1$ loss that acts quadratic for small errors and linear for large errors (robust to outliers).
* **Prioritized Experience Replay (PER)**: (Optional) Biases sampling toward transitions with higher temporal-difference (TD) errors using a Sum-Tree data structure.
* **Learning Rate Scheduler**: Periodically decays learning rate to stabilize policy convergence.
* **Gradient Clipping**: Clips gradients at 1.0 to prevent exploding gradients.

---

## 🚀 Getting Started

### 1. Prerequisites
You must have Python 3.8+ and **SUMO (Simulation of Urban Mobility)** installed on your machine.
Make sure you set the `SUMO_HOME` environment variable pointing to your SUMO installation folder:

* **Windows PowerShell**:
  ```powershell
  $env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
  ```
* **Windows CMD**:
  ```cmd
  set SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo
  ```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Training
To train the main **DQN** agent:
```bash
python train_dqn.py
```
*Checkpoints are saved under `results_dqn/checkpoints/` and the best model is stored at `results_dqn/best_model/best_dqn_model.pth`.*

To resume training from a checkpoint:
```bash
python train_dqn.py --resume results_dqn/checkpoints/checkpoint_ep200.pth
```

### 4. Running Baseline Trainers (Optional)
To train **Q-Learning** or **PPO** from scratch to generate comparison data:
* **Q-Learning**:
  ```python
  # Run a python script or call baselines/q_learning_trainer.py
  ```
* **PPO**:
  ```python
  # Run a python script or call baselines/ppo_trainer.py
  ```

### 5. Evaluation and Comparison
Evaluate all models side-by-side and output performance plots:
```bash
python evaluate_dqn.py --episodes 10
```
This loads the best models from `results_dqn/`, `results_qlearning/`, and `results_ppo/`, runs them for 10 evaluation episodes, creates a performance summary table in CSV format (`results/evaluation_results.csv`), and generates comparative plots under `results/plots/model_comparison.png`.

---

## 📈 Logging and TensorBoard
All training logs (rewards, queue lengths, losses, speeds, exploration rates) are automatically pushed to TensorBoard. You can view them by running:
```bash
tensorboard --logdir results_dqn/tensorboard_logs
```
Then navigate to `http://localhost:6006` in your browser.
