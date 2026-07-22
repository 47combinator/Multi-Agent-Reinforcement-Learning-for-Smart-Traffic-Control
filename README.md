# Multi-Agent Reinforcement Learning for Smart Traffic Control

> A unified research project comparing three RL algorithms — **PPO**, **DQN**, and **Q-Learning** — for adaptive traffic signal control using SUMO and TraCI.

---

## Project Structure

```
├── ppo/               ← Proximal Policy Optimization agent (Pratyush)
├── dqn/               ← Deep Q-Network agent (teammate)
├── qlearning/         ← Tabular Q-Learning agent (teammate)
├── sumo_env/          ← Shared SUMO simulation files (single intersection)
├── benchmark/         ← Unified evaluation: runs all 3 agents, same seeds
│   ├── benchmark.py
│   └── plots/
├── assets/            ← Report figures
└── PPO_TRAINING_REPORT.md
```

---

## Shared Environment

All three agents use the **same** SUMO intersection network:
- `sumo_env/single_intersection.net.xml` — 4-way intersection
- `sumo_env/single_intersection.rou.xml` — NS: 400 veh/hr | EW: 250 veh/hr
- Traffic Light ID: **`center`**
- State space: 8-dimensional (queue + waiting time per lane)
- Action space: 4 discrete phases (NS Green, EW Green, NS Extend, EW Extend)

---

## Running Each Agent

### PPO
```bash
python ppo/main.py --mode train --timesteps 500000
python ppo/main.py --mode evaluate --episodes 10
```

### Q-Learning
```bash
python qlearning/main.py --mode train
python qlearning/main.py --mode evaluate
```

### DQN
```bash
cd dqn
python train_dqn.py
python evaluate_dqn.py --episodes 10
```

---

## Unified Benchmark (All 3 Together)

After all agents are trained, compare them side-by-side:

```bash
python benchmark/benchmark.py --episodes 10 --seed 42
```

This will:
1. Load the best saved model from each agent
2. Run all on the same SUMO simulation with identical seeds
3. Output a comparison table and save a bar chart to `benchmark/plots/`

---

## Results Summary

Evaluated over 10 episodes per agent on the same SUMO simulation with identical seeds (seed=42).

| Algorithm          | Mean Wait Time (s) | Mean Queue Length | Std Reward |
|:-------------------|:-------------------|:-----------------|:-----------|
| **Q-Learning**     | **519.0** 🏆       | **26.53** 🏆     | ±10.94     |
| **DQN (Double+Dueling)** | 522.2         | 26.74            | ±4.53      |
| **Fixed-Time**     | 575.8              | 27.30            | ±4.68      |
| **PPO**            | 593.2              | 30.23            | ±28.08     |

> **Note:** Raw reward values are not directly comparable across agents because PPO and Q-Learning
> use a normalized, clipped reward function while DQN uses an unnormalized variant.
> The traffic metrics above (wait time, queue length) are objective and comparable.

---

## Requirements

1. **Install SUMO** from [sumo.dlr.de](https://sumo.dlr.de/docs/Downloads.php) and set the `SUMO_HOME` environment variable.
2. Install Python dependencies:
```bash
pip install -r requirements.txt
```
