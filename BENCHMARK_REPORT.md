# Multi-Agent RL Traffic Control — Final Benchmark Report

This report summarizes the final unified evaluation of all three reinforcement learning agents (PPO, DQN, and Tabular Q-Learning) alongside a Fixed-Time baseline. All agents were evaluated on the **exact same SUMO simulation** (4-way intersection) across 10 identical random seeds (seed=42).

---

## Benchmark Results — Traffic Metrics

The primary comparison uses **objective traffic metrics** (waiting time, queue length) rather than raw reward values, since the reward functions differ across agent implementations.

> **Lower** Wait Time and Queue Length = **Better** performance.

| Algorithm              | Mean Wait Time (s) | Mean Queue Length | Reward Std Dev |
|:-----------------------|:------------------:|:----------------:|:--------------:|
| **Q-Learning (Tabular)** | **519.0** | **26.53** | +/-10.94 |
| **DQN (Double+Dueling)** | 522.2 | 26.74 | +/-4.53 |
| **Fixed-Time Baseline**  | 575.8 | 27.30 | +/-4.68 |
| **PPO (SB3)**            | 593.2 | 30.23 | +/-28.08 |

*Evaluated over 10 episodes, 720 steps per episode. Wait times and queues are per-step averages across all 4 incoming lanes.*

### Improvement Over Fixed-Time Baseline

| Algorithm | Wait Time Reduction | Queue Reduction |
|:----------|:-------------------:|:---------------:|
| Q-Learning | **-9.9%** | **-2.8%** |
| DQN | **-9.3%** | **-2.1%** |
| PPO | +3.0% (worse) | +10.7% (worse) |

---

## Analysis & Key Findings

### 1. Value-Based Methods Dominate

Both **Q-Learning** and **DQN** significantly outperformed the Fixed-Time baseline and PPO on objective traffic metrics. They achieved nearly identical performance (~520s wait, ~26.5 queue), suggesting that the value-based approach is fundamentally better suited for this discrete-action traffic control task.

### 2. DQN is the Most Consistent Agent

While Q-Learning edged out DQN on raw metrics, DQN exhibited the **lowest variance** (std dev +/-4.53 vs Q-Learning's +/-10.94). This indicates DQN's learned policy is more robust across different traffic seeds — an important property for real-world deployment.

### 3. PPO Underperformed on Discrete Actions

PPO was designed for continuous action spaces. Forcing it into a 4-discrete-action domain makes it harder for the policy gradient to converge efficiently compared to value-based methods. PPO's high variance (+/-28.08) further confirms instability in this setting.

### 4. Reward Function Caveat

> **Important:** The raw reward values are **not directly comparable** across agents.
> PPO and Q-Learning use a normalized, clipped reward (output in [-1, 1] per step),
> while DQN uses an unnormalized variant. This is why we report traffic metrics
> as the primary comparison.

---

## Agent Implementation Details

| Property | PPO | DQN | Q-Learning |
|:---------|:----|:----|:-----------|
| **Framework** | Stable-Baselines3 | Custom PyTorch | Custom (tabular) |
| **Architecture** | Actor-Critic MLP | Double Dueling DQN | Q-Table |
| **State Space** | 8-dim continuous | 8-dim continuous | 18-dim discretized |
| **Action Space** | 4 discrete | 4 discrete | 4 discrete |
| **Training Episodes** | 500K timesteps | 1000 episodes | ~5000 episodes |
| **Reward Clipping** | [-1, 1] | None | [-1, 1] |
| **Key Technique** | Clipped surrogate objective | Experience replay + soft target updates | Epsilon-greedy decay |

---

## Reproducibility

| Parameter | Value |
|:----------|:------|
| SUMO Version | 1.20.0 |
| Python | 3.9+ |
| Random Seed | 42 |
| Episodes | 10 |
| Steps/Episode | 720 (= 3600 simulated seconds) |
| Traffic Flow | NS: 400 veh/hr, EW: 250 veh/hr |
| Intersection | Single 4-way, ID: `center` |
