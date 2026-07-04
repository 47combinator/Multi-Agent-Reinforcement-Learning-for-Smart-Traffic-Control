# Q-Learning Traffic Signal Controller

A standalone, completely independent Q-Learning Reinforcement Learning architecture for intelligent traffic signal control in SUMO. 

## Project Overview

This project implements a custom Q-Learning tabular agent to optimize traffic light phases. Unlike continuous state-space continuous state-space models that rely on deep neural networks, this architecture discretizes the continuous state space extracted from the environment and maps it to a discrete action space using a Q-Table. 

It explores traffic patterns using an epsilon-greedy policy and iteratively updates its action-value estimates using the Bellman Equation to minimize traffic wait times and queue lengths.

## Folder Structure

```
qlearning/
??? agent/
?   ??? qlearning_agent.py      # Core Q-Learning Agent and Bellman Update logic
??? config/
?   ??? qlearning.yaml          # Configuration (hyperparameters, paths, episodes)
??? environment/
?   ??? reward_calculator.py    # Calculates reward (negative wait time/queue)
?   ??? state_extractor.py      # Extracts environment state via TraCI
?   ??? traffic_env.py          # Custom Gymnasium-like SUMO Environment
??? evaluation/
?   ??? q_evaluator.py          # Benchmarks RL against fixed-time SUMO baseline
??? results/                    # Generated output directories
?   ??? best_model/             # Saved optimum Q-Table model JSON
?   ??? checkpoints/            # Intermediate model JSONs
?   ??? logs/                   # monitor.csv metrics
?   ??? tensorboard_logs/       # TensorBoard events
??? utils/                      # Logging and reproducibility utilities
??? visualization/              # Metric plotting tools
??? main.py                     # Entry point for training and evaluation
```

## Installation

Ensure you have Python 3.9+ and SUMO installed.

1. **Install SUMO**: Download and install [Eclipse SUMO](https://www.eclipse.org/sumo/). Ensure the `SUMO_HOME` environment variable is set to your installation directory (e.g., `C:\Program Files (x86)\Eclipse\Sumo`).
2. **Install Python dependencies**:
   ```bash
   pip install traci numpy pyyaml matplotlib tensorboard
   ```

*(Note: `stable-baselines3` and `torch` are NOT required for this standalone project).*

## Running Training

To train the Q-Learning model from scratch, execute:
```bash
python qlearning/main.py --mode train
```

To override the default number of episodes (1500) defined in the config:
```bash
python qlearning/main.py --mode train --train-episodes 500
```

## Running Evaluation

To evaluate your trained model deterministically against the default SUMO fixed-time baseline:
```bash
python qlearning/main.py --mode evaluate --episodes 10
```

*By default, evaluation will attempt to load the optimal model from `qlearning/results/best_model/q_model_center.json`. You can specify a custom model path using `--model`.*

## Monitoring Performance

During or after training, you can visualize the learning curves using TensorBoard:
```bash
tensorboard --logdir qlearning/results/tensorboard_logs
```
