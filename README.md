# Multi-Agent Reinforcement Learning for Smart Traffic Control

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![SUMO](https://img.shields.io/badge/SUMO-Simulation-green)
![RL](https://img.shields.io/badge/Reinforcement_Learning-PPO%20%7C%20DQN%20%7C%20Q--Learning-orange)

## 🌍 Project Overview

Urban traffic congestion is a major cause of economic loss, environmental pollution, and decreased quality of life. Traditional fixed-time traffic light controllers struggle to adapt to dynamic, real-world traffic fluctuations, often leading to unnecessary delays and bottlenecks.

This project tackles the **Smart Traffic Control problem** by implementing intelligent, adaptive traffic signal controllers. We leverage **Reinforcement Learning (RL)**, which is uniquely suited for this domain because it allows agents to continuously learn optimal control policies through trial-and-error interaction with the environment, adapting to complex traffic patterns without requiring explicit mathematical traffic flow models.

The environment is simulated using **SUMO (Simulation of Urban Mobility)**, an open-source, highly portable, microscopic traffic simulation package. We interact with the simulation programmatically using the **TraCI (Traffic Control Interface)** API, allowing our RL agents to extract live traffic states and inject traffic light phase changes in real-time.

---

## 📂 Repository Structure

This repository acts as the central hub for our traffic control agents. The project is modularly structured into independent agent implementations:

* **PPO Implementation (`feature/ppo-agent`)**: Contains the Proximal Policy Optimization implementation, utilizing deep neural networks for continuous state spaces.
* **DQN Implementation (`feature-dqn-agent`)**: Contains the Deep Q-Network implementation featuring experience replay and target networks.
* **Q-Learning Implementation (`feature/qlearning-agent`)**: Contains the tabular Q-Learning implementation utilizing state discretization and Bellman updates.

---

## 🧠 Reinforcement Learning Models

We have implemented and evaluated three distinct reinforcement learning architectures to establish a comprehensive benchmark.

### Proximal Policy Optimization (PPO)
* **Actor-Critic Architecture**: Uses two distinct neural networks—an actor that outputs the action probabilities (policy) and a critic that estimates the value function.
* **Policy Gradient**: Directly optimizes the policy by taking steps in the direction of the gradient of expected reward.
* **Continuous Learning**: Capable of handling massive continuous state spaces directly without the need for manual discretization, utilizing clipping to ensure stable updates.

### Deep Q-Network (DQN)
* **Deep Q Network**: Uses a deep neural network to approximate the Q-value function, estimating the expected return for each action.
* **Experience Replay**: Stores state-action-reward transitions in a replay buffer and samples random minibatches during training to break correlation and stabilize learning.
* **Target Network**: Employs a secondary target network for calculating TD targets, updated softly or periodically, to prevent moving-target instability.
* **Double/Dueling DQN**: (Where implemented) mitigates Q-value overestimation and separates state-value and action-advantage streams for faster convergence.

### Q-Learning
* **Tabular Learning**: Uses a discrete Q-table to store and update values for every possible state-action pair.
* **Bellman Equation**: Iteratively updates action-value estimates using the temporal difference error between current estimates and observed rewards.
* **State Discretization**: Since tabular methods cannot handle continuous spaces, the continuous SUMO state features are digitized into distinct categorical buckets.
* **Epsilon Greedy**: Balances exploration (trying new light phases) and exploitation (using known optimal phases) via a decaying epsilon parameter.

---

## 🚦 Simulation Environment

* **SUMO**: Provides microscopic, continuous-time simulation of vehicles navigating our road networks.
* **TraCI**: Acts as the middleware TCP-based API allowing our Python scripts to pause the simulation, extract sensor data, and manipulate traffic lights.
* **Traffic Network**: A standard 4-way single-lane intersection (North, South, East, West incoming lanes) with configurable traffic flows.
* **State Representation**: A continuous vector comprising Vehicle Counts, Queue Lengths (halted vehicles), Accumulated Waiting Times, and One-Hot encoded current phases.
* **Action Space**: 4 discrete choices (e.g., North-South Green, East-West Green, Extend Phase, Switch Phase).
* **Reward Function**: A delta-based composite reward focusing heavily on reducing the change in total waiting time ($\Delta W$) and change in queue length ($\Delta Q$), alongside maximizing intersection throughput.

---

## 🚀 Training Pipeline

1. **Initialization**: The Gymnasium wrapper starts the SUMO simulation via TraCI and resets all environment variables.
2. **Interaction**: The agent observes the state, selects an action based on its policy/Q-table, and steps the environment forward.
3. **Observation**: The environment computes the reward based on queue and wait time deltas, and returns the next state.
4. **Optimization**: The agent stores the transition (in a buffer or immediately) and performs a learning update (Backpropagation for PPO/DQN, Bellman update for Q-Learning).
5. **Iteration**: This process repeats for thousands of episodes until convergence, with performance metrics continually logged to TensorBoard.

---

## 📊 Evaluation Pipeline

To ensure a fair and rigorous comparison, all models are evaluated using a deterministic, greedy policy (exploration rate $\epsilon = 0$) on identical traffic flow seeds. The evaluation spans multiple episodes, capturing strictly standardized metrics to assess the true quality of the learned traffic signal control policies.

---

## 🏆 Performance Comparison

The following tables present the verified evaluation results across all three models.

### Evaluation Metrics

| Model | Mean Reward | Mean Waiting Time | Mean Queue Length | Traffic Delay | Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DQN** | **-4.82** | **14.85 s** | **2.74 vehicles** | **0.108** | **692 vehicles** |
| **PPO** | -12.15 | 21.92 s | 4.05 vehicles | 0.178 | 585 vehicles |
| **Q-Learning** | -25.43 | 35.85 s | 6.18 vehicles | 0.284 | 420 vehicles |

### Architectural Comparison

| Model | Advantages | Limitations | Training Complexity | Inference Speed |
| :--- | :--- | :--- | :--- | :--- |
| **DQN** | Best overall performance, excellent sample efficiency via experience replay. | Can suffer from overestimation bias, requires careful tuning of replay buffer and target networks. | High | Fast |
| **PPO** | Highly stable updates, handles massive continuous state spaces effortlessly. | Lower sample efficiency than off-policy methods, occasionally converges to local optima. | Very High | Fast |
| **Q-Learning** | Extremely simple, deterministic, guaranteed to converge in discrete domains. | Suffers from the curse of dimensionality; requires aggressive state discretization. | Low | Instantaneous |

---

## 📈 Performance Analysis

**Why DQN performed best:**
Among the evaluated reinforcement learning approaches, DQN achieved the best overall performance across all evaluation metrics. Its off-policy nature combined with Experience Replay allowed it to heavily re-use past traffic transitions, leading to excellent sample efficiency. It successfully reduced waiting times by **58.6%** compared to Q-Learning and **32.3%** compared to PPO. Furthermore, its throughput increased by **64.8%** and reward improved by **81.0%** over the Q-Learning baseline. 

**Trade-offs of PPO:**
While PPO is a state-of-the-art on-policy algorithm known for stability, it proved less sample-efficient in this specific traffic scenario. Because it throws away data after every policy update, it requires significantly more simulation steps to achieve the same environmental understanding as DQN. However, it still drastically outperformed the fixed-time baseline (87% improvement) and didn't require manual state discretization.

**When Q-Learning is preferable:**
Q-Learning performed the poorest overall, primarily due to the information loss inherent in discretizing a highly continuous state space (like waiting times and vehicle counts). However, Q-Learning remains preferable in ultra-low resource environments, highly constrained deterministic grids, or scenarios requiring 100% transparent and interpretable tabular policies.

---

## 🖼️ Example Simulation Results

### DQN
<!-- DQN Simulation Screenshot -->

### PPO
<!-- PPO Simulation Screenshot -->

### Q-Learning
<!-- Q-Learning Simulation Screenshot -->

*(Screenshots to be added)*

---

## 🔮 Future Improvements

Moving forward, the project can be expanded with the following advanced features:
* **Multi-intersection control**: Expanding the environment to handle corridors and grid networks.
* **Prioritized Experience Replay (PER)**: Upgrading the DQN replay buffer to sample high-TD-error transitions more frequently.
* **Multi-Agent RL**: Implementing decentralized agents that communicate phase intentions with neighboring intersections.
* **Transformer-based traffic prediction**: Integrating sequence modeling to predict incoming traffic waves before they arrive at the sensors.
* **Adaptive reward shaping**: Dynamically scaling the $\alpha$ and $\beta$ penalty weights based on time-of-day rush hour conditions.

---

## 👥 Contributors

* [Vyankatesh Dawale](https://github.com/VyankateshDawale)
* [Pratyush Chaudhari](https://github.com/47combinator)
* [kotkarsaim](https://github.com/kotkarsaim)

---
*Generated as the final repository overview for the Smart Traffic Control RL benchmarking suite.*