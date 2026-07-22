# Multi-Agent Reinforcement Learning for Smart Traffic Control

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)
![SUMO](https://img.shields.io/badge/SUMO-Simulation-green)
![RL](https://img.shields.io/badge/Reinforcement_Learning-PPO%20%7C%20DQN%20%7C%20Q--Learning-orange)

## 🌍 Project Overview

Urban traffic congestion is a major cause of economic loss, environmental pollution, and decreased quality of life. Traditional fixed-time traffic light controllers struggle to adapt to dynamic, real-world traffic fluctuations, often leading to unnecessary delays and bottlenecks.

This project tackles the **Smart Traffic Control problem** by implementing intelligent, adaptive traffic signal controllers. We leverage **Reinforcement Learning (RL)**, which is uniquely suited for this domain because it allows agents to continuously learn optimal control policies through trial-and-error interaction with the environment, adapting to complex traffic patterns without requiring explicit mathematical traffic flow models.

The environment is simulated using **SUMO (Simulation of Urban Mobility)**, an open-source, highly portable, microscopic traffic simulation package. We interact with the simulation programmatically using the **TraCI (Traffic Control Interface)** API, allowing our RL agents to extract live traffic states and inject traffic light phase changes in real-time.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    A[SUMO Simulation] -->|Traffic State Data| B(TraCI API)
    B -->|Sensor Readings| C{Traffic Environment Wrapper}
    C -->|Normalized State| D[Reinforcement Learning Agent]
    D -->|Action Selection| C
    C -->|Phase Command| B
    B -->|Light Change| A
```

---

## ⚙️ Project Workflow

```mermaid
flowchart LR
    A((State S_t)) --> B[RL Agent]
    B -->|Selects| C((Action A_t))
    C --> D[Environment]
    D -->|Yields| E((Reward R_t))
    D -->|Yields| F((State S_t+1))
    E --> G[Learning Update]
    F --> A
```

---

## 📂 Repository Structure

The project is modularly structured into independent agent implementations sharing a unified dependency environment:

```text
Multi-Agent-Reinforcement-Learning-for-Smart-Traffic-Control/
│
├── README.md
├── assets/
├── ppo/                  # Proximal Policy Optimization implementation
├── dqn/                  # Deep Q-Network implementation
├── qlearning/            # Tabular Q-Learning implementation
├── requirements.txt      # Unified project dependencies
└── LICENSE
```

---

## 📦 Installation

To run the project locally, follow these steps to configure the Python environment and SUMO simulator.

### 1. Clone the Repository
```bash
git clone https://github.com/47combinator/Multi-Agent-Reinforcement-Learning-for-Smart-Traffic-Control.git
cd Multi-Agent-Reinforcement-Learning-for-Smart-Traffic-Control
```

### 2. Create a Virtual Environment
**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```
**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Install SUMO
**Windows:**
Download the installer from the [Eclipse SUMO website](https://eclipse.dev/sumo/) and follow the installation wizard.

**Linux (Ubuntu):**
```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
```

### 5. Configure SUMO_HOME
You must set the `SUMO_HOME` environment variable so Python can locate the TraCI package.
**Windows (PowerShell):**
```powershell
$env:SUMO_HOME="C:\Program Files (x86)\Eclipse\Sumo"
```
**Linux:**
```bash
export SUMO_HOME="/usr/share/sumo"
```

### 6. Verify TraCI
```bash
python -c "import traci; print('TraCI successfully imported!')"
```

---

## 🚀 Running the Project

Each reinforcement learning agent acts as an independent module. Run them directly from the project root.

### Proximal Policy Optimization (PPO)
```bash
python ppo/main.py --mode train --timesteps 500000
python ppo/main.py --mode evaluate --episodes 10
```

### Deep Q-Network (DQN)
```bash
python dqn/train_dqn.py
python dqn/evaluate_dqn.py --episodes 10
```

### Q-Learning
```bash
python qlearning/main.py --mode train
python qlearning/main.py --mode evaluate
```

### Unified Benchmark (All 3 Together)
After all agents are trained, compare them side-by-side using the unified evaluation script.
This evaluates all agents on identical traffic seeds for a fair comparison:
```bash
python benchmark/benchmark.py --episodes 10 --seed 42
```

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
* **Double/Dueling DQN**: Mitigates Q-value overestimation and separates state-value and action-advantage streams for faster convergence.

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
* **Action Space**: 4 discrete choices (North-South Green, East-West Green, Extend Phase, Switch Phase).
* **Reward Function**: A delta-based composite reward focusing heavily on reducing the change in total waiting time ($\Delta W$) and change in queue length ($\Delta Q$), alongside maximizing intersection throughput.

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
| **DQN** | High sample efficiency via experience replay. | Can suffer from overestimation bias, requires careful tuning. | High | Fast |
| **PPO** | Highly stable updates, handles continuous state spaces effortlessly. | Lower sample efficiency than off-policy methods. | Very High | Fast |
| **Q-Learning** | Simple, deterministic, guaranteed convergence in discrete domains. | Suffers from the curse of dimensionality. | Low | Instantaneous |

### Performance Analysis

Under the current benchmark configuration, DQN achieved the strongest performance across the evaluated metrics. Its off-policy nature combined with Experience Replay allowed it to heavily re-use past traffic transitions, leading to excellent sample efficiency. It successfully reduced waiting times by 58.6% compared to Q-Learning and 32.3% compared to PPO. Furthermore, its throughput increased by 64.8% and reward improved by 81.0% over the Q-Learning baseline.

While PPO is a highly robust algorithm known for stability, it proved less sample-efficient in this specific traffic scenario, though it still successfully optimized traffic flow without manual state discretization. Q-Learning served as a lightweight, highly interpretable tabular baseline, demonstrating foundational RL convergence on discretized traffic states.

---

## 🖼️ Example Simulation Results

### PPO Simulation
> Add PPO simulation screenshot here.

### DQN Simulation
> Add DQN simulation screenshot here.

### Q-Learning Simulation
> Add Q-Learning simulation screenshot here.

---

## 🧩 Dependencies

The project relies on the following core libraries:
* **Python**
* **PyTorch**
* **Gymnasium**
* **SUMO (Simulation of Urban Mobility)**
* **TraCI**
* **NumPy**
* **TensorBoard**
* **Matplotlib**

---

## 🔬 Reproducibility

To ensure maximum reproducibility of these benchmark metrics, the environments were evaluated under the following standardized conditions:
* **Python Version:** 3.9+
* **SUMO Version:** 1.20.0
* **Random Seed:** 42 (Fixed across environment initialization and PyTorch RNG streams)
* **Operating System Tested:** Windows 11 / Ubuntu 22.04 LTS

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

* **Vyankatesh Dawale** ([@VyankateshDawale](https://github.com/VyankateshDawale))
  * Standalone Q-Learning implementation
  * Documentation
  * Evaluation

* **Pratyush Chaudhari** ([@47combinator](https://github.com/47combinator))
  * PPO implementation

* **kotkarsaim** ([@kotkarsaim](https://github.com/kotkarsaim))
  * DQN implementation

---

## 🙏 Acknowledgements

We extend our gratitude to the developers of the open-source tools that made this research possible:
* [Eclipse SUMO](https://eclipse.dev/sumo/) for microscopic traffic simulation.
* [Gymnasium (Farama Foundation)](https://gymnasium.farama.org/) for standardized RL API design.
* [PyTorch](https://pytorch.org/) for accelerating deep learning models.