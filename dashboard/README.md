# Smart Traffic Control - Enterprise Dashboard

A premium, SaaS-grade visualization and control center for Multi-Agent Reinforcement Learning (MARL) traffic management algorithms (PPO, DQN, Q-Learning). 

Built with **Next.js 15, Tailwind CSS, Recharts**, and **Flask WebSockets**.

---

## 🏗️ Architecture

```mermaid
graph TD
    subgraph Frontend [Next.js 15 Dashboard]
        UI[React UI Components]
        Charts[Recharts Telemetry]
        Export[CSV/JSON Exporter]
    end

    subgraph Backend [Flask API]
        WS[Flask-SocketIO Server]
        Runners[RL Agent Runners]
        State[Simulation State Manager]
    end

    subgraph Simulation [Eclipse SUMO]
        TraCI[TraCI Interface]
        SUMO_GUI[SUMO Environment]
    end

    subgraph RL Models
        PPO[(PPO Best Model)]
        DQN[(DQN Best Model)]
        QL[(Q-Learning Best Model)]
    end

    %% Connections
    UI <-->|WebSocket Events| WS
    Charts <..|Live Metric Stream| WS
    WS <--> Runners
    Runners <-->|Step Action| TraCI
    TraCI <--> SUMO_GUI
    Runners -.->|Load Weights| PPO
    Runners -.->|Load Weights| DQN
    Runners -.->|Load Weights| QL
    
    classDef default fill:#1a1d27,stroke:#3a3f55,stroke-width:1px,color:#c8d0e0;
    classDef highlight fill:#1e3a8a,stroke:#3b82f6,stroke-width:2px,color:#fff;
    class Frontend highlight
    class Backend highlight
```

## 🚀 Quick Start

### 1. Start the Flask WebSocket Server
This server handles the Python RL models and the TraCI SUMO connection.
```bash
# In the root project directory
pip install -r requirements.txt
python api/app.py
```
*The server will run on `http://localhost:5000`*

### 2. Start the Next.js Dashboard
This runs the React frontend application.
```bash
# Open a new terminal
cd dashboard
npm install
npm run dev
```
*The dashboard will run on `http://localhost:3000`*

## ✨ Features
- **Live Telemetry:** Watch PPO, DQN, and Q-Learning agents interact with the intersection in real-time.
- **Dynamic Control:** Start, Pause, Resume, and Stop the SUMO simulation seamlessly.
- **Enterprise Benchmarks:** Visually compare algorithms using multi-objective Radar charts and Bar charts.
- **Data Export:** Instantly export simulation telemetry and benchmark comparisons to CSV or JSON formats.
- **Accessible & Responsive:** Fully responsive App Router layout with keyboard navigability and ARIA compliances.
