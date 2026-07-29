import sys
import os
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SUMO_ENV_CFG = str(ROOT / "sumo_env" / "single_intersection.sumocfg")

def run_agent_stream(model_type, n_episodes, seed, use_gui, state):
    """
    Generator that yields metrics step-by-step for the given model type.
    """
    if model_type == "Fixed-Time":
        yield from stream_fixed_time(n_episodes, seed, use_gui, state)
    elif model_type == "PPO":
        yield from stream_ppo(n_episodes, seed, use_gui, state)
    elif model_type == "DQN":
        yield from stream_dqn(n_episodes, seed, use_gui, state)
    elif model_type == "Q-Learning":
        yield from stream_qlearning(n_episodes, seed, use_gui, state)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def stream_fixed_time(n_episodes, seed, use_gui, state, port=8830):
    from ppo.environment.traffic_env import TrafficEnv

    PHASES = 4
    for ep in range(n_episodes):
        if state["should_stop"]: break
        
        env = TrafficEnv(sumocfg_path=SUMO_ENV_CFG, traci_port=port + ep, seed=seed + ep, use_gui=use_gui)
        obs, _ = env.reset(seed=seed + ep)
        done, step, ep_reward = False, 0, 0.0

        while not done:
            if state["should_stop"]: break
            
            action = step % PHASES
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            step += 1
            
            # Send metric update
            yield {
                "episode": ep + 1,
                "step": step,
                "reward": float(ep_reward),
                "wait_time": float(info.get("total_waiting_time", 0)),
                "queue_length": float(info.get("total_queue", 0)),
                "throughput": float(info.get("total_throughput", 0)),
                "action": int(action)
            }

        env.close()

def stream_ppo(n_episodes, seed, use_gui, state, port=8820):
    from stable_baselines3 import PPO
    from ppo.environment.traffic_env import TrafficEnv

    model_path = ROOT / "ppo" / "results" / "best_model" / "best_model"
    if not model_path.with_suffix(".zip").exists():
        raise FileNotFoundError("PPO model not found.")

    model = PPO.load(str(model_path), device="cpu")

    for ep in range(n_episodes):
        if state["should_stop"]: break
        
        env = TrafficEnv(sumocfg_path=SUMO_ENV_CFG, traci_port=port + ep, seed=seed + ep, use_gui=use_gui)
        obs, _ = env.reset(seed=seed + ep)
        done, step, ep_reward = False, 0, 0.0

        while not done:
            if state["should_stop"]: break
            
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            ep_reward += reward
            step += 1
            
            yield {
                "episode": ep + 1,
                "step": step,
                "reward": float(ep_reward),
                "wait_time": float(info.get("total_waiting_time", 0)),
                "queue_length": float(info.get("total_queue", 0)),
                "throughput": float(info.get("total_throughput", 0)),
                "action": int(action)
            }

        env.close()

def stream_qlearning(n_episodes, seed, use_gui, state, port=8825):
    ql_root = ROOT / "qlearning"
    from qlearning.environment.traffic_env import TrafficEnv as QLTrafficEnv
    from qlearning.agent.qlearning_agent import QLearningAgent

    model_path = ROOT / "qlearning" / "results" / "best_model" / "q_model_center.json"
    if not model_path.exists():
        raise FileNotFoundError("Q-Learning model not found.")

    agent = QLearningAgent(action_space_n=4, agent_id="center")
    agent.load(str(model_path))
    agent.is_training = False

    ql_cfg_candidate = ql_root.parent / "sumo_env" / "single_intersection.sumocfg"
    ql_cfg = str(ql_cfg_candidate) if ql_cfg_candidate.exists() else SUMO_ENV_CFG

    for ep in range(n_episodes):
        if state["should_stop"]: break
        
        env = QLTrafficEnv(sumocfg_path=ql_cfg, traci_port=port + ep, seed=seed + ep, use_gui=use_gui)
        obs, _ = env.reset(seed=seed + ep)
        done, step, ep_reward = False, 0, 0.0

        while not done:
            if state["should_stop"]: break
            
            action = agent.choose_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            step += 1
            
            yield {
                "episode": ep + 1,
                "step": step,
                "reward": float(ep_reward),
                "wait_time": float(info.get("total_waiting_time", 0)),
                "queue_length": float(info.get("total_queue", 0)),
                "throughput": float(info.get("total_throughput", 0)),
                "action": int(action)
            }

        env.close()

def stream_dqn(n_episodes, seed, use_gui, state, port=8835):
    dqn_root = ROOT / "dqn"
    from dqn.environment.traffic_env import TrafficEnv as DQNTrafficEnv
    from dqn.models.dqn import DQNAgent
    from dqn.config import ENV_CONFIG, DQN_CONFIG

    model_path = dqn_root / "results_dqn" / "best_model" / "best_dqn_model.pth"
    if not model_path.exists():
        raise FileNotFoundError("DQN model not found.")

    cfg = dict(ENV_CONFIG)
    cfg["sumocfg_path"] = str(ROOT / "sumo_env" / "single_intersection.sumocfg")
    cfg["traci_port"]   = port
    cfg["use_gui"]      = use_gui
    cfg["seed"]         = seed

    obs_dim    = 8
    n_actions  = 4
    agent = DQNAgent(state_size=obs_dim, action_size=n_actions, config=DQN_CONFIG)
    agent.load(str(model_path))
    agent.set_evaluation()

    for ep in range(n_episodes):
        if state["should_stop"]: break
        
        cfg["traci_port"] = port + ep
        cfg["seed"] = seed + ep
        env = DQNTrafficEnv(**cfg)
        obs, _ = env.reset(seed=seed + ep)
        done, step, ep_reward = False, 0, 0.0

        while not done:
            if state["should_stop"]: break
            
            action = agent.choose_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            step += 1
            
            yield {
                "episode": ep + 1,
                "step": step,
                "reward": float(ep_reward),
                "wait_time": float(info.get("total_waiting_time", 0)),
                "queue_length": float(info.get("total_queue", 0)),
                "throughput": float(info.get("total_throughput", 0)),
                "action": int(action)
            }

        env.close()
