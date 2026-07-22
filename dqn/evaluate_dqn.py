"""
evaluate_dqn.py — Evaluation and Comparison Pipeline
=====================================================

Usage:
    python evaluate_dqn.py [--episodes 10] [--gui]
"""

import os
import sys

from pathlib import Path
_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_ROOT_DIR))

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from dqn.environment.traffic_env import TrafficEnv
from dqn.models.dqn import DQNAgent
from baselines.q_learning import QLearningAgent
from baselines.ppo import PPOAgent
from config import ENV_CONFIG, DQN_CONFIG, PPO_CONFIG
from dqn.utils import get_logger, TrafficPlotter

logger = get_logger(__name__)

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DQN, Q-Learning, and PPO Comparison Evaluation")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes.")
    parser.add_argument("--gui", action="store_true", default=False, help="Open SUMO-GUI.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for evaluation.")
    return parser.parse_args()


class RandomAgent:
    """Fallback agent that selects random actions."""
    def __init__(self, action_size: int):
        self.action_size = action_size

    def choose_action(self, state: np.ndarray) -> int:
        return int(np.random.randint(self.action_size))


def run_evaluation(agent: Any, env: TrafficEnv, n_episodes: int, seed: int) -> dict:
    """
    Run evaluation episodes and collect traffic KPIs.
    """
    rewards = []
    waiting_times = []
    queues = []
    throughputs = []
    delays = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        
        ep_wait_times = []
        ep_queues = []
        ep_delays = []
        ep_throughput = 0

        while not done:
            # Check if agent has choose_action method
            if hasattr(agent, "choose_action"):
                action = agent.choose_action(obs)
            else:
                action = int(np.random.randint(env.action_space.n))

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            ep_reward += reward
            ep_wait_times.append(info.get("total_waiting_time", 0.0))
            ep_queues.append(info.get("total_queue", 0.0))
            ep_delays.append(info.get("mean_delay", 0.0))
            ep_throughput += info.get("metrics/step_throughput", 0.0)

        rewards.append(ep_reward)
        waiting_times.append(np.mean(ep_wait_times) if ep_wait_times else 0.0)
        queues.append(np.mean(ep_queues) if ep_queues else 0.0)
        delays.append(np.mean(ep_delays) if ep_delays else 0.0)
        throughputs.append(ep_throughput)

    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_waiting_time": float(np.mean(waiting_times)),
        "mean_queue": float(np.mean(queues)),
        "mean_delay": float(np.mean(delays)),
        "throughput": float(np.mean(throughputs))
    }


def main():
    args = get_args()
    
    # Setup directories
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True, parents=True)
    
    plotter = TrafficPlotter(results_dir=str(results_dir))

    # Initialize evaluation environment
    env = TrafficEnv(
        sumocfg_path=ENV_CONFIG["sumocfg_path"],
        tls_id=ENV_CONFIG["tls_id"],
        lane_ids=ENV_CONFIG["lane_ids"],
        delta_t=ENV_CONFIG["delta_t"],
        max_steps=ENV_CONFIG["max_steps"],
        traci_port=ENV_CONFIG["eval_traci_port"],
        use_gui=args.gui,
        seed=args.seed
    )

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. Load DQN Agent ────────────────────────────────────────────────────
    dqn_path = Path("results_dqn/best_model/best_dqn_model.pth")
    if dqn_path.exists():
        logger.info(f"Loading DQN agent from {dqn_path}")
        dqn_agent = DQNAgent(state_size, action_size, DQN_CONFIG, device)
        dqn_agent.load(str(dqn_path))
        dqn_agent.set_evaluation()
    else:
        logger.warning(f"DQN checkpoint not found at {dqn_path}. Using Random Agent for comparison.")
        dqn_agent = RandomAgent(action_size)

    # ── 2. Load Q-Learning Agent ─────────────────────────────────────────────
    ql_path = Path("results_qlearning/best_model/q_model_center.json")
    if ql_path.exists():
        logger.info(f"Loading Q-Learning agent from {ql_path}")
        ql_agent = QLearningAgent(action_size, agent_id="center")
        ql_agent.load(str(ql_path))
        ql_agent.set_evaluation()
    else:
        logger.warning(f"Q-Learning checkpoint not found at {ql_path}. Using Random Agent for comparison.")
        ql_agent = RandomAgent(action_size)

    # ── 3. Load PPO Agent ────────────────────────────────────────────────────
    ppo_path = Path("results_ppo/best_model/best_ppo_model.pth")
    if ppo_path.exists():
        logger.info(f"Loading PPO agent from {ppo_path}")
        ppo_agent = PPOAgent(state_size, action_size, PPO_CONFIG, device)
        ppo_agent.load(str(ppo_path))
        ppo_agent.set_evaluation()
    else:
        logger.warning(f"PPO checkpoint not found at {ppo_path}. Using Random Agent for comparison.")
        ppo_agent = RandomAgent(action_size)

    # ── Run Evaluations ─────────────────────────────────────────────────────
    logger.info(f"Starting evaluation of agents for {args.episodes} episodes...")

    logger.info("Evaluating Q-Learning baseline...")
    ql_results = run_evaluation(ql_agent, env, args.episodes, args.seed)
    logger.info(f"Q-Learning results: {ql_results}")

    logger.info("Evaluating PPO baseline...")
    ppo_results = run_evaluation(ppo_agent, env, args.episodes, args.seed)
    logger.info(f"PPO results: {ppo_results}")

    logger.info("Evaluating DQN Agent...")
    dqn_results = run_evaluation(dqn_agent, env, args.episodes, args.seed)
    logger.info(f"DQN results: {dqn_results}")

    env.close()

    # ── Save Results to CSV ──────────────────────────────────────────────────
    results_df = pd.DataFrame([
        {"Agent": "Q-Learning", **ql_results},
        {"Agent": "PPO", **ppo_results},
        {"Agent": "DQN", **dqn_results}
    ])
    
    results_csv = results_dir / "evaluation_results.csv"
    results_df.to_csv(results_csv, index=False)
    logger.info(f"Saved evaluation results to {results_csv}")

    # ── Display Performance Summary Table ─────────────────────────────────────
    print("\n" + "=" * 80)
    print("                      EVALUATION PERFORMANCE SUMMARY")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("=" * 80 + "\n")

    # ── Plot Visual Performance Comparison Charts ──────────────────────────────
    try:
        plot_path = plotter.plot_model_comparison(dqn_results, ql_results, ppo_results)
        logger.info(f"Saved performance comparison plots to: {plot_path}")
    except Exception as e:
        logger.error(f"Failed to generate comparative plots: {e}")

if __name__ == "__main__":
    from typing import Any
    main()
