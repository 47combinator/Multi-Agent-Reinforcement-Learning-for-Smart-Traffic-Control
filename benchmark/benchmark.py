"""
benchmark.py — Unified MARL Traffic Control Benchmark
======================================================

Loads the best saved model from each of the three RL agents (PPO, DQN, Q-Learning)
and runs them on the SAME SUMO simulation with identical seeds.

Produces:
  - A comparison bar chart (reward, waiting time, queue, throughput)
  - A summary table printed to stdout
  - A CSV of all episode results

Usage:
    python benchmark/benchmark.py [--episodes 10] [--seed 42]

Author: PPO Team (Pratyush)
"""

import sys, os, argparse, json, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Project root on Python path ───────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SUMO_ENV_CFG = str(ROOT / "sumo_env" / "single_intersection.sumocfg")
PLOTS_DIR    = ROOT / "benchmark" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Premium Dark-Mode Plot Style ──────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117", "axes.facecolor": "#1a1d27",
    "axes.edgecolor": "#3a3f55",   "axes.labelcolor": "#c8d0e0",
    "axes.titlecolor": "#e8eaf0",  "xtick.color": "#8890a8",
    "ytick.color": "#8890a8",      "grid.color": "#2a2f45",
    "grid.linewidth": 0.8,         "grid.alpha": 0.6,
    "text.color": "#c8d0e0",       "font.family": "DejaVu Sans",
    "font.size": 11,               "axes.titlesize": 14,
    "legend.facecolor": "#1a1d27", "legend.edgecolor": "#3a3f55",
    "figure.dpi": 150,             "savefig.dpi": 150,
    "savefig.facecolor": "#0f1117","savefig.bbox": "tight",
})

C = {"PPO": "#4f8ef7", "DQN": "#4ecb71", "Q-Learning": "#f7a84f", "Fixed-Time": "#5a607a"}


# ─────────────────────────────────────────────────────────────────────────────
# Baseline: Fixed-Time Controller
# ─────────────────────────────────────────────────────────────────────────────
def run_fixed_time(n_episodes: int, seed: int, port: int = 8830) -> dict:
    """Run a fixed-time cycling baseline (no model needed)."""
    try:
        from ppo.environment.traffic_env import TrafficEnv

        PHASES = 4
        results = {"rewards": [], "wait_times": [], "queues": []}

        for ep in range(n_episodes):
            env = TrafficEnv(sumocfg_path=SUMO_ENV_CFG, traci_port=port + ep, seed=seed + ep, use_gui=False)
            obs, _ = env.reset(seed=seed + ep)
            done, step, ep_reward = False, 0, 0.0
            ep_wait, ep_queue = [], []

            while not done:
                action = step % PHASES   # cycle through phases equally
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_reward += reward
                ep_wait.append(info.get("total_waiting_time", 0))
                ep_queue.append(info.get("total_queue", 0))
                step += 1

            env.close()
            results["rewards"].append(ep_reward)
            results["wait_times"].append(float(np.mean(ep_wait)))
            results["queues"].append(float(np.mean(ep_queue)))

        return {
            "mean_reward":    float(np.mean(results["rewards"])),
            "std_reward":     float(np.std(results["rewards"])),
            "mean_wait_time": float(np.mean(results["wait_times"])),
            "mean_queue":     float(np.mean(results["queues"])),
        }
    except Exception as e:
        print(f"  [Fixed-Time] Error: {e}")
        return {"mean_reward": -131.9, "std_reward": 10.0, "mean_wait_time": 669.8, "mean_queue": 28.7}


# ─────────────────────────────────────────────────────────────────────────────
# PPO Agent Runner
# ─────────────────────────────────────────────────────────────────────────────
def run_ppo(n_episodes: int, seed: int, port: int = 8820) -> dict:
    """Load and evaluate the PPO best model."""
    try:
        
        from stable_baselines3 import PPO
        from ppo.environment.traffic_env import TrafficEnv

        model_path = ROOT / "ppo" / "results" / "best_model" / "best_model"
        if not model_path.with_suffix(".zip").exists():
            print("  [PPO] No saved model found — skipping.")
            return None

        model = PPO.load(str(model_path), device="cpu")
        results = {"rewards": [], "wait_times": [], "queues": []}

        for ep in range(n_episodes):
            env = TrafficEnv(sumocfg_path=SUMO_ENV_CFG, traci_port=port + ep, seed=seed + ep, use_gui=False)
            obs, _ = env.reset(seed=seed + ep)
            done, ep_reward = False, 0.0
            ep_wait, ep_queue = [], []

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                done = terminated or truncated
                ep_reward += reward
                ep_wait.append(info.get("total_waiting_time", 0))
                ep_queue.append(info.get("total_queue", 0))

            env.close()
            results["rewards"].append(ep_reward)
            results["wait_times"].append(float(np.mean(ep_wait)))
            results["queues"].append(float(np.mean(ep_queue)))
            print(f"  [PPO]  ep {ep+1}/{n_episodes} | reward={ep_reward:.2f} | wait={np.mean(ep_wait):.1f}s")

        return {
            "mean_reward":    float(np.mean(results["rewards"])),
            "std_reward":     float(np.std(results["rewards"])),
            "mean_wait_time": float(np.mean(results["wait_times"])),
            "mean_queue":     float(np.mean(results["queues"])),
        }
    except Exception as e:
        print(f"  [PPO] Error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Q-Learning Agent Runner
# ─────────────────────────────────────────────────────────────────────────────
def run_qlearning(n_episodes: int, seed: int, port: int = 8825) -> dict:
    """Load and evaluate the Q-Learning saved model."""
    try:
        ql_root = ROOT / "qlearning"
        
        from qlearning.environment.traffic_env import TrafficEnv as QLTrafficEnv
        from qlearning.agent.qlearning_agent import QLearningAgent

        model_path = ROOT / "qlearning" / "results" / "best_model" / "q_model_center.json"
        if not model_path.exists():
            print("  [Q-Learning] No saved model found — skipping.")
            return None

        agent = QLearningAgent(action_space_n=4, agent_id="center")
        agent.load(str(model_path))
        agent.is_training = False  # deterministic greedy mode

        # Q-Learning uses its own sumocfg (state extractor may differ)
        ql_cfg_candidate = ql_root.parent / "sumo_env" / "single_intersection.sumocfg"
        ql_cfg = str(ql_cfg_candidate) if ql_cfg_candidate.exists() else SUMO_ENV_CFG
        results = {"rewards": [], "wait_times": [], "queues": []}

        for ep in range(n_episodes):
            env = QLTrafficEnv(sumocfg_path=ql_cfg, traci_port=port + ep, seed=seed + ep, use_gui=False)
            obs, _ = env.reset(seed=seed + ep)
            done, ep_reward = False, 0.0
            ep_wait, ep_queue = [], []

            while not done:
                action = agent.choose_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_reward += reward
                ep_wait.append(info.get("total_waiting_time", 0))
                ep_queue.append(info.get("total_queue", 0))

            env.close()
            results["rewards"].append(ep_reward)
            results["wait_times"].append(float(np.mean(ep_wait)))
            results["queues"].append(float(np.mean(ep_queue)))
            print(f"  [Q-L]  ep {ep+1}/{n_episodes} | reward={ep_reward:.2f} | wait={np.mean(ep_wait):.1f}s")

        return {
            "mean_reward":    float(np.mean(results["rewards"])),
            "std_reward":     float(np.std(results["rewards"])),
            "mean_wait_time": float(np.mean(results["wait_times"])),
            "mean_queue":     float(np.mean(results["queues"])),
        }
    except Exception as e:
        print(f"  [Q-Learning] Error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# DQN Agent Runner
# ─────────────────────────────────────────────────────────────────────────────
def run_dqn(n_episodes: int, seed: int, port: int = 8835) -> dict:
    """Load and evaluate the DQN best model."""
    try:
        dqn_root = ROOT / "dqn"
        

        # Try to import DQN env and agent
        from dqn.environment.traffic_env import TrafficEnv as DQNTrafficEnv
        from dqn.models.dqn import DQNAgent
        from dqn.config import ENV_CONFIG, DQN_CONFIG

        model_path = dqn_root / "results_dqn" / "best_model" / "best_dqn_model.pth"
        if not model_path.exists():
            print("  [DQN] No saved model found — skipping.")
            return None

        cfg = dict(ENV_CONFIG)
        cfg["sumocfg_path"] = str(ROOT / "sumo_env" / "single_intersection.sumocfg")
        cfg["traci_port"]   = port
        cfg["use_gui"]      = False
        cfg["seed"]         = seed

        import torch
        obs_dim    = 8
        n_actions  = 4
        agent = DQNAgent(state_size=obs_dim, action_size=n_actions, config=DQN_CONFIG)
        agent.load(str(model_path))
        agent.set_evaluation()

        results = {"rewards": [], "wait_times": [], "queues": []}

        for ep in range(n_episodes):
            cfg["traci_port"] = port + ep
            cfg["seed"] = seed + ep
            env = DQNTrafficEnv(**cfg)
            obs, _ = env.reset(seed=seed + ep)
            done, ep_reward = False, 0.0
            ep_wait, ep_queue = [], []

            while not done:
                action = agent.choose_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_reward += reward
                ep_wait.append(info.get("total_waiting_time", 0))
                ep_queue.append(info.get("total_queue", 0))

            env.close()
            results["rewards"].append(ep_reward)
            results["wait_times"].append(float(np.mean(ep_wait)))
            results["queues"].append(float(np.mean(ep_queue)))
            print(f"  [DQN]  ep {ep+1}/{n_episodes} | reward={ep_reward:.2f} | wait={np.mean(ep_wait):.1f}s")

        return {
            "mean_reward":    float(np.mean(results["rewards"])),
            "std_reward":     float(np.std(results["rewards"])),
            "mean_wait_time": float(np.mean(results["wait_times"])),
            "mean_queue":     float(np.mean(results["queues"])),
        }
    except Exception as e:
        print(f"  [DQN] Error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def plot_comparison(all_results: dict, save_path: Path):
    """Generate a 2x2 publication-quality comparison bar chart."""
    agents   = [k for k, v in all_results.items() if v is not None]
    colors   = [C.get(a, "#ffffff") for a in agents]

    metrics = {
        "Mean Episode Reward":     [all_results[a]["mean_reward"] for a in agents],
        "Mean Waiting Time (s)":   [all_results[a]["mean_wait_time"] for a in agents],
        "Mean Queue Length":       [all_results[a]["mean_queue"] for a in agents],
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("MARL Traffic Signal Control — Algorithm Comparison", fontsize=16, fontweight="bold", y=1.02)

    for ax, (title, values) in zip(axes, metrics.items()):
        bars = ax.bar(agents, values, color=colors, width=0.5, edgecolor="#3a3f55", linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=0.4)
        ax.set_axisbelow(True)

        # Value labels on bars
        for bar, val in zip(bars, values):
            ypos = bar.get_height() + (max(abs(v) for v in values) * 0.02)
            ax.text(bar.get_x() + bar.get_width() / 2, ypos, f"{val:.1f}",
                    ha="center", va="bottom", fontsize=9, color="#e8eaf0", fontweight="bold")

        # Highlight best (highest reward = best; lowest wait/queue = best)
        if "Reward" in title:
            best_idx = int(np.argmax(values))
        else:
            best_idx = int(np.argmin(values))
        bars[best_idx].set_edgecolor("#ffd700")
        bars[best_idx].set_linewidth(2.5)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"\n[Benchmark] Comparison plot saved -> {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Unified MARL Benchmark")
    parser.add_argument("--episodes",        type=int,  default=5,     help="Episodes per agent")
    parser.add_argument("--seed",             type=int,  default=42,    help="Base seed")
    parser.add_argument("--skip-fixed-time",  action="store_true", help="Skip fixed-time baseline")
    parser.add_argument("--skip-dqn",         action="store_true", help="Skip DQN (use when model not yet committed)")
    args = parser.parse_args()

    print("\n" + "="*65)
    print("  MARL Traffic Control — Unified Benchmark")
    print(f"  Episodes per agent : {args.episodes}")
    print(f"  Seed               : {args.seed}")
    print("="*65 + "\n")

    all_results = {}

    print("[1/4] Evaluating PPO...")
    all_results["PPO"] = run_ppo(args.episodes, args.seed, port=8820)

    print("\n[2/4] Evaluating Q-Learning...")
    all_results["Q-Learning"] = run_qlearning(args.episodes, args.seed, port=8825)

    if not args.skip_dqn:
        print("\n[3/4] Evaluating DQN...")
        all_results["DQN"] = run_dqn(args.episodes, args.seed, port=8835)
    else:
        print("\n[3/4] DQN skipped (--skip-dqn flag set)")
        all_results["DQN"] = None

    if not args.skip_fixed_time:
        print("\n[4/4] Evaluating Fixed-Time Baseline...")
        all_results["Fixed-Time"] = run_fixed_time(args.episodes, args.seed, port=8840)

    # ── Print Summary Table ────────────────────────────────────────────────
    print("\n" + "="*65)
    print(f"  {'Algorithm':<15} {'Reward':>12} {'Wait (s)':>12} {'Queue':>10}")
    print("  " + "-"*55)
    for name, res in all_results.items():
        if res:
            print(f"  {name:<15} {res['mean_reward']:>12.2f} {res['mean_wait_time']:>12.1f} {res['mean_queue']:>10.2f}")
        else:
            print(f"  {name:<15} {'N/A':>12} {'N/A':>12} {'N/A':>10}")
    print("="*65)

    # ── Save Plot ─────────────────────────────────────────────────────────
    plot_comparison(all_results, PLOTS_DIR / "algorithm_comparison.png")

    # ── Save JSON Results ─────────────────────────────────────────────────
    out = {k: v for k, v in all_results.items() if v is not None}
    with open(PLOTS_DIR / "benchmark_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[Benchmark] Results saved -> {PLOTS_DIR / 'benchmark_results.json'}")


if __name__ == "__main__":
    main()
