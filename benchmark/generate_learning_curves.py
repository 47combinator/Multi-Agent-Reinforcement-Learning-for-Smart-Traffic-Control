"""
generate_learning_curves.py — Plot training curves for all 3 agents
====================================================================
Reads training history CSVs and plots reward vs episode for all agents
on the same axes for a direct comparison.

Usage:
    python benchmark/generate_learning_curves.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = ROOT / "benchmark" / "plots"
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

COLORS = {"DQN": "#4ecb71", "PPO": "#4f8ef7", "Q-Learning": "#f7a84f"}

def smooth(values, window=20):
    """Simple moving average for smoothing noisy curves."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid')


def load_dqn_history():
    """Load DQN training history CSV."""
    csv_path = ROOT / "dqn" / "results_dqn" / "training_history.csv"
    if not csv_path.exists():
        print(f"[WARN] DQN history not found at {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    return df


def load_ppo_history():
    """Load PPO evaluation results CSV."""
    csv_path = ROOT / "ppo" / "results" / "evaluation_results.csv"
    if not csv_path.exists():
        print(f"[WARN] PPO history not found at {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    return df


def load_qlearning_history():
    """Try to find Q-Learning training history."""
    candidates = [
        ROOT / "qlearning" / "results" / "training_history.csv",
        ROOT / "qlearning" / "results" / "training_log.csv",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    print("[WARN] Q-Learning training history not found")
    return None


def plot_wait_time_comparison():
    """Generate a wait-time learning curve comparison plot."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Training Progression - All RL Agents", fontsize=16, fontweight="bold")

    # === Plot 1: Wait Time over Training ===
    ax1 = axes[0]
    ax1.set_title("Average Waiting Time During Training")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Avg Waiting Time (s)")
    ax1.grid(True, alpha=0.4)

    # DQN
    dqn = load_dqn_history()
    if dqn is not None and "avg_waiting_time" in dqn.columns:
        episodes = dqn["episode"].values
        waits = dqn["avg_waiting_time"].values
        smoothed = smooth(waits, window=30)
        offset = len(waits) - len(smoothed)
        ax1.plot(episodes[offset:], smoothed, color=COLORS["DQN"], linewidth=2, label="DQN", alpha=0.9)
        ax1.fill_between(episodes, waits, alpha=0.1, color=COLORS["DQN"])

    # PPO
    ppo = load_ppo_history()
    if ppo is not None:
        wait_col = None
        for col in ["mean_wait", "avg_waiting_time", "waiting_time", "mean_waiting_time"]:
            if col in ppo.columns:
                wait_col = col
                break
        if wait_col:
            episodes_ppo = np.arange(len(ppo))
            waits_ppo = ppo[wait_col].values
            ax1.plot(episodes_ppo, waits_ppo, color=COLORS["PPO"], linewidth=2, label="PPO", alpha=0.9)

    # Q-Learning
    ql = load_qlearning_history()
    if ql is not None:
        wait_col = None
        for col in ["avg_waiting_time", "mean_wait", "waiting_time"]:
            if col in ql.columns:
                wait_col = col
                break
        if wait_col:
            episodes_ql = ql.index.values if "episode" not in ql.columns else ql["episode"].values
            waits_ql = ql[wait_col].values
            smoothed_ql = smooth(waits_ql, window=30)
            offset_ql = len(waits_ql) - len(smoothed_ql)
            ax1.plot(episodes_ql[offset_ql:], smoothed_ql, color=COLORS["Q-Learning"], linewidth=2, label="Q-Learning", alpha=0.9)

    ax1.legend(loc="upper right", fontsize=10)

    # === Plot 2: Reward over Training ===
    ax2 = axes[1]
    ax2.set_title("Episode Reward During Training")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Reward")
    ax2.grid(True, alpha=0.4)

    if dqn is not None and "reward" in dqn.columns:
        episodes = dqn["episode"].values
        rewards = dqn["reward"].values
        smoothed = smooth(rewards, window=30)
        offset = len(rewards) - len(smoothed)
        ax2.plot(episodes[offset:], smoothed, color=COLORS["DQN"], linewidth=2, label="DQN (smoothed)", alpha=0.9)
        ax2.fill_between(episodes, rewards, alpha=0.08, color=COLORS["DQN"])

    if ppo is not None:
        rew_col = None
        for col in ["mean_reward", "reward", "episode_reward"]:
            if col in ppo.columns:
                rew_col = col
                break
        if rew_col:
            episodes_ppo = np.arange(len(ppo))
            rewards_ppo = ppo[rew_col].values
            ax2.plot(episodes_ppo, rewards_ppo, color=COLORS["PPO"], linewidth=2, label="PPO", alpha=0.9)

    if ql is not None:
        rew_col = None
        for col in ["reward", "episode_reward", "mean_reward"]:
            if col in ql.columns:
                rew_col = col
                break
        if rew_col:
            episodes_ql = ql.index.values if "episode" not in ql.columns else ql["episode"].values
            rewards_ql = ql[rew_col].values
            smoothed_ql = smooth(rewards_ql, window=30)
            offset_ql = len(rewards_ql) - len(smoothed_ql)
            ax2.plot(episodes_ql[offset_ql:], smoothed_ql, color=COLORS["Q-Learning"], linewidth=2, label="Q-Learning (smoothed)", alpha=0.9)

    ax2.legend(loc="lower right", fontsize=10)
    ax2.annotate("Note: Reward scales differ across agents\n(DQN unnormalized, PPO/Q-Learning clipped [-1,1])",
                 xy=(0.02, 0.02), xycoords="axes fraction", fontsize=8,
                 color="#8890a8", style="italic")

    plt.tight_layout()
    save_path = PLOTS_DIR / "training_curves_comparison.png"
    plt.savefig(save_path)
    plt.close()
    print(f"[OK] Training curves saved -> {save_path}")


if __name__ == "__main__":
    plot_wait_time_comparison()
