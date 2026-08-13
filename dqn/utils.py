"""
utils.py — Helper Utilities and Plotting
========================================

Architecture Role:
    Provides utility functions for seeding, logging, and generating
    academic-grade, dark-mode research plots comparing Q-Learning, PPO, and DQN.
"""

import random
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib

# Set non-interactive backend for server environments
matplotlib.use("Agg")

# ── Dark Mode Style Configuration ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor": "#1a1d27",
    "axes.edgecolor": "#3a3f55",
    "axes.labelcolor": "#c8d0e0",
    "axes.titlecolor": "#e8eaf0",
    "xtick.color": "#8890a8",
    "ytick.color": "#8890a8",
    "grid.color": "#2a2f45",
    "grid.linewidth": 0.8,
    "grid.alpha": 0.6,
    "text.color": "#c8d0e0",
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "legend.facecolor": "#1a1d27",
    "legend.edgecolor": "#3a3f55",
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.facecolor": "#0f1117",
    "savefig.bbox": "tight",
})

# Custom Premium HSL Palettes
C_BLUE = "#4f8ef7"
C_GREEN = "#4ecb71"
C_RED = "#f76f6f"
C_ORANGE = "#f7a84f"
C_PURPLE = "#a04ff7"
C_CYAN = "#4fcdf7"
C_GRAY = "#5a607a"


def set_global_seed(seed: int):
    """
    Ensure reproducible runs by seeding all source generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_logger(name: str) -> logging.Logger:
    """
    Formulate a standardized logging stream.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger


def _ema_smooth(values: List[float], alpha: float = 0.08) -> np.ndarray:
    """Exponential Moving Average smoothing."""
    if len(values) == 0:
        return np.array([])
    smoothed = np.zeros(len(values))
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


class TrafficPlotter:
    """
    Renders high-quality visualization diagrams and performance plots.
    """

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def plot_dqn_training(
        self,
        episodes: List[int],
        rewards: List[float],
        losses: List[float],
        wait_times: List[float],
        queues: List[float],
        epsilons: List[float]
    ) -> str:
        """
        Generate training dashboard plots for DQN containing:
        1. Episode Reward with EMA
        2. Loss curve
        3. Mean Waiting Time
        4. Mean Queue Length & Exploration Rate
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("Deep Q-Network (DQN) Training Dashboard", fontsize=18, fontweight="bold", y=0.98)

        # 1. Reward
        ax = axes[0, 0]
        ax.plot(episodes, rewards, color=C_BLUE, alpha=0.3, label="Raw Episode Reward")
        if len(rewards) > 5:
            ax.plot(episodes, _ema_smooth(rewards), color=C_BLUE, linewidth=2.5, label="Reward (EMA)")
        ax.set_title("Episode Rewards")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.grid(True)
        ax.legend()

        # 2. Loss
        ax = axes[0, 1]
        valid_losses = [l for l in losses if l > 0]
        valid_eps = [episodes[i] for i, l in enumerate(losses) if l > 0]
        if valid_losses:
            ax.plot(valid_eps, valid_losses, color=C_RED, alpha=0.3, label="Raw Loss")
            ax.plot(valid_eps, _ema_smooth(valid_losses, alpha=0.1), color=C_RED, linewidth=2.0, label="Loss (EMA)")
        ax.set_title("Huber Optimization Loss")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        ax.grid(True)
        ax.legend()

        # 3. Waiting Time
        ax = axes[1, 0]
        ax.plot(episodes, wait_times, color=C_ORANGE, alpha=0.3, label="Raw Waiting Time")
        if len(wait_times) > 5:
            ax.plot(episodes, _ema_smooth(wait_times), color=C_ORANGE, linewidth=2.5, label="Wait Time (EMA)")
        ax.set_title("Average Waiting Time")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Waiting Time (s)")
        ax.grid(True)
        ax.legend()

        # 4. Queue Length and Epsilon
        ax = axes[1, 1]
        ax.plot(episodes, queues, color=C_GREEN, alpha=0.3, label="Raw Queue Length")
        if len(queues) > 5:
            ax.plot(episodes, _ema_smooth(queues), color=C_GREEN, linewidth=2.5, label="Queue Length (EMA)")
        ax.set_ylabel("Queue Length (veh)", color=C_GREEN)
        ax.tick_params(axis='y', labelcolor=C_GREEN)
        ax.grid(True)

        ax2 = ax.twinx()
        ax2.plot(episodes, epsilons, color=C_PURPLE, linestyle="--", linewidth=1.5, label="Epsilon")
        ax2.set_ylabel("Exploration (Epsilon)", color=C_PURPLE)
        ax2.tick_params(axis='y', labelcolor=C_PURPLE)
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        ax.set_title("Queue Length & Exploration rate")
        ax.set_xlabel("Episode")

        plt.tight_layout()
        path = str(self.plots_dir / "dqn_training_dashboard.png")
        plt.savefig(path)
        plt.close(fig)
        return path

    def plot_model_comparison(
        self,
        dqn_results: Dict[str, float],
        q_results: Dict[str, float],
        ppo_results: Dict[str, float]
    ) -> str:
        """
        Produce bar charts comparing DQN, Q-Learning, and PPO on evaluation metrics:
        - Mean Waiting Time (s)
        - Mean Queue Length (vehicles)
        - Mean Episode Reward
        - Throughput (vehicles cleared)
        - Traffic Delay (average deviation)
        """
        metrics = {
            "Mean Reward": ("mean_reward", True),
            "Mean Waiting Time (s)": ("mean_waiting_time", False),
            "Mean Queue Length (veh)": ("mean_queue", False),
            "Traffic Delay (deviation)": ("mean_delay", False),
            "Throughput (veh)": ("throughput", True)
        }

        fig, axes = plt.subplots(3, 2, figsize=(14, 14))
        fig.suptitle("Model Evaluation Comparison\n(DQN vs Q-Learning vs PPO)", fontsize=18, fontweight="bold", y=0.98)

        models = ["Q-Learning", "PPO", "DQN"]
        colors = [C_GRAY, C_PURPLE, C_BLUE]

        for i, (title, (key, higher_better)) in enumerate(metrics.items()):
            ax = axes[i // 2, i % 2]
            
            # Values for each model
            val_q = q_results.get(key, 0.0)
            val_ppo = ppo_results.get(key, 0.0)
            val_dqn = dqn_results.get(key, 0.0)
            values = [val_q, val_ppo, val_dqn]

            bars = ax.bar(models, values, color=colors, width=0.45, edgecolor="#0f1117", linewidth=1.5)
            
            # Annotate values
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.01 * max(values) if max(values) != 0 else 0.01),
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=10, color="#e8eaf0"
                )

            # Determine best performance and highlight with a green border
            best_idx = -1
            if higher_better:
                # Higher is better
                best_idx = int(np.argmax(values))
            else:
                # Lower is better (waiting time, queue, delay)
                best_idx = int(np.argmin(values))
            
            bars[best_idx].set_edgecolor(C_GREEN)
            bars[best_idx].set_linewidth(2.0)

            ax.set_title(title)
            ax.grid(True, axis="y")
            ax.set_ylabel("Value")

        # Hide the 6th empty subplot
        axes[2, 1].axis("off")

        plt.tight_layout()
        path = str(self.plots_dir / "model_comparison.png")
        plt.savefig(path)
        plt.close(fig)
        return path
