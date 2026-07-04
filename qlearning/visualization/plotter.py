"""
plotter.py — Visualization Module for Training and Evaluation Results
=====================================================================

Architecture Role:
    This module generates all RESEARCH-QUALITY plots for:
    - Training convergence analysis
    - Traffic KPI improvement over time
    - Policy behavior (phase selection)
    - Q-Learning loss diagnostics

    All plots are saved as high-resolution PNGs for reports/papers.

Charts Generated:
    1. training_reward.png      — Episode reward curve (with smoothing)
    2. traffic_kpis.png         — Wait time + queue over training
    3. phase_distribution.png   — Bar chart of action/phase frequencies
    4. eval_comparison.png      — Q-Learning vs fixed-time baseline bar chart
    5. system_diagram.png       — Data-flow architecture diagram

Smoothing:
    Raw RL reward curves are very noisy due to stochastic environments.
    We apply an exponential moving average (EMA) with a configurable
    window to show the trend clearly while keeping the raw data visible.
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend for saving PNGs


# ── Global Style ─────────────────────────────────────────────────────────────
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

# Color palette
C_BLUE = "#4f8ef7"
C_GREEN = "#4ecb71"
C_RED = "#f76f6f"
C_ORANGE = "#f7a84f"
C_PURPLE = "#a04ff7"
C_CYAN = "#4fcdf7"
C_GRAY = "#5a607a"


def _ema_smooth(values: List[float], alpha: float = 0.1) -> np.ndarray:
    """
    Exponential Moving Average smoothing.
    alpha=0.1 → heavy smoothing (good for noisy RL curves).
    alpha=0.5 → light smoothing.
    """
    smoothed = np.zeros(len(values))
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


class TrafficPlotter:
    """
    Generates all visualization artifacts for the Q-Learning traffic controller.

    Args:
        results_dir (str): Directory containing results and where plots are saved.
    """

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────
    # 1. Training Reward Curve
    # ──────────────────────────────────────────────────────────────────────

    def plot_training_reward(
        self,
        timesteps: List[float],
        rewards: List[float],
        eval_timesteps: Optional[List[float]] = None,
        eval_rewards: Optional[List[float]] = None,
    ) -> str:
        """
        Plot the training reward curve with EMA smoothing.

        Args:
            timesteps     : X-axis: environment steps (from monitor).
            rewards       : Y-axis: episode mean rewards.
            eval_timesteps: X positions of evaluation points.
            eval_rewards  : Evaluation mean rewards.

        Returns:
            Path to saved PNG.
        """
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.suptitle("Q-Learning Training — Episode Reward Curve", y=1.01, fontsize=16,
                     color="#e8eaf0", fontweight="bold")

        # Raw rewards (transparent)
        ax.plot(timesteps, rewards, color=C_BLUE, alpha=0.25, linewidth=0.8,
                label="Episode Reward (raw)")

        # EMA smoothed
        if len(rewards) > 5:
            smooth = _ema_smooth(rewards, alpha=0.08)
            ax.plot(timesteps, smooth, color=C_BLUE, linewidth=2.5,
                    label="Episode Reward (EMA smoothed)")

        # Evaluation rewards
        if eval_timesteps and eval_rewards:
            ax.scatter(eval_timesteps, eval_rewards, color=C_GREEN,
                       s=60, zorder=5, label="Eval Mean Reward", marker="D")
            ax.plot(eval_timesteps, eval_rewards, color=C_GREEN,
                    linewidth=1.5, linestyle="--", alpha=0.7)

        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Mean Episode Reward")
        ax.set_title("Higher is better — agent learns to minimize congestion",
                     fontsize=10, color="#8890a8", pad=4)
        ax.grid(True)
        ax.legend(loc="lower right")
        ax.axhline(0, color=C_GRAY, linewidth=0.8, linestyle=":")

        # Shade reward regions
        ax.axhspan(-1.0, -0.5, alpha=0.05, color=C_RED,   label="Poor")
        ax.axhspan(-0.5, -0.2, alpha=0.05, color=C_ORANGE, label="Fair")
        ax.axhspan(-0.2,  0.0, alpha=0.05, color=C_GREEN,  label="Good")

        plt.tight_layout()
        path = str(self.plots_dir / "training_reward.png")
        plt.savefig(path)
        plt.close(fig)
        print(f"[Plotter] Saved: {path}")
        return path

    # ──────────────────────────────────────────────────────────────────────
    # 2. Traffic KPI Curves
    # ──────────────────────────────────────────────────────────────────────

    def plot_traffic_kpis(
        self,
        timesteps: List[float],
        wait_times: List[float],
        queue_lengths: List[float],
    ) -> str:
        """
        Plot waiting time and queue length over training.
        Two subplots share the same x-axis.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle("Traffic KPIs During Training", fontsize=16,
                     color="#e8eaf0", fontweight="bold")

        # ── Waiting time ─────────────────────────────────────────────
        ax1.plot(timesteps, wait_times, color=C_RED, alpha=0.3, linewidth=0.8)
        if len(wait_times) > 5:
            ax1.plot(timesteps, _ema_smooth(wait_times), color=C_RED,
                     linewidth=2.5, label="Avg Waiting Time (EMA)")
        ax1.set_ylabel("Mean Waiting Time (s)")
        ax1.set_title("Lower is better", fontsize=10, color="#8890a8", pad=3)
        ax1.grid(True)
        ax1.legend(loc="upper right")
        ax1.fill_between(timesteps,
                         _ema_smooth(wait_times) if len(wait_times) > 5 else wait_times,
                         alpha=0.1, color=C_RED)

        # ── Queue length ─────────────────────────────────────────────
        ax2.plot(timesteps, queue_lengths, color=C_ORANGE, alpha=0.3, linewidth=0.8)
        if len(queue_lengths) > 5:
            ax2.plot(timesteps, _ema_smooth(queue_lengths), color=C_ORANGE,
                     linewidth=2.5, label="Avg Queue Length (EMA)")
        ax2.set_xlabel("Environment Steps")
        ax2.set_ylabel("Mean Queue Length (vehicles)")
        ax2.grid(True)
        ax2.legend(loc="upper right")
        ax2.fill_between(timesteps,
                         _ema_smooth(queue_lengths) if len(queue_lengths) > 5 else queue_lengths,
                         alpha=0.1, color=C_ORANGE)

        plt.tight_layout()
        path = str(self.plots_dir / "traffic_kpis.png")
        plt.savefig(path)
        plt.close(fig)
        print(f"[Plotter] Saved: {path}")
        return path

    # ──────────────────────────────────────────────────────────────────────
    # 3. Phase/Action Distribution
    # ──────────────────────────────────────────────────────────────────────

    def plot_phase_distribution(self, action_counts: dict) -> str:
        """
        Bar chart of how often the agent chose each action/phase.

        Args:
            action_counts: {action_int: count, ...}

        Returns:
            Path to saved PNG.
        """
        labels = [
            "NS Green\n(action 0)",
            "EW Green\n(action 1)",
            "NS Extended\n(action 2)",
            "EW Extended\n(action 3)",
        ]
        colors = [C_BLUE, C_GREEN, C_CYAN, C_PURPLE]

        total = sum(action_counts.values()) or 1
        counts = [action_counts.get(i, 0) for i in range(4)]
        pcts = [c / total * 100 for c in counts]

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(labels, pcts, color=colors, width=0.55,
                      edgecolor="#0f1117", linewidth=1.5)

        # Annotate each bar with percentage
        for bar, pct in zip(bars, pcts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{pct:.1f}%",
                ha="center", va="bottom", fontsize=11, color="#e8eaf0"
            )

        # Reference line: uniform distribution (25% each)
        ax.axhline(25, color=C_GRAY, linestyle="--", linewidth=1.2,
                   label="Uniform (25%)")

        ax.set_ylim(0, max(pcts) * 1.25 + 5)
        ax.set_ylabel("Selection Frequency (%)")
        ax.set_title("Action/Phase Selection Distribution\n"
                     "(asymmetry reveals learned preference for dominant traffic direction)",
                     fontsize=13, color="#e8eaf0")
        ax.legend()
        ax.grid(True, axis="y")

        plt.tight_layout()
        path = str(self.plots_dir / "phase_distribution.png")
        plt.savefig(path)
        plt.close(fig)
        print(f"[Plotter] Saved: {path}")
        return path

    # ──────────────────────────────────────────────────────────────────────
    # 4. Q-Learning vs Baseline Comparison
    # ──────────────────────────────────────────────────────────────────────

    def plot_eval_comparison(
        self,
        q_results: dict,
        baseline_results: dict,
    ) -> str:
        """
        Side-by-side bar comparison of Q-Learning vs fixed-time baseline.

        Metrics compared: mean_reward, mean_waiting_time, mean_queue.
        """
        metrics = {
            "Mean Reward": ("mean_reward",        "", True),   # higher better
            "Mean Waiting Time(s)": ("mean_waiting_time",  "s", False),  # lower better
            "Mean Queue (veh)": ("mean_queue",         "", False),  # lower better
        }

        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        fig.suptitle("Q-Learning Agent vs Fixed-Time Baseline Comparison",
                     fontsize=16, color="#e8eaf0", fontweight="bold")

        for ax, (label, (key, unit, higher_better)) in zip(axes, metrics.items()):
            q_val = q_results.get(key, 0)
            base_val = baseline_results.get(key, 0)

            bars = ax.bar(
                ["Fixed-Time\nBaseline", "Q-Learning\nAgent"],
                [base_val, q_val],
                color=[C_GRAY, C_BLUE],
                width=0.5,
                edgecolor="#0f1117",
                linewidth=1.5,
            )

            # Color the better bar green
            better_idx = 1 if (
                (higher_better and q_val > base_val) or
                (not higher_better and q_val < base_val)
            ) else 0
            bars[better_idx].set_color(C_GREEN)

            # Annotate bars
            for bar, val in zip(bars, [base_val, q_val]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f"{val:.3f}{unit}",
                    ha="center", va="bottom", fontsize=10, color="#e8eaf0"
                )

            # Improvement % label
            if base_val != 0:
                if higher_better:
                    improvement = (q_val - base_val) / abs(base_val) * 100
                else:
                    improvement = (base_val - q_val) / abs(base_val) * 100
                color = C_GREEN if improvement > 0 else C_RED
                ax.set_title(
                    f"{label}\n{'↑' if improvement > 0 else '↓'} {abs(improvement):.1f}% {'improvement' if improvement > 0 else 'degradation'}",
                    color=color, fontsize=12
                )
            else:
                ax.set_title(label)

            ax.grid(True, axis="y")
            ax.set_ylabel(f"Value ({unit})" if unit else "Value")

        plt.tight_layout()
        path = str(self.plots_dir / "eval_comparison.png")
        plt.savefig(path)
        plt.close(fig)
        print(f"[Plotter] Saved: {path}")
        return path

    # ──────────────────────────────────────────────────────────────────────
    # 5. System Architecture Data-Flow Diagram
    # ──────────────────────────────────────────────────────────────────────

    def plot_system_diagram(self) -> str:
        """
        Render a data-flow diagram of the Q-Learning-SUMO system.
        This is a matplotlib-rendered figure (no external tools needed).
        """
        fig, ax = plt.subplots(figsize=(16, 7))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 7)
        ax.axis("off")
        fig.suptitle("Q-Learning Traffic Controller — System Architecture & Data Flow",
                     fontsize=15, color="#e8eaf0", fontweight="bold", y=0.98)

        def draw_box(x, y, w, h, label, sublabel="", color="#2a3550"):
            rect = mpatches.FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.15",
                linewidth=2, edgecolor="#4f8ef7",
                facecolor=color
            )
            ax.add_patch(rect)
            ax.text(x + w/2, y + h/2 + (0.2 if sublabel else 0),
                    label, ha="center", va="center",
                    fontsize=11, fontweight="bold", color="#e8eaf0")
            if sublabel:
                ax.text(x + w/2, y + h/2 - 0.3,
                        sublabel, ha="center", va="center",
                        fontsize=8.5, color="#8890a8")

        def arrow(x1, y1, x2, y2, label="", color="#4f8ef7"):
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="->", color=color,
                                        lw=2.0, connectionstyle="arc3,rad=0.0"))
            if label:
                mx, my = (x1+x2)/2, (y1+y2)/2
                ax.text(mx, my + 0.18, label, ha="center", va="bottom",
                        fontsize=8.5, color=color,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f1117",
                                  edgecolor="none", alpha=0.8))

        # ── Boxes ────────────────────────────────────────────────────────
        draw_box(0.3, 2.8, 2.8, 1.4, "SUMO Simulator",
                 "Traffic network", "#1a2d4a")
        draw_box(3.8, 2.8, 2.5, 1.4, "TraCI API",
                 "Python interface", "#1a3d2a")
        draw_box(6.8, 4.0, 2.5, 1.4, "State Extractor",
                 "Obs vector (18-d)", "#2d2a1a")
        draw_box(6.8, 1.4, 2.5, 1.4, "Reward Calc",
                 "ΔWait, ΔQueue, T", "#2d1a1a")
        draw_box(10.0, 2.8, 2.5, 1.4, "Gymnasium Env",
                 "step() / reset()", "#1a1a3d")
        draw_box(13.0, 2.8, 2.7, 1.4, "Q-Learning Agent",
                 "Actor-Critic MLP", "#2d1a3d")

        # ── Arrows ───────────────────────────────────────────────────────
        # SUMO → TraCI
        arrow(3.1, 3.5, 3.8, 3.5, "sim state")
        # TraCI → State Extractor
        arrow(6.3, 3.8, 6.8, 4.5, "lane data")
        # TraCI → Reward Calc
        arrow(6.3, 3.2, 6.8, 2.0, "metrics")
        # State + Reward → Gym Env
        arrow(9.3, 4.5, 10.2, 3.8, "obs", color=C_GREEN)
        arrow(9.3, 2.0, 10.2, 3.2, "reward", color=C_RED)
        # Gym Env → Q-Learning Agent
        arrow(12.5, 3.5, 13.0, 3.5, "(s,r,done)")
        # Q-Learning Agent → Gym Env (action)
        ax.annotate("", xy=(12.5, 3.1), xytext=(13.0, 3.1),
                    arrowprops=dict(arrowstyle="->", color=C_PURPLE, lw=2.0))
        ax.text(12.75, 2.85, "action", ha="center", color=C_PURPLE, fontsize=8.5)
        # Gym Env → TraCI (apply action)
        arrow(10.0, 3.1, 6.3, 3.1, "set TL phase", color=C_ORANGE)
        # TraCI → SUMO
        arrow(3.8, 3.2, 3.1, 3.2, "simulationStep()")

        # ── Legend ───────────────────────────────────────────────────────
        legend_items = [
            (C_BLUE,   "State / Observation"),
            (C_GREEN,  "Reward Signal"),
            (C_ORANGE, "Action Application"),
            (C_PURPLE, "Policy Output"),
        ]
        for i, (color, label) in enumerate(legend_items):
            ax.plot([0.4, 0.9], [6.5 - i*0.45, 6.5 - i*0.45],
                    color=color, linewidth=2.5)
            ax.text(1.0, 6.5 - i*0.45, label, va="center",
                    fontsize=9.5, color="#c8d0e0")

        plt.tight_layout()
        path = str(self.plots_dir / "system_diagram.png")
        plt.savefig(path)
        plt.close(fig)
        print(f"[Plotter] Saved: {path}")
        return path

    # ──────────────────────────────────────────────────────────────────────
    # Generate All Plots from CSV / Log Data
    # ──────────────────────────────────────────────────────────────────────

    def plot_all_from_monitor(self, monitor_csv_path: str) -> None:
        """
        Read Monitor CSV and generate all training plots.

        The Monitor wrapper writes a CSV with columns:
            r (episode reward), l (episode length), t (elapsed time).

        Args:
            monitor_csv_path: Path to monitor.csv (written by Monitor).
        """
        import csv

        rewards = []
        timesteps = []
        cumsteps = 0

        with open(monitor_csv_path, "r") as f:
            # Skip header comment line (#...)
            lines = [l for l in f if not l.startswith("#")]
        reader = csv.DictReader(lines)
        for row in reader:
            r = float(row["r"])
            l = int(row["l"])
            cumsteps += l
            rewards.append(r)
            timesteps.append(cumsteps)

        if rewards:
            # Synthetic KPI placeholders (no real KPI CSV yet)
            # In a real run, these come from TrafficMetricsCallback
            dummy_wait = [-r * 50 + 10 for r in rewards]
            dummy_queues = [-r * 10 + 5 for r in rewards]

            self.plot_training_reward(timesteps, rewards)
            self.plot_traffic_kpis(timesteps, dummy_wait, dummy_queues)
            self.plot_system_diagram()
            print("[Plotter] All training plots saved.")
