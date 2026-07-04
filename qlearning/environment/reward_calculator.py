"""
reward_calculator.py — Composite Reward Function for Traffic Control
=====================================================================

Architecture Role:
    This module is the TEACHER of our RL system.
    It defines WHAT the agent should optimize.
    The reward signal is the only feedback the Q-Learning agent receives;
    everything about the learned policy flows from this design.

Reward Design Philosophy:
    Three objectives must be balanced:
      1. Minimize waiting time   → vehicles shouldn't sit idle
      2. Minimize queue length   → avoid spillback and gridlock
      3. Maximize throughput     → the intersection should serve traffic

    We use a DELTA-BASED reward:
        r_t = -α·ΔW_t - β·ΔQ_t + γ·T_t

    Where:
        ΔW_t = W_t - W_{t-1}  (change in total waiting time; positive = worsening)
        ΔQ_t = Q_t - Q_{t-1}  (change in total queue; positive = worsening)
        T_t  = vehicles that left the intersection this step (throughput)
        α, β, γ = weighting coefficients (configurable)

    Why deltas instead of absolute values?
        Absolute: r = -W_t penalizes the agent just for being in a congested
                  state, even if it's actively improving. This is unfair.
        Delta:    r = -ΔW_t only penalizes WORSENING. If the agent reduces
                  congestion (ΔW < 0), it gets a POSITIVE reward.
        Result:   Much denser, more informative gradient signal.

    Final reward is clipped to [-1, 1] for training stability.
    This prevents a single catastrophic step from dominating gradient updates.

Data Flow:
    StateExtractor.get_raw_metrics() [step t-1]
        → stored as previous_metrics
    StateExtractor.get_raw_metrics() [step t]
        → compute deltas → combine with throughput → normalize → clip
"""

import sys
import os
import numpy as np

# TraCI import for throughput measurement
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import traci

# ---------------------------------------------------------------------------
# Normalization denominators for each reward component.
# These are chosen to map typical values into a roughly [-1, 1] range
# BEFORE final clipping, so clipping rarely activates under normal traffic.
# ---------------------------------------------------------------------------
# Maximum expected change in total waiting time per step (seconds).
# 20 vehicles × 1 second/step = 20 if every vehicle waits one more second.
NORM_DELTA_WAIT = 20.0

# Maximum expected change in total queue length per step.
# Realistically, at most 4-5 vehicles join/leave queue per step.
NORM_DELTA_QUEUE = 5.0

# Maximum throughput per step (vehicles that cleared the intersection).
# At 3600 veh/hr total demand, max step throughput ≈ 1 vehicle/second.
NORM_THROUGHPUT = 2.0


class RewardCalculator:
    """
    Calculates the per-step scalar reward for the Q-Learning agent.

    Attributes:
        alpha (float)         : Weight for waiting time delta component.
        beta  (float)         : Weight for queue length delta component.
        gamma (float)         : Weight for throughput component.
        _prev_metrics (dict)  : Raw metrics from the previous simulation step.
        _arrived_prev (int)   : Vehicles that had arrived at end of prev step.
    """

    def __init__(self, alpha: float = 0.4, beta: float = 0.3, gamma: float = 0.3):
        """
        Args:
            alpha : Penalty weight for increasing total waiting time.
            beta  : Penalty weight for increasing total queue length.
            gamma : Bonus weight for vehicles completing their journey.

        The weights must satisfy: alpha + beta + gamma ≈ 1.0 (not required,
        but keeps the reward scale interpretable before normalization).
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Previous step state — initialized on reset()
        self._prev_metrics = None
        self._arrived_prev = 0   # cumulative arrived vehicles at t-1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, initial_metrics: dict) -> None:
        """
        Reset the calculator at the start of a new episode.
        Must be called AFTER SUMO has started and an initial observation
        has been collected.

        Args:
            initial_metrics: Raw metrics dict from StateExtractor.get_raw_metrics()
                             at t=0.
        """
        self._prev_metrics = initial_metrics
        # Get cumulative arrived count at episode start (usually 0)
        try:
            self._arrived_prev = traci.simulation.getArrivedNumber()
        except Exception:
            self._arrived_prev = 0

    def compute_reward(self, current_metrics: dict) -> tuple[float, dict]:
        """
        Compute the scalar reward for the current simulation step.

        Args:
            current_metrics: Raw metrics dict from StateExtractor.get_raw_metrics()
                             at the CURRENT step t.

        Returns:
            reward     (float): Clipped scalar reward in [-1.0, 1.0].
            components (dict) : Breakdown of each reward term for logging.
        """
        if self._prev_metrics is None:
            # Safety fallback: return 0 if reset() wasn't called
            self._prev_metrics = current_metrics
            return 0.0, {}

        # ── Component 1: Waiting time delta ─────────────────────────────
        # ΔW = current_total_wait - previous_total_wait
        # If ΔW > 0: congestion worsened → negative reward.
        # If ΔW < 0: congestion improved → positive reward.
        delta_wait = (
            current_metrics["total_waiting_time"]
            - self._prev_metrics["total_waiting_time"]
        )
        reward_wait = -self.alpha * (delta_wait / NORM_DELTA_WAIT)

        # ── Component 2: Queue length delta ─────────────────────────────
        delta_queue = (
            current_metrics["total_queue"]
            - self._prev_metrics["total_queue"]
        )
        reward_queue = -self.beta * (delta_queue / NORM_DELTA_QUEUE)

        # ── Component 3: Throughput ──────────────────────────────────────
        # getArrivedNumber() returns CUMULATIVE arrivals since sim start.
        # We take the difference to get STEP-LEVEL throughput.
        try:
            arrived_now = traci.simulation.getArrivedNumber()
            step_arrived = max(0, arrived_now - self._arrived_prev)
            self._arrived_prev = arrived_now
        except Exception:
            step_arrived = 0

        reward_throughput = self.gamma * (step_arrived / NORM_THROUGHPUT)

        # ── Combine and clip ─────────────────────────────────────────────
        raw_reward = reward_wait + reward_queue + reward_throughput
        reward = float(np.clip(raw_reward, -1.0, 1.0))

        # Update previous metrics for next step
        self._prev_metrics = current_metrics

        # Return detailed breakdown for TensorBoard logging
        components = {
            "reward/total": reward,
            "reward/waiting_delta": reward_wait,
            "reward/queue_delta": reward_queue,
            "reward/throughput": reward_throughput,
            "metrics/delta_wait": delta_wait,
            "metrics/delta_queue": delta_queue,
            "metrics/step_throughput": step_arrived,
        }

        return reward, components

    def get_info_dict(self, current_metrics: dict, reward_components: dict) -> dict:
        """
        Build the 'info' dict returned by env.step().
        Used for logging.

        Args:
            current_metrics   : From StateExtractor.get_raw_metrics()
            reward_components : From compute_reward()

        Returns:
            dict compatible with Gymnasium's step() info contract.
        """
        return {
            # Raw traffic KPIs (for plotter and evaluator)
            "total_waiting_time": current_metrics["total_waiting_time"],
            "total_queue": current_metrics["total_queue"],
            "vehicle_counts": current_metrics["vehicle_counts"],
            # Reward breakdown
            **reward_components,
        }
