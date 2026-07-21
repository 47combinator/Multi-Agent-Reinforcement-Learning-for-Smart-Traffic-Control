"""
qlearning_agent.py — Tabular Q-Learning Agent for Traffic Signal Control
========================================================================

Architecture Role:
    This module implements the Q-Learning algorithm.
    It manages the Q-table, action selection (epsilon-greedy), and Q-value updates.
    State discretization is handled internally since the environment provides continuous states.
"""

import os
import json
import numpy as np
from typing import Dict, Tuple
from utils.logger import get_logger

logger = get_logger(__name__)


class QLearningAgent:
    """
    A tabular Q-Learning agent with epsilon-greedy exploration.

    Accommodates state discretization for continuous state spaces.
    Designed to manage a single independent Q-table. For multi-agent
    scenarios, instantiate one QLearningAgent per traffic signal.
    """

    def __init__(
        self,
        action_space_n: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9995,
        bins: int = 6,
        agent_id: str = "center"
    ):
        """
        Args:
            action_space_n (int): Number of discrete actions.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Initial exploration rate.
            epsilon_min (float): Minimum exploration rate.
            epsilon_decay (float): Epsilon decay factor per step/episode.
            bins (int): Number of discrete bins per continuous state dimension.
            agent_id (str): Identifier for this agent/traffic signal.
        """
        self.action_space_n = action_space_n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.bins = bins
        self.agent_id = agent_id

        # The Q-table is a dictionary mapping discrete state tuples to a numpy array of Q-values
        self.q_table: Dict[Tuple, np.ndarray] = {}

        self.is_training = True

    def _discretize_state(self, state: np.ndarray) -> Tuple:
        """
        Convert continuous state vector (values bounded [0, 1]) to a discrete tuple.
        """
        # Ensure state values are clamped between 0 and 1 before binning
        clipped_state = np.clip(state, 0.0, 1.0)
        # Scale to [0, bins] and cast to int, bounding max value to bins - 1
        discrete_state = np.minimum((clipped_state * self.bins).astype(int), self.bins - 1)
        return tuple(discrete_state.tolist())

    def _get_q_values(self, discrete_state: Tuple) -> np.ndarray:
        """Retrieve Q-values for a given state, initializing if unseen."""
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.action_space_n, dtype=np.float32)
        return self.q_table[discrete_state]

    def choose_action(self, state: np.ndarray) -> int:
        """
        Select an action using epsilon-greedy policy.
        """
        discrete_state = self._discretize_state(state)

        if self.is_training and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_space_n))

        q_values = self._get_q_values(discrete_state)
        # Break ties randomly by shuffling indices of max Q-values
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return int(np.random.choice(best_actions))

    def update_q_value(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, terminated: bool) -> float:
        """
        Update the Q-table based on the Bellman equation.

        Returns:
            The temporal difference (TD) error.
        """
        discrete_state = self._discretize_state(state)
        discrete_next_state = self._discretize_state(next_state)

        q_values = self._get_q_values(discrete_state)
        next_q_values = self._get_q_values(discrete_next_state)

        current_q = q_values[action]
        max_next_q = 0.0 if terminated else np.max(next_q_values)

        td_target = reward + self.gamma * max_next_q
        td_error = td_target - current_q

        # Update Q-value
        q_values[action] = current_q + self.alpha * td_error
        self.q_table[discrete_state] = q_values

        return float(td_error)

    def learn(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, terminated: bool) -> float:
        """Alias for update_q_value for compatibility or specific integrations."""
        return self.update_q_value(state, action, reward, next_state, terminated)

    def decay_epsilon(self) -> None:
        """Decay the exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def set_training(self, mode: bool = True) -> None:
        """Set agent to training mode (enables exploration)."""
        self.is_training = mode

    def set_evaluation(self) -> None:
        """Set agent to evaluation mode (greedy action selection)."""
        self.is_training = False

    def save(self, filepath: str) -> None:
        """Save the Q-table to a JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Convert tuple keys to strings for JSON serialization
        serializable_q_table = {
            str(k): v.tolist() for k, v in self.q_table.items()
        }
        with open(filepath, "w") as f:
            json.dump({
                "agent_id": self.agent_id,
                "q_table": serializable_q_table,
                "epsilon": self.epsilon
            }, f, indent=4)
        logger.info(f"[{self.agent_id}] Saved Q-table with {len(self.q_table)} states to {filepath}")

    def load(self, filepath: str) -> None:
        """Load the Q-table from a JSON file."""
        if not os.path.exists(filepath):
            logger.warning(f"[{self.agent_id}] Q-table file not found at {filepath}. Starting fresh.")
            return

        with open(filepath, "r") as f:
            data = json.load(f)

        self.q_table = {}
        for k_str, v_list in data["q_table"].items():
            # Convert string "(0, 1, ...)" back to tuple
            k_tuple = tuple(map(int, k_str.strip("()").split(", ")))
            self.q_table[k_tuple] = np.array(v_list, dtype=np.float32)

        if "epsilon" in data:
            self.epsilon = data["epsilon"]

        logger.info(f"[{self.agent_id}] Loaded Q-table with {len(self.q_table)} states from {filepath}")
