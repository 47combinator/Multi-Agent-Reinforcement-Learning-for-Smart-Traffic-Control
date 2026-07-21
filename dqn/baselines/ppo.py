"""
baselines/ppo.py — PPO wrapper for DQN comparison
==================================================
Wraps the stable-baselines3 PPO model for use in evaluate_dqn.py.
"""

import sys
import numpy as np
from pathlib import Path


class PPOAgent:
    """
    Thin wrapper around stable_baselines3 PPO for uniform agent interface.
    """

    def __init__(self, model_path: str = None):
        self._model = None
        if model_path:
            self.load(model_path)

    def load(self, model_path: str):
        try:
            from stable_baselines3 import PPO
            self._model = PPO.load(model_path, device="cpu")
        except Exception as e:
            print(f"[PPOAgent] Could not load model from {model_path}: {e}")
            self._model = None

    def choose_action(self, state: np.ndarray) -> int:
        if self._model is None:
            return 0
        action, _ = self._model.predict(state, deterministic=True)
        return int(action)

    def set_evaluation(self): pass
    def set_training(self):   pass
    is_training = False
