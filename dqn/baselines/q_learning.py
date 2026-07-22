"""
baselines/q_learning.py — Q-Learning wrapper for DQN comparison
================================================================
Wraps the qlearning/ QLearningAgent for use in evaluate_dqn.py.
"""

import sys
from pathlib import Path

# Add qlearning package to path
_ql_root = Path(__file__).resolve().parent.parent.parent / "qlearning"
sys.path.insert(0, str(_ql_root))

try:
    from agent.qlearning_agent import QLearningAgent  # noqa: F401
except ImportError:
    # Fallback stub if qlearning package not available
    import numpy as np

    class QLearningAgent:
        """Stub if qlearning package is not installed."""
        def __init__(self, *args, **kwargs): pass
        def choose_action(self, state): return 0
        def load(self, path): pass
        def set_evaluation(self): pass
        is_training = False
