"""
reproducibility.py — Global Seed Management for Reproducible Experiments
=========================================================================

Reproducibility in RL requires seeding FOUR independent sources:
    1. Python's built-in random module
    2. NumPy's RNG
    3. PyTorch (CPU and CUDA)
    4. The environment itself (SUMO seed passed via TraCI)

Without this, two runs of the same code with the same hyperparameters
can produce completely different results — making debugging impossible
and comparisons invalid.
"""

import random
import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """
    Set deterministic seeds for all RNG sources.

    Args:
        seed: Integer seed value. Use different seeds for different
              experimental conditions (not just 42 every time).

    Note:
        SUMO/TraCI seeding is handled separately in TrafficEnv._start_sumo()
        via the --seed command-line argument.

        CuDNN determinism: torch.backends.cudnn.deterministic = True
        slows GPU computation but ensures exact reproducibility.
        Disable for faster training if exact reproducibility is not needed.
    """
    # Python stdlib
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (all GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # CuDNN reproducibility (slightly slower on GPU)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[Reproducibility] Global seed set to: {seed}")
