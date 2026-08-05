"""
config.py — Global Configuration and Hyperparameters
===================================================
"""

import os
from pathlib import Path
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
SUMO_FILES_DIR = ROOT_DIR / "sumo_env"
RESULTS_DIR = BASE_DIR / "results"

# SUMO configuration path
SUMOCFG_PATH = str(SUMO_FILES_DIR / "single_intersection.sumocfg")

# Hardware acceleration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Environment configuration
ENV_CONFIG = {
    "sumocfg_path": SUMOCFG_PATH,
    "tls_id": "center",
    "lane_ids": ["n2c_0", "s2c_0", "e2c_0", "w2c_0"],
    "delta_t": 5,
    "max_steps": 720,       # Max steps per episode (720 steps = 3600 seconds)
    "traci_port": 8813,
    "eval_traci_port": 8814,
    "use_gui": False,
    "seed": 42,
    # Reward weights
    "alpha": 0.4,           # Waiting time penalty weight
    "beta": 0.3,            # Queue length penalty weight
    "gamma": 0.3            # Throughput reward weight
}

# DQN Hyperparameters
DQN_CONFIG = {
    "gamma": 0.99,
    "lr": 1e-4,
    "buffer_size": 100000,
    "batch_size": 64,
    "target_update_freq": 1000,     # How often to copy weights to target network (in steps)
    "tau": 0.005,                   # For soft target updates: target = tau * online + (1 - tau) * target
    "use_soft_update": True,        # Set True for soft update, False for hard update every target_update_freq steps
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.995,         # Epsilon decay rate per episode
    "num_episodes": 15,
    "max_steps": 1000,              # Max steps per episode to override env limit if needed
    "optimizer": "Adam",
    "grad_clip": 1.0,               # Gradient clipping threshold
    "huber_delta": 1.0,             # Delta parameter for Huber Loss
    "lr_decay_step": 200,           # Learning rate scheduler step size (episodes)
    "lr_decay_gamma": 0.5,          # Learning rate scheduler decay rate
    
    # DQN Architecture flags
    "double_dqn": True,             # Use Double DQN
    "dueling_dqn": True,            # Use Dueling DQN Architecture
    "prioritized_replay": False     # Set to True if using Prioritized Experience Replay
}

# PPO Hyperparameters (for baseline comparison)
PPO_CONFIG = {
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "c1": 0.5,                      # Value loss coefficient
    "c2": 0.01,                     # Entropy coefficient
    "batch_size": 64,
    "mini_batch_size": 16,
    "update_epochs": 10,
    "num_episodes": 1000
}

# General settings
LOG_FREQ = 1                        # Logging to console frequency (episodes)
EVAL_FREQ = 10                      # Evaluation frequency (episodes)
N_EVAL_EPISODES = 5                 # Number of episodes for evaluation
CHECKPOINT_FREQ = 100               # Checkpoint save frequency (episodes)
EARLY_STOPPING_PATIENCE = 15        # Number of evaluations with no improvement before stopping
