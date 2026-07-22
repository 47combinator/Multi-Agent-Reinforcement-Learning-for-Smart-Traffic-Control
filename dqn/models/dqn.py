"""
dqn.py — Deep Q-Network Agent
==============================

Implements:
  - Dueling DQN network architecture
  - Double DQN target computation
  - Soft target network updates
  - Prioritised Experience Replay (optional)
  - Huber loss optimisation
  - Epsilon-greedy exploration with decay

Compatible with train_dqn.py interface:
    DQNAgent(state_size, action_size, config, device)
    agent.step(obs, action, reward, next_obs, done)   → loss
    agent.choose_action(obs)                           → int
    agent.set_evaluation() / set_training()
    agent.save(path) / load(path)
"""

import os
import random
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ─────────────────────────────────────────────────────────────────────────────
# Neural Network
# ─────────────────────────────────────────────────────────────────────────────

class DuelingDQNNetwork(nn.Module):
    """
    Dueling DQN architecture.
    Splits the final fully-connected layers into:
      - Value stream  V(s)
      - Advantage stream A(s, a)
    Combined as: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
    """

    def __init__(self, state_size: int, action_size: int, hidden: list = None):
        super().__init__()
        hidden = hidden or [256, 128, 64]

        # Shared feature extractor
        layers = []
        in_dim = state_size
        for h in hidden[:-1]:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.shared = nn.Sequential(*layers)

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(in_dim, hidden[-1]),
            nn.ReLU(),
            nn.Linear(hidden[-1], 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(in_dim, hidden[-1]),
            nn.ReLU(),
            nn.Linear(hidden[-1], action_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features  = self.shared(x)
        value     = self.value_stream(features)       # [B, 1]
        advantage = self.advantage_stream(features)   # [B, A]
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q


class StandardDQNNetwork(nn.Module):
    """Standard DQN without dueling (used when dueling_dqn=False)."""

    def __init__(self, state_size: int, action_size: int, hidden: list = None):
        super().__init__()
        hidden = hidden or [256, 128, 64]
        layers, in_dim = [], state_size
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, action_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Standard uniform experience replay buffer."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions,      dtype=np.int64),
            np.array(rewards,      dtype=np.float32),
            np.stack(next_states),
            np.array(dones,        dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────────────────────
# DQN Agent
# ─────────────────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    Full DQN agent with Double DQN + Dueling architecture + Soft target updates.

    Args:
        state_size  : Dimension of the observation vector.
        action_size : Number of discrete actions.
        config      : DQN_CONFIG dict from config.py.
        device      : torch.device.
    """

    def __init__(
        self,
        state_size  : int,
        action_size : int,
        config      : dict,
        device      : torch.device = None,
        # Allow calling as DQNAgent(obs_dim=8, n_actions=4, config=...) from benchmark
        obs_dim     : int = None,
        n_actions   : int = None,
    ):
        # Support alternate keyword names used by benchmark.py
        if obs_dim    is not None: state_size  = obs_dim
        if n_actions  is not None: action_size = n_actions

        self.state_size  = state_size
        self.action_size = action_size
        self.config      = config
        self.device      = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparams
        self.gamma          = config.get("gamma", 0.99)
        self.lr             = config.get("lr", 1e-4)
        self.batch_size     = config.get("batch_size", 64)
        self.buffer_size    = config.get("buffer_size", 100_000)
        self.tau            = config.get("tau", 0.005)
        self.use_soft_update= config.get("use_soft_update", True)
        self.target_update_freq = config.get("target_update_freq", 1000)
        self.grad_clip      = config.get("grad_clip", 1.0)
        self.huber_delta    = config.get("huber_delta", 1.0)
        self.double_dqn     = config.get("double_dqn", True)
        self.dueling_dqn    = config.get("dueling_dqn", True)
        self.use_per        = config.get("prioritized_replay", False)

        # Exploration
        self.epsilon        = config.get("epsilon_start", 1.0)
        self.epsilon_min    = config.get("epsilon_end",   0.01)
        self.epsilon_decay  = config.get("epsilon_decay", 0.995)

        self.is_training    = True
        self.t_step         = 0   # total update steps

        # Networks
        NetCls = DuelingDQNNetwork if self.dueling_dqn else StandardDQNNetwork
        self.qnetwork_local  = NetCls(state_size, action_size).to(self.device)
        self.qnetwork_target = NetCls(state_size, action_size).to(self.device)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        self.qnetwork_target.eval()

        # Optimiser + LR scheduler
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size = config.get("lr_decay_step", 200),
            gamma     = config.get("lr_decay_gamma", 0.5),
        )

        # Replay buffer
        self.memory = ReplayBuffer(self.buffer_size)

    # ── Action Selection ──────────────────────────────────────────────────────

    def choose_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if self.is_training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            q_values = self.qnetwork_local(state_t)
        if self.is_training:
            self.qnetwork_local.train()
        return int(q_values.argmax(dim=1).item())

    # ── Learning Step ─────────────────────────────────────────────────────────

    def step(
        self,
        state     : np.ndarray,
        action    : int,
        reward    : float,
        next_state: np.ndarray,
        done      : bool,
    ) -> float:
        """Store transition and, if buffer ready, perform a learning step. Returns loss (0 if no update)."""
        self.memory.push(state, action, reward, next_state, done)

        if len(self.memory) < self.batch_size:
            return 0.0

        self.t_step += 1
        loss = self._learn()

        # Target network update
        if self.use_soft_update:
            self._soft_update()
        elif self.t_step % self.target_update_freq == 0:
            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

        return loss

    def _learn(self) -> float:
        """Sample a batch and update Q-network. Returns scalar loss."""
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Current Q-values
        q_current = self.qnetwork_local(states_t).gather(1, actions_t)

        # Target Q-values — Double DQN
        with torch.no_grad():
            if self.double_dqn:
                best_actions = self.qnetwork_local(next_states_t).argmax(dim=1, keepdim=True)
                q_next = self.qnetwork_target(next_states_t).gather(1, best_actions)
            else:
                q_next = self.qnetwork_target(next_states_t).max(dim=1, keepdim=True)[0]

            q_target = rewards_t + self.gamma * q_next * (1 - dones_t)

        # Huber loss
        loss = F.huber_loss(q_current, q_target, delta=self.huber_delta)

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), self.grad_clip)
        self.optimizer.step()

        return float(loss.item())

    def _soft_update(self):
        """θ_target ← τ·θ_local + (1-τ)·θ_target"""
        for tp, lp in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            tp.data.copy_(self.tau * lp.data + (1.0 - self.tau) * tp.data)

    # ── Exploration Decay ─────────────────────────────────────────────────────

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ── Mode Switching ────────────────────────────────────────────────────────

    def set_training(self):
        self.is_training = True
        self.qnetwork_local.train()

    def set_evaluation(self):
        self.is_training = False
        self.qnetwork_local.eval()

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, filepath: str):
        """Save model weights and training state."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "qnetwork_local_state_dict" : self.qnetwork_local.state_dict(),
            "qnetwork_target_state_dict": self.qnetwork_target.state_dict(),
            "optimizer_state_dict"      : self.optimizer.state_dict(),
            "scheduler_state_dict"      : self.scheduler.state_dict(),
            "epsilon"                   : self.epsilon,
            "t_step"                    : self.t_step,
            "state_size"                : self.state_size,
            "action_size"               : self.action_size,
            "config"                    : self.config,
        }, filepath)

    def load(self, filepath: str):
        """Load model weights."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.qnetwork_local.load_state_dict(checkpoint["qnetwork_local_state_dict"])
        self.qnetwork_target.load_state_dict(checkpoint["qnetwork_target_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_min)
        self.t_step  = checkpoint.get("t_step",  0)
