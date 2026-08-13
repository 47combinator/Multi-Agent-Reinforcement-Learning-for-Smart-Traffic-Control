"""
networks.py — Custom Actor-Critic Neural Network Architecture
=============================================================

Architecture Role:
    This module defines the BRAIN of the PPO agent:
    the Actor network (policy) and Critic network (value function).

Why customize instead of using SB3 defaults?
    SB3's default MlpPolicy uses separate Actor/Critic networks with
    no shared layers. For traffic control:
      - Shared layers learn general traffic representations
        (queue dynamics, congestion patterns) used by BOTH actor and critic.
      - Separate heads specialize: actor learns WHAT to do, critic
        learns HOW GOOD the state is.
    This parameter sharing improves sample efficiency.

Architecture:

    Input (18-dim obs)
         │
    ┌────▼─────────────────────────────────────────────────┐
    │  SHARED BACKBONE (shared between actor and critic)   │
    │  Linear(18 → 256) → LayerNorm → ReLU               │
    │  Linear(256 → 128) → LayerNorm → ReLU              │
    └────────────────────────────┬─────────────────────────┘
                    ┌────────────┴────────────┐
              ┌─────▼──────┐           ┌──────▼──────┐
              │  ACTOR HEAD│           │ CRITIC HEAD │
              │  Linear    │           │  Linear     │
              │  (128→64)  │           │  (128→64)   │
              │  → ReLU    │           │  → ReLU     │
              │  Linear    │           │  Linear     │
              │  (64→4)    │           │  (64→1)     │
              │  → logits  │           │  → V(s)     │
              └────────────┘           └─────────────┘

Why LayerNorm instead of BatchNorm?
    In RL, batch statistics fluctuate wildly between rollouts (unlike
    supervised learning). LayerNorm normalizes per-sample, per-layer,
    making it robust to the non-stationary distribution of RL data.

Why ReLU?
    Simple, stable, fast. Avoids vanishing gradients for this depth.
    Tanh (SB3 default) saturates at ±1; ReLU avoids this.

SB3 Integration:
    SB3 expects a custom policy via 'policy_kwargs':
        policy_kwargs = {
            "features_extractor_class"  : nn.Identity (we do own extraction),
            "net_arch"                  : [],  # we bypass SB3's arch builder
            "activation_fn"             : nn.ReLU,
        }
    We use a cleaner approach: subclass ActorCriticPolicy directly.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, List, Optional, Union, Type
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule


class TrafficSharedBackbone(nn.Module):
    """
    Shared feature extraction backbone used by both Actor and Critic.

    This network takes the raw observation and produces a latent
    representation that captures traffic state semantics.

    Args:
        obs_dim     (int): Input observation dimension (18).
        hidden1     (int): Size of first hidden layer.
        hidden2     (int): Size of second hidden layer.
    """

    def __init__(self, obs_dim: int = 18, hidden1: int = 256, hidden2: int = 128):
        super().__init__()

        self.backbone = nn.Sequential(
            # Layer 1: Project raw obs into a rich representation
            nn.Linear(obs_dim, hidden1),
            nn.LayerNorm(hidden1),  # Normalize across features (not batch)
            nn.ReLU(),

            # Layer 2: Compress and refine representation
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.ReLU(),
        )

        # Output dimension is used by SB3 to size the policy/value heads
        self.output_dim = hidden2

        # Weight initialization: orthogonal init is standard for RL policies
        # It ensures good gradient flow at the start of training.
        self._init_weights()

    def _init_weights(self) -> None:
        """Orthogonal initialization for all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs (Tensor): Shape (batch_size, obs_dim)

        Returns:
            Tensor: Shape (batch_size, hidden2) — latent features
        """
        return self.backbone(obs)


class TrafficActorCriticPolicy(ActorCriticPolicy):
    """
    Custom Actor-Critic policy for traffic signal control.

    Plugs into SB3's PPO by subclassing ActorCriticPolicy.
    SB3 automatically calls _build_mlp_extractor() to set up the
    feature extractor, then uses it for both the policy and value networks.

    The key override is _build_mlp_extractor(), where we inject our
    custom shared backbone instead of SB3's default separate MLPs.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule: Schedule,
        *args,
        **kwargs,
    ):
        # net_arch=[] tells SB3 to NOT build its own MLP layers,
        # since we manage the architecture ourselves.
        kwargs["net_arch"]      = []
        kwargs["activation_fn"] = nn.ReLU

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

        # Force weight initialization after parent __init__
        # (parent calls _build_mlp_extractor internally)
        self._initialize_policy_weights()

    def _build_mlp_extractor(self) -> None:
        """
        Override SB3's default MLP builder with our custom backbone.

        SB3 calls this during __init__ to populate:
            self.mlp_extractor  : used for feature extraction
            self.latent_dim_pi  : actor head input dim
            self.latent_dim_vf  : critic head input dim
        """
        obs_dim = self.observation_space.shape[0]

        # Our shared backbone
        self.mlp_extractor = TrafficSharedBackbone(
            obs_dim  = obs_dim,
            hidden1  = 256,
            hidden2  = 128,
        )

        # Actor head: 128 → 64 → n_actions
        self.policy_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # Critic head: 128 → 64 → 1
        self.value_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # Tell SB3 the output dims of actor/critic latent spaces
        # SB3 appends its own final linear layers: latent→n_actions and latent→1
        self.latent_dim_pi = 64
        self.latent_dim_vf = 64

    def _initialize_policy_weights(self) -> None:
        """
        Apply orthogonal initialization to policy and value heads.
        Actor output layer uses gain=0.01 (small init for stable policy),
        Critic output layer uses gain=1.0 (standard).
        """
        # Find the final linear layers added by SB3
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if "action_net" in name:
                    # Small init for policy output → near-uniform initial policy
                    nn.init.orthogonal_(module.weight, gain=0.01)
                    nn.init.zeros_(module.bias)
                elif "value_net" in name:
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        obs         : torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the full Actor-Critic network.

        Args:
            obs          : Observation tensor (batch, obs_dim)
            deterministic: If True, take argmax (eval mode). If False, sample.

        Returns:
            actions     : Sampled or argmax actions
            values      : State values V(s) from critic
            log_probs   : Log probabilities of the sampled actions
        """
        # 1. Shared backbone: obs → latent features
        features = self.extract_features(obs, self.pi_features_extractor)
        latent   = self.mlp_extractor(features)

        # 2. Actor head → action distribution
        pi_latent = self.policy_net(latent)
        distribution = self._get_action_dist_from_latent(pi_latent)

        # 3. Critic head → state value
        vf_latent = self.value_net(latent)
        values    = self.value_net_output(vf_latent)  # SB3's final linear layer

        # 4. Sample action
        actions   = distribution.get_actions(deterministic=deterministic)
        log_probs = distribution.log_prob(actions)

        return actions, values, log_probs
