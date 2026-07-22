"""
ppo_agent.py — PPO Agent Builder and Manager
============================================

Architecture Role:
    This module owns the PPO ALGORITHM itself.
    It wraps Stable-Baselines3's PPO with:
      - Our custom Actor-Critic architecture
      - Research-quality hyperparameter configuration
      - Model save/load helpers
      - A detailed explanation of every hyperparameter

PPO Algorithm Recap (Implementation Details):
─────────────────────────────────────────────
    The PPO training loop has two phases:

    PHASE 1 — ROLLOUT COLLECTION:
        The current policy π_θ interacts with the environment for
        N steps, collecting (s_t, a_t, r_t, s_{t+1}, done) tuples
        into a RolloutBuffer of size n_steps.

    PHASE 2 — POLICY UPDATE (runs n_epochs times):
        For each mini-batch of size batch_size:

        a) Compute Advantage using GAE:
           δ_t = r_t + γ·V(s_{t+1}) - V(s_t)      ← TD error
           Â_t = Σ_{k=0}^{T} (γλ)^k · δ_{t+k}     ← GAE

        b) Compute probability ratio:
           r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)

        c) PPO-Clip objective:
           L^CLIP = E[min(r_t·Â_t, clip(r_t, 1-ε, 1+ε)·Â_t)]

        d) Value loss (MSE between predicted and TD-target values):
           L^VF = E[(V_θ(s_t) - V_target)²]

        e) Entropy bonus (encourage exploration):
           H = E[-Σ π log π]

        f) Total loss (MINIMIZED by gradient descent):
           L = -L^CLIP + c₁·L^VF - c₂·H

Key PPO Hyperparameters:
────────────────────────
    n_steps (2048):
        Number of environment steps collected per rollout.
        Too small → high variance estimates.
        Too large → stale data, slow updates.
        2048 is standard for continuous/discrete control tasks.

    batch_size (64):
        Mini-batch size for each SGD update within an epoch.
        Must divide n_steps evenly.
        Larger batches → more stable gradients but more memory.

    n_epochs (10):
        How many passes over the rollout data per update.
        More epochs → better use of data but risk of overfitting
        to the current rollout (PPO's clipping limits this).

    gamma (0.99):
        Discount factor. γ=0.99 means future rewards are important
        but not too far: effective horizon ≈ 1/(1-γ) = 100 steps.
        For long-horizon traffic control, 0.99 is appropriate.

    gae_lambda (0.95):
        GAE smoothing. λ=1 → full Monte Carlo return (high variance).
        λ=0 → pure TD(0) (high bias). λ=0.95 is the standard balance.

    clip_range (0.2):
        PPO's key innovation. The policy ratio r_t is clipped to
        [1-0.2, 1+0.2] = [0.8, 1.2], preventing large policy updates
        that could destabilize training. The Schulman et al. paper
        recommends 0.1–0.3.

    ent_coef (0.01):
        Entropy bonus coefficient. Encourages the policy to remain
        stochastic (exploratory). Too high → random policy.
        Too low → premature convergence.

    vf_coef (0.5):
        Value function loss weight in the combined loss.
        Balances actor vs. critic update magnitude.

    learning_rate (3e-4):
        Adam optimizer learning rate. 3e-4 is the "golden rule"
        for Adam in deep RL (Andrej Karpathy's recommendation).

    max_grad_norm (0.5):
        Gradient clipping threshold. Prevents exploding gradients
        which are common in RL due to non-stationary data distribution.
"""

import os
from pathlib import Path
from typing import Optional

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from ppo.environment.traffic_env import TrafficEnv


# ---------------------------------------------------------------------------
# Default hyperparameters — can be overridden by hyperparams.yaml
# ---------------------------------------------------------------------------
DEFAULT_HYPERPARAMS = {
    # Rollout buffer size (steps collected before each update)
    "n_steps"        : 2048,

    # Mini-batch size for SGD
    "batch_size"     : 64,

    # Epochs over rollout data per update
    "n_epochs"       : 10,

    # Discount factor
    "gamma"          : 0.99,

    # GAE lambda (bias-variance trade-off for advantage estimation)
    "gae_lambda"     : 0.95,

    # PPO clipping parameter epsilon
    "clip_range"     : 0.2,

    # Entropy bonus coefficient (exploration)
    "ent_coef"       : 0.01,

    # Value function loss coefficient
    "vf_coef"        : 0.5,

    # Adam learning rate
    "learning_rate"  : 3e-4,

    # Gradient norm clipping
    "max_grad_norm"  : 0.5,

    # Normalize advantages within each mini-batch (recommended: True)
    "normalize_advantage": True,

    # Device: "cpu" or "cuda" or "auto"
    "device"         : "auto",
}


class PPOTrafficAgent:
    """
    Manages the PPO agent lifecycle: build, train, save, load.

    This is the primary interface between the Trainer and SB3's PPO.
    It abstracts away SB3 API details and exposes a clean interface.

    Args:
        env_fn    (callable)  : Factory function that returns a TrafficEnv.
        n_envs    (int)       : Number of parallel environments (≥1).
        log_dir   (str)       : Directory for TensorBoard logs.
        hyperparams_path (str): Path to hyperparams.yaml (optional).
        seed      (int)       : Global random seed.
    """

    def __init__(
        self,
        env_fn           : callable,
        n_envs           : int  = 1,
        log_dir          : str  = "results/logs",
        hyperparams_path : str  = None,
        seed             : int  = 42,
    ):
        self.env_fn    = env_fn
        self.n_envs    = n_envs
        self.log_dir   = log_dir
        self.seed      = seed

        # Load hyperparameters (YAML overrides defaults)
        self.hyperparams = self._load_hyperparams(hyperparams_path)

        # Build vectorized environment
        # DummyVecEnv: sequential (safe for TraCI; avoids port conflicts)
        # SubprocVecEnv: parallel (requires unique ports per env)
        self.vec_env = DummyVecEnv([env_fn] * n_envs)

        # Build PPO model
        self.model = self._build_model()

    def _load_hyperparams(self, path: Optional[str]) -> dict:
        """
        Load hyperparameters from YAML, falling back to defaults.

        YAML format:
            ppo:
              n_steps: 2048
              batch_size: 64
              ...
        """
        params = DEFAULT_HYPERPARAMS.copy()
        if path and Path(path).exists():
            with open(path, "r") as f:
                yaml_data = yaml.safe_load(f)
            if yaml_data and "ppo" in yaml_data:
                params.update(yaml_data["ppo"])
        return params

    def _build_model(self) -> PPO:
        """
        Instantiate SB3's PPO with our configuration.

        policy="MlpPolicy": SB3 uses this string to look up the policy class.
        We use "MlpPolicy" (the default) here because using our custom
        TrafficActorCriticPolicy requires subclassing in a way that
        can conflict with SB3's internal policy registry.

        Instead, we pass our architecture via policy_kwargs:

        SB3 2.x net_arch format (CHANGED from SB3 1.x):
            In SB3 >= 2.0, net_arch must be a dict with 'pi' and 'vf' keys.
            Each value is the FULL list of hidden layer widths for that head.
            The old list-with-dict format [256, dict(pi=[64], vf=[64])] is
            no longer valid and causes a TypeError at model creation.

            Our architecture:
              Input(18) -> Linear(256) -> ReLU -> Linear(128) -> ReLU  [shared]
                       -> Linear(64) -> ReLU -> Linear(n_actions)      [actor]
                       -> Linear(64) -> ReLU -> Linear(1)              [critic]

            Expressed as net_arch = dict(pi=[256,128,64], vf=[256,128,64]).
            SB3 deduces shared vs. separate layers automatically.

        activation_fn=nn.ReLU : Use ReLU in all MLP layers.
        ortho_init=True       : Orthogonal weight initialization (standard in RL).
        """
        import torch.nn as nn

        # SB3 >= 2.0 requires net_arch as a dict, NOT a list-with-dict.
        # pi = policy (actor) head layer sizes
        # vf = value  (critic) head layer sizes
        policy_kwargs = {
            "net_arch"     : dict(pi=[256, 128, 64], vf=[256, 128, 64]),
            "activation_fn": nn.ReLU,
            "ortho_init"   : True,
        }

        h = self.hyperparams
        model = PPO(
            policy               = "MlpPolicy",
            env                  = self.vec_env,
            # ── Core PPO hyperparameters ──────────────────────────────
            n_steps              = h["n_steps"],
            batch_size           = h["batch_size"],
            n_epochs             = h["n_epochs"],
            gamma                = h["gamma"],
            gae_lambda           = h["gae_lambda"],
            clip_range           = h["clip_range"],
            ent_coef             = h["ent_coef"],
            vf_coef              = h["vf_coef"],
            learning_rate        = h["learning_rate"],
            max_grad_norm        = h["max_grad_norm"],
            normalize_advantage  = h["normalize_advantage"],
            # ── Infrastructure ────────────────────────────────────────
            policy_kwargs        = policy_kwargs,
            # TensorBoard is optional. Probe at runtime: if tensorboard is
            # not importable (e.g. Python 3.14 compat issues), pass None so
            # SB3 falls back to stdout logging only. Training is unaffected.
            tensorboard_log      = self._resolve_tb_log_dir(),
            seed                 = self.seed,
            device               = h.get("device", "auto"),
            verbose              = 1,  # print training progress to stdout
        )
        return model

    def _resolve_tb_log_dir(self):
        """Return the TensorBoard log dir if tensorboard is importable, else None."""
        try:
            import tensorboard  # noqa: F401
            return self.log_dir
        except ImportError:
            print(
                "[PPOAgent] tensorboard not importable — disabling TB logging. "
                "All metrics will appear in stdout via verbose=1."
            )
            return None

    def train(self, total_timesteps: int, callbacks: list = None) -> PPO:
        """
        Run the PPO training loop.

        Args:
            total_timesteps: Total environment steps to train for.
            callbacks      : List of SB3 callback objects.

        Returns:
            The trained PPO model.
        """
        self.model.learn(
            total_timesteps   = total_timesteps,
            callback          = callbacks,
            tb_log_name       = "PPO_Traffic",
            reset_num_timesteps = True,
            progress_bar      = True,
        )
        return self.model

    def save(self, path: str) -> None:
        """
        Save the model to disk.
        SB3 saves policy weights, optimizer state, and hyperparameters.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        print(f"[PPOAgent] Model saved to: {path}")

    def load(self, path: str) -> None:
        """
        Load a saved model from disk.
        The environment is automatically reattached.
        """
        self.model = PPO.load(path, env=self.vec_env, device=self.hyperparams.get("device", "auto"))
        print(f"[PPOAgent] Model loaded from: {path}")

    def predict(self, obs, deterministic: bool = True):
        """
        Run inference with the trained policy.

        Args:
            obs         : Observation array/tensor.
            deterministic: If True, take argmax (greedy policy).
                           If False, sample from distribution (stochastic).

        Returns:
            action, state (SB3 convention)
        """
        return self.model.predict(obs, deterministic=deterministic)

    def close(self) -> None:
        """Close all environments in the VecEnv."""
        self.vec_env.close()
