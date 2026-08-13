"""
trainer.py — PPO Training Pipeline with Callbacks
==================================================

Architecture Role:
    This module orchestrates the TRAINING LOOP.
    It wires together the PPOTrafficAgent, SB3 callbacks, logging,
    and checkpointing into a single reproducible training run.

SB3 Callbacks Used:
─────────────────────
    EvalCallback:
        Periodically runs the policy on a separate evaluation environment
        (no exploration noise) and tracks mean reward.
        Saves the best model seen so far (based on eval mean reward).
        Prevents overfitting to the current rollout distribution.

    CheckpointCallback:
        Saves the model to disk every N steps.
        Enables resuming training after interruption.

    TrafficMetricsCallback (custom):
        After each rollout, logs traffic-specific KPIs to TensorBoard:
        - Mean waiting time per episode
        - Mean queue length per episode
        - Phase selection distribution

    StopTrainingOnRewardThreshold:
        Automatically stops training if the agent achieves a target
        mean reward in evaluation (useful for convergence detection).

Training Loop (SB3 internal):
    for each training iteration:
        1. ROLLOUT: collect n_steps from vec_env using current policy
        2. COMPUTE: advantages (GAE), returns
        3. UPDATE: run n_epochs SGD passes over the rollout buffer
        4. LOG: losses, metrics to TensorBoard
        5. CALLBACKS: eval, checkpoint, custom metrics
"""

import os
import time
from pathlib import Path
from typing import Optional, Callable

import numpy as np
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
    BaseCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.vec_env import DummyVecEnv

from ppo.environment.traffic_env import TrafficEnv
from ppo.agent.ppo_agent import PPOTrafficAgent
from ppo.utils.logger import get_logger
from ppo.utils.reproducibility import set_global_seed

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Custom Callback: Traffic Metrics Logger
# ---------------------------------------------------------------------------

class TrafficMetricsCallback(BaseCallback):
    """
    Custom SB3 callback that logs traffic-specific metrics to TensorBoard.

    SB3's BaseCallback provides hooks at key points in the training loop:
        _on_rollout_start()  : Before collecting a new rollout
        _on_step()           : After each environment step
        _on_rollout_end()    : After a full rollout is collected
        _on_training_end()   : After all training is done

    We use _on_rollout_end() to aggregate and log episode statistics
    from the info dicts collected during the rollout.

    How info dicts reach us:
        TrafficEnv.step() returns an 'info' dict.
        SB3 stores these in self.locals['infos'] during rollout collection.
        At rollout end, we process them here.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)
        self._episode_rewards    : list = []
        self._episode_wait_times : list = []
        self._episode_queues     : list = []
        self._action_counts      : dict = {0: 0, 1: 0, 2: 0, 3: 0}

    def _on_step(self) -> bool:
        """
        Called after each env step.
        We collect episode info when an episode terminates.
        Returns True to continue training (False would stop it).
        """
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep = info["episode"]
                self._episode_rewards.append(ep.get("r", 0.0))
                self._episode_wait_times.append(ep.get("mean_waiting_time", 0.0))
                self._episode_queues.append(ep.get("mean_queue", 0.0))

            # Track action distribution across all steps
            actions = self.locals.get("actions", [])
            if hasattr(actions, "__iter__"):
                for a in actions:
                    if int(a) in self._action_counts:
                        self._action_counts[int(a)] += 1

        return True  # continue training

    def _on_rollout_end(self) -> None:
        """
        Called after a complete rollout (n_steps steps collected).
        Log aggregated traffic metrics to TensorBoard.
        """
        if self._episode_rewards:
            # Log episode-level metrics
            self.logger.record(
                "traffic/mean_ep_reward",
                float(np.mean(self._episode_rewards))
            )
            self.logger.record(
                "traffic/mean_waiting_time",
                float(np.mean(self._episode_wait_times))
            )
            self.logger.record(
                "traffic/mean_queue_length",
                float(np.mean(self._episode_queues))
            )

            # Log phase selection distribution
            total_actions = sum(self._action_counts.values()) or 1
            for action_idx, count in self._action_counts.items():
                phase_name = ["NS_green", "EW_green", "NS_extend", "EW_extend"][action_idx]
                self.logger.record(
                    f"traffic/action_{phase_name}_pct",
                    count / total_actions * 100
                )

            # Reset accumulators for next rollout
            self._episode_rewards    = []
            self._episode_wait_times = []
            self._episode_queues     = []
            self._action_counts      = {0: 0, 1: 0, 2: 0, 3: 0}


# ---------------------------------------------------------------------------
# Main Trainer Class
# ---------------------------------------------------------------------------

class PPOTrainer:
    """
    Manages the complete PPO training pipeline.

    Responsibilities:
        - Set up directories (logs, checkpoints, best models)
        - Build training and evaluation environments
        - Construct and configure all callbacks
        - Run the training loop via PPOTrafficAgent
        - Save the final model

    Args:
        sumocfg_path     (str) : Path to SUMO config file.
        results_dir      (str) : Base directory for all outputs.
        total_timesteps  (int) : Total environment steps to train for.
        eval_freq        (int) : Steps between evaluation runs.
        n_eval_episodes  (int) : Episodes per evaluation run.
        checkpoint_freq  (int) : Steps between checkpoint saves.
        seed             (int) : Random seed for reproducibility.
        hyperparams_path (str) : Path to hyperparams.yaml.
        traci_port       (int) : TraCI port for training env.
        eval_traci_port  (int) : TraCI port for eval env (must differ!).
    """

    def __init__(
        self,
        sumocfg_path      : str,
        results_dir       : str  = "results",
        total_timesteps   : int  = 500_000,
        eval_freq         : int  = 10_000,
        n_eval_episodes   : int  = 5,
        checkpoint_freq   : int  = 50_000,
        seed              : int  = 42,
        hyperparams_path  : str  = None,
        traci_port        : int  = 8813,
        eval_traci_port   : int  = 8814,
    ):
        self.sumocfg_path      = sumocfg_path
        self.results_dir       = Path(results_dir)
        self.total_timesteps   = total_timesteps
        self.eval_freq         = eval_freq
        self.n_eval_episodes   = n_eval_episodes
        self.checkpoint_freq   = checkpoint_freq
        self.seed              = seed
        self.hyperparams_path  = hyperparams_path
        self.traci_port        = traci_port
        self.eval_traci_port   = eval_traci_port

        # Set up output directories
        self._setup_dirs()

        # Set global random seed for reproducibility
        set_global_seed(seed)

    def _setup_dirs(self) -> None:
        """Create all required output directories."""
        self.log_dir        = self.results_dir / "tensorboard_logs"
        self.checkpoint_dir = self.results_dir / "checkpoints"
        self.best_model_dir = self.results_dir / "best_model"
        self.final_model_dir = self.results_dir / "final_model"

        for d in [self.log_dir, self.checkpoint_dir, self.best_model_dir, self.final_model_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _make_train_env(self) -> callable:
        """
        Factory function for the training environment.
        Used by DummyVecEnv to create env instances.
        """
        cfg       = self.sumocfg_path
        port      = self.traci_port
        seed      = self.seed

        def _make():
            env = TrafficEnv(
                sumocfg_path = cfg,
                traci_port   = port,
                seed         = seed,
                use_gui      = False,
            )
            return env

        return _make

    def _make_eval_env(self) -> DummyVecEnv:
        """
        Create a separate environment for evaluation.
        Must use a DIFFERENT TraCI port than training env!

        Why separate eval env?
            Evaluation uses deterministic actions (no exploration).
            If we evaluated on the training env, it would interrupt
            the rollout collection and corrupt the training state.
        """
        cfg  = self.sumocfg_path
        port = self.eval_traci_port
        seed = self.seed + 1000  # different seed for eval diversity

        def _make():
            return TrafficEnv(
                sumocfg_path = cfg,
                traci_port   = port,
                seed         = seed,
                use_gui      = False,
            )

        return DummyVecEnv([_make])

    def _build_callbacks(self, eval_env: DummyVecEnv) -> CallbackList:
        """
        Construct and combine all SB3 callbacks.

        Callback execution order in SB3:
            Each callback's _on_step() is called in list order.
            EvalCallback runs evaluation when its internal step counter
            reaches eval_freq.
        """
        # ── 1. Stop if eval reward exceeds threshold ──────────────────
        # -0.5 is a reasonable "good" reward for our clipped [-1,1] range.
        stop_callback = StopTrainingOnRewardThreshold(
            reward_threshold = -0.3,  # adjust based on domain
            verbose          = 1,
        )

        # ── 2. Periodic evaluation + best model saving ────────────────
        eval_callback = EvalCallback(
            eval_env            = eval_env,
            best_model_save_path= str(self.best_model_dir),
            log_path            = str(self.log_dir),
            eval_freq           = self.eval_freq,
            n_eval_episodes     = self.n_eval_episodes,
            deterministic       = True,   # greedy eval
            render              = False,
            callback_on_new_best= stop_callback,  # stop if best is good enough
            verbose             = 1,
        )

        # ── 3. Periodic checkpoint saves ─────────────────────────────
        checkpoint_callback = CheckpointCallback(
            save_freq   = self.checkpoint_freq,
            save_path   = str(self.checkpoint_dir),
            name_prefix = "ppo_traffic",
            verbose     = 1,
        )

        # ── 4. Custom traffic metrics ─────────────────────────────────
        metrics_callback = TrafficMetricsCallback(verbose=0)

        return CallbackList([eval_callback, checkpoint_callback, metrics_callback])

    def train(self) -> PPOTrafficAgent:
        """
        Run the full training pipeline.

        Returns:
            The trained PPOTrafficAgent (model accessible via agent.model).
        """
        logger.info("=" * 60)
        logger.info("Starting PPO Traffic Signal Controller Training")
        logger.info(f"  Total timesteps : {self.total_timesteps:,}")
        logger.info(f"  Results dir     : {self.results_dir}")
        logger.info(f"  TensorBoard     : tensorboard --logdir {self.log_dir}")
        logger.info("=" * 60)

        # Build environments
        train_env_fn = self._make_train_env()
        eval_env     = self._make_eval_env()

        # Build agent
        agent = PPOTrafficAgent(
            env_fn           = train_env_fn,
            n_envs           = 1,
            log_dir          = str(self.log_dir),
            hyperparams_path = self.hyperparams_path,
            seed             = self.seed,
        )

        # Build callbacks
        callbacks = self._build_callbacks(eval_env)

        # ── TRAIN ─────────────────────────────────────────────────────
        start_time = time.time()
        logger.info("Training started...")

        agent.train(
            total_timesteps = self.total_timesteps,
            callbacks       = callbacks,
        )

        elapsed = time.time() - start_time
        logger.info(f"Training finished in {elapsed/60:.1f} minutes.")

        # Save final model
        final_path = str(self.final_model_dir / "ppo_traffic_final")
        agent.save(final_path)
        logger.info(f"Final model saved to: {final_path}")

        # Clean up environments — MUST be done before Python exits.
        # Without explicit close(), the DummyVecEnv holds the SUMO subprocess
        # open. When Python exits, the TraCI TCP socket is reset forcefully,
        # causing SUMO to print "tcpip::Socket::recvAndCheck: peer shutdown".
        eval_env.close()   # close evaluation environment first
        agent.close()       # close training environment + SUMO subprocess

        return agent
