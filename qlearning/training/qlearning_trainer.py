"""
qlearning_trainer.py — Q-Learning Training Pipeline
===================================================

Architecture Role:
    Orchestrates the TRAINING LOOP for Q-Learning.
    Handles environments, handling environments, logging to TensorBoard,
    checkpointing, and periodic evaluations.
"""

import time
import csv
from pathlib import Path
from typing import Dict

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from qlearning.environment.traffic_env import TrafficEnv
from qlearning.agent.qlearning_agent import QLearningAgent
from qlearning.utils.logger import get_logger
from qlearning.utils.reproducibility import set_global_seed

logger = get_logger(__name__)


class QLearningTrainer:
    """
    Manages the complete Q-Learning training pipeline.

    Args:
        sumocfg_path     (str) : Path to SUMO config file.
        results_dir      (str) : Base directory for all outputs.
        total_episodes   (int) : Total episodes to train for.
        eval_freq        (int) : Episodes between evaluation runs.
        n_eval_episodes  (int) : Episodes per evaluation run.
        checkpoint_freq  (int) : Episodes between checkpoint saves.
        seed             (int) : Random seed.
        hyperparams      (dict): Q-Learning configuration dict.
        traci_port       (int) : TraCI port for training env.
        eval_traci_port  (int) : TraCI port for eval env.
    """

    def __init__(
        self,
        sumocfg_path: str,
        results_dir: str = "results_qlearning",
        total_episodes: int = 1500,
        eval_freq: int = 50,
        n_eval_episodes: int = 5,
        checkpoint_freq: int = 100,
        seed: int = 42,
        hyperparams: dict = None,
        traci_port: int = 8816,
        eval_traci_port: int = 8817,
    ):
        self.sumocfg_path = sumocfg_path
        self.results_dir = Path(results_dir)
        self.total_episodes = total_episodes
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.checkpoint_freq = checkpoint_freq
        self.seed = seed
        self.hyperparams = hyperparams or {}
        self.traci_port = traci_port
        self.eval_traci_port = eval_traci_port

        # Q-Learning params
        ql_params = self.hyperparams.get("qlearning", {})
        self.alpha = ql_params.get("alpha", 0.1)
        self.gamma = ql_params.get("gamma", 0.99)
        self.epsilon_start = ql_params.get("epsilon_start", 1.0)
        self.epsilon_min = ql_params.get("epsilon_min", 0.05)
        self.epsilon_decay = ql_params.get("epsilon_decay", 0.9995)
        self.bins = ql_params.get("bins", 6)

        # Set up output directories
        self._setup_dirs()
        set_global_seed(seed)

        # Setup TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # Setup Monitor CSV for visualization plotter
        self._setup_monitor()

    def _setup_dirs(self) -> None:
        self.log_dir = self.results_dir / "tensorboard_logs"
        self.checkpoint_dir = self.results_dir / "checkpoints"
        self.best_model_dir = self.results_dir / "best_model"
        self.final_model_dir = self.results_dir / "final_model"
        self.monitor_dir = self.results_dir / "logs"

        for d in [self.log_dir, self.checkpoint_dir, self.best_model_dir, self.final_model_dir, self.monitor_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _setup_monitor(self) -> None:
        """Create a monitor.csv compatible with plotting utilities."""
        self.monitor_path = self.monitor_dir / "monitor.csv"
        # Write monitor header (comment line + csv header)
        with open(self.monitor_path, "w", newline="") as f:
            f.write('# {"t_start": 0.0, "env_id": "TrafficEnv"}\n')
            writer = csv.writer(f)
            writer.writerow(["r", "l", "t"])
        self.t_start = time.time()

    def _log_monitor(self, ep_reward: float, ep_length: int) -> None:
        """Append episode results to monitor.csv."""
        t_elapsed = round(time.time() - self.t_start, 2)
        with open(self.monitor_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([round(ep_reward, 4), ep_length, t_elapsed])

    def train(self) -> Dict[str, QLearningAgent]:
        """
        Run the Q-Learning training loop.
        Returns a dict of agents (accommodating multi-agent setups).
        """
        logger.info("=" * 60)
        logger.info("Starting Q-Learning Traffic Signal Controller Training")
        logger.info(f"  Total episodes  : {self.total_episodes}")
        logger.info(f"  Results dir     : {self.results_dir}")
        logger.info(f"  TensorBoard     : tensorboard --logdir {self.log_dir}")
        logger.info("=" * 60)

        env = TrafficEnv(
            sumocfg_path=self.sumocfg_path,
            traci_port=self.traci_port,
            seed=self.seed,
            use_gui=False,
        )

        # Initialize Q-Learning agent
        # The environment returns a dict if multi-agent, but we know it's a single agent
        # We will wrap it in a dict to accommodate future multi-agent
        agents: Dict[str, QLearningAgent] = {
            "center": QLearningAgent(
                action_space_n=env.action_space.n,
                alpha=self.alpha,
                gamma=self.gamma,
                epsilon=self.epsilon_start,
                epsilon_min=self.epsilon_min,
                epsilon_decay=self.epsilon_decay,
                bins=self.bins,
                agent_id="center"
            )
        }

        best_eval_reward = -float('inf')
        total_steps = 0

        start_time = time.time()
        logger.info("Training started...")

        for episode in range(1, self.total_episodes + 1):
            obs, _ = env.reset(seed=self.seed + episode)
            done = False
            ep_reward = 0.0
            ep_wait_times = []
            ep_queues = []
            ep_steps = 0

            # Action frequency tracking
            action_counts = {i: 0 for i in range(env.action_space.n)}

            # Only controlling "center" for now
            agent = agents["center"]

            while not done:
                action = agent.choose_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Train
                agent.learn(obs, action, reward, next_obs, terminated)

                # Accumulate stats
                ep_reward += reward
                ep_wait_times.append(info.get("total_waiting_time", 0.0))
                ep_queues.append(info.get("total_queue", 0.0))
                action_counts[action] += 1

                obs = next_obs
                ep_steps += 1
                total_steps += 1

            # Decay epsilon per episode
            agent.decay_epsilon()

            # Write monitor.csv
            self._log_monitor(ep_reward, ep_steps)

            # Log to TensorBoard
            self.writer.add_scalar("traffic/mean_ep_reward", ep_reward, total_steps)
            self.writer.add_scalar("traffic/mean_waiting_time", np.mean(ep_wait_times), total_steps)
            self.writer.add_scalar("traffic/mean_queue_length", np.mean(ep_queues), total_steps)
            self.writer.add_scalar("hyperparameters/epsilon", agent.epsilon, total_steps)

            for a_idx, count in action_counts.items():
                phase_name = ["NS_green", "EW_green", "NS_extend", "EW_extend"][a_idx]
                pct = count / max(1, ep_steps) * 100
                self.writer.add_scalar(f"traffic/action_{phase_name}_pct", pct, total_steps)

            # Console logging
            if episode % 10 == 0:
                logger.info(
                    f"Ep {episode:4d} | Steps: {total_steps:7d} | "
                    f"Reward: {ep_reward:7.2f} | Eps: {agent.epsilon:.3f} | "
                    f"Wait: {np.mean(ep_wait_times):5.1f}s | Queue: {np.mean(ep_queues):5.1f}"
                )

            # Evaluation and Checkpointing
            if episode % self.eval_freq == 0:
                eval_reward = self._evaluate_agents(agents, total_steps)
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    for agent_id, a in agents.items():
                        a.save(str(self.best_model_dir / f"q_model_{agent_id}.json"))
                    logger.info(f"New best model saved at episode {episode} with eval reward {eval_reward:.3f}")

            if episode % self.checkpoint_freq == 0:
                for agent_id, a in agents.items():
                    a.save(str(self.checkpoint_dir / f"q_model_{agent_id}_ep{episode}.json"))

        env.close()

        # Save final model
        for agent_id, a in agents.items():
            a.save(str(self.final_model_dir / f"q_model_{agent_id}_final.json"))

        elapsed = time.time() - start_time
        logger.info(f"Training finished in {elapsed/60:.1f} minutes.")
        self.writer.close()

        return agents

    def _evaluate_agents(self, agents: Dict[str, QLearningAgent], global_step: int) -> float:
        """Run deterministic evaluation during training."""
        eval_env = TrafficEnv(
            sumocfg_path=self.sumocfg_path,
            traci_port=self.eval_traci_port,
            seed=self.seed + 1000,
            use_gui=False,
        )

        # Set evaluation mode
        for a in agents.values():
            a.set_evaluation()

        eval_rewards = []
        for i in range(self.n_eval_episodes):
            obs, _ = eval_env.reset(seed=self.seed + 2000 + i)
            done = False
            ep_reward = 0.0

            while not done:
                # Currently single agent 'center'
                action = agents["center"].choose_action(obs)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                ep_reward += reward
                done = terminated or truncated

            eval_rewards.append(ep_reward)

        eval_env.close()

        # Restore training mode
        for a in agents.values():
            a.set_training()

        mean_reward = float(np.mean(eval_rewards))
        self.writer.add_scalar("eval/mean_reward", mean_reward, global_step)
        logger.info(f"--- Evaluation at step {global_step} | Mean Reward: {mean_reward:.3f} ---")
        return mean_reward
