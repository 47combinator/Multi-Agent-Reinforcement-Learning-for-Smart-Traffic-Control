"""
q_evaluator.py — Policy Evaluation for Q-Learning
=================================================

Architecture Role:
    Evaluates a trained Q-Learning policy deterministically.
    Mirrors evaluation standards to produce the same CSV output and dict 
    format as fixed-time baseline comparisons. with TrafficPlotter.
"""

import csv
from pathlib import Path
import numpy as np

from qlearning.environment.traffic_env import TrafficEnv
from qlearning.agent.qlearning_agent import QLearningAgent
from qlearning.utils.logger import get_logger

logger = get_logger(__name__)


class QLearningEvaluator:
    """
    Evaluates a trained Q-Learning policy on the traffic environment.
    """

    def __init__(
        self,
        model_path: str,
        sumocfg_path: str,
        n_episodes: int = 10,
        traci_port: int = 8815,
        results_dir: str = "results_qlearning",
        seed: int = 9999,
        use_gui: bool = False,
    ):
        self.model_path = model_path
        self.sumocfg_path = sumocfg_path
        self.n_episodes = n_episodes
        self.traci_port = traci_port
        self.results_dir = Path(results_dir)
        self.seed = seed
        self.use_gui = use_gui

        self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading Q-Learning model from: {model_path}")
        # Note: action_space_n is temporarily dummy, it doesn't matter for evaluation
        # since it's just doing lookups. We assume 4 based on env.
        self.agent = QLearningAgent(action_space_n=4)
        self.agent.load(model_path)
        self.agent.set_evaluation()

    def evaluate(self) -> dict:
        """Run n_episodes evaluation rollouts deterministically."""
        logger.info(f"Evaluating Q-Learning policy over {self.n_episodes} episodes...")

        episode_metrics = []

        for ep_idx in range(self.n_episodes):
            ep_seed = self.seed + ep_idx

            env = TrafficEnv(
                sumocfg_path=self.sumocfg_path,
                traci_port=self.traci_port,
                seed=ep_seed,
                use_gui=self.use_gui,
            )

            obs, _ = env.reset(seed=ep_seed)
            ep_reward = 0.0
            ep_wait_times = []
            ep_queues = []
            done = False

            while not done:
                action = self.agent.choose_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                ep_wait_times.append(info.get("total_waiting_time", 0.0))
                ep_queues.append(info.get("total_queue", 0.0))
                done = terminated or truncated

            env.close()

            ep_stats = {
                "episode": ep_idx + 1,
                "total_reward": round(ep_reward, 4),
                "mean_waiting_time": round(float(np.mean(ep_wait_times)), 2),
                "mean_queue": round(float(np.mean(ep_queues)), 2),
                "max_waiting_time": round(float(np.max(ep_wait_times)), 2),
                "max_queue": round(float(np.max(ep_queues)), 2),
            }
            episode_metrics.append(ep_stats)

            logger.info(
                f"  Episode {ep_idx+1:2d} | "
                f"Reward: {ep_stats['total_reward']:7.3f} | "
                f"AvgWait: {ep_stats['mean_waiting_time']:6.1f}s | "
                f"AvgQueue: {ep_stats['mean_queue']:5.1f}"
            )

        rewards = [m["total_reward"] for m in episode_metrics]
        wait_times = [m["mean_waiting_time"] for m in episode_metrics]
        queues = [m["mean_queue"] for m in episode_metrics]

        results = {
            "n_episodes": self.n_episodes,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_waiting_time": float(np.mean(wait_times)),
            "std_waiting_time": float(np.std(wait_times)),
            "mean_queue": float(np.mean(queues)),
            "std_queue": float(np.std(queues)),
            "episode_data": episode_metrics,
        }

        logger.info("\n" + "=" * 50)
        logger.info("EVALUATION SUMMARY")
        logger.info(f"  Mean Reward       : {results['mean_reward']:.4f} ± {results['std_reward']:.4f}")
        logger.info(f"  Mean Waiting Time : {results['mean_waiting_time']:.2f}s ± {results['std_waiting_time']:.2f}s")
        logger.info(f"  Mean Queue Length : {results['mean_queue']:.2f} ± {results['std_queue']:.2f}")
        logger.info("=" * 50)

        self._save_csv(episode_metrics)
        return results

    def _save_csv(self, episode_metrics: list) -> None:
        csv_path = self.results_dir / "evaluation_results.csv"
        if not episode_metrics:
            return
        fieldnames = list(episode_metrics[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(episode_metrics)
        logger.info(f"Evaluation results saved to: {csv_path}")

    def evaluate_baseline(self) -> dict:
        """
        Evaluate the FIXED-TIME baseline controller.

        The baseline lets SUMO's default traffic light cycle run
        without any RL intervention. This is the standard benchmark.

        We apply action 0 always (hold NS green), which lets
        SUMO's internal phase duration timers control the light.
        In our env, we use action cycling with period matching the
        default 32+3+32+3 = 70 second cycle.
        """
        logger.info("Evaluating FIXED-TIME baseline...")

        episode_metrics = []

        for ep_idx in range(self.n_episodes):
            ep_seed = self.seed + 5000 + ep_idx

            env = TrafficEnv(
                sumocfg_path=self.sumocfg_path,
                traci_port=self.traci_port,
                seed=ep_seed,
                use_gui=False,
            )

            obs, _ = env.reset(seed=ep_seed)

            ep_reward = 0.0
            ep_wait_times = []
            ep_queues = []
            done = False
            step_count = 0

            # Fixed-time policy: alternate NS/EW every 7 steps
            # (7 steps × 5s/step = 35s ≈ standard green phase duration)
            while not done:
                # Fixed-time: cycle NS green for 7 steps, then EW for 7
                fixed_action = 0 if (step_count // 7) % 2 == 0 else 1
                obs, reward, terminated, truncated, info = env.step(fixed_action)

                ep_reward += reward
                ep_wait_times.append(info.get("total_waiting_time", 0.0))
                ep_queues.append(info.get("total_queue", 0.0))
                step_count += 1
                done = terminated or truncated

            env.close()

            ep_stats = {
                "episode": ep_idx + 1,
                "total_reward": round(ep_reward, 4),
                "mean_waiting_time": round(float(np.mean(ep_wait_times)), 2),
                "mean_queue": round(float(np.mean(ep_queues)), 2),
            }
            episode_metrics.append(ep_stats)

        rewards = [m["total_reward"] for m in episode_metrics]
        wait_times = [m["mean_waiting_time"] for m in episode_metrics]
        queues = [m["mean_queue"] for m in episode_metrics]

        results = {
            "controller": "fixed_time_baseline",
            "mean_reward": float(np.mean(rewards)),
            "mean_waiting_time": float(np.mean(wait_times)),
            "mean_queue": float(np.mean(queues)),
        }

        logger.info(f"Baseline — Mean Reward: {results['mean_reward']:.4f} | "
                    f"Wait: {results['mean_waiting_time']:.2f}s | "
                    f"Queue: {results['mean_queue']:.2f}")

        return results
