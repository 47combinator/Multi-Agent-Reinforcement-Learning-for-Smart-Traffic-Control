"""
train_dqn.py — Deep Q-Network Training Pipeline
==============================================

Usage:
    python train_dqn.py [--resume RESULTS_DIR/checkpoints/checkpoint_epX.pth]
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from env.traffic_env import TrafficEnv
from models.dqn import DQNAgent
from config import ENV_CONFIG, DQN_CONFIG, LOG_FREQ, EVAL_FREQ, N_EVAL_EPISODES, CHECKPOINT_FREQ, EARLY_STOPPING_PATIENCE
from utils import get_logger, set_global_seed, TrafficPlotter

logger = get_logger(__name__)

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DQN Traffic Signal Controller Training")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pth file to resume training from.")
    parser.add_argument("--seed", type=int, default=DQN_CONFIG.get("seed", 42), help="Random seed.")
    parser.add_argument("--gui", action="store_true", default=False, help="Enable SUMO-GUI.")
    return parser.parse_args()


def evaluate(agent: DQNAgent, env_config: dict, n_episodes: int, seed: int) -> dict:
    """
    Evaluate agent deterministically on a separate environment instance.
    """
    eval_env = TrafficEnv(
        sumocfg_path=env_config["sumocfg_path"],
        tls_id=env_config["tls_id"],
        lane_ids=env_config["lane_ids"],
        delta_t=env_config["delta_t"],
        max_steps=env_config["max_steps"],
        traci_port=env_config["eval_traci_port"],
        use_gui=False,
        seed=seed
    )

    agent.set_evaluation()
    
    eval_rewards = []
    eval_wait_times = []
    eval_queues = []
    eval_throughputs = []
    eval_delays = []

    for ep in range(n_episodes):
        obs, _ = eval_env.reset(seed=seed + 1000 + ep)
        done = False
        ep_reward = 0.0
        
        ep_wait_times = []
        ep_queues = []
        ep_delays = []
        ep_throughput = 0

        while not done:
            action = agent.choose_action(obs)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            
            ep_reward += reward
            ep_wait_times.append(info.get("total_waiting_time", 0.0))
            ep_queues.append(info.get("total_queue", 0.0))
            ep_delays.append(info.get("mean_delay", 0.0))
            ep_throughput += info.get("metrics/step_throughput", 0.0)

        eval_rewards.append(ep_reward)
        eval_wait_times.append(np.mean(ep_wait_times) if ep_wait_times else 0.0)
        eval_queues.append(np.mean(ep_queues) if ep_queues else 0.0)
        eval_delays.append(np.mean(ep_delays) if ep_delays else 0.0)
        eval_throughputs.append(ep_throughput)

    eval_env.close()
    agent.set_training()

    return {
        "reward": float(np.mean(eval_rewards)),
        "waiting_time": float(np.mean(eval_wait_times)),
        "queue": float(np.mean(eval_queues)),
        "throughput": float(np.mean(eval_throughputs)),
        "delay": float(np.mean(eval_delays))
    }


def main():
    args = get_args()
    
    # Seeding
    set_global_seed(args.seed)

    # Set up folders
    results_dir = Path("results_dqn")
    results_dir.mkdir(exist_ok=True, parents=True)
    
    log_dir = results_dir / "tensorboard_logs"
    checkpoint_dir = results_dir / "checkpoints"
    best_model_dir = results_dir / "best_model"
    
    for d in [log_dir, checkpoint_dir, best_model_dir]:
        d.mkdir(exist_ok=True, parents=True)

    # Initialize TensorBoard Writer
    writer = SummaryWriter(log_dir=str(log_dir))
    plotter = TrafficPlotter(results_dir=str(results_dir))

    # Initialize Environment
    env = TrafficEnv(
        sumocfg_path=ENV_CONFIG["sumocfg_path"],
        tls_id=ENV_CONFIG["tls_id"],
        lane_ids=ENV_CONFIG["lane_ids"],
        delta_t=ENV_CONFIG["delta_t"],
        max_steps=ENV_CONFIG["max_steps"],
        traci_port=ENV_CONFIG["traci_port"],
        use_gui=args.gui,
        seed=args.seed
    )

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Initialize Agent
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        config=DQN_CONFIG,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    start_episode = 1
    best_eval_reward = -float("inf")
    early_stop_counter = 0

    # Resume Training
    if args.resume:
        logger.info(f"Resuming training from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=agent.device)
        agent.qnetwork_local.load_state_dict(checkpoint['qnetwork_local_state_dict'])
        agent.qnetwork_target.load_state_dict(checkpoint['qnetwork_target_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.epsilon = checkpoint['epsilon']
        agent.t_step = checkpoint['t_step']
        if 'scheduler_state_dict' in checkpoint:
            agent.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Deduce starting episode from file name if possible, or start from 1
        # filename is usually checkpoint_epX.pth
        try:
            fn = Path(args.resume).stem
            start_episode = int(fn.split("_ep")[-1]) + 1
        except Exception:
            start_episode = 1

    # Keep track of history for CSV saving
    history = {
        "episode": [],
        "reward": [],
        "loss": [],
        "avg_waiting_time": [],
        "avg_queue": [],
        "avg_delay": [],
        "throughput": [],
        "epsilon": [],
        "time": []
    }

    logger.info("=" * 60)
    logger.info("Starting DQN Traffic Signal Controller Training")
    logger.info(f"  State size     : {state_size}")
    logger.info(f"  Action size    : {action_size}")
    logger.info(f"  Device         : {agent.device}")
    logger.info(f"  Double DQN     : {agent.double_dqn}")
    logger.info(f"  Dueling DQN    : {agent.dueling_dqn}")
    logger.info(f"  Prioritized RX : {agent.use_per}")
    logger.info(f"  Resume         : {args.resume is not None}")
    logger.info(f"  GUI            : {args.gui}")
    logger.info("=" * 60)

    start_time = time.time()
    total_steps = agent.t_step

    for episode in range(start_episode, DQN_CONFIG["num_episodes"] + 1):
        ep_start_time = time.time()
        obs, _ = env.reset(seed=args.seed + episode)
        done = False
        ep_reward = 0.0
        
        ep_wait_times = []
        ep_queues = []
        ep_delays = []
        ep_losses = []
        ep_throughput = 0
        ep_steps = 0

        while not done:
            action = agent.choose_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Learn and optimize
            loss = agent.step(obs, action, reward, next_obs, terminated)
            
            ep_reward += reward
            ep_wait_times.append(info.get("total_waiting_time", 0.0))
            ep_queues.append(info.get("total_queue", 0.0))
            ep_delays.append(info.get("mean_delay", 0.0))
            ep_throughput += info.get("metrics/step_throughput", 0.0)
            if loss > 0:
                ep_losses.append(loss)

            obs = next_obs
            ep_steps += 1
            total_steps += 1

        # Decay exploration
        agent.decay_epsilon()
        # Decay learning rate
        agent.scheduler.step()

        # Gather metrics
        mean_wait = float(np.mean(ep_wait_times)) if ep_wait_times else 0.0
        mean_queue = float(np.mean(ep_queues)) if ep_queues else 0.0
        mean_delay = float(np.mean(ep_delays)) if ep_delays else 0.0
        mean_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        ep_duration = time.time() - ep_start_time

        # Save to history dictionary
        history["episode"].append(episode)
        history["reward"].append(ep_reward)
        history["loss"].append(mean_loss)
        history["avg_waiting_time"].append(mean_wait)
        history["avg_queue"].append(mean_queue)
        history["avg_delay"].append(mean_delay)
        history["throughput"].append(ep_throughput)
        history["epsilon"].append(agent.epsilon)
        history["time"].append(ep_duration)

        # Log to TensorBoard
        writer.add_scalar("train/episode_reward", ep_reward, episode)
        writer.add_scalar("train/loss", mean_loss, episode)
        writer.add_scalar("train/waiting_time", mean_wait, episode)
        writer.add_scalar("train/queue_length", mean_queue, episode)
        writer.add_scalar("train/delay", mean_delay, episode)
        writer.add_scalar("train/throughput", ep_throughput, episode)
        writer.add_scalar("train/epsilon", agent.epsilon, episode)
        writer.add_scalar("train/lr", agent.optimizer.param_groups[0]["lr"], episode)

        # Console logging
        if episode % LOG_FREQ == 0:
            elapsed_min = (time.time() - start_time) / 60
            logger.info(
                f"Episode {episode:4d}/{DQN_CONFIG['num_episodes']} | "
                f"Steps: {total_steps:7d} | "
                f"Reward: {ep_reward:7.2f} | "
                f"Wait: {mean_wait:5.1f}s | "
                f"Queue: {mean_queue:4.1f} | "
                f"Delay: {mean_delay:5.3f} | "
                f"Throughput: {ep_throughput:4.0f} | "
                f"Loss: {mean_loss:6.4f} | "
                f"Eps: {agent.epsilon:.3f} | "
                f"Time: {elapsed_min:5.1f}m"
            )

        # Periodic Evaluation
        if episode % EVAL_FREQ == 0:
            eval_metrics = evaluate(agent, ENV_CONFIG, N_EVAL_EPISODES, args.seed)
            writer.add_scalar("eval/mean_reward", eval_metrics["reward"], episode)
            writer.add_scalar("eval/mean_waiting_time", eval_metrics["waiting_time"], episode)
            writer.add_scalar("eval/mean_queue_length", eval_metrics["queue"], episode)
            writer.add_scalar("eval/mean_delay", eval_metrics["delay"], episode)
            writer.add_scalar("eval/throughput", eval_metrics["throughput"], episode)
            
            logger.info(
                f"[EVALUATION] Episode {episode} | "
                f"Reward: {eval_metrics['reward']:.2f} | "
                f"Wait: {eval_metrics['waiting_time']:.1f}s | "
                f"Queue: {eval_metrics['queue']:.1f} | "
                f"Delay: {eval_metrics['delay']:.3f} | "
                f"Throughput: {eval_metrics['throughput']:.0f}"
            )

            # Check if best model
            if eval_metrics["reward"] > best_eval_reward:
                best_eval_reward = eval_metrics["reward"]
                agent.save(str(best_model_dir / "best_dqn_model.pth"))
                logger.info(f"--> Saved new best DQN model with eval reward {best_eval_reward:.2f}")
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                logger.info(f"Eval reward did not improve. Early stop counter: {early_stop_counter}/{EARLY_STOPPING_PATIENCE}")

            # Check early stopping
            if early_stop_counter >= EARLY_STOPPING_PATIENCE:
                logger.warning(f"Early stopping triggered! Training stopped at episode {episode}.")
                break

        # Save regular checkpoints
        if episode % CHECKPOINT_FREQ == 0:
            agent.save(str(checkpoint_dir / f"checkpoint_ep{episode}.pth"))

    env.close()

    # Save final model
    agent.save(str(results_dir / "final_dqn_model.pth"))
    logger.info("Training completed.")

    # Save training history to CSV
    df = pd.DataFrame(history)
    df.to_csv(results_dir / "training_history.csv", index=False)
    logger.info(f"Saved training history to {results_dir / 'training_history.csv'}")

    # Generate training plots
    try:
        plotter.plot_dqn_training(
            episodes=history["episode"],
            rewards=history["reward"],
            losses=history["loss"],
            wait_times=history["avg_waiting_time"],
            queues=history["avg_queue"],
            epsilons=history["epsilon"]
        )
        logger.info("Saved training dashboard plots.")
    except Exception as e:
        logger.warning(f"Failed to generate training plots: {e}")

    writer.close()

if __name__ == "__main__":
    main()
