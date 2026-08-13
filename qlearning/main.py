"""
main.py — Entry Point for the Q-Learning Traffic Signal Controller
====================================================================

Usage:
    # Train from scratch
    python main.py --mode train

    # Train with custom episodes
    python main.py --mode train --episodes 1500

    # Evaluate a saved model
    python main.py --mode evaluate --model results/best_model/q_model_center.json

    # Generate all plots from a monitor CSV
    python main.py --mode visualize --monitor results/logs/monitor.csv

    # Evaluate + compare vs baseline
    python main.py --mode evaluate --baseline
"""

import argparse
import sys
import os
from pathlib import Path

# ── Make sure qlearning/ is on the Python path when running from project root ──
_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_ROOT_DIR))

if "SUMO_HOME" not in os.environ:
    raise EnvironmentError(
        "\n[ERROR] SUMO_HOME environment variable is not set.\n"
        "Please set it:\n"
        "  Windows PowerShell: $env:SUMO_HOME = 'D:\\projects'\n"
        "  Windows CMD:        set SUMO_HOME=D:\\projects\n"
    )


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Q-Learning Traffic Signal Controller — Main Entry Point",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", type=str, choices=["train", "evaluate",
                        "visualize", "diagram"], default="train", help="Operating mode.")
    parser.add_argument("--train-episodes", type=int, default=None, help="Override total_episodes from config.")
    parser.add_argument("--model", type=str, default=None, help="Path to saved model JSON for evaluate mode.")
    parser.add_argument("--monitor", type=str, default=None, help="Path to monitor CSV for visualize mode.")
    parser.add_argument("--results-dir", type=str, default=str(_THIS_DIR / "results"), help="Directory for outputs.")
    parser.add_argument("--config", type=str, default=str(_THIS_DIR / "config" /
                        "qlearning.yaml"), help="Path to qlearning.yaml.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--baseline", action="store_true", default=False,
                        help="Run fixed-time baseline comparison in evaluate.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes.")
    parser.add_argument("--gui", action="store_true", default=False, help="Open SUMO-GUI.")
    return parser.parse_args()


def get_sumocfg_path() -> str:
    return str(_THIS_DIR.parent / "sumo_env" / "single_intersection.sumocfg")


def run_train(args: argparse.Namespace) -> None:
    import yaml
    from training.qlearning_trainer import QLearningTrainer

    config = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}

    train_cfg = config.get("training", {})
    total_episodes = args.train_episodes or train_cfg.get("total_episodes", 1500)

    trainer = QLearningTrainer(
        sumocfg_path=get_sumocfg_path(),
        results_dir=args.results_dir,
        total_episodes=total_episodes,
        eval_freq=train_cfg.get("eval_freq", 50),
        n_eval_episodes=train_cfg.get("n_eval_episodes", 5),
        checkpoint_freq=train_cfg.get("checkpoint_freq", 100),
        seed=args.seed,
        hyperparams=config,
        traci_port=config.get("environment", {}).get("traci_port", 8816),
        eval_traci_port=config.get("environment", {}).get("eval_traci_port", 8817),
    )

    trainer.train()

    print("\n[main] Training complete.")
    print(f"[main] Best model: {args.results_dir}/best_model/q_model_center.json")
    print(f"[main] View TensorBoard: tensorboard --logdir {args.results_dir}/tensorboard_logs")


def run_evaluate(args: argparse.Namespace) -> None:
    from evaluation.q_evaluator import QLearningEvaluator
    from visualization.plotter import TrafficPlotter

    model_path = args.model
    if model_path is None:
        model_path = str(Path(args.results_dir) / "best_model" / "q_model_center.json")

    if not Path(model_path).exists():
        print(f"[ERROR] Model not found at: {model_path}")
        sys.exit(1)

    evaluator = QLearningEvaluator(
        model_path=model_path,
        sumocfg_path=get_sumocfg_path(),
        n_episodes=args.episodes,
        traci_port=8818,
        results_dir=args.results_dir,
        seed=args.seed + 9999,
        use_gui=args.gui,
    )

    q_results = evaluator.evaluate()

    if args.baseline:
        baseline_results = evaluator.evaluate_baseline()
        plotter = TrafficPlotter(results_dir=args.results_dir)
        try:
            plotter.plot_eval_comparison(q_results, baseline_results)
            print("\n[main] Comparison plot saved to plots/eval_comparison.png")
        except Exception as e:
            print(f"[WARN] Could not plot baseline comparison: {e}")

    print("\n[main] Evaluation results saved.")


def run_visualize(args: argparse.Namespace) -> None:
    from visualization.plotter import TrafficPlotter
    plotter = TrafficPlotter(results_dir=args.results_dir)
    try:
        plotter.plot_system_diagram()
    except Exception as e:
        print(f"[WARN] Failed to plot diagram: {e}")

    monitor_path = args.monitor or str(Path(args.results_dir) / "logs" / "monitor.csv")
    if Path(monitor_path).exists():
        try:
            plotter.plot_all_from_monitor(monitor_path)
            print(f"\n[main] All plots saved to {args.results_dir}/plots/")
        except Exception as e:
            print(f"[WARN] Plotting from monitor failed: {e}")
    else:
        print(f"[WARN] Monitor CSV not found at {monitor_path}.")


def run_diagram(args: argparse.Namespace) -> None:
    from visualization.plotter import TrafficPlotter
    plotter = TrafficPlotter(results_dir=args.results_dir)
    path = plotter.plot_system_diagram()
    print(f"\n[main] Architecture diagram saved to: {path}")


if __name__ == "__main__":
    args = get_args()
    print("\n" + "=" * 65)
    print("  Q-Learning Traffic Signal Controller")
    print(f"  Mode : {args.mode.upper()}")
    print(f"  Seed : {args.seed}")
    print("=" * 65 + "\n")

    mode_dispatch = {
        "train": run_train,
        "evaluate": run_evaluate,
        "visualize": run_visualize,
        "diagram": run_diagram,
    }
    mode_dispatch[args.mode](args)
