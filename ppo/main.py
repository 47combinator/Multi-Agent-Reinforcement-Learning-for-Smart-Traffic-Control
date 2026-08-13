"""
main.py — Entry Point for the PPO Traffic Signal Controller
============================================================

Usage:
    # Train from scratch
    python main.py --mode train

    # Train with custom timesteps
    python main.py --mode train --timesteps 1000000

    # Evaluate a saved model
    python main.py --mode evaluate --model results/best_model/best_model

    # Generate all plots from a monitor CSV
    python main.py --mode visualize --monitor results/logs/monitor.csv

    # Quick smoke test (10k steps)
    python main.py --mode train --timesteps 10000

    # Evaluate + compare vs baseline
    python main.py --mode evaluate --baseline

    # Generate the system architecture diagram only
    python main.py --mode diagram
"""

import argparse
import sys
import os
from pathlib import Path

# ── Make sure ppo/ is on the Python path when running from project root ──
_THIS_DIR = Path(__file__).resolve().parent  # ppo/
_ROOT_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_ROOT_DIR))

# ── SUMO_HOME must be set before importing any TraCI module ─────────────
if "SUMO_HOME" not in os.environ:
    raise EnvironmentError(
        "\n[ERROR] SUMO_HOME environment variable is not set.\n"
        "Please set it:\n"
        "  Windows PowerShell: $env:SUMO_HOME = 'D:\\projects'\n"
        "  Windows CMD:        set SUMO_HOME=D:\\projects\n"
    )


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description = "PPO Traffic Signal Controller — Main Entry Point",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        type    = str,
        choices = ["train", "evaluate", "visualize", "diagram"],
        default = "train",
        help    = "Operating mode.",
    )
    parser.add_argument(
        "--timesteps",
        type    = int,
        default = None,
        help    = "Override total_timesteps from config.",
    )
    parser.add_argument(
        "--model",
        type    = str,
        default = None,
        help    = "Path to saved model .zip for evaluate mode.",
    )
    parser.add_argument(
        "--monitor",
        type    = str,
        default = None,
        help    = "Path to SB3 monitor CSV for visualize mode.",
    )
    parser.add_argument(
        "--results-dir",
        type    = str,
        default = str(_THIS_DIR / "results"),
        help    = "Directory for all outputs.",
    )
    parser.add_argument(
        "--config",
        type    = str,
        default = str(_THIS_DIR / "config" / "hyperparams.yaml"),
        help    = "Path to hyperparams.yaml.",
    )
    parser.add_argument(
        "--seed",
        type    = int,
        default = 42,
        help    = "Random seed.",
    )
    parser.add_argument(
        "--baseline",
        action  = "store_true",
        default = False,
        help    = "In evaluate mode, also run fixed-time baseline comparison.",
    )
    parser.add_argument(
        "--episodes",
        type    = int,
        default = 10,
        help    = "Number of evaluation episodes.",
    )
    parser.add_argument(
        "--gui",
        action  = "store_true",
        default = False,
        help    = "Open SUMO-GUI during evaluation (slow; for debugging).",
    )

    return parser.parse_args()


def get_sumocfg_path() -> str:
    """Resolve absolute path to the SUMO config file."""
    return str(_THIS_DIR.parent / "sumo_env" / "single_intersection.sumocfg")


# ─────────────────────────────────────────────────────────────────────────────
# MODE: TRAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_train(args: argparse.Namespace) -> None:
    """
    Launch the full PPO training pipeline.

    Steps:
        1. Load hyperparams.yaml
        2. Build training + evaluation environments
        3. Construct PPO agent and callbacks
        4. Call model.learn(total_timesteps)
        5. Save final model to results/final_model/
    """
    import yaml
    from training.trainer import PPOTrainer

    # Load config
    config = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}

    train_cfg = config.get("training", {})
    total_ts  = args.timesteps or train_cfg.get("total_timesteps", 500_000)

    trainer = PPOTrainer(
        sumocfg_path     = get_sumocfg_path(),
        results_dir      = args.results_dir,
        total_timesteps  = total_ts,
        eval_freq        = train_cfg.get("eval_freq", 10_000),
        n_eval_episodes  = train_cfg.get("n_eval_episodes", 5),
        checkpoint_freq  = train_cfg.get("checkpoint_freq", 50_000),
        seed             = args.seed,
        hyperparams_path = args.config,
        traci_port       = config.get("environment", {}).get("traci_port", 8813),
        eval_traci_port  = config.get("environment", {}).get("eval_traci_port", 8814),
    )

    agent = trainer.train()

    print("\n[main] Training complete.")
    print(f"[main] Best model: {args.results_dir}/best_model/best_model.zip")
    print(f"[main] View TensorBoard: tensorboard --logdir {args.results_dir}/tensorboard_logs")


# ─────────────────────────────────────────────────────────────────────────────
# MODE: EVALUATE
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluate(args: argparse.Namespace) -> None:
    """
    Run deterministic evaluation of a trained model.

    If --baseline flag is given, also evaluates fixed-time control
    and generates a comparison plot.
    """
    from evaluation.evaluator import PPOEvaluator
    from visualization.plotter import TrafficPlotter

    # Resolve model path
    model_path = args.model
    if model_path is None:
        # Default: look for best model saved by EvalCallback
        model_path = str(Path(args.results_dir) / "best_model" / "best_model")
    if not Path(model_path + ".zip").exists() and not Path(model_path).exists():
        print(f"[ERROR] Model not found at: {model_path}")
        print("       Train first with: python main.py --mode train")
        sys.exit(1)

    evaluator = PPOEvaluator(
        model_path   = model_path,
        sumocfg_path = get_sumocfg_path(),
        n_episodes   = args.episodes,
        traci_port   = 8815,
        results_dir  = args.results_dir,
        seed         = args.seed + 9999,
        use_gui      = args.gui,   # --gui flag opens sumo-gui
    )

    # Evaluate PPO agent
    ppo_results = evaluator.evaluate()

    if args.baseline:
        # Evaluate fixed-time baseline
        baseline_results = evaluator.evaluate_baseline()

        # Generate comparison plot
        plotter = TrafficPlotter(results_dir=args.results_dir)
        plotter.plot_eval_comparison(ppo_results, baseline_results)
        print("\n[main] Comparison plot saved to results/plots/eval_comparison.png")

    print("\n[main] Evaluation results saved to results/evaluation_results.csv")


# ─────────────────────────────────────────────────────────────────────────────
# MODE: VISUALIZE
# ─────────────────────────────────────────────────────────────────────────────

def run_visualize(args: argparse.Namespace) -> None:
    """
    Generate all training plots from a monitor CSV log.
    """
    from visualization.plotter import TrafficPlotter

    plotter = TrafficPlotter(results_dir=args.results_dir)

    # System diagram (no data required)
    plotter.plot_system_diagram()

    # Training curves (requires monitor CSV)
    monitor_path = args.monitor
    if monitor_path is None:
        # Default SB3 monitor location
        monitor_path = str(Path(args.results_dir) / "logs" / "monitor.csv")

    if Path(monitor_path).exists():
        plotter.plot_all_from_monitor(monitor_path)
    else:
        print(f"[WARN] Monitor CSV not found at {monitor_path}.")
        print("       Run training first, or pass --monitor <path>")
        print("       System diagram was saved to results/plots/system_diagram.png")

    # Phase distribution (dummy data for demo)
    plotter.plot_phase_distribution({0: 320, 1: 280, 2: 60, 3: 50})
    print("\n[main] All plots saved to results/plots/")


# ─────────────────────────────────────────────────────────────────────────────
# MODE: DIAGRAM
# ─────────────────────────────────────────────────────────────────────────────

def run_diagram(args: argparse.Namespace) -> None:
    """Generate only the system architecture diagram."""
    from visualization.plotter import TrafficPlotter
    plotter = TrafficPlotter(results_dir=args.results_dir)
    path = plotter.plot_system_diagram()
    print(f"\n[main] Architecture diagram saved to: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = get_args()

    print("\n" + "=" * 65)
    print("  PPO Traffic Signal Controller")
    print(f"  Mode : {args.mode.upper()}")
    print(f"  Seed : {args.seed}")
    print(f"  SUMO : {os.environ.get('SUMO_HOME', 'NOT SET')}")
    print("=" * 65 + "\n")

    mode_dispatch = {
        "train"    : run_train,
        "evaluate" : run_evaluate,
        "visualize": run_visualize,
        "diagram"  : run_diagram,
    }

    mode_dispatch[args.mode](args)
