"""
watch_agent.py — Watch the trained PPO agent control traffic in SUMO-GUI
=========================================================================
Run from the project root:
    python ppo/watch_agent.py

This opens SUMO-GUI so you can visually inspect the agent's decisions.
Use this AFTER training — it loads the saved model from results/best_model/.

Controls in SUMO-GUI:
    Play button  : Start the simulation
    Step button  : Advance one step manually
    Speed slider : Control simulation speed (set to 100% for real-time)
    Delay input  : Add ms delay between steps (useful to slow down and observe)
"""

import sys, os
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault('SUMO_HOME', 'D:/projects')

from pathlib import Path
from stable_baselines3 import PPO
from ppo.environment.traffic_env import TrafficEnv

# ── Configuration ─────────────────────────────────────────────────────────────
RESULTS_DIR  = Path("ppo/results")
MODEL_PATH   = RESULTS_DIR / "best_model" / "best_model"   # saved by EvalCallback
FALLBACK     = RESULTS_DIR / "final_model" / "ppo_traffic_final"

N_EPISODES   = 2       # how many episodes to watch
TRACI_PORT   = 8816    # use a different port to avoid conflicts

# ── Load model ────────────────────────────────────────────────────────────────
model_path = MODEL_PATH if MODEL_PATH.with_suffix(".zip").exists() else FALLBACK
if not model_path.with_suffix(".zip").exists() and not model_path.exists():
    print("[watch_agent] ERROR: No saved model found.")
    print("  Run training first: python ppo/main.py --mode train")
    sys.exit(1)

print(f"[watch_agent] Loading model from: {model_path}")
model = PPO.load(str(model_path), device="cpu")

# ── Watch episodes ────────────────────────────────────────────────────────────
for ep in range(N_EPISODES):
    print(f"\n[watch_agent] Episode {ep + 1}/{N_EPISODES} — opening SUMO-GUI...")
    print("  → Press PLAY in the GUI window to start the simulation.")

    env = TrafficEnv(
        traci_port = TRACI_PORT,
        seed       = 2000 + ep,
        use_gui    = True,   # ← opens sumo-gui
    )

    obs, _ = env.reset()
    total_reward = 0.0
    step = 0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        step += 1
        done = terminated or truncated

        # Print agent decision every 10 steps so you can follow along
        if step % 10 == 0:
            phase_names = {0: "NS-Green", 1: "EW-Green", 2: "NS-Extend", 3: "EW-Extend"}
            wt  = info.get("total_waiting_time", 0)
            q   = info.get("total_queue", 0)
            print(f"  step={step:3d} | action={phase_names[int(action)]:<12} | "
                  f"reward={reward:+.3f} | wait={wt:.0f}s | queue={q}")

    env.close()
    print(f"[watch_agent] Episode {ep+1} done — total_reward={total_reward:.2f}, steps={step}")

print("\n[watch_agent] All episodes complete.")
