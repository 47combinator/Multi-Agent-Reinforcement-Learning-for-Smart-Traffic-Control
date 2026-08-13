"""
watch_agent.py - Watch the trained PPO agent control traffic in SUMO-GUI
=========================================================================
Run from the project root:
    python ppo/watch_agent.py

This opens SUMO-GUI so you can visually inspect the agent's decisions.
Use this AFTER training - it loads the saved model from results/best_model/.

Controls in SUMO-GUI:
    Play button  : Start the simulation
    Step button  : Advance one step manually
    Delay slider : Already set to slow-down via --delay flag (default 200ms)
"""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("SUMO_HOME", "C:\\Program Files (x86)\\Eclipse\\Sumo")

import argparse
from stable_baselines3 import PPO
from ppo.environment.traffic_env import TrafficEnv

# ── Configuration ─────────────────────────────────────────────────────────────
RESULTS_DIR  = Path("ppo/results")
MODEL_PATH   = RESULTS_DIR / "best_model" / "best_model"
FALLBACK     = RESULTS_DIR / "final_model" / "ppo_traffic_final"

N_EPISODES   = 2
TRACI_PORT   = 8816

PHASE_NAMES  = {0: "NS-Green ", 1: "EW-Green ", 2: "NS-Extend", 3: "EW-Extend"}

# ── Load model ────────────────────────────────────────────────────────────────
model_path = MODEL_PATH if MODEL_PATH.with_suffix(".zip").exists() else FALLBACK
if not model_path.with_suffix(".zip").exists() and not model_path.exists():
    print("[PPO watch] ERROR: No saved model found.")
    print("  Run: python ppo/main.py --mode train")
    sys.exit(1)

print("=" * 60)
print("  PPO Agent - SUMO-GUI Visualisation")
print("=" * 60)
print(f"  Model  : {model_path}")
print(f"  Episodes: {N_EPISODES}")
print()
print("  Car colours in SUMO-GUI:")
print("    BLUE   = cars from North")
print("    RED    = cars from South")
print("    GREEN  = cars from East")
print("    YELLOW = cars from West")
print()
print("  After the window opens:")
print("  1. Click the PLAY button (triangle) to start")
print("  2. Use the Delay slider to slow down if needed")
print("=" * 60)

model = PPO.load(str(model_path), device="cpu")

# ── Watch episodes ────────────────────────────────────────────────────────────
for ep in range(N_EPISODES):
    print(f"\n[PPO] Episode {ep + 1}/{N_EPISODES} - opening SUMO-GUI...")

    env = TrafficEnv(
        traci_port = TRACI_PORT,
        seed       = 2000 + ep,
        use_gui    = True,
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

        if step % 10 == 0:
            wt = info.get("total_waiting_time", 0)
            q  = info.get("total_queue", 0)
            print(f"  step={step:3d} | {PHASE_NAMES[int(action)]} | "
                  f"reward={reward:+.3f} | wait={wt:.0f}s | queue={q}")

    env.close()
    print(f"[PPO] Episode {ep+1} done - total_reward={total_reward:.2f}, steps={step}")

print("\n[PPO] All episodes complete.")
