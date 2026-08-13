"""
watch_qlearning.py - Watch the trained Q-Learning agent control traffic in SUMO-GUI
====================================================================================
Run from the project root:
    python qlearning/watch_qlearning.py

This opens SUMO-GUI so you can visually watch the Q-Learning agent controlling
the intersection. Use this AFTER training.

Controls in SUMO-GUI:
    Play button  : Start the simulation
    Delay slider : Slow down the simulation to watch clearly
"""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("SUMO_HOME", "C:\\Program Files (x86)\\Eclipse\\Sumo")

from qlearning.environment.traffic_env import TrafficEnv
from qlearning.agent.qlearning_agent import QLearningAgent

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH   = Path("qlearning/results/best_model/q_model_center.json")
SUMOCFG_PATH = Path("sumo_env/single_intersection.sumocfg")
N_EPISODES   = 2
TRACI_PORT   = 8822

PHASE_NAMES  = {0: "NS-Green ", 1: "EW-Green ", 2: "NS-Extend", 3: "EW-Extend"}

# ── Load model ────────────────────────────────────────────────────────────────
if not MODEL_PATH.exists():
    print("[Q-Learning watch] ERROR: No saved model found at:", MODEL_PATH)
    print("  Run: python qlearning/main.py --mode train")
    sys.exit(1)

print("=" * 60)
print("  Q-Learning Agent - SUMO-GUI Visualisation")
print("=" * 60)
print(f"  Model   : {MODEL_PATH}")
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

agent = QLearningAgent(action_space_n=4)
agent.load(str(MODEL_PATH))
agent.set_evaluation()

# ── Watch episodes ────────────────────────────────────────────────────────────
for ep in range(N_EPISODES):
    print(f"\n[Q-Learning] Episode {ep + 1}/{N_EPISODES} - opening SUMO-GUI...")

    env = TrafficEnv(
        sumocfg_path = str(SUMOCFG_PATH),
        traci_port   = TRACI_PORT,
        seed         = 4000 + ep,
        use_gui      = True,
    )

    obs, _ = env.reset(seed=4000 + ep)
    total_reward = 0.0
    step = 0
    done = False

    while not done:
        action = agent.choose_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        done = terminated or truncated

        if step % 10 == 0:
            wt = info.get("total_waiting_time", 0)
            q  = info.get("total_queue", 0)
            print(f"  step={step:3d} | {PHASE_NAMES.get(action, str(action))} | "
                  f"reward={reward:+.3f} | wait={wt:.0f}s | queue={q}")

    env.close()
    print(f"[Q-Learning] Episode {ep+1} done - total_reward={total_reward:.2f}, steps={step}")

print("\n[Q-Learning] All episodes complete.")
