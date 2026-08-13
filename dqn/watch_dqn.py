"""
watch_dqn.py - Watch the trained DQN agent control traffic in SUMO-GUI
=======================================================================
Run from the project root:
    python dqn/watch_dqn.py

This opens SUMO-GUI so you can visually watch the DQN agent controlling
the intersection. Use this AFTER training.

Controls in SUMO-GUI:
    Play button  : Start the simulation
    Delay slider : Slow down the simulation to watch clearly
"""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("SUMO_HOME", "C:\\Program Files (x86)\\Eclipse\\Sumo")

import torch
from dqn.environment.traffic_env import TrafficEnv
from dqn.models.dqn import DQNAgent
from dqn.config import ENV_CONFIG, DQN_CONFIG

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH  = Path("dqn/results_dqn/best_model/best_dqn_model.pth")
N_EPISODES  = 2
TRACI_PORT  = 8820

PHASE_NAMES = {0: "NS-Green ", 1: "EW-Green ", 2: "NS-Extend", 3: "EW-Extend"}

# ── Load model ────────────────────────────────────────────────────────────────
if not MODEL_PATH.exists():
    print("[DQN watch] ERROR: No saved model found at:", MODEL_PATH)
    print("  Run: python dqn/train_dqn.py")
    sys.exit(1)

print("=" * 60)
print("  DQN Agent - SUMO-GUI Visualisation")
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a temporary env just to get observation/action space sizes
tmp_env = TrafficEnv(
    sumocfg_path = ENV_CONFIG["sumocfg_path"],
    tls_id       = ENV_CONFIG["tls_id"],
    lane_ids     = ENV_CONFIG["lane_ids"],
    delta_t      = ENV_CONFIG["delta_t"],
    max_steps    = ENV_CONFIG["max_steps"],
    traci_port   = TRACI_PORT,
    use_gui      = False,
    seed         = 0,
)
state_size  = tmp_env.observation_space.shape[0]
action_size = tmp_env.action_space.n
tmp_env.close()

agent = DQNAgent(state_size, action_size, DQN_CONFIG, device)
agent.load(str(MODEL_PATH))
agent.set_evaluation()

# ── Watch episodes ────────────────────────────────────────────────────────────
for ep in range(N_EPISODES):
    print(f"\n[DQN] Episode {ep + 1}/{N_EPISODES} - opening SUMO-GUI...")

    env = TrafficEnv(
        sumocfg_path = ENV_CONFIG["sumocfg_path"],
        tls_id       = ENV_CONFIG["tls_id"],
        lane_ids     = ENV_CONFIG["lane_ids"],
        delta_t      = ENV_CONFIG["delta_t"],
        max_steps    = ENV_CONFIG["max_steps"],
        traci_port   = TRACI_PORT,
        use_gui      = True,
        seed         = 3000 + ep,
    )

    obs, _ = env.reset(seed=3000 + ep)
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
    print(f"[DQN] Episode {ep+1} done - total_reward={total_reward:.2f}, steps={step}")

print("\n[DQN] All episodes complete.")
