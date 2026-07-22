"""
Quick SUMO integration test — runs 3 env steps to verify TraCI connectivity.
Run from the project root: python ppo/test_env.py
"""
import sys, os
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault('SUMO_HOME', 'D:/projects')

from ppo.environment.traffic_env import TrafficEnv

print("Creating TrafficEnv...")
env = TrafficEnv(traci_port=8813, seed=42)
print("Resetting environment (starts SUMO + 100 warmup steps)...")
obs, info = env.reset()
print(f"  Obs shape  : {obs.shape}")
print(f"  Obs range  : [{obs.min():.3f}, {obs.max():.3f}]")
print(f"  Obs dtype  : {obs.dtype}")
print()
print("Running 3 steps with random actions...")
for i in range(3):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    wt = info.get("total_waiting_time", 0)
    q  = info.get("total_queue", 0)
    print(f"  Step {i+1}: action={action}  reward={reward:.4f}"
          f"  wait={wt:.1f}s  queue={q}"
          f"  done={terminated or truncated}")
env.close()
print()
print("=== SUMO ENVIRONMENT TEST PASSED ===")
