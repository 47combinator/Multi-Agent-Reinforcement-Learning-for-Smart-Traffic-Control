"""
test_project.py — Unit Tests and Verification Suite
===================================================

This script verifies the correctness of the neural networks, replay buffers,
environment logic, and agents. It mocks TraCI so it can be run on any system
without requiring SUMO to be installed.

Usage:
    pip install unittest-xml-reporting  (optional)
    python -m unittest test_project.py
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch

# Mock traci module before importing project modules
import sys
sys.modules['traci'] = MagicMock()
import traci

# Mock traci simulation sub-methods
traci.simulation = MagicMock()
traci.simulation.getMinExpectedNumber.return_value = 1
traci.simulation.getTime.return_value = 10.0
traci.simulation.getArrivedNumber.return_value = 0

# Mock traci lane sub-methods
traci.lane = MagicMock()
traci.lane.getLastStepVehicleNumber.return_value = 2
traci.lane.getLastStepHaltingNumber.return_value = 1
traci.lane.getWaitingTime.return_value = 15.0
traci.lane.getLastStepOccupancy.return_value = 25.0
traci.lane.getLastStepMeanSpeed.return_value = 10.0

# Mock traci traffic light sub-methods
traci.trafficlight = MagicMock()
traci.trafficlight.getPhase.return_value = 0
traci.trafficlight.getNextSwitch.return_value = 30.0
traci.trafficlight.getPhaseDuration.return_value = 42.0

# Now import the project modules
from env.state_extractor import StateExtractor, OBS_SIZE
from env.reward_calculator import RewardCalculator, NORM_DELTA_WAIT
from env.traffic_env import TrafficEnv
from models.network import QNetwork
from models.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from models.dqn import DQNAgent
from baselines.q_learning import QLearningAgent
from baselines.ppo import PPOAgent


class TestTrafficEnvironment(unittest.TestCase):
    """Verifies StateExtractor, RewardCalculator, and TrafficEnv."""

    def test_state_extractor_dimensions(self):
        extractor = StateExtractor("center", ["n2c_0", "s2c_0", "e2c_0", "w2c_0"])
        obs = extractor.get_observation()
        
        self.assertEqual(obs.shape, (OBS_SIZE,))
        self.assertTrue(np.all(obs >= 0.0) and np.all(obs <= 1.0))
        
        metrics = extractor.get_raw_metrics()
        self.assertIn("vehicle_counts", metrics)
        self.assertIn("mean_delay", metrics)

    def test_reward_calculator(self):
        calculator = RewardCalculator(alpha=0.4, beta=0.3, gamma=0.3)
        
        initial_metrics = {
            "total_waiting_time": 10.0,
            "total_queue": 2,
            "vehicle_counts": [1, 1, 0, 0]
        }
        calculator.reset(initial_metrics)
        
        current_metrics = {
            "total_waiting_time": 15.0,  # Wait time increased (+5.0)
            "total_queue": 3,            # Queue increased (+1)
            "vehicle_counts": [1, 1, 1, 0]
        }
        
        reward, components = calculator.compute_reward(current_metrics)
        
        # waiting delta penalty: -0.4 * (5.0 / 20.0) = -0.1
        # queue delta penalty: -0.3 * (1.0 / 5.0) = -0.06
        # throughput reward: 0.3 * (0.0 / 2.0) = 0.0 (since arrived_prev is mocked)
        self.assertAlmostEqual(components["reward/waiting_delta"], -0.1)
        self.assertAlmostEqual(components["reward/queue_delta"], -0.06)
        self.assertTrue(-1.0 <= reward <= 1.0)

    @patch('env.traffic_env.traci')
    def test_env_reset_step(self, mock_traci):
        # Setup mock return values inside environment imports
        mock_traci.simulation.getMinExpectedNumber.return_value = 5
        
        env = TrafficEnv(sumocfg_path="dummy.sumocfg", traci_port=9999)
        obs, info = env.reset(seed=42)
        
        self.assertEqual(obs.shape, (OBS_SIZE,))
        
        # Test stepping
        next_obs, reward, terminated, truncated, info = env.step(1)
        self.assertEqual(next_obs.shape, (OBS_SIZE,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)


class TestQNetworks(unittest.TestCase):
    """Verifies standard and dueling network layers and outputs."""

    def test_standard_dqn_output_shape(self):
        state_dim = 25
        action_dim = 4
        net = QNetwork(state_dim, action_dim, dueling=False)
        
        dummy_input = torch.randn(8, state_dim)
        output = net(dummy_input)
        
        self.assertEqual(output.shape, (8, action_dim))

    def test_dueling_dqn_output_shape(self):
        state_dim = 25
        action_dim = 4
        net = QNetwork(state_dim, action_dim, dueling=True)
        
        dummy_input = torch.randn(8, state_dim)
        output = net(dummy_input)
        
        self.assertEqual(output.shape, (8, action_dim))


class TestReplayBuffers(unittest.TestCase):
    """Verifies replay buffer sampling and prioritization."""

    def test_uniform_replay_buffer(self):
        device = torch.device("cpu")
        buffer = ReplayBuffer(action_size=4, buffer_size=100, batch_size=4, device=device)
        
        for _ in range(10):
            buffer.push(np.zeros(25), 1, 1.0, np.zeros(25), False)
            
        self.assertEqual(len(buffer), 10)
        
        states, actions, rewards, next_states, dones = buffer.sample()
        self.assertEqual(states.shape, (4, 25))
        self.assertEqual(actions.shape, (4, 1))

    def test_prioritized_replay_buffer(self):
        device = torch.device("cpu")
        buffer = PrioritizedReplayBuffer(action_size=4, buffer_size=32, batch_size=4, device=device)
        
        for i in range(10):
            buffer.push(np.ones(25) * i, i % 4, float(i), np.ones(25) * (i + 1), False)
            
        self.assertEqual(len(buffer), 10)
        
        (states, actions, rewards, next_states, dones), weights, idxs = buffer.sample()
        self.assertEqual(states.shape, (4, 25))
        self.assertEqual(weights.shape, (4,))
        self.assertEqual(len(idxs), 4)
        
        # Test priority updates
        buffer.update_priorities(idxs, np.array([0.1, 0.2, 0.3, 0.4]))


class TestAgents(unittest.TestCase):
    """Verifies DQN, Q-Learning, and PPO agent choose action and learn logic."""

    def test_dqn_agent(self):
        config = {
            "gamma": 0.99,
            "lr": 1e-4,
            "batch_size": 4,
            "tau": 0.005,
            "use_soft_update": True,
            "epsilon_start": 0.5,
            "epsilon_end": 0.01,
            "epsilon_decay": 0.99,
            "double_dqn": True,
            "dueling_dqn": True,
            "prioritized_replay": False
        }
        device = torch.device("cpu")
        agent = DQNAgent(state_size=25, action_size=4, config=config, device=device)
        
        action = agent.choose_action(np.zeros(25))
        self.assertTrue(0 <= action < 4)
        
        # Push experiences to trigger learning
        for _ in range(10):
            agent.step(np.zeros(25), 1, 1.0, np.zeros(25), False)
            
        self.assertEqual(len(agent.memory), 10)

    def test_q_learning_agent(self):
        agent = QLearningAgent(action_space_n=4, bins=6)
        
        state = np.array([0.1, 0.5, 0.9] * 8 + [0.2])  # Length 25 state
        action = agent.choose_action(state)
        self.assertTrue(0 <= action < 4)
        
        td_error = agent.learn(state, action, 1.0, state, False)
        self.assertIsInstance(td_error, float)

    def test_ppo_agent(self):
        config = {
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_epsilon": 0.2,
            "c1": 0.5,
            "c2": 0.01,
            "mini_batch_size": 2,
            "batch_size": 4,
            "update_epochs": 2,
            "lr": 3e-4
        }
        device = torch.device("cpu")
        agent = PPOAgent(state_size=25, action_size=4, config=config, device=device)
        
        action = agent.choose_action(np.zeros(25))
        self.assertTrue(0 <= action < 4)
        
        # Store some transitions
        for _ in range(4):
            agent.store_transition(np.zeros(25), 1, -0.5, 1.0, False)
            
        loss = agent.learn()
        self.assertIsInstance(loss, float)


if __name__ == "__main__":
    unittest.main()
