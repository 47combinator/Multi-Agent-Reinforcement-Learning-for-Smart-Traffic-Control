"""
state_extractor.py — Observation Builder for the Traffic Environment
=====================================================================

Architecture Role:
    This module is the EYES of our RL system.
    It owns all TraCI read calls and translates raw SUMO data into a
    clean, normalized NumPy vector that the Q-Learning agent can process.

Why a separate module?
    Separation of concerns: the Gymnasium env handles episodes/actions,
    the state extractor handles SUMO data queries exclusively.
    This makes it easy to swap/extend features without touching env logic.

Data Flow:
    SUMO simulation
        → TraCI API calls (vehicle counts, queues, waiting times, phase)
        → Raw values (unnormalized)
        → Feature normalization (all values mapped to [0, 1])
        → Concatenated flat NumPy array (dtype float32)
        → Gymnasium environment → Q-Learning policy

Observation Vector Layout (18 dimensions):
    Indices 0-3   : vehicle count per incoming lane (n, s, e, w)
    Indices 4-7   : queue (halting vehicles) per lane
    Indices 8-11  : mean waiting time per lane (seconds)
    Indices 12-15 : one-hot encoding of current TL phase (0-3)
    Index  16     : elapsed time in current phase (normalized)
    Index  17     : elapsed time in episode (normalized)
"""

import sys
import os
import numpy as np

# ---------------------------------------------------------------------------
# TraCI import — SUMO_HOME must point to the SUMO installation root.
# The 'tools' subdirectory contains the Python TraCI package.
# ---------------------------------------------------------------------------
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    raise EnvironmentError(
        "SUMO_HOME is not set. Please set it to your SUMO installation path."
    )
import traci


# ---------------------------------------------------------------------------
# Normalization constants — chosen from domain knowledge.
# Setting these conservatively (max realistic values) ensures values stay
# in [0, 1] under normal traffic conditions without clipping valid data.
# ---------------------------------------------------------------------------
MAX_VEHICLES_PER_LANE = 20      # lane capacity at jam density (~5m/vehicle on 100m lane)
MAX_QUEUE_PER_LANE = 20      # maximum halting vehicles per lane
MAX_WAITING_TIME = 200.0   # seconds — teleport threshold in sumocfg is 300s
MAX_PHASE_DURATION = 60.0    # seconds — max expected green phase duration
MAX_EPISODE_STEPS = 3600    # seconds — aligns with sumocfg end time

# Number of TL phases we model (must match tlLogic in .net.xml)
NUM_PHASES = 4

# Observation vector size — must match TrafficEnv.observation_space
OBS_SIZE = 18  # 4 + 4 + 4 + 4 (one-hot) + 1 + 1


class StateExtractor:
    """
    Extracts and normalizes the traffic state from SUMO via TraCI.

    Attributes:
        tls_id (str)       : Traffic Light System ID as defined in .net.xml
        lane_ids (list)    : Ordered list of incoming lane IDs to monitor.
                             Order determines which index each lane occupies
                             in the observation vector (n=0, s=1, e=2, w=3).
        _step_count (int)  : Tracks elapsed simulation steps (set by env).
    """

    def __init__(self, tls_id: str, lane_ids: list[str]):
        """
        Args:
            tls_id  : SUMO traffic light ID (e.g., "center")
            lane_ids: List of 4 incoming lane IDs in order [N, S, E, W].
                      Each lane ID has the format "<edge_id>_<lane_index>",
                      e.g., "n2c_0" means edge "n2c", lane 0.
        """
        self.tls_id = tls_id
        self.lane_ids = lane_ids
        self._step_count = 0   # updated externally by TrafficEnv

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_observation(self) -> np.ndarray:
        """
        Build and return the full normalized observation vector.

        Returns:
            obs (np.ndarray): Shape (OBS_SIZE,), dtype float32.
                              All values are in [0.0, 1.0].
        """
        vehicle_counts = self._get_vehicle_counts()    # shape (4,)
        queue_lengths = self._get_queue_lengths()      # shape (4,)
        waiting_times = self._get_waiting_times()      # shape (4,)
        phase_one_hot = self._get_phase_one_hot()      # shape (4,)
        phase_elapsed = self._get_phase_elapsed_norm()  # scalar
        episode_elapsed = self._get_episode_elapsed_norm()  # scalar

        # Concatenate all features into a single flat array.
        # np.concatenate requires all inputs to be array-like.
        obs = np.concatenate([
            vehicle_counts,
            queue_lengths,
            waiting_times,
            phase_one_hot,
            [phase_elapsed],
            [episode_elapsed],
        ]).astype(np.float32)

        # Defensive clip: ensures no float precision error pushes values
        # outside [0, 1], which would violate the observation_space bounds.
        obs = np.clip(obs, 0.0, 1.0)

        assert obs.shape == (OBS_SIZE,), \
            f"Unexpected obs shape: {obs.shape}, expected ({OBS_SIZE},)"
        return obs

    def get_raw_metrics(self) -> dict:
        """
        Return un-normalized metrics for reward calculation and logging.
        Called by RewardCalculator and TrafficEnv after each step.

        Returns:
            dict with keys: 'total_waiting_time', 'total_queue',
                            'vehicle_counts', 'waiting_times', 'queues'
        """
        vehicle_counts = [
            traci.lane.getLastStepVehicleNumber(lid) for lid in self.lane_ids
        ]
        queues = [
            traci.lane.getLastStepHaltingNumber(lid) for lid in self.lane_ids
        ]
        waiting_times = [
            traci.lane.getWaitingTime(lid) for lid in self.lane_ids
        ]
        return {
            "vehicle_counts": vehicle_counts,
            "queues": queues,
            "waiting_times": waiting_times,
            "total_waiting_time": sum(waiting_times),
            "total_queue": sum(queues),
        }

    # ------------------------------------------------------------------
    # Private feature extraction methods
    # Each method handles one group of features and returns a
    # normalized numpy array.
    # ------------------------------------------------------------------

    def _get_vehicle_counts(self) -> np.ndarray:
        """
        Query SUMO for the number of vehicles that entered each lane
        during the last simulation step.

        TraCI call: lane.getLastStepVehicleNumber(lane_id)
            Returns: integer count of vehicles in the lane.

        Normalization: divide by MAX_VEHICLES_PER_LANE.
        """
        counts = np.array(
            [traci.lane.getLastStepVehicleNumber(lid) for lid in self.lane_ids],
            dtype=np.float32,
        )
        return counts / MAX_VEHICLES_PER_LANE

    def _get_queue_lengths(self) -> np.ndarray:
        """
        Query SUMO for the number of HALTING (stopped) vehicles per lane.
        A vehicle is 'halting' if its speed < 0.1 m/s.

        TraCI call: lane.getLastStepHaltingNumber(lane_id)
            Returns: integer count of halting vehicles.

        This is the standard proxy for queue length in traffic engineering.
        Normalization: divide by MAX_QUEUE_PER_LANE.
        """
        halting = np.array(
            [traci.lane.getLastStepHaltingNumber(lid) for lid in self.lane_ids],
            dtype=np.float32,
        )
        return halting / MAX_QUEUE_PER_LANE

    def _get_waiting_times(self) -> np.ndarray:
        """
        Query SUMO for the accumulated waiting time of ALL vehicles in each lane.

        TraCI call: lane.getWaitingTime(lane_id)
            Returns: float, sum of waiting times of all current vehicles (seconds).

        Note: This is a lane-level aggregate, not per-vehicle.
        Normalization: divide by (MAX_WAITING_TIME * MAX_VEHICLES_PER_LANE)
                       since it's a sum across all vehicles in the lane.
        """
        wait_total = MAX_WAITING_TIME * MAX_VEHICLES_PER_LANE

        waiting = np.array(
            [traci.lane.getWaitingTime(lid) for lid in self.lane_ids],
            dtype=np.float32,
        )
        return waiting / wait_total

    def _get_phase_one_hot(self) -> np.ndarray:
        """
        Get the current traffic light phase as a one-hot encoded vector.

        Why one-hot (not integer)?
            The phase integer (0, 1, 2, 3) has no ordinal meaning to the
            network. Phase 0 is not 'less than' phase 2 in any useful sense.
            One-hot encoding treats each phase as a distinct categorical feature,
            which is standard practice in RL for discrete state variables.

        TraCI call: trafficlight.getPhase(tls_id)
            Returns: integer phase index.
        """
        phase = traci.trafficlight.getPhase(self.tls_id)
        one_hot = np.zeros(NUM_PHASES, dtype=np.float32)
        # Guard against unexpected phase indices from SUMO
        if 0 <= phase < NUM_PHASES:
            one_hot[phase] = 1.0
        return one_hot

    def _get_phase_elapsed_norm(self) -> float:
        """
        How long has the current phase been active (normalized)?

        Gives the agent temporal context: it can learn that a very long
        green phase should probably switch, even if queues are still building.

        TraCI call: trafficlight.getNextSwitch(tls_id)
            Returns: simulation time (seconds) at which the phase will next change.

        elapsed = current_time - (next_switch - phase_duration)
        """
        try:
            next_switch = traci.trafficlight.getNextSwitch(self.tls_id)
            phase_duration = traci.trafficlight.getPhaseDuration(self.tls_id)
            current_time = traci.simulation.getTime()
            # Time since phase started
            phase_start = next_switch - phase_duration
            elapsed = current_time - phase_start
            return float(np.clip(elapsed / MAX_PHASE_DURATION, 0.0, 1.0))
        except Exception:
            return 0.0  # safe fallback during edge cases

    def _get_episode_elapsed_norm(self) -> float:
        """
        Fraction of the episode that has elapsed.
        Helps the agent reason about end-of-episode urgency.
        """
        return float(np.clip(self._step_count / MAX_EPISODE_STEPS, 0.0, 1.0))
