"""
traffic_env.py — Core Gymnasium Environment for Traffic Signal Control
=======================================================================

Architecture Role:
    This is the CENTRAL HUB of the entire system.
    It implements the Gymnasium Env interface that SB3's PPO expects,
    and orchestrates all other modules (StateExtractor, RewardCalculator).

Gymnasium Contract:
    Every Gymnasium environment must implement:
        reset()  → obs, info           : Start a new episode
        step(a)  → obs, reward, terminated, truncated, info : Take one action
        close()  → None                : Clean up resources

Episode Structure:
    ┌─────────────────────────────────────────────────────────┐
    │  reset()                                                │
    │    ├─ Start SUMO subprocess via TraCI                   │
    │    ├─ Warm-up: simulate N steps with default TL logic   │
    │    └─ Return initial observation                        │
    │                                                         │
    │  loop until done:                                       │
    │    step(action)                                         │
    │      ├─ Apply action: set TL phase via TraCI            │
    │      ├─ Simulate DELTA_T seconds                        │
    │      ├─ Collect observation (StateExtractor)            │
    │      ├─ Compute reward (RewardCalculator)               │
    │      └─ Check termination condition                     │
    │                                                         │
    │  close()                                                │
    │    └─ TraCI disconnect → SUMO process terminates        │
    └─────────────────────────────────────────────────────────┘

Action → TL Phase Mapping:
    Action 0 → Phase 0 (GGrr): North-South GREEN (straight)
    Action 1 → Phase 2 (rrGG): East-West GREEN   (straight)
    Action 2 → Phase 0 with extended duration (NS + turning)
    Action 3 → Phase 2 with extended duration (EW + turning)

    Note on yellow phases:
        When switching from NS to EW (or vice versa), a mandatory
        yellow phase (3 seconds) is automatically inserted to match
        real-world traffic light behavior. This prevents the agent
        from learning to instantly flip phases (which would be unsafe
        and cause SUMO warnings).

TraCI Port Management:
    Each SUMO instance needs a unique TraCI port so multiple
    environments can run in parallel (for VecEnv). The port is
    passed as a constructor argument.
"""

import os
import sys
import time
import subprocess
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path

# ---------------------------------------------------------------------------
# TraCI import — requires SUMO_HOME to be set
# ---------------------------------------------------------------------------
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    raise EnvironmentError(
        "SUMO_HOME environment variable is not set. "
        "Please set it to your SUMO installation directory."
    )
import traci
import traci.constants as tc  # SUMO subscription constants (optional)

from dqn.environment.state_extractor import StateExtractor, OBS_SIZE
from dqn.environment.reward_calculator import RewardCalculator


# ---------------------------------------------------------------------------
# Environment constants
# ---------------------------------------------------------------------------

# Simulation seconds to advance per agent step.
# DELTA_T = 5 means the agent makes a decision every 5 simulated seconds.
# Smaller = more decisions, slower training. Larger = less responsive control.
DELTA_T = 5

# Maximum number of agent steps per episode.
# 3600s / 5s per step = 720 steps per episode.
MAX_STEPS = 720

# Minimum green duration (in simulation steps).
# Prevents the agent from oscillating phases too rapidly.
# 10s minimum green is a standard traffic engineering requirement.
MIN_GREEN_STEPS = 10   # 10 * DELTA_T = 10 simulated seconds minimum green

# Yellow phase duration in simulation seconds (fixed, non-agent-controlled).
YELLOW_DURATION = 3

# TL phase indices in the .net.xml tlLogic definition
PHASE_NS_GREEN  = 0   # "GGrrGGrr" — North-South gets green
PHASE_NS_YELLOW = 1   # "yyrryyрр" — North-South yellow transition
PHASE_EW_GREEN  = 2   # "rrGGrrGG" — East-West gets green
PHASE_EW_YELLOW = 3   # "rryyrrуу" — East-West yellow transition

# Incoming lane IDs (must match edge IDs in .net.xml + "_0" lane suffix)
# Order: [North, South, East, West]
INCOMING_LANES = ["n2c_0", "s2c_0", "e2c_0", "w2c_0"]

# Traffic Light System ID (matches id="center" in .net.xml)
TLS_ID = "center"


class TrafficEnv(gym.Env):
    """
    Gymnasium environment wrapping a SUMO single-intersection simulation.

    The environment models a 4-way signalized intersection.
    The PPO agent controls the traffic light to minimize congestion.

    Args:
        sumocfg_path (str) : Absolute path to the .sumocfg file.
        tls_id       (str) : Traffic light ID in SUMO network.
        lane_ids     (list): Incoming lane IDs to monitor.
        delta_t      (int) : Simulation seconds per agent step.
        max_steps    (int) : Maximum steps before episode ends (truncation).
        traci_port   (int) : TCP port for TraCI connection.
                             Use different ports for parallel envs.
        use_gui      (bool): If True, open sumo-gui (for debugging only).
        seed         (int) : Random seed for reproducibility.
        alpha (float)      : Waiting time reward weight.
        beta  (float)      : Queue length reward weight.
        gamma (float)      : Throughput reward weight.
    """

    # SB3 requires metadata for rendering
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        sumocfg_path : str  = None,
        tls_id       : str  = TLS_ID,
        lane_ids     : list = None,
        delta_t      : int  = DELTA_T,
        max_steps    : int  = MAX_STEPS,
        traci_port   : int  = 8813,
        use_gui      : bool = False,
        seed         : int  = 42,
        alpha        : float = 0.4,
        beta         : float = 0.3,
        gamma        : float = 0.3,
    ):
        super().__init__()

        # ── Configuration ───────────────────────────────────────────────
        if sumocfg_path is None:
            # Default: relative to this file's location
            _here = Path(__file__).resolve().parent.parent
            sumocfg_path = str(
                _here.parent / "sumo_env" / "single_intersection.sumocfg"
            )
        self.sumocfg_path = sumocfg_path
        self.tls_id       = tls_id
        self.lane_ids     = lane_ids if lane_ids else INCOMING_LANES
        self.delta_t      = delta_t
        self.max_steps    = max_steps
        self.traci_port   = traci_port
        self.use_gui      = use_gui
        self._seed        = seed

        # ── Gymnasium spaces ────────────────────────────────────────────
        # Observation: OBS_SIZE-dimensional box, all values in [0, 1].
        # Using float32 to match PyTorch tensor dtype.
        self.observation_space = spaces.Box(
            low   = 0.0,
            high  = 1.0,
            shape = (OBS_SIZE,),
            dtype = np.float32,
        )

        # Action: 4 discrete choices (see module docstring).
        # SB3's PPO with Discrete action space uses a Categorical policy.
        self.action_space = spaces.Discrete(4)

        # ── Sub-modules ─────────────────────────────────────────────────
        self.state_extractor = StateExtractor(
            tls_id   = tls_id,
            lane_ids = self.lane_ids,
        )
        self.reward_calculator = RewardCalculator(alpha, beta, gamma)

        # ── Episode state ───────────────────────────────────────────────
        self._step_count     = 0       # steps taken in current episode
        self._current_phase  = PHASE_NS_GREEN  # current TL phase
        self._phase_step_count = 0     # steps since last phase change
        self._sumo_running   = False   # SUMO subprocess flag
        self._episode_reward = 0.0     # accumulated reward (for logging)

        # Per-episode KPI accumulators (filled by step(), reset in reset())
        self._ep_waiting_times = []
        self._ep_queues        = []
        self._ep_throughputs   = []

    # ------------------------------------------------------------------
    # Gymnasium Core Methods
    # ------------------------------------------------------------------

    def reset(
        self,
        seed    : int  = None,
        options : dict = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Reset the environment to start a new episode.

        Sequence:
            1. Close previous SUMO instance (if any)
            2. Start SUMO with a new random seed
            3. Warm-up: run N steps with default TL to fill the network
            4. Initialize sub-modules
            5. Return first observation

        Args:
            seed    : Optional seed override (used by SB3's VecEnv).
            options : Unused (Gymnasium API requirement).

        Returns:
            obs  (np.ndarray): Initial observation vector.
            info (dict)      : Empty dict (Gymnasium 0.26+ requirement).
        """
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed

        # Step 1: Shut down existing SUMO instance
        self._close_sumo()

        # Step 2: Start new SUMO instance
        self._start_sumo()

        # Step 3: Warm-up — run 100 steps with default TL logic
        #         so that vehicles populate the network before RL starts.
        #         Without this, the agent sees an empty network at episode start.
        for _ in range(100):
            traci.simulationStep()

        # Step 4: Reset internal state
        self._step_count       = 0
        self._current_phase    = PHASE_NS_GREEN
        self._phase_step_count = 0
        self._episode_reward   = 0.0
        self._ep_waiting_times = []
        self._ep_queues        = []
        self._ep_throughputs   = []

        # Sync StateExtractor step counter
        self.state_extractor._step_count = 0

        # Initialize reward calculator with current state
        initial_metrics = self.state_extractor.get_raw_metrics()
        self.reward_calculator.reset(initial_metrics)

        # Step 5: Get first observation
        obs = self.state_extractor.get_observation()
        return obs, {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one agent step in the environment.

        A single 'step' = DELTA_T simulated seconds.
        Within those seconds, SUMO runs autonomously; the agent only
        sets the TL phase at the START of each step.

        Args:
            action (int): Integer in {0, 1, 2, 3}.

        Returns:
            obs        (np.ndarray): New observation after DELTA_T seconds.
            reward     (float)     : Scalar reward for this step.
            terminated (bool)      : True if the simulation ended naturally.
            truncated  (bool)      : True if max_steps was reached.
            info       (dict)      : Metrics dict for logging.
        """
        # ── 1. Apply action: set traffic light phase ─────────────────────
        self._apply_action(action)

        # ── 2. Advance simulation by DELTA_T seconds ─────────────────────
        for _ in range(self.delta_t):
            traci.simulationStep()

        # ── 3. Increment counters ─────────────────────────────────────────
        self._step_count              += 1
        self._phase_step_count        += 1
        self.state_extractor._step_count = self._step_count

        # ── 4. Collect observations ───────────────────────────────────────
        obs             = self.state_extractor.get_observation()
        current_metrics = self.state_extractor.get_raw_metrics()

        # ── 5. Compute reward ─────────────────────────────────────────────
        reward, reward_components = self.reward_calculator.compute_reward(
            current_metrics
        )
        self._episode_reward += reward

        # ── 6. Accumulate episode KPIs ────────────────────────────────────
        self._ep_waiting_times.append(current_metrics["total_waiting_time"])
        self._ep_queues.append(current_metrics["total_queue"])

        # ── 7. Check termination conditions ──────────────────────────────
        # terminated: SUMO simulation reached its end naturally
        terminated = traci.simulation.getMinExpectedNumber() <= 0

        # truncated: we ran out of steps (SB3 handles this as a time limit)
        truncated = self._step_count >= self.max_steps

        # ── 8. Build info dict ────────────────────────────────────────────
        info = self.reward_calculator.get_info_dict(
            current_metrics, reward_components
        )
        # Add episode-level stats when the episode ends
        if terminated or truncated:
            info["episode"] = {
                "r"                       : self._episode_reward,
                "l"                       : self._step_count,
                "mean_waiting_time"       : float(np.mean(self._ep_waiting_times))
                    if self._ep_waiting_times else 0.0,
                "mean_queue"              : float(np.mean(self._ep_queues))
                    if self._ep_queues else 0.0,
            }

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Clean up: close TraCI connection and terminate SUMO process."""
        self._close_sumo()

    # ------------------------------------------------------------------
    # Action Application
    # ------------------------------------------------------------------

    def _apply_action(self, action: int) -> None:
        """
        Translate a discrete action integer into a SUMO TL phase change.

        Action-to-Phase mapping:
            0 → NS Green  (Phase 0): Give green to North-South traffic
            1 → EW Green  (Phase 2): Give green to East-West traffic
            2 → NS Green  (Phase 0): Same as 0 (kept for future extension
                                     to NS left-turn phase)
            3 → EW Green  (Phase 2): Same as 1 (kept for future extension
                                     to EW left-turn phase)

        Yellow phase enforcement:
            If the agent wants to switch from NS to EW (or vice versa),
            we first insert a YELLOW phase (3s) before the new green.
            This is mandatory in real-world traffic engineering and
            prevents SUMO from generating collision warnings.

        Minimum green enforcement:
            If the agent tries to switch before MIN_GREEN_STEPS steps,
            the action is ignored and the current phase is held.
            This prevents pathological oscillation.

        Args:
            action (int): Agent's chosen action {0, 1, 2, 3}.
        """
        # Map action integer to target SUMO phase index
        # Actions 0,2 → NS green; Actions 1,3 → EW green
        target_phase = PHASE_NS_GREEN if action in [0, 2] else PHASE_EW_GREEN

        # Determine if this is a phase SWITCH (NS↔EW)
        current_green = self._current_phase  # always NS_GREEN or EW_GREEN
        is_switch = (target_phase != current_green)

        # Enforce minimum green duration
        if is_switch and self._phase_step_count < MIN_GREEN_STEPS:
            # Too early to switch — hold current phase
            # (No TraCI call needed; SUMO keeps the current phase)
            return

        if is_switch:
            # Insert yellow transition phase
            yellow_phase = (
                PHASE_NS_YELLOW if current_green == PHASE_NS_GREEN
                else PHASE_EW_YELLOW
            )
            # Override the TL program to yellow immediately
            traci.trafficlight.setPhase(self.tls_id, yellow_phase)
            # Advance simulation for yellow duration (within this step)
            # Note: YELLOW_DURATION < DELTA_T so this fits within one step
            for _ in range(min(YELLOW_DURATION, self.delta_t - 1)):
                traci.simulationStep()

            # Set the new green phase
            traci.trafficlight.setPhase(self.tls_id, target_phase)
            self._current_phase    = target_phase
            self._phase_step_count = 0  # reset phase timer
        else:
            # Hold current phase — explicitly set to avoid SUMO auto-advancing
            traci.trafficlight.setPhase(self.tls_id, target_phase)

    # ------------------------------------------------------------------
    # SUMO Lifecycle Management
    # ------------------------------------------------------------------

    def _start_sumo(self) -> None:
        """
        Launch SUMO as a subprocess and connect TraCI to it.

        SUMO is started with:
            --seed <seed>     : Different seed each episode for stochasticity
            --start           : Begin simulation immediately (no wait)
            --quit-on-end     : Auto-terminate when simulation ends
            --no-step-log     : Suppress per-step console output

        TraCI connection:
            traci.start() forks the SUMO subprocess AND connects.
            It blocks until SUMO is ready to accept commands.
        """
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"

        sumo_cmd = [
            sumo_binary,
            "--configuration-file", self.sumocfg_path,
            "--seed",               str(self._seed),
            "--start",              # begin immediately
            "--quit-on-end",        # terminate when simulation time ends
            "--no-step-log",        # suppress verbose step output
            "--waiting-time-memory","200",  # track wait times 200s back
        ]

        # traci.start() both starts SUMO and opens the TraCI socket.
        # label=str(port) allows multiple simultaneous environments.
        traci.start(sumo_cmd, port=self.traci_port, label=str(self.traci_port))
        self._sumo_running = True

    def _close_sumo(self) -> None:
        """
        Safely close the TraCI connection.
        If SUMO is already closed, this is a no-op.
        """
        if self._sumo_running:
            try:
                traci.switch(str(self.traci_port))  # select this connection
                traci.close()
            except Exception:
                pass  # already closed; ignore
            finally:
                self._sumo_running = False

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def render(self) -> None:
        """Rendering is handled by SUMO-GUI. This is a no-op for headless."""
        pass

    def get_episode_stats(self) -> dict:
        """
        Return aggregated statistics for the current/last episode.
        Used by the Evaluator.
        """
        return {
            "total_reward"     : self._episode_reward,
            "steps"            : self._step_count,
            "mean_waiting_time": float(np.mean(self._ep_waiting_times))
                if self._ep_waiting_times else 0.0,
            "mean_queue"       : float(np.mean(self._ep_queues))
                if self._ep_queues else 0.0,
        }
