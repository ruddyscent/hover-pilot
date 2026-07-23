import math
import time
from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hoverpilot.rflink.client import RFLinkClient
from hoverpilot.rflink.models import FlightAxisState, RFControlAction
from hoverpilot.rflink.protocol import state_looks_uninitialized
from hoverpilot.training.hover import (
    ElevatorHoverFeatures,
    REWARD_PROFILE_ELEVATOR,
    REWARD_PROFILE_STANDARD,
    REALFLIGHT_VERTICAL_HOVER_INCLINATION_DEG,
    STANDARD_HOVER_INCLINATION_DEG,
    RewardConfig,
    RewardBreakdown,
    compute_elevator_hover_features,
    compute_elevator_recovery_target_deg,
    compute_reward,
    project_onto_target_heading,
)

TRAINER_RESET_REASONS = {
    "trainer_reset",
    "trainer_reset_button",
    "trainer_repositioned",
}
BOOL_FIELD_THRESHOLD = 0.5
DEFAULT_RESET_WAIT_SECONDS = 8.0
DEFAULT_RESET_POLL_INTERVAL_SECONDS = 0.05


class HoverTaskProfile(str, Enum):
    STANDARD = "full"
    ELEVATOR = "elevator"

    @property
    def reward_profile(self) -> str:
        if self is HoverTaskProfile.ELEVATOR:
            return REWARD_PROFILE_ELEVATOR
        return REWARD_PROFILE_STANDARD

    @property
    def observation_dim(self) -> int:
        if self is HoverTaskProfile.ELEVATOR:
            return 6
        return 13

    @property
    def anchor_heading_to_reset_state(self) -> bool:
        return self is HoverTaskProfile.ELEVATOR

    @property
    def require_vertical_start(self) -> bool:
        return self is HoverTaskProfile.ELEVATOR

    @property
    def realflight_reference_inclination_deg(self) -> float:
        if self.require_vertical_start:
            return REALFLIGHT_VERTICAL_HOVER_INCLINATION_DEG
        return STANDARD_HOVER_INCLINATION_DEG


STANDARD_HOVER_TASK = HoverTaskProfile.STANDARD
ELEVATOR_HOVER_TASK = HoverTaskProfile.ELEVATOR


@dataclass
class EpisodeLifecycleResult:
    ready: bool
    started: bool
    terminated: bool
    truncated: bool
    reason: Optional[str] = None


@dataclass
class EpisodeBoundaryAssessment:
    readiness: EpisodeLifecycleResult
    reset_reason: Optional[str]
    can_start: bool
    pre_reset_wait: bool


def state_to_observation(
    state: FlightAxisState,
    *,
    target_x_m: float = 0.0,
    target_y_m: float = 0.0,
    target_altitude_agl_m: float = 1.5,
) -> np.ndarray:
    """Build a compact hover-oriented observation vector.

    These fields capture the variables most directly tied to stationary hover:
    planar position, altitude, attitude, world-frame velocity, and body rates.
    We intentionally omit less essential telemetry to keep the first RL interface
    compact and easier to tune.
    """

    azimuth_rad = np.deg2rad(state.m_azimuth_DEG)
    observation = np.asarray(
        [
            (state.m_aircraftPositionX_MTR - target_x_m) / 8.0,
            (state.m_aircraftPositionY_MTR - target_y_m) / 8.0,
            (state.m_altitudeAGL_MTR - target_altitude_agl_m) / 3.0,
            state.m_roll_DEG / 45.0,
            state.m_inclination_DEG / 45.0,
            np.sin(azimuth_rad),
            np.cos(azimuth_rad),
            state.m_velocityWorldU_MPS / 10.0,
            state.m_velocityWorldV_MPS / 10.0,
            state.m_velocityWorldW_MPS / 10.0,
            state.m_pitchRate_DEGpSEC / 180.0,
            state.m_rollRate_DEGpSEC / 180.0,
            state.m_yawRate_DEGpSEC / 180.0,
        ],
        dtype=np.float32,
    )
    return np.clip(observation, -5.0, 5.0).astype(np.float32, copy=False)


def elevator_features_to_observation(
    elevator_features: ElevatorHoverFeatures,
    *,
    config: RewardConfig,
) -> np.ndarray:
    """Normalize elevator hover features using the recovery-target frame."""

    target_inclination_error_deg = compute_elevator_recovery_target_deg(
        elevator_features.longitudinal_position_error_m,
        elevator_features.longitudinal_velocity_mps,
        position_gain_deg_per_m=(
            config.elevator_recovery_position_gain_deg_per_m
        ),
        velocity_gain_deg_per_mps=(
            config.elevator_recovery_velocity_gain_deg_per_mps
        ),
        inclination_limit_deg=config.elevator_recovery_inclination_limit_deg,
    )
    inclination_tracking_error_deg = (
        elevator_features.inclination_error_deg
        - target_inclination_error_deg
    )
    observation = np.asarray(
        [
            inclination_tracking_error_deg / config.inclination_error_scale_deg,
            elevator_features.pitch_rate_deg_s / config.pitch_rate_scale_deg_s,
            elevator_features.longitudinal_position_error_m
            / config.longitudinal_position_scale_m,
            elevator_features.longitudinal_velocity_mps
            / config.velocity_error_scale_mps,
            elevator_features.altitude_error_m / config.altitude_error_scale_m,
            elevator_features.vertical_velocity_mps
            / config.velocity_error_scale_mps,
        ],
        dtype=np.float32,
    )
    return np.clip(observation, -5.0, 5.0).astype(np.float32, copy=False)


def gym_action_to_rf_action(action: Union[np.ndarray, list, tuple]) -> RFControlAction:
    action_array = np.asarray(action, dtype=np.float32).reshape(-1)
    if action_array.shape != (4,):
        raise ValueError("action must have shape (4,) for [aileron, elevator, throttle, rudder]")

    clipped = np.clip(
        action_array,
        np.asarray([-1.0, -1.0, 0.0, -1.0], dtype=np.float32),
        np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    )
    return RFControlAction(
        aileron=float(clipped[0]),
        elevator=float(clipped[1]),
        throttle=float(clipped[2]),
        rudder=float(clipped[3]),
    )


def _mark_terminal_failure(
    reward_breakdown: RewardBreakdown,
    *,
    reason: str,
    terminal_failure_reward: float,
) -> RewardBreakdown:
    extra_penalty = 0.0 if reward_breakdown.terminated else terminal_failure_reward
    return replace(
        reward_breakdown,
        reward=reward_breakdown.reward + extra_penalty,
        terminal_penalty=reward_breakdown.terminal_penalty + extra_penalty,
        terminated=True,
        termination_reason=reason,
    )


class HoverPilotHoverEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        host: str,
        port: int,
        reward_config: Optional[RewardConfig] = None,
        max_episode_steps: Optional[int] = None,
        sleep_interval_s: float = 0.0,
        anchor_target_to_reset_state: bool = True,
        task_profile: HoverTaskProfile = STANDARD_HOVER_TASK,
        reset_button_threshold: float = 0.5,
        lost_components_threshold: float = 0.5,
        physics_time_reset_tolerance_s: float = 1.0e-3,
        max_reset_wait_seconds: float = DEFAULT_RESET_WAIT_SECONDS,
        reset_poll_interval_seconds: float = DEFAULT_RESET_POLL_INTERVAL_SECONDS,
        ready_controller_active_threshold: Optional[float] = None,
        ready_running_threshold: Optional[float] = None,
        ready_locked_threshold: float = BOOL_FIELD_THRESHOLD,
        require_nonzero_physics_time_for_ready: bool = True,
        allow_ground_contact_at_ready: bool = True,
        minimum_start_altitude_agl_m: float = 0.25,
        start_groundspeed_threshold_mps: float = 1.0,
        start_airspeed_threshold_mps: float = 1.5,
        start_body_rate_threshold_deg_s: float = 60.0,
        elevator_start_inclination_tolerance_deg: float = 0.5,
        reposition_speed_threshold_mps: float = 0.5,
        reset_teleport_distance_m: float = 2.0,
        client_factory: Optional[Callable[[], RFLinkClient]] = None,
    ):
        super().__init__()
        self.host = host
        self.port = port
        self.max_episode_steps = max_episode_steps
        self.sleep_interval_s = sleep_interval_s
        self.anchor_target_to_reset_state = anchor_target_to_reset_state
        self.task_profile = task_profile
        base_reward_config = RewardConfig() if reward_config is None else reward_config
        self.reward_config = replace(
            base_reward_config,
            profile=task_profile.reward_profile,
        )
        self.reset_button_threshold = reset_button_threshold
        self.lost_components_threshold = lost_components_threshold
        self.physics_time_reset_tolerance_s = physics_time_reset_tolerance_s
        self.max_reset_wait_seconds = max_reset_wait_seconds
        self.reset_poll_interval_seconds = reset_poll_interval_seconds
        self.ready_controller_active_threshold = ready_controller_active_threshold
        self.ready_running_threshold = ready_running_threshold
        self.ready_locked_threshold = ready_locked_threshold
        self.require_nonzero_physics_time_for_ready = require_nonzero_physics_time_for_ready
        self.allow_ground_contact_at_ready = allow_ground_contact_at_ready
        self.minimum_start_altitude_agl_m = minimum_start_altitude_agl_m
        self.start_groundspeed_threshold_mps = start_groundspeed_threshold_mps
        self.start_airspeed_threshold_mps = start_airspeed_threshold_mps
        self.start_body_rate_threshold_deg_s = start_body_rate_threshold_deg_s
        self.elevator_start_inclination_tolerance_deg = (
            elevator_start_inclination_tolerance_deg
        )
        self.reposition_speed_threshold_mps = reposition_speed_threshold_mps
        self.reset_teleport_distance_m = reset_teleport_distance_m
        self._client_factory = (
            client_factory if client_factory is not None else lambda: RFLinkClient(self.host, self.port)
        )
        self._client = None  # type: Optional[RFLinkClient]
        self._episode_steps = 0
        self._last_state = None  # type: Optional[FlightAxisState]
        self._pending_episode_start = None  # type: Optional[Tuple[FlightAxisState, str]]
        self._waiting_for_reset = False
        self._episode_started = False
        self._ground_contact_started_at_s = None  # type: Optional[float]
        self._previous_elevator = 0.0
        self._last_longitudinal_position_rate_mps = 0.0

        self.action_space = spaces.Box(
            low=np.asarray([-1.0, -1.0, 0.0, -1.0], dtype=np.float32),
            high=np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(task_profile.observation_dim,),
            dtype=np.float32,
        )

    def _compute_elevator_features(
        self,
        state: FlightAxisState,
    ) -> ElevatorHoverFeatures:
        position_rate_mps = self._last_longitudinal_position_rate_mps
        previous_state = self._last_state
        if previous_state is None:
            position_rate_mps = 0.0
        else:
            delta_time_s = (
                state.m_currentPhysicsTime_SEC
                - previous_state.m_currentPhysicsTime_SEC
            )
            delta_x_m = (
                state.m_aircraftPositionX_MTR
                - previous_state.m_aircraftPositionX_MTR
            )
            delta_y_m = (
                state.m_aircraftPositionY_MTR
                - previous_state.m_aircraftPositionY_MTR
            )
            displacement_m = math.hypot(delta_x_m, delta_y_m)
            if (
                delta_time_s > self.physics_time_reset_tolerance_s
                and displacement_m < self.reset_teleport_distance_m
            ):
                position_rate_mps = project_onto_target_heading(
                    delta_x_m / delta_time_s,
                    delta_y_m / delta_time_s,
                    self.reward_config.target_azimuth_deg,
                )
            elif (
                delta_time_s < -self.physics_time_reset_tolerance_s
                or displacement_m >= self.reset_teleport_distance_m
            ):
                position_rate_mps = 0.0
        return compute_elevator_hover_features(
            state,
            target_x_m=self.reward_config.target_x_m,
            target_y_m=self.reward_config.target_y_m,
            target_altitude_agl_m=self.reward_config.target_altitude_agl_m,
            target_azimuth_deg=self.reward_config.target_azimuth_deg,
            longitudinal_position_rate_mps=position_rate_mps,
        )

    def _state_to_observation(
        self,
        state: FlightAxisState,
        *,
        elevator_features: Optional[ElevatorHoverFeatures] = None,
    ) -> np.ndarray:
        if self.task_profile == ELEVATOR_HOVER_TASK:
            features = (
                elevator_features
                if elevator_features is not None
                else self._compute_elevator_features(state)
            )
            return elevator_features_to_observation(
                features,
                config=self.reward_config,
            )
        return state_to_observation(
            state,
            target_x_m=self.reward_config.target_x_m,
            target_y_m=self.reward_config.target_y_m,
            target_altitude_agl_m=self.reward_config.target_altitude_agl_m,
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self._episode_steps = 0
        self._last_state = None
        self._pending_episode_start = None
        self._waiting_for_reset = False
        self._episode_started = False
        self._ground_contact_started_at_s = None
        self._previous_elevator = 0.0
        self._last_longitudinal_position_rate_mps = 0.0
        self.close()
        self._client = self._client_factory()
        self._client.connect()

        ready_action = self._safe_start_action()
        if options and "initial_action" in options:
            ready_action = gym_action_to_rf_action(options["initial_action"])

        state, episode_start_reason = self._wait_for_ready_state(ready_action)
        self._previous_elevator = ready_action.elevator
        return self._start_episode_from_state(state, episode_start_reason=episode_start_reason)

    def step(self, action: np.ndarray):
        if self._client is None:
            raise RuntimeError("environment must be reset() before step()")

        if self.sleep_interval_s > 0.0:
            time.sleep(self.sleep_interval_s)

        rf_action = gym_action_to_rf_action(action)
        state = self._client.step(rf_action)
        elevator_features = (
            self._compute_elevator_features(state)
            if self.task_profile == ELEVATOR_HOVER_TASK
            else None
        )
        elevator_delta = rf_action.elevator - self._previous_elevator
        self._previous_elevator = rf_action.elevator
        ground_contact_duration_s = self._update_ground_contact_duration(state)
        reward_breakdown = compute_reward(
            state,
            self.reward_config,
            episode_started=self._episode_started,
            ground_contact_duration_s=ground_contact_duration_s,
            elevator_delta=elevator_delta,
            elevator_features=elevator_features,
        )
        readiness = self.compute_episode_start_status(state)
        trainer_reset_reason = self._detect_trainer_reset(state)
        parked_reason = self._detect_parked_episode_boundary(state)
        lifecycle = EpisodeLifecycleResult(
            ready=readiness.ready,
            started=self._episode_started,
            terminated=reward_breakdown.terminated,
            truncated=False,
            reason=reward_breakdown.termination_reason,
        )

        if trainer_reset_reason is not None:
            self._waiting_for_reset = True
            self._episode_started = False
            self._pending_episode_start = (state, trainer_reset_reason)
            if self._assess_episode_boundary(
                state,
                require_reset_boundary=False,
                pending_reset_reason=trainer_reset_reason,
            ).can_start:
                self._waiting_for_reset = False
            reward_breakdown = _mark_terminal_failure(
                reward_breakdown,
                reason=trainer_reset_reason,
                terminal_failure_reward=self.reward_config.terminal_failure_reward,
            )
            lifecycle = replace(lifecycle, terminated=True, started=False, reason=trainer_reset_reason)

        if (
            parked_reason is not None
            and trainer_reset_reason is None
            and (
                not reward_breakdown.terminated
                or reward_breakdown.termination_reason == "altitude_too_low"
            )
        ):
            self._waiting_for_reset = True
            self._episode_started = False
            reward_breakdown = _mark_terminal_failure(
                reward_breakdown,
                reason=parked_reason,
                terminal_failure_reward=self.reward_config.terminal_failure_reward,
            )
            lifecycle = replace(lifecycle, terminated=True, started=False, reason=parked_reason)

        self._episode_steps += 1
        truncated = self.max_episode_steps is not None and self._episode_steps >= self.max_episode_steps
        lifecycle = replace(lifecycle, truncated=truncated)

        if reward_breakdown.terminated and trainer_reset_reason is None:
            self._waiting_for_reset = True
            self._episode_started = False
            lifecycle = replace(lifecycle, started=False)

        observation = self._state_to_observation(
            state,
            elevator_features=elevator_features,
        )
        info = self._build_info(
            state=state,
            reward_breakdown=reward_breakdown,
            truncated=truncated,
            reset=False,
            episode_start_reason=None,
            waiting_for_reset=self._waiting_for_reset,
            lifecycle=lifecycle,
            readiness=readiness,
            ground_contact_duration_s=ground_contact_duration_s,
            elevator_features=elevator_features,
        )
        if elevator_features is not None:
            self._last_longitudinal_position_rate_mps = (
                elevator_features.longitudinal_velocity_mps
            )
        self._last_state = state
        return (
            observation,
            float(reward_breakdown.reward),
            bool(reward_breakdown.terminated),
            bool(truncated),
            info,
        )

    def poll_wait_for_next_episode(
        self,
        action: Optional[Union[np.ndarray, list, tuple]] = None,
    ) -> Tuple[bool, np.ndarray, Dict[str, Any]]:
        if self._client is None:
            raise RuntimeError("environment must be reset() before waiting for the next episode")

        if self._pending_episode_start is not None:
            pending_state, reason = self._pending_episode_start
            assessment = self._assess_episode_boundary(
                pending_state,
                require_reset_boundary=False,
                pending_reset_reason=reason,
            )
            if assessment.can_start:
                self._pending_episode_start = None
                self._previous_elevator = 0.0
                observation, info = self._start_episode_from_state(pending_state, episode_start_reason=reason)
                return True, observation, info

        wait_action = self._safe_start_action() if action is None else gym_action_to_rf_action(action)
        state = self._poll_state(wait_action, interval_s=self.reset_poll_interval_seconds)
        pending_reason = self._pending_episode_start[1] if self._pending_episode_start is not None else None
        assessment = self._assess_episode_boundary(
            state,
            require_reset_boundary=self._waiting_for_reset,
            pending_reset_reason=pending_reason,
        )
        if assessment.reset_reason is not None:
            self._pending_episode_start = (state, assessment.reset_reason)
            pending_reason = assessment.reset_reason

        if self._pending_episode_start is not None and assessment.can_start:
            self._pending_episode_start = None
            self._previous_elevator = wait_action.elevator
            observation, info = self._start_episode_from_state(state, episode_start_reason=pending_reason)
            return True, observation, info
        lifecycle = EpisodeLifecycleResult(
            ready=assessment.readiness.ready,
            started=False,
            terminated=False,
            truncated=False,
            reason=assessment.readiness.reason,
        )
        elevator_features = (
            self._compute_elevator_features(state)
            if self.task_profile == ELEVATOR_HOVER_TASK
            else None
        )
        info = self._build_info(
            state=state,
            reward_breakdown=None,
            truncated=False,
            reset=False,
            episode_start_reason=None,
            waiting_for_reset=True,
            lifecycle=lifecycle,
            readiness=assessment.readiness,
            ground_contact_duration_s=0.0,
            elevator_features=elevator_features,
        )
        if elevator_features is not None:
            self._last_longitudinal_position_rate_mps = (
                elevator_features.longitudinal_velocity_mps
            )
        self._last_state = state
        return (
            False,
            self._state_to_observation(
                state,
                elevator_features=elevator_features,
            ),
            info,
        )

    def wait_for_next_episode(
        self,
        action: Optional[Union[np.ndarray, list, tuple]] = None,
    ):
        while True:
            started, observation, info = self.poll_wait_for_next_episode(action=action)
            if started:
                return observation, info

    def continue_after_truncation(self):
        """Start a new time-limit segment without resetting the live aircraft."""

        if self._last_state is None or not self._episode_started:
            raise RuntimeError("cannot continue a truncated episode without an active aircraft state")
        if self._waiting_for_reset:
            raise RuntimeError("cannot continue a truncated episode while waiting for trainer reset")

        self._episode_steps = 0
        state = self._last_state
        elevator_features = (
            self._compute_elevator_features(state)
            if self.task_profile == ELEVATOR_HOVER_TASK
            else None
        )
        readiness = self.compute_episode_start_status(state)
        lifecycle = EpisodeLifecycleResult(
            ready=readiness.ready,
            started=True,
            terminated=False,
            truncated=False,
            reason="time_limit_continuation",
        )
        info = self._build_info(
            state=state,
            reward_breakdown=None,
            truncated=False,
            reset=False,
            episode_start_reason="time_limit_continuation",
            waiting_for_reset=False,
            lifecycle=lifecycle,
            readiness=readiness,
            ground_contact_duration_s=0.0,
            elevator_features=elevator_features,
        )
        observation = self._state_to_observation(
            state,
            elevator_features=elevator_features,
        )
        return observation, info

    def render(self):
        return None

    def close(self):
        if self._client is not None:
            self._client.close()
            self._client = None

    def compute_episode_start_status(self, state: FlightAxisState) -> EpisodeLifecycleResult:
        # These flags are used conservatively as operational readiness hints.
        # RealFlight Link does not guarantee perfect semantics for every trainer mode,
        # so controller/engine checks are configurable instead of hard-required by default.
        if state_looks_uninitialized(state):
            return EpisodeLifecycleResult(False, False, False, False, "uninitialized_state")
        if self.require_nonzero_physics_time_for_ready and state.m_currentPhysicsTime_SEC <= 0.0:
            return EpisodeLifecycleResult(False, False, False, False, "physics_time_not_started")
        if state.m_isLocked > self.ready_locked_threshold:
            return EpisodeLifecycleResult(False, False, False, False, "vehicle_locked")
        if self._is_lost_components_active(state):
            return EpisodeLifecycleResult(False, False, False, False, "lost_components")
        if state.m_currentAircraftStatus in self.reward_config.known_terminal_aircraft_status_codes:
            return EpisodeLifecycleResult(False, False, False, False, "aircraft_status_terminal")
        if (
            self.ready_controller_active_threshold is not None
            and state.m_flightAxisControllerIsActive < self.ready_controller_active_threshold
        ):
            return EpisodeLifecycleResult(False, False, False, False, "controller_inactive")
        if (
            self.ready_running_threshold is not None
            and state.m_anEngineIsRunning < self.ready_running_threshold
        ):
            return EpisodeLifecycleResult(False, False, False, False, "engine_stopped")
        if not self.allow_ground_contact_at_ready and self._is_touching_ground(state):
            return EpisodeLifecycleResult(False, False, False, False, "touching_ground")
        return EpisodeLifecycleResult(True, True, False, False, None)

    def _wait_for_ready_state(self, action: RFControlAction) -> Tuple[FlightAxisState, str]:
        deadline = time.monotonic() + self.max_reset_wait_seconds
        last_state = None  # type: Optional[FlightAxisState]
        last_reason = "reset_timeout"
        startup_sync_required = False
        pending_start_reason = None  # type: Optional[str]
        while time.monotonic() <= deadline:
            state = self._poll_state(action, interval_s=self.reset_poll_interval_seconds)
            assessment = self._assess_episode_boundary(
                state,
                require_reset_boundary=startup_sync_required,
                pending_reset_reason=pending_start_reason,
            )
            self._last_state = state
            last_state = state
            last_reason = assessment.readiness.reason or last_reason
            if not startup_sync_required and assessment.pre_reset_wait:
                startup_sync_required = True
                self._waiting_for_reset = True

            if assessment.reset_reason is not None:
                pending_start_reason = assessment.reset_reason
                startup_sync_required = False

            if assessment.can_start and not startup_sync_required:
                self._waiting_for_reset = False
                return state, pending_start_reason or "reset_ready"
        diagnostics = "none"
        if last_state is not None:
            diagnostics = self._format_readiness_diagnostics(last_state)
        raise TimeoutError(f"timed out waiting for ready episode state: {last_reason}; {diagnostics}")

    def _poll_state(self, action: RFControlAction, *, interval_s: float) -> FlightAxisState:
        if interval_s > 0.0:
            time.sleep(interval_s)
        return self._client.request_state(action)

    def _safe_start_action(self) -> RFControlAction:
        return RFControlAction.safe_idle()

    def _start_episode_from_state(self, state: FlightAxisState, episode_start_reason: str):
        self._episode_steps = 0
        self._pending_episode_start = None
        self._waiting_for_reset = False
        self._episode_started = True
        self._ground_contact_started_at_s = None
        if self.anchor_target_to_reset_state:
            self.reward_config = replace(
                self.reward_config,
                target_x_m=state.m_aircraftPositionX_MTR,
                target_y_m=state.m_aircraftPositionY_MTR,
                target_altitude_agl_m=state.m_altitudeAGL_MTR,
            )
        if self.task_profile.anchor_heading_to_reset_state:
            self.reward_config = replace(
                self.reward_config,
                target_azimuth_deg=state.m_azimuth_DEG,
            )
        self._last_longitudinal_position_rate_mps = 0.0
        self._last_state = state
        elevator_features = (
            self._compute_elevator_features(state)
            if self.task_profile == ELEVATOR_HOVER_TASK
            else None
        )
        readiness = self.compute_episode_start_status(state)
        lifecycle = EpisodeLifecycleResult(
            ready=readiness.ready,
            started=True,
            terminated=False,
            truncated=False,
            reason=episode_start_reason,
        )
        observation = self._state_to_observation(
            state,
            elevator_features=elevator_features,
        )
        info = self._build_info(
            state=state,
            reward_breakdown=None,
            truncated=False,
            reset=True,
            episode_start_reason=episode_start_reason,
            waiting_for_reset=False,
            lifecycle=lifecycle,
            readiness=readiness,
            ground_contact_duration_s=0.0,
            elevator_features=elevator_features,
        )
        return observation, info

    def _build_info(
        self,
        *,
        state: FlightAxisState,
        reward_breakdown: Optional[RewardBreakdown],
        truncated: bool,
        reset: bool,
        episode_start_reason: Optional[str],
        waiting_for_reset: bool,
        lifecycle: EpisodeLifecycleResult,
        readiness: EpisodeLifecycleResult,
        ground_contact_duration_s: float,
        elevator_features: Optional[ElevatorHoverFeatures] = None,
    ) -> Dict[str, Any]:
        debug_state = {
            "x_m": state.m_aircraftPositionX_MTR,
            "y_m": state.m_aircraftPositionY_MTR,
            "altitude_agl_m": state.m_altitudeAGL_MTR,
            "roll_deg": state.m_roll_DEG,
            "azimuth_deg": state.m_azimuth_DEG,
            "inclination_deg": state.m_inclination_DEG,
            "pitch_rate_deg_s": state.m_pitchRate_DEGpSEC,
            "velocity_world_u_mps": state.m_velocityWorldU_MPS,
            "velocity_world_v_mps": state.m_velocityWorldV_MPS,
            "velocity_world_w_mps": state.m_velocityWorldW_MPS,
            "controller_active": state.m_flightAxisControllerIsActive,
            "physics_time_s": state.m_currentPhysicsTime_SEC,
            "reset_button_pressed": state.m_resetButtonHasBeenPressed,
            "lost_components": state.m_hasLostComponents,
            "aircraft_status": state.m_currentAircraftStatus,
            "touching_ground": state.m_isTouchingGround,
            "engine_running": state.m_anEngineIsRunning,
            "vehicle_locked": state.m_isLocked,
            "ground_contact_duration_s": ground_contact_duration_s,
        }
        debug_state["distance_from_cylinder_axis_m"] = math.hypot(
            state.m_aircraftPositionX_MTR
            - self.reward_config.target_x_m,
            state.m_aircraftPositionY_MTR
            - self.reward_config.target_y_m,
        )

        info = {  # type: Dict[str, Any]
            "state_summary": state.summary(),
            "debug_state": debug_state,
            "target_hover": {
                "x_m": self.reward_config.target_x_m,
                "y_m": self.reward_config.target_y_m,
                "altitude_agl_m": self.reward_config.target_altitude_agl_m,
                "roll_deg": self.reward_config.target_roll_deg,
                "inclination_deg": (
                    self.task_profile.realflight_reference_inclination_deg
                ),
                "azimuth_deg": self.reward_config.target_azimuth_deg,
            },
            "episode_step": self._episode_steps,
            "reset": reset,
            "truncated": truncated,
            "episode_start_reason": episode_start_reason,
            "waiting_for_reset": waiting_for_reset,
            "episode_lifecycle": asdict(lifecycle),
            "episode_readiness": asdict(readiness),
        }
        if reward_breakdown is not None:
            info["reward_breakdown"] = asdict(reward_breakdown)
            info["termination_reason"] = reward_breakdown.termination_reason
        if self.task_profile == ELEVATOR_HOVER_TASK:
            features = (
                elevator_features
                if elevator_features is not None
                else self._compute_elevator_features(state)
            )
            info["elevator_hover_features"] = asdict(features)
            info["elevator_recovery_target_deg"] = (
                reward_breakdown.target_inclination_error_deg
                if reward_breakdown is not None
                else compute_elevator_recovery_target_deg(
                    features.longitudinal_position_error_m,
                    features.longitudinal_velocity_mps,
                    position_gain_deg_per_m=(
                        self.reward_config.elevator_recovery_position_gain_deg_per_m
                    ),
                    velocity_gain_deg_per_mps=(
                        self.reward_config.elevator_recovery_velocity_gain_deg_per_mps
                    ),
                    inclination_limit_deg=(
                        self.reward_config.elevator_recovery_inclination_limit_deg
                    ),
                )
            )
        return info

    def _detect_trainer_reset(self, state: FlightAxisState) -> Optional[str]:
        if state.m_resetButtonHasBeenPressed >= self.reset_button_threshold:
            return "trainer_reset_button"

        previous_state = self._last_state
        if previous_state is None:
            return None

        if (
            state.m_currentPhysicsTime_SEC + self.physics_time_reset_tolerance_s
            < previous_state.m_currentPhysicsTime_SEC
        ):
            return "trainer_reset"

        if self._looks_like_reset_teleport(previous_state, state):
            return "trainer_repositioned"

        return None

    def _detect_parked_episode_boundary(self, state: FlightAxisState) -> Optional[str]:
        if not self._episode_started:
            return None
        if not self._is_parked_on_ground_state(state):
            return None
        return "parked_on_ground"

    def _update_ground_contact_duration(self, state: FlightAxisState) -> float:
        if not self._episode_started or not self._is_touching_ground(state):
            self._ground_contact_started_at_s = None
            return 0.0
        if self._ground_contact_started_at_s is None:
            self._ground_contact_started_at_s = time.monotonic()
            return 0.0
        return time.monotonic() - self._ground_contact_started_at_s

    def _is_lost_components_active(self, state: FlightAxisState) -> bool:
        return state.m_hasLostComponents > self.lost_components_threshold

    def _is_touching_ground(self, state: FlightAxisState) -> bool:
        return state.m_isTouchingGround > self.reward_config.touching_ground_threshold


    def _format_readiness_diagnostics(self, state: FlightAxisState) -> str:
        return (
            f"locked={state.m_isLocked:.1f} lost={state.m_hasLostComponents:.1f} "
            f"engine={state.m_anEngineIsRunning:.1f} ctrl={state.m_flightAxisControllerIsActive:.1f} "
            f"ground={state.m_isTouchingGround:.1f} status={state.m_currentAircraftStatus:.1f} "
            f"inc={state.m_inclination_DEG:.1f} gs={state.m_groundspeed_MPS:.2f} "
            f"air={state.m_airspeed_MPS:.2f} physics_t={state.m_currentPhysicsTime_SEC:.3f}"
        )

    def _looks_like_reset_teleport(self, previous_state: FlightAxisState, state: FlightAxisState) -> bool:
        # RealFlight Hover Trainer does not always expose a dedicated reset flag.
        # As a fallback, treat a sudden jump out of a crash-wait state as a
        # trainer-driven reposition rather than a boundary failure.
        if not self.compute_episode_start_status(state).ready:
            return False
        if (
            self._is_low_altitude_wait_state(previous_state)
            and not self._is_low_altitude_wait_state(state)
            and (
                state.m_altitudeAGL_MTR
                - previous_state.m_altitudeAGL_MTR
                >= self.reset_teleport_distance_m / 2.0
            )
        ):
            return True
        return (
            self._planar_distance(previous_state, state)
            >= self.reset_teleport_distance_m
            and self._is_reset_like_stationary_state(state)
            and self._is_inactive_reset_state(state)
        )

    def _assess_episode_boundary(
        self,
        state: FlightAxisState,
        *,
        require_reset_boundary: bool,
        pending_reset_reason: Optional[str],
    ) -> EpisodeBoundaryAssessment:
        readiness = self.compute_episode_start_status(state)
        reset_reason = self._detect_trainer_reset(state)
        effective_reset_reason = reset_reason or pending_reset_reason
        pre_reset_wait = self._is_pre_reset_wait_state(state)
        if effective_reset_reason in TRAINER_RESET_REASONS and not self._is_low_altitude_wait_state(state):
            pre_reset_wait = False
        can_start = readiness.ready and not pre_reset_wait
        if can_start and not self._is_start_stable_state(state):
            can_start = False
        if can_start and effective_reset_reason in TRAINER_RESET_REASONS:
            return EpisodeBoundaryAssessment(readiness, reset_reason, True, pre_reset_wait)
        if can_start and require_reset_boundary:
            can_start = False
        if can_start and self._is_reset_like_stationary_state(state) and self._is_inactive_reset_state(state):
            can_start = False
        return EpisodeBoundaryAssessment(readiness, reset_reason, can_start, pre_reset_wait)

    def _is_pre_reset_wait_state(self, state: FlightAxisState) -> bool:
        return (
            self._is_parked_on_ground_state(state)
            or self._is_low_altitude_wait_state(state)
            or (
                self._is_inactive_reset_state(state)
                and not self._is_start_stable_state(state)
            )
            or (
                self._is_reset_like_stationary_state(state)
                and self._is_inactive_reset_state(state)
            )
        )

    def _is_low_altitude_wait_state(self, state: FlightAxisState) -> bool:
        # The hover trainer's pre-reset crash states often remain very close to the
        # ground. Treat low AGL itself as a strong "not started yet" signal even if
        # the aircraft still has some residual motion.
        return state.m_altitudeAGL_MTR <= self.minimum_start_altitude_agl_m

    def _is_parked_on_ground_state(self, state: FlightAxisState) -> bool:
        return self._is_low_altitude_wait_state(state) and self._is_reset_like_stationary_state(state)

    def _is_start_stable_state(self, state: FlightAxisState) -> bool:
        # A valid episode start should not begin mid-crash or mid-fall. Require a
        # modestly stable state before declaring the episode active.
        attitude_ready = (
            not self.task_profile.require_vertical_start
            or abs(
                state.m_inclination_DEG
                - REALFLIGHT_VERTICAL_HOVER_INCLINATION_DEG
            )
            <= self.elevator_start_inclination_tolerance_deg
        )
        return (
            attitude_ready
            and state.m_groundspeed_MPS <= self.start_groundspeed_threshold_mps
            and state.m_airspeed_MPS <= self.start_airspeed_threshold_mps
            and abs(state.m_pitchRate_DEGpSEC) <= self.start_body_rate_threshold_deg_s
            and abs(state.m_rollRate_DEGpSEC) <= self.start_body_rate_threshold_deg_s
            and abs(state.m_yawRate_DEGpSEC) <= self.start_body_rate_threshold_deg_s
        )

    def _is_reset_like_stationary_state(self, state: FlightAxisState) -> bool:
        rate_threshold_deg_s = 5.0
        return (
            state.m_groundspeed_MPS <= self.reposition_speed_threshold_mps
            and state.m_airspeed_MPS <= self.reposition_speed_threshold_mps
            and abs(state.m_pitchRate_DEGpSEC) <= rate_threshold_deg_s
            and abs(state.m_rollRate_DEGpSEC) <= rate_threshold_deg_s
            and abs(state.m_yawRate_DEGpSEC) <= rate_threshold_deg_s
        )

    def _is_inactive_reset_state(self, state: FlightAxisState) -> bool:
        return (
            state.m_flightAxisControllerIsActive <= BOOL_FIELD_THRESHOLD
            and state.m_anEngineIsRunning <= BOOL_FIELD_THRESHOLD
        )

    def _planar_distance(self, previous_state: FlightAxisState, state: FlightAxisState) -> float:
        dx = state.m_aircraftPositionX_MTR - previous_state.m_aircraftPositionX_MTR
        dy = state.m_aircraftPositionY_MTR - previous_state.m_aircraftPositionY_MTR
        return float((dx * dx + dy * dy) ** 0.5)
