from dataclasses import dataclass, field
import math
from typing import Optional, Tuple

from hoverpilot.rflink.models import FlightAxisState


REWARD_PROFILE_STANDARD = "standard"
REWARD_PROFILE_ELEVATOR = "elevator"
STANDARD_HOVER_INCLINATION_DEG = 0.0
REALFLIGHT_VERTICAL_HOVER_INCLINATION_DEG = 90.0


@dataclass
class RewardConfig:
    profile: str = REWARD_PROFILE_STANDARD
    target_x_m: float = 0.0
    target_y_m: float = 0.0
    target_altitude_agl_m: float = 1.5
    target_roll_deg: float = 0.0
    target_azimuth_deg: float = 0.0
    trainer_cylinder_radius_m: float = 6.0
    min_altitude_agl_m: float = 0.2
    position_error_weight: float = 0.15
    altitude_error_weight: float = 0.2
    attitude_error_weight: float = 0.01
    survival_reward: float = 1.0
    elevator_position_error_weight: float = 0.2
    elevator_altitude_error_weight: float = 0.1
    inclination_error_weight: float = 1.0
    elevator_recovery_position_gain_deg_per_m: float = 2.0
    elevator_recovery_velocity_gain_deg_per_mps: float = 3.0
    elevator_recovery_inclination_limit_deg: float = 30.0
    pitch_rate_weight: float = 0.5
    velocity_error_weight: float = 0.2
    elevator_smoothness_weight: float = 0.01
    inclination_error_scale_deg: float = 15.0
    pitch_rate_scale_deg_s: float = 30.0
    longitudinal_position_scale_m: float = 4.0
    altitude_error_scale_m: float = 1.5
    velocity_error_scale_mps: float = 5.0
    max_normalized_error_squared: float = 16.0
    boundary_proximity_weight: float = 0.75
    terminal_failure_reward: float = -25.0
    proximity_penalty_margin_ratio: float = 0.25
    controller_active_threshold: Optional[float] = None
    lost_components_threshold: float = 0.5
    locked_threshold: float = 0.5
    engine_running_threshold: float = 0.5
    touching_ground_threshold: float = 0.5
    terminate_on_engine_stopped: bool = False
    terminate_on_touching_ground: bool = True
    ground_contact_grace_seconds: float = 0.0
    known_terminal_aircraft_status_codes: Tuple[float, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.trainer_cylinder_radius_m)
            or self.trainer_cylinder_radius_m <= 0.0
        ):
            raise ValueError("trainer_cylinder_radius_m must be greater than zero")
        observation_scales = {
            "inclination_error_scale_deg": self.inclination_error_scale_deg,
            "pitch_rate_scale_deg_s": self.pitch_rate_scale_deg_s,
            "longitudinal_position_scale_m": (
                self.longitudinal_position_scale_m
            ),
            "altitude_error_scale_m": self.altitude_error_scale_m,
            "velocity_error_scale_mps": self.velocity_error_scale_mps,
        }
        for name, value in observation_scales.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and greater than zero")
        recovery_parameters = {
            "elevator_recovery_position_gain_deg_per_m": (
                self.elevator_recovery_position_gain_deg_per_m
            ),
            "elevator_recovery_velocity_gain_deg_per_mps": (
                self.elevator_recovery_velocity_gain_deg_per_mps
            ),
            "elevator_recovery_inclination_limit_deg": (
                self.elevator_recovery_inclination_limit_deg
            ),
        }
        for name, value in recovery_parameters.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class ElevatorHoverFeatures:
    """State features that elevator-only hover can directly influence."""

    inclination_error_deg: float
    pitch_rate_deg_s: float
    longitudinal_position_error_m: float
    longitudinal_velocity_mps: float
    altitude_error_m: float
    vertical_velocity_mps: float


@dataclass
class TerminationResult:
    terminated: bool
    termination_reason: Optional[str] = None


@dataclass
class RewardBreakdown:
    reward: float
    survival_reward: float
    position_penalty: float
    altitude_penalty: float
    attitude_penalty: float
    angular_rate_penalty: float
    velocity_penalty: float
    action_smoothness_penalty: float
    boundary_proximity_penalty: float
    terminal_penalty: float
    target_inclination_error_deg: float
    inclination_tracking_error_deg: float
    terminated: bool
    termination_reason: Optional[str]



def compute_termination(
    state: FlightAxisState,
    config: RewardConfig,
    *,
    episode_started: bool = True,
    ground_contact_duration_s: float = 0.0,
) -> TerminationResult:
    if state.m_hasLostComponents > config.lost_components_threshold:
        return TerminationResult(True, "lost_components")

    x_error = state.m_aircraftPositionX_MTR - config.target_x_m
    y_error = state.m_aircraftPositionY_MTR - config.target_y_m
    distance_from_cylinder_axis = math.hypot(x_error, y_error)
    if distance_from_cylinder_axis > config.trainer_cylinder_radius_m:
        return TerminationResult(True, "outside_trainer_cylinder")

    altitude_agl_m = state.m_altitudeAGL_MTR
    if altitude_agl_m < config.min_altitude_agl_m:
        return TerminationResult(True, "altitude_too_low")
    if state.m_isLocked > config.locked_threshold:
        return TerminationResult(True, "vehicle_locked")

    if (
        config.controller_active_threshold is not None
        and state.m_flightAxisControllerIsActive < config.controller_active_threshold
    ):
        return TerminationResult(True, "controller_inactive")

    if (
        episode_started
        and config.terminate_on_engine_stopped
        and state.m_anEngineIsRunning < config.engine_running_threshold
    ):
        return TerminationResult(True, "engine_stopped")

    if (
        episode_started
        and config.terminate_on_touching_ground
        and state.m_isTouchingGround > config.touching_ground_threshold
        and ground_contact_duration_s >= config.ground_contact_grace_seconds
    ):
        return TerminationResult(True, "touching_ground")

    if state.m_currentAircraftStatus in config.known_terminal_aircraft_status_codes:
        return TerminationResult(True, "aircraft_status_terminal")

    return TerminationResult(False, None)



def compute_reward(
    state: FlightAxisState,
    config: RewardConfig,
    *,
    episode_started: bool = True,
    ground_contact_duration_s: float = 0.0,
    elevator_delta: float = 0.0,
    elevator_features: Optional[ElevatorHoverFeatures] = None,
) -> RewardBreakdown:
    termination = compute_termination(
        state,
        config,
        episode_started=episode_started,
        ground_contact_duration_s=ground_contact_duration_s,
    )

    if config.profile == REWARD_PROFILE_ELEVATOR:
        features = (
            elevator_features
            if elevator_features is not None
            else compute_elevator_hover_features(
                state,
                target_x_m=config.target_x_m,
                target_y_m=config.target_y_m,
                target_altitude_agl_m=config.target_altitude_agl_m,
                target_azimuth_deg=config.target_azimuth_deg,
            )
        )
        position_penalty = config.elevator_position_error_weight * (
            _bounded_normalized_square(
                features.longitudinal_position_error_m,
                config.longitudinal_position_scale_m,
                config.max_normalized_error_squared,
            )
        )
        altitude_penalty = config.elevator_altitude_error_weight * (
            _bounded_normalized_square(
                features.altitude_error_m,
                config.altitude_error_scale_m,
                config.max_normalized_error_squared,
            )
        )
        target_inclination_error_deg = (
            compute_elevator_recovery_target_deg(
                features.longitudinal_position_error_m,
                features.longitudinal_velocity_mps,
                position_gain_deg_per_m=(
                    config.elevator_recovery_position_gain_deg_per_m
                ),
                velocity_gain_deg_per_mps=(
                    config.elevator_recovery_velocity_gain_deg_per_mps
                ),
                inclination_limit_deg=(
                    config.elevator_recovery_inclination_limit_deg
                ),
            )
        )
        inclination_tracking_error_deg = (
            features.inclination_error_deg - target_inclination_error_deg
        )
        attitude_penalty = config.inclination_error_weight * (
            _bounded_normalized_square(
                inclination_tracking_error_deg,
                config.inclination_error_scale_deg,
                config.max_normalized_error_squared,
            )
        )
        angular_rate_penalty = config.pitch_rate_weight * (
            _bounded_normalized_square(
                features.pitch_rate_deg_s,
                config.pitch_rate_scale_deg_s,
                config.max_normalized_error_squared,
            )
        )
        velocity_penalty = config.velocity_error_weight * (
            _bounded_normalized_square(
                features.longitudinal_velocity_mps,
                config.velocity_error_scale_mps,
                config.max_normalized_error_squared,
            )
            + _bounded_normalized_square(
                features.vertical_velocity_mps,
                config.velocity_error_scale_mps,
                config.max_normalized_error_squared,
            )
        )
        action_smoothness_penalty = config.elevator_smoothness_weight * (
            elevator_delta / 2.0
        ) ** 2
        survival_reward = config.survival_reward
    elif config.profile == REWARD_PROFILE_STANDARD:
        x_error = state.m_aircraftPositionX_MTR - config.target_x_m
        y_error = state.m_aircraftPositionY_MTR - config.target_y_m
        altitude_error = (
            state.m_altitudeAGL_MTR - config.target_altitude_agl_m
        )
        position_penalty = config.position_error_weight * (
            (x_error * x_error) + (y_error * y_error)
        )
        altitude_penalty = config.altitude_error_weight * abs(altitude_error)
        attitude_penalty = config.attitude_error_weight * (
            abs(angular_error_deg(state.m_roll_DEG, config.target_roll_deg))
            + abs(
                angular_error_deg(
                    state.m_inclination_DEG,
                    STANDARD_HOVER_INCLINATION_DEG,
                )
            )
        )
        angular_rate_penalty = 0.0
        velocity_penalty = 0.0
        action_smoothness_penalty = 0.0
        survival_reward = 0.0
        target_inclination_error_deg = 0.0
        inclination_tracking_error_deg = 0.0
    else:
        raise ValueError(f"unsupported reward profile: {config.profile!r}")

    boundary_proximity_penalty = _compute_boundary_proximity_penalty(state, config)
    terminal_penalty = config.terminal_failure_reward if termination.terminated else 0.0

    reward = survival_reward - (
        position_penalty
        + altitude_penalty
        + attitude_penalty
        + angular_rate_penalty
        + velocity_penalty
        + action_smoothness_penalty
        + boundary_proximity_penalty
    ) + terminal_penalty

    return RewardBreakdown(
        reward=reward,
        survival_reward=survival_reward,
        position_penalty=position_penalty,
        altitude_penalty=altitude_penalty,
        attitude_penalty=attitude_penalty,
        angular_rate_penalty=angular_rate_penalty,
        velocity_penalty=velocity_penalty,
        action_smoothness_penalty=action_smoothness_penalty,
        boundary_proximity_penalty=boundary_proximity_penalty,
        terminal_penalty=terminal_penalty,
        target_inclination_error_deg=target_inclination_error_deg,
        inclination_tracking_error_deg=inclination_tracking_error_deg,
        terminated=termination.terminated,
        termination_reason=termination.termination_reason,
    )


def angular_error_deg(value_deg: float, target_deg: float) -> float:
    """Return the shortest signed angular error in [-180, 180)."""

    return (value_deg - target_deg + 180.0) % 360.0 - 180.0


def signed_vertical_inclination_error_deg(
    inclination_deg: float,
    azimuth_deg: float,
    target_azimuth_deg: float,
) -> float:
    """Project nose-up attitude error onto the reset elevator plane.

    RealFlight reports the vertical nose-up hover at ``inclination=90``.
    Around that Euler singularity, azimuth and roll jump by 180 degrees, so
    ``inclination - 90`` alone assigns the same sign to opposite pitch errors.
    Projecting by the heading difference preserves the physically meaningful
    elevator error: zero is nose-up, positive and negative are opposite tilts.
    """

    heading_error_rad = math.radians(
        angular_error_deg(azimuth_deg, target_azimuth_deg)
    )
    return (
        inclination_deg - REALFLIGHT_VERTICAL_HOVER_INCLINATION_DEG
    ) * math.cos(heading_error_rad)


def project_onto_target_heading(
    world_x: float,
    world_y: float,
    target_azimuth_deg: float,
) -> float:
    """Project a horizontal world vector onto RealFlight's heading axis.

    RealFlight azimuth is clockwise from its negative world-Y axis, so the
    corresponding unit vector is ``(sin(azimuth), -cos(azimuth))``.
    """

    heading_rad = math.radians(target_azimuth_deg)
    return world_x * math.sin(heading_rad) - world_y * math.cos(heading_rad)


def compute_elevator_hover_features(
    state: FlightAxisState,
    *,
    target_x_m: float,
    target_y_m: float,
    target_altitude_agl_m: float,
    target_azimuth_deg: float,
    longitudinal_position_rate_mps: Optional[float] = None,
) -> ElevatorHoverFeatures:
    """Compute the shared observation, reward, and telemetry feature set."""

    longitudinal_position_error_m = project_onto_target_heading(
        state.m_aircraftPositionX_MTR - target_x_m,
        state.m_aircraftPositionY_MTR - target_y_m,
        target_azimuth_deg,
    )
    longitudinal_velocity_mps = (
        project_onto_target_heading(
            state.m_velocityWorldU_MPS,
            state.m_velocityWorldV_MPS,
            target_azimuth_deg,
        )
        if longitudinal_position_rate_mps is None
        else float(longitudinal_position_rate_mps)
    )
    return ElevatorHoverFeatures(
        inclination_error_deg=signed_vertical_inclination_error_deg(
            state.m_inclination_DEG,
            state.m_azimuth_DEG,
            target_azimuth_deg,
        ),
        pitch_rate_deg_s=state.m_pitchRate_DEGpSEC,
        longitudinal_position_error_m=longitudinal_position_error_m,
        longitudinal_velocity_mps=longitudinal_velocity_mps,
        altitude_error_m=state.m_altitudeAGL_MTR - target_altitude_agl_m,
        vertical_velocity_mps=state.m_velocityWorldW_MPS,
    )


def compute_elevator_recovery_target_deg(
    longitudinal_position_error_m: float,
    longitudinal_velocity_mps: float,
    *,
    position_gain_deg_per_m: float = 2.0,
    velocity_gain_deg_per_mps: float = 3.0,
    inclination_limit_deg: float = 30.0,
) -> float:
    """Return the signed tilt target that drives longitudinal drift to zero.

    The velocity input is the observed longitudinal position-error rate.
    Outward motion strengthens the restoring tilt, while inward motion starts
    braking before the aircraft crosses origin. At the hover origin the target
    remains vertical (zero signed inclination error).
    """

    parameters = (
        position_gain_deg_per_m,
        velocity_gain_deg_per_mps,
        inclination_limit_deg,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in parameters):
        raise ValueError(
            "elevator recovery gains and limit must be finite and non-negative"
        )
    unconstrained_target = -(
        position_gain_deg_per_m * longitudinal_position_error_m
        + velocity_gain_deg_per_mps * longitudinal_velocity_mps
    )
    limit = inclination_limit_deg
    return max(-limit, min(unconstrained_target, limit))


def _bounded_normalized_square(
    value: float,
    scale: float,
    maximum: float,
) -> float:
    return min((value / scale) ** 2, maximum)



def _compute_boundary_proximity_penalty(
    state: FlightAxisState,
    config: RewardConfig,
) -> float:
    dx = state.m_aircraftPositionX_MTR - config.target_x_m
    dy = state.m_aircraftPositionY_MTR - config.target_y_m
    distance_from_cylinder_axis = math.hypot(dx, dy)
    return config.boundary_proximity_weight * _boundary_edge_penalty(
        distance_to_edge=(
            config.trainer_cylinder_radius_m
            - distance_from_cylinder_axis
        ),
        limit=config.trainer_cylinder_radius_m,
        margin_ratio=config.proximity_penalty_margin_ratio,
    )


def _boundary_edge_penalty(
    distance_to_edge: float,
    limit: float,
    margin_ratio: float,
) -> float:
    if limit <= 0.0:
        return 0.0

    margin = max(limit * margin_ratio, 1.0e-6)
    if distance_to_edge >= margin:
        return 0.0
    if distance_to_edge <= 0.0:
        return 1.0

    normalized = 1.0 - (distance_to_edge / margin)
    return normalized * normalized
