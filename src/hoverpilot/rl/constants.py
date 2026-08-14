"""Shared PPO control modes, defaults, and policy constants."""

from __future__ import annotations

import numpy as np

WAITING_LOG_INTERVAL_S = 0.75
DEFAULT_WAIT_ACTION = (0.0, 0.0, 0.0, 0.0)
DEFAULT_INITIAL_ACTION = (0.0, 0.0, 0.55, 0.0)
CONTROL_MODE_ALL = "all"
CONTROL_MODE_ELEVATOR = "elevator"
CONTROL_MODE_AILERON = "aileron"
CONTROL_MODE_RUDDER = "rudder"
CONTROL_MODE_THROTTLE = "throttle"
CONTROL_MODE_ELEVATOR_THROTTLE = "elevator-throttle"
CONTROL_MODE_AILERON_THROTTLE = "aileron-throttle"
CONTROL_MODE_RUDDER_THROTTLE = "rudder-throttle"
CONTROL_MODES = (
    CONTROL_MODE_ALL,
    CONTROL_MODE_AILERON,
    CONTROL_MODE_ELEVATOR,
    CONTROL_MODE_RUDDER,
    CONTROL_MODE_THROTTLE,
    CONTROL_MODE_ELEVATOR_THROTTLE,
    CONTROL_MODE_AILERON_THROTTLE,
    CONTROL_MODE_RUDDER_THROTTLE,
)
CONNECTION_EPISODE_CONTROL_MODES = {
    CONTROL_MODE_AILERON,
    CONTROL_MODE_RUDDER,
    CONTROL_MODE_THROTTLE,
    CONTROL_MODE_ELEVATOR_THROTTLE,
    CONTROL_MODE_AILERON_THROTTLE,
    CONTROL_MODE_RUDDER_THROTTLE,
}
POLICY_PRESET_NONE = "none"
POLICY_PRESET_ELEVATOR_PD = "elevator-pd"
POLICY_PRESETS = (POLICY_PRESET_NONE, POLICY_PRESET_ELEVATOR_PD)
PPO_CHECKPOINT_FORMAT = "hoverpilot-ppo"
PPO_CHECKPOINT_VERSION = 2
_ELEVATOR_PD_PRIOR_WEIGHT = np.asarray(
    [[-1.00, 1.50, 0.0, 0.0, 0.0, 0.0]],
    dtype=np.float32,
)
_ELEVATOR_PPO_INITIAL_GAIN = np.asarray(
    [0.55, 0.45],
    dtype=np.float32,
)
_AILERON_PPO_INITIAL_GAIN = np.asarray(
    [0.80, 2.50],
    dtype=np.float32,
)
_AILERON_PPO_INITIAL_TRIM = 0.78
_RUDDER_PPO_INITIAL_GAIN = np.asarray(
    [1.50, 1.20],
    dtype=np.float32,
)
_THROTTLE_PPO_INITIAL_GAIN = np.asarray(
    [1.50, 2.00],
    dtype=np.float32,
)
_ALL_CONTROLS_RESIDUAL_SCALE = 0.2
_THROTTLE_PPO_INITIAL_TRIM = 0.65
_ELEVATOR_PD_PRIOR_LIMIT = 0.5
_ELEVATOR_PD_RESIDUAL_LIMIT = 0.2
_ELEVATOR_EFFECTIVE_RESTORING_ACTION = 0.2
_ELEVATOR_OBSERVATION_CONFIG_FIELDS = (
    "inclination_error_scale_deg",
    "pitch_rate_scale_deg_s",
    "longitudinal_position_scale_m",
    "altitude_error_scale_m",
    "velocity_error_scale_mps",
    "elevator_recovery_position_gain_deg_per_m",
    "elevator_recovery_velocity_gain_deg_per_mps",
    "elevator_recovery_inclination_limit_deg",
)
_AILERON_OBSERVATION_CONFIG_FIELDS = (
    "roll_error_scale_deg",
    "roll_rate_scale_deg_s",
)
_RUDDER_RECOVERY_CONFIG_FIELDS = (
    "rudder_recovery_position_gain_deg_per_m",
    "rudder_recovery_velocity_gain_deg_per_mps",
    "rudder_recovery_angle_limit_deg",
)
_RUDDER_OBSERVATION_CONFIG_FIELDS = (
    "rudder_angle_error_scale_deg",
    "yaw_rate_scale_deg_s",
)
_THROTTLE_OBSERVATION_CONFIG_FIELDS = (
    "altitude_error_scale_m",
    "velocity_error_scale_mps",
)
_AILERON_THROTTLE_OBSERVATION_CONFIG_FIELDS = (
    *_AILERON_OBSERVATION_CONFIG_FIELDS,
    *_THROTTLE_OBSERVATION_CONFIG_FIELDS,
)
_RUDDER_THROTTLE_OBSERVATION_CONFIG_FIELDS = (
    *_RUDDER_OBSERVATION_CONFIG_FIELDS,
    *_THROTTLE_OBSERVATION_CONFIG_FIELDS,
)
DEFAULT_TIMESTEPS = 50_000
DEFAULT_ELEVATOR_TIMESTEPS = 300_000
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_EVAL_EPISODES = 10
DEFAULT_ENTROPY_COEF = 0.0001
DEFAULT_POLICY_STD = 0.08
