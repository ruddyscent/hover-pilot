"""Validation rules for PPO trainer configuration."""

from __future__ import annotations

import math

from .checkpoints import _validate_fixed_throttle
from .config import PPOConfig
from .constants import CONTROL_MODES
from .runtime import _validate_rflink_settings


def validate_trainer_config(config: PPOConfig) -> None:
    """Reject invalid trainer settings before allocating an environment or model."""
    if config.timesteps <= 0:
        raise ValueError("timesteps must be greater than zero")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be greater than zero")
    if config.eval_episodes <= 0:
        raise ValueError("eval_episodes must be greater than zero")
    if config.reward_scale <= 0.0:
        raise ValueError("reward_scale must be greater than zero")
    if config.entropy_coef is not None and config.entropy_coef < 0.0:
        raise ValueError("entropy_coef must be non-negative")
    if config.policy_initial_std is not None and config.policy_initial_std <= 0.0:
        raise ValueError("policy_initial_std must be greater than zero")
    if config.control_mode not in CONTROL_MODES:
        raise ValueError(
            f"Unsupported control mode {config.control_mode!r}; choose one of {CONTROL_MODES}."
        )
    _validate_rflink_settings(config)
    if config.checkpoint_interval_steps < 0:
        raise ValueError("checkpoint_interval_steps must be non-negative")
    if config.eval_interval_steps < 0:
        raise ValueError("eval_interval_steps must be non-negative")
    if config.telemetry_log_interval_steps < 0:
        raise ValueError("telemetry_log_interval_steps must be non-negative")
    if (
        not math.isfinite(config.episode_start_idle_seconds)
        or config.episode_start_idle_seconds < 0.0
    ):
        raise ValueError(
            "episode_start_idle_seconds must be a finite non-negative value"
        )
    if config.episode_start_idle_curriculum_steps < 0:
        raise ValueError("episode_start_idle_curriculum_steps must be non-negative")
    if (
        not math.isfinite(config.episode_start_idle_curriculum_start_seconds)
        or config.episode_start_idle_curriculum_start_seconds < 0.0
        or config.episode_start_idle_curriculum_start_seconds
        > config.episode_start_idle_seconds
    ):
        raise ValueError(
            "episode_start_idle_curriculum_start_seconds must be finite "
            "and between zero and episode_start_idle_seconds"
        )
    if (
        not math.isfinite(config.episode_start_handoff_seconds)
        or config.episode_start_handoff_seconds < 0.0
    ):
        raise ValueError(
            "episode_start_handoff_seconds must be a finite non-negative value"
        )
    _validate_fixed_throttle(
        config.episode_start_idle_throttle,
        "episode start idle",
    )
