"""Versioned PPO checkpoint serialization and validation."""

from __future__ import annotations

import math
import os
from dataclasses import replace
from typing import Dict, Mapping

import torch

from hoverpilot.training.hover import RewardConfig

from .config import PPOCheckpoint
from .constants import (
    CONTROL_MODES,
    CONTROL_MODE_AILERON,
    CONTROL_MODE_AILERON_THROTTLE,
    CONTROL_MODE_ALL,
    CONTROL_MODE_ELEVATOR,
    CONTROL_MODE_RUDDER,
    CONTROL_MODE_RUDDER_THROTTLE,
    CONTROL_MODE_THROTTLE,
    POLICY_PRESET_ELEVATOR_PD,
    POLICY_PRESETS,
    PPO_CHECKPOINT_FORMAT,
    PPO_CHECKPOINT_SUPPORTED_VERSIONS,
    PPO_CHECKPOINT_VERSION,
    _AILERON_OBSERVATION_CONFIG_FIELDS,
    _AILERON_THROTTLE_OBSERVATION_CONFIG_FIELDS,
    _ELEVATOR_OBSERVATION_CONFIG_FIELDS,
    _RUDDER_OBSERVATION_CONFIG_FIELDS,
    _RUDDER_RECOVERY_CONFIG_FIELDS,
    _RUDDER_THROTTLE_OBSERVATION_CONFIG_FIELDS,
    _THROTTLE_OBSERVATION_CONFIG_FIELDS,
)
from .models import ActorCritic

def _load_checkpoint_mapping(checkpoint_path: str) -> Mapping[str, object]:
    resolved_path = os.path.abspath(os.path.expanduser(checkpoint_path))
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"PPO checkpoint does not exist: {resolved_path}")

    try:
        checkpoint = torch.load(resolved_path, map_location="cpu", weights_only=True)
    except TypeError:
        # PyTorch 1.12 does not expose weights_only. HoverPilot's RL extra still
        # supports that release on older Python installations.
        checkpoint = torch.load(resolved_path, map_location="cpu")

    if not isinstance(checkpoint, Mapping):
        raise ValueError(
            f"PPO checkpoint must contain a state dictionary, got {type(checkpoint).__name__}"
        )
    return checkpoint


def _validate_policy_state_dict(
    state_dict: object,
) -> Mapping[str, torch.Tensor]:
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("PPO checkpoint contains an empty or invalid model state dictionary")
    if not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state_dict.items()
    ):
        raise ValueError("PPO checkpoint model state dictionary must map string names to tensors")
    return state_dict


def _validate_policy_preset(preset: object, control_mode: str) -> str:
    if preset not in POLICY_PRESETS:
        raise ValueError(
            f"Unsupported policy preset {preset!r}; choose one of {POLICY_PRESETS}."
        )
    if (
        preset == POLICY_PRESET_ELEVATOR_PD
        and control_mode != CONTROL_MODE_ELEVATOR
    ):
        raise ValueError(
            "The 'elevator-pd' policy preset requires elevator control mode"
        )
    return str(preset)


def _validate_fixed_throttle(value: object, source: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{source} elevator_fixed_throttle must be in [0, 1]")
    return float(value)


def _observation_config_fields(control_mode: str) -> tuple[str, ...]:
    if control_mode == CONTROL_MODE_ALL:
        return tuple(
            dict.fromkeys(
                (
                    *_AILERON_OBSERVATION_CONFIG_FIELDS,
                    *_ELEVATOR_OBSERVATION_CONFIG_FIELDS,
                    *_THROTTLE_OBSERVATION_CONFIG_FIELDS,
                    *_RUDDER_OBSERVATION_CONFIG_FIELDS,
                    *_RUDDER_RECOVERY_CONFIG_FIELDS,
                )
            )
        )
    if control_mode == CONTROL_MODE_RUDDER_THROTTLE:
        return _RUDDER_THROTTLE_OBSERVATION_CONFIG_FIELDS
    if control_mode == CONTROL_MODE_AILERON_THROTTLE:
        return _AILERON_THROTTLE_OBSERVATION_CONFIG_FIELDS
    if control_mode == CONTROL_MODE_AILERON:
        return _AILERON_OBSERVATION_CONFIG_FIELDS
    if control_mode == CONTROL_MODE_RUDDER:
        return _RUDDER_OBSERVATION_CONFIG_FIELDS
    if control_mode == CONTROL_MODE_THROTTLE:
        return _THROTTLE_OBSERVATION_CONFIG_FIELDS
    return _ELEVATOR_OBSERVATION_CONFIG_FIELDS


def _observation_config_from_reward_config(
    reward_config: RewardConfig,
    control_mode: str,
) -> Dict[str, float]:
    fields = _observation_config_fields(control_mode)
    return {
        name: float(getattr(reward_config, name))
        for name in fields
    }


def _validate_observation_config(
    value: object,
    control_mode: str,
) -> Dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError("PPO checkpoint observation_config must be a mapping")
    fields = _observation_config_fields(control_mode)
    expected_fields = set(fields)
    if set(value) != expected_fields:
        raise ValueError(
            "PPO checkpoint observation_config fields do not match the "
            f"current {control_mode} observation"
        )
    observation_config: Dict[str, float] = {}
    for name in fields:
        raw_value = value[name]
        if (
            isinstance(raw_value, bool)
            or not isinstance(raw_value, (int, float))
        ):
            raise ValueError(
                f"PPO checkpoint observation_config {name} must be numeric"
            )
        observation_config[name] = float(raw_value)
    try:
        RewardConfig(**observation_config)
    except ValueError as exc:
        raise ValueError(
            f"PPO checkpoint observation_config is invalid: {exc}"
        ) from exc
    return observation_config


def _apply_observation_config(
    reward_config: RewardConfig,
    observation_config: Mapping[str, float],
) -> RewardConfig:
    return replace(reward_config, **dict(observation_config))


def load_policy_checkpoint(checkpoint_path: str) -> PPOCheckpoint:
    checkpoint = _load_checkpoint_mapping(checkpoint_path)
    checkpoint_format = checkpoint.get("checkpoint_format")
    format_version = checkpoint.get("format_version")
    if (
        checkpoint_format != PPO_CHECKPOINT_FORMAT
        or format_version not in PPO_CHECKPOINT_SUPPORTED_VERSIONS
    ):
        raise ValueError(
            "Unsupported PPO checkpoint "
            f"format={checkpoint_format!r} version={format_version!r}"
        )

    control_mode = checkpoint.get("control_mode")
    if control_mode not in CONTROL_MODES:
        raise ValueError(
            f"PPO checkpoint has unsupported control_mode={control_mode!r}"
        )
    training_step = checkpoint.get("training_step", 0)
    if isinstance(training_step, bool) or not isinstance(training_step, int) or training_step < 0:
        raise ValueError("PPO checkpoint training_step must be a non-negative integer")
    best_mean_reward = checkpoint.get("best_mean_reward")
    if best_mean_reward is not None and (
        isinstance(best_mean_reward, bool)
        or not isinstance(best_mean_reward, (int, float))
        or not math.isfinite(best_mean_reward)
    ):
        raise ValueError("PPO checkpoint best_mean_reward must be finite or null")

    return PPOCheckpoint(
        model_state_dict=_validate_policy_state_dict(
            checkpoint.get("model_state_dict")
        ),
        control_mode=control_mode,
        policy_preset=_validate_policy_preset(
            checkpoint.get("policy_preset"),
            control_mode,
        ),
        elevator_fixed_throttle=_validate_fixed_throttle(
            checkpoint.get("elevator_fixed_throttle"),
            "PPO checkpoint",
        ),
        observation_config=_validate_observation_config(
            checkpoint.get("observation_config"),
            str(control_mode),
        ),
        experiment_metadata=(
            dict(checkpoint.get("experiment_metadata", {}))
            if isinstance(checkpoint.get("experiment_metadata", {}), Mapping)
            else {}
        ),
        format_version=int(format_version),
        optimizer_state_dict=(
            checkpoint.get("optimizer_state_dict")
            if isinstance(checkpoint.get("optimizer_state_dict"), Mapping)
            else None
        ),
        scheduler_state_dict=(
            checkpoint.get("scheduler_state_dict")
            if isinstance(checkpoint.get("scheduler_state_dict"), Mapping)
            else None
        ),
        training_step=training_step,
        rng_state=(
            dict(checkpoint.get("rng_state", {}))
            if isinstance(checkpoint.get("rng_state", {}), Mapping)
            else {}
        ),
        evaluation_history=tuple(
            item
            for item in checkpoint.get("evaluation_history", ())
            if isinstance(item, Mapping)
        ),
        best_mean_reward=(
            float(best_mean_reward)
            if isinstance(best_mean_reward, (int, float))
            else None
        ),
        environment_config=(
            dict(checkpoint.get("environment_config", {}))
            if isinstance(checkpoint.get("environment_config", {}), Mapping)
            else {}
        ),
        reward_config=(
            dict(checkpoint.get("reward_config", {}))
            if isinstance(checkpoint.get("reward_config", {}), Mapping)
            else {}
        ),
    )


def build_policy_checkpoint(
    model: ActorCritic,
    *,
    control_mode: str,
    elevator_fixed_throttle: float,
    reward_config: RewardConfig,
    experiment_metadata: Mapping[str, object] | None = None,
    optimizer_state_dict: Mapping[str, object] | None = None,
    scheduler_state_dict: Mapping[str, object] | None = None,
    training_step: int = 0,
    rng_state: Mapping[str, object] | None = None,
    evaluation_history: tuple[Mapping[str, object], ...] = (),
    best_mean_reward: float | None = None,
    environment_config: Mapping[str, object] | None = None,
    full_reward_config: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    """Build the portable, versioned representation of a PPO policy."""

    if control_mode not in CONTROL_MODES:
        raise ValueError(
            f"Unsupported control mode {control_mode!r}; choose one of {CONTROL_MODES}."
        )
    policy_preset = _validate_policy_preset(
        model.policy_preset,
        control_mode,
    )
    fixed_throttle = _validate_fixed_throttle(
        elevator_fixed_throttle,
        "PPO checkpoint",
    )
    checkpoint: Dict[str, object] = {
        "checkpoint_format": PPO_CHECKPOINT_FORMAT,
        "format_version": PPO_CHECKPOINT_VERSION,
        "model_state_dict": model.state_dict(),
        "control_mode": control_mode,
        "policy_preset": policy_preset,
        "elevator_fixed_throttle": fixed_throttle,
        "observation_config": _observation_config_from_reward_config(
            reward_config,
            control_mode,
        ),
        "experiment_metadata": dict(experiment_metadata or {}),
        "optimizer_state_dict": optimizer_state_dict,
        "scheduler_state_dict": scheduler_state_dict,
        "training_step": int(training_step),
        "rng_state": dict(rng_state or {}),
        "evaluation_history": list(evaluation_history),
        "best_mean_reward": best_mean_reward,
        "environment_config": dict(environment_config or {}),
        "reward_config": dict(full_reward_config or {}),
    }
    return checkpoint
