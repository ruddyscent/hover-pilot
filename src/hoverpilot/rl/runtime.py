"""Environment construction, action mapping, device, and episode helpers."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch

from hoverpilot.envs import (
    AILERON_HOVER_TASK,
    AILERON_THROTTLE_HOVER_TASK,
    ELEVATOR_HOVER_TASK,
    ELEVATOR_THROTTLE_HOVER_TASK,
    RUDDER_HOVER_TASK,
    RUDDER_THROTTLE_HOVER_TASK,
    STANDARD_HOVER_TASK,
    THROTTLE_HOVER_TASK,
    HoverPilotHoverEnv,
)
from hoverpilot.rflink.client import RFLinkClient
from hoverpilot.training.hover import RewardConfig
from hoverpilot.utils.logger import format_debug_state

from .config import PPOConfig, PPOPlayConfig
from .constants import (
    CONNECTION_EPISODE_CONTROL_MODES,
    CONTROL_MODE_AILERON,
    CONTROL_MODE_AILERON_THROTTLE,
    CONTROL_MODE_ALL,
    CONTROL_MODE_ELEVATOR,
    CONTROL_MODE_ELEVATOR_THROTTLE,
    CONTROL_MODE_RUDDER,
    CONTROL_MODE_RUDDER_THROTTLE,
    CONTROL_MODE_THROTTLE,
    DEFAULT_ELEVATOR_TIMESTEPS,
    DEFAULT_EVAL_EPISODES,
    DEFAULT_LEARNING_RATE,
    DEFAULT_TIMESTEPS,
    WAITING_LOG_INTERVAL_S,
    _AILERON_PPO_INITIAL_TRIM,
    _THROTTLE_PPO_INITIAL_TRIM,
)

def _build_hover_env(
    config: Union[PPOConfig, PPOPlayConfig],
    control_mode: str,
    reward_config: Optional[RewardConfig] = None,
) -> HoverPilotHoverEnv:
    return HoverPilotHoverEnv(
        host=config.host,
        port=config.port,
        reward_config=reward_config,
        max_episode_steps=config.max_episode_steps,
        sleep_interval_s=config.sleep_interval_s,
        task_profile=(
            {
                CONTROL_MODE_ALL: STANDARD_HOVER_TASK,
                CONTROL_MODE_AILERON: AILERON_HOVER_TASK,
                CONTROL_MODE_ELEVATOR: ELEVATOR_HOVER_TASK,
                CONTROL_MODE_RUDDER: RUDDER_HOVER_TASK,
                CONTROL_MODE_THROTTLE: THROTTLE_HOVER_TASK,
                CONTROL_MODE_ELEVATOR_THROTTLE: (
                    ELEVATOR_THROTTLE_HOVER_TASK
                ),
                CONTROL_MODE_AILERON_THROTTLE: (
                    AILERON_THROTTLE_HOVER_TASK
                ),
                CONTROL_MODE_RUDDER_THROTTLE: (
                    RUDDER_THROTTLE_HOVER_TASK
                ),
            }[control_mode]
        ),
        start_body_rate_threshold_deg_s=(
            180.0
            if control_mode in CONNECTION_EPISODE_CONTROL_MODES
            else 60.0
        ),
        start_inclination_tolerance_deg=(
            5.0
            if control_mode in {
                CONTROL_MODE_RUDDER,
                CONTROL_MODE_THROTTLE,
                CONTROL_MODE_ELEVATOR_THROTTLE,
                CONTROL_MODE_AILERON_THROTTLE,
                CONTROL_MODE_RUDDER_THROTTLE,
            }
            else 0.5
        ),
        client_factory=lambda: RFLinkClient(
            config.host,
            config.port,
            socket_timeout_s=config.rflink_socket_timeout_s,
            request_attempts=config.rflink_request_attempts,
            retry_backoff_s=config.rflink_retry_backoff_s,
        ),
    )


def _validate_rflink_settings(
    config: Union[PPOConfig, PPOPlayConfig],
) -> None:
    if config.rflink_socket_timeout_s <= 0.0:
        raise ValueError("rflink_socket_timeout_s must be greater than zero")
    if config.rflink_request_attempts < 1:
        raise ValueError("rflink_request_attempts must be at least 1")
    if config.rflink_retry_backoff_s < 0.0:
        raise ValueError("rflink_retry_backoff_s must be non-negative")


def _resolve_training_defaults(config: PPOConfig) -> PPOConfig:
    elevator_mode = config.control_mode in {
        CONTROL_MODE_ELEVATOR,
        CONTROL_MODE_ELEVATOR_THROTTLE,
    }
    return replace(
        config,
        timesteps=(
            config.timesteps
            if config.timesteps is not None
            else (
                DEFAULT_ELEVATOR_TIMESTEPS
                if elevator_mode
                else DEFAULT_TIMESTEPS
            )
        ),
        learning_rate=(
            config.learning_rate
            if config.learning_rate is not None
            else DEFAULT_LEARNING_RATE
        ),
        eval_episodes=(
            config.eval_episodes
            if config.eval_episodes is not None
            else DEFAULT_EVAL_EPISODES
        ),
    )

def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but PyTorch cannot access a CUDA device.")
    if requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise ValueError("MPS was requested, but PyTorch cannot access an MPS device.")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device {requested!r}; choose auto, cpu, cuda, or mps.")
    return torch.device(requested)


def _policy_action_space(
    control_mode: str,
    env_action_space: gym.spaces.Box,
) -> gym.spaces.Box:
    if control_mode in {
        CONTROL_MODE_ELEVATOR_THROTTLE,
        CONTROL_MODE_AILERON_THROTTLE,
        CONTROL_MODE_RUDDER_THROTTLE,
    }:
        return gym.spaces.Box(
            low=np.asarray([-1.0, 0.0], dtype=np.float32),
            high=np.asarray([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
    if control_mode == CONTROL_MODE_THROTTLE:
        return gym.spaces.Box(
            low=np.asarray([0.0], dtype=np.float32),
            high=np.asarray([1.0], dtype=np.float32),
            dtype=np.float32,
        )
    if control_mode in {
        CONTROL_MODE_AILERON,
        CONTROL_MODE_ELEVATOR,
        CONTROL_MODE_RUDDER,
    }:
        return gym.spaces.Box(
            low=np.asarray([-1.0], dtype=np.float32),
            high=np.asarray([1.0], dtype=np.float32),
            dtype=np.float32,
        )
    return env_action_space


def _expand_policy_action(
    policy_action: np.ndarray,
    policy_action_space: gym.spaces.Box,
    control_mode: str,
    elevator_fixed_throttle: float,
) -> np.ndarray:
    action = np.asarray(policy_action, dtype=np.float32).reshape(-1)
    if action.shape != policy_action_space.shape:
        raise ValueError(
            f"policy action must have shape {policy_action_space.shape}, "
            f"got {action.shape}"
        )
    action = np.clip(
        action,
        policy_action_space.low,
        policy_action_space.high,
    )
    if control_mode == CONTROL_MODE_ELEVATOR_THROTTLE:
        return np.asarray(
            [0.0, action[0], action[1], 0.0],
            dtype=np.float32,
        )
    if control_mode == CONTROL_MODE_AILERON_THROTTLE:
        return np.asarray(
            [action[0], 0.0, action[1], 0.0],
            dtype=np.float32,
        )
    if control_mode == CONTROL_MODE_RUDDER_THROTTLE:
        return np.asarray(
            [0.0, 0.0, action[1], action[0]],
            dtype=np.float32,
        )
    if control_mode == CONTROL_MODE_AILERON:
        return np.asarray(
            [action[0], 0.0, elevator_fixed_throttle, 0.0],
            dtype=np.float32,
        )
    if control_mode == CONTROL_MODE_ELEVATOR:
        return np.asarray(
            [0.0, action[0], elevator_fixed_throttle, 0.0],
            dtype=np.float32,
        )
    if control_mode == CONTROL_MODE_RUDDER:
        return np.asarray(
            [0.0, 0.0, elevator_fixed_throttle, action[0]],
            dtype=np.float32,
        )
    if control_mode == CONTROL_MODE_THROTTLE:
        return np.asarray(
            [0.0, 0.0, action[0], 0.0],
            dtype=np.float32,
        )
    return action


def _initial_env_action(
    control_mode: str,
    elevator_fixed_throttle: float,
    default_action: Tuple[float, float, float, float],
) -> np.ndarray:
    if control_mode == CONTROL_MODE_ALL:
        return np.asarray(
            [
                _AILERON_PPO_INITIAL_TRIM,
                0.0,
                _THROTTLE_PPO_INITIAL_TRIM,
                0.0,
            ],
            dtype=np.float32,
        )
    if control_mode in {
        CONTROL_MODE_THROTTLE,
        CONTROL_MODE_ELEVATOR_THROTTLE,
        CONTROL_MODE_RUDDER_THROTTLE,
    }:
        return np.asarray(
            [0.0, 0.0, _THROTTLE_PPO_INITIAL_TRIM, 0.0],
            dtype=np.float32,
        )
    if control_mode == CONTROL_MODE_AILERON_THROTTLE:
        return np.asarray(
            [
                _AILERON_PPO_INITIAL_TRIM,
                0.0,
                _THROTTLE_PPO_INITIAL_TRIM,
                0.0,
            ],
            dtype=np.float32,
        )
    if control_mode in {
        CONTROL_MODE_AILERON,
        CONTROL_MODE_ELEVATOR,
        CONTROL_MODE_RUDDER,
    }:
        return np.asarray(
            [0.0, 0.0, elevator_fixed_throttle, 0.0],
            dtype=np.float32,
        )
    return np.asarray(default_action, dtype=np.float32)

def reset_env_with_wait(
    env: gym.Env,
    *,
    action: Optional[Union[np.ndarray, list, tuple]] = None,
    initial_action: Optional[Union[np.ndarray, list, tuple]] = None,
    require_reset_boundary: bool = False,
):
    if getattr(env, "_waiting_for_reset", False):
        poll_wait = getattr(env, "poll_wait_for_next_episode", None)
        if not callable(poll_wait):
            raise RuntimeError("environment reports waiting-for-reset but does not expose poll_wait_for_next_episode()")
        return _wait_for_episode_start(env, poll_wait=poll_wait, action=action)

    try:
        reset_options = None
        if initial_action is not None or require_reset_boundary:
            reset_options = {}
            if initial_action is not None:
                reset_options["initial_action"] = initial_action
            if require_reset_boundary:
                reset_options["require_reset_boundary"] = True
        return env.reset(options=reset_options)
    except TimeoutError as exc:
        poll_wait = getattr(env, "poll_wait_for_next_episode", None)
        if not callable(poll_wait):
            raise

        print(f"waiting for trainer reset before episode | {exc}")
        return _wait_for_episode_start(env, poll_wait=poll_wait, action=action)


def continue_env_after_truncation(env: gym.Env):
    continue_segment = getattr(env, "continue_after_truncation", None)
    if not callable(continue_segment):
        raise RuntimeError(
            "environment truncated a live episode but does not expose "
            "continue_after_truncation()"
        )
    return continue_segment()


def _wait_for_episode_start(
    env: gym.Env,
    *,
    poll_wait,
    action: Optional[Union[np.ndarray, list, tuple]],
):
    del env
    last_wait_log_at = 0.0
    while True:
        started, observation, info = poll_wait(action=action)
        if started:
            return observation, info
        now = time.monotonic()
        if now - last_wait_log_at >= WAITING_LOG_INTERVAL_S:
            print(f"waiting for trainer reset | {format_debug_state(info.get('debug_state'))}")
            last_wait_log_at = now
