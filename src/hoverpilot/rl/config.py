"""Configuration and checkpoint data structures for PPO workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Tuple

import torch

from hoverpilot.config import HOST, PORT
from hoverpilot.training.hover import RewardConfig

from .constants import (
    CONTROL_MODE_ALL,
    DEFAULT_INITIAL_ACTION,
    DEFAULT_WAIT_ACTION,
    POLICY_PRESET_NONE,
)

@dataclass
class PPOConfig:
    host: str = HOST
    port: int = PORT
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    max_episode_steps: Optional[int] = 300
    sleep_interval_s: float = 0.0

    timesteps: Optional[int] = None
    n_steps: int = 1024
    batch_size: int = 64
    epochs: int = 5
    learning_rate: Optional[float] = None
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    target_kl: Optional[float] = 0.02
    reward_scale: float = 0.1
    value_coef: float = 0.5
    entropy_coef: Optional[float] = None
    policy_initial_std: Optional[float] = None
    max_grad_norm: float = 0.5
    seed: Optional[int] = None
    save_path: str = "ppo_hoverpilot.pt"
    resume_from: Optional[str] = None
    eval_episodes: Optional[int] = None
    log_interval: int = 1
    telemetry_log_interval_steps: int = 25
    initial_action: Tuple[float, float, float, float] = DEFAULT_INITIAL_ACTION
    wait_action: Tuple[float, float, float, float] = DEFAULT_WAIT_ACTION
    tensorboard_log_dir: Optional[str] = "runs/hoverpilot-ppo"
    device: str = "auto"
    control_mode: str = CONTROL_MODE_ALL
    policy_preset: str = POLICY_PRESET_NONE
    elevator_fixed_throttle: float = 0.55
    episode_start_idle_seconds: float = 0.0
    episode_start_idle_throttle: float = 0.66
    episode_start_idle_curriculum_steps: int = 0
    episode_start_idle_curriculum_start_seconds: float = 0.0
    episode_start_handoff_seconds: float = 0.1
    rflink_socket_timeout_s: float = 3.0
    rflink_request_attempts: int = 4
    rflink_retry_backoff_s: float = 0.1
    checkpoint_interval_steps: int = 1024


@dataclass
class PPOPlayConfig:
    checkpoint_path: str
    host: str = HOST
    port: int = PORT
    max_episode_steps: Optional[int] = None
    sleep_interval_s: float = 0.0
    device: str = "auto"
    episodes: int = 0
    log_interval_steps: int = 25
    initial_action: Tuple[float, float, float, float] = DEFAULT_INITIAL_ACTION
    wait_action: Tuple[float, float, float, float] = DEFAULT_WAIT_ACTION
    rflink_socket_timeout_s: float = 3.0
    rflink_request_attempts: int = 4
    rflink_retry_backoff_s: float = 0.1


@dataclass(frozen=True)
class PPOCheckpoint:
    model_state_dict: Mapping[str, torch.Tensor]
    control_mode: str
    policy_preset: str
    elevator_fixed_throttle: float
    observation_config: Mapping[str, float]
