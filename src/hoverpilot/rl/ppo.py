from __future__ import annotations

import argparse
import math
import os
import random
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Dict, List, Mapping, Optional, Tuple, Union

import gymnasium as gym
import numpy as np

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.distributions import Normal
except ImportError as exc:
    raise ImportError("PyTorch is required to use PPO training. Install it with `pip install torch`.") from exc

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from hoverpilot.config import HOST, PORT
from hoverpilot.envs import (
    ELEVATOR_HOVER_TASK,
    STANDARD_HOVER_TASK,
    HoverPilotHoverEnv,
    elevator_features_to_observation,
)
from hoverpilot.rl.elevator_diagnostics import diagnose_elevator_response
from hoverpilot.rflink.client import RFLinkClient
from hoverpilot.training.hover import (
    ElevatorHoverFeatures,
    RewardConfig,
    angular_error_deg,
)
from hoverpilot.utils.logger import format_debug_state


WAITING_LOG_INTERVAL_S = 0.75
DEFAULT_WAIT_ACTION = (0.0, 0.0, 0.0, 0.0)
DEFAULT_INITIAL_ACTION = (0.0, 0.0, 0.55, 0.0)
CONTROL_MODE_ALL = "all"
CONTROL_MODE_ELEVATOR = "elevator"
CONTROL_MODES = (CONTROL_MODE_ALL, CONTROL_MODE_ELEVATOR)
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
DEFAULT_TIMESTEPS = 50_000
DEFAULT_ELEVATOR_TIMESTEPS = 300_000
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_ELEVATOR_LEARNING_RATE = 1e-4
DEFAULT_EVAL_EPISODES = 3
DEFAULT_ELEVATOR_EVAL_EPISODES = 10
DEFAULT_ELEVATOR_ENTROPY_COEF = 0.0001
DEFAULT_ELEVATOR_POLICY_STD = 0.08


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
    rflink_socket_timeout_s: float = 3.0
    rflink_request_attempts: int = 4
    rflink_retry_backoff_s: float = 0.1
    checkpoint_interval_steps: int = 1024


@dataclass(frozen=True)
class PPOCheckpoint:
    model_state_dict: Mapping[str, torch.Tensor]
    control_mode: str
    policy_preset: str
    elevator_fixed_throttle: float
    observation_config: Mapping[str, float]


def _build_hover_env(
    config: PPOConfig,
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
            ELEVATOR_HOVER_TASK
            if control_mode == CONTROL_MODE_ELEVATOR
            else STANDARD_HOVER_TASK
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
    config: PPOConfig,
) -> None:
    if config.rflink_socket_timeout_s <= 0.0:
        raise ValueError("rflink_socket_timeout_s must be greater than zero")
    if config.rflink_request_attempts < 1:
        raise ValueError("rflink_request_attempts must be at least 1")
    if config.rflink_retry_backoff_s < 0.0:
        raise ValueError("rflink_retry_backoff_s must be non-negative")


def _resolve_training_defaults(config: PPOConfig) -> PPOConfig:
    elevator_mode = config.control_mode == CONTROL_MODE_ELEVATOR
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
            else (
                DEFAULT_ELEVATOR_LEARNING_RATE
                if elevator_mode
                else DEFAULT_LEARNING_RATE
            )
        ),
        eval_episodes=(
            config.eval_episodes
            if config.eval_episodes is not None
            else (
                DEFAULT_ELEVATOR_EVAL_EPISODES
                if elevator_mode
                else DEFAULT_EVAL_EPISODES
            )
        ),
    )


class ActorCritic(nn.Module):
    _SQUASH_EPSILON = 1.0e-6

    def __init__(
        self,
        observation_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        *,
        initial_policy_std: float = 0.25,
        policy_preset: str = POLICY_PRESET_NONE,
    ):
        super().__init__()
        if initial_policy_std <= 0.0:
            raise ValueError("initial_policy_std must be greater than zero")
        action_low_tensor = torch.as_tensor(action_low, dtype=torch.float32)
        action_high_tensor = torch.as_tensor(action_high, dtype=torch.float32)
        if action_low_tensor.shape != action_high_tensor.shape:
            raise ValueError("action bounds must have matching shapes")
        action_dim = int(action_low_tensor.numel())
        if policy_preset == POLICY_PRESET_NONE:
            prior_weight_tensor = torch.zeros(
                (action_dim, observation_dim),
                dtype=torch.float32,
            )
            prior_limit = 0.0
            residual_limit = 0.0
        elif (
            policy_preset == POLICY_PRESET_ELEVATOR_PD
            and observation_dim == 6
            and action_dim == 1
        ):
            prior_weight_tensor = torch.as_tensor(
                _ELEVATOR_PD_PRIOR_WEIGHT.copy(),
                dtype=torch.float32,
            )
            prior_limit = _ELEVATOR_PD_PRIOR_LIMIT
            residual_limit = _ELEVATOR_PD_RESIDUAL_LIMIT
        else:
            raise ValueError(
                f"Policy preset {policy_preset!r} is not valid for "
                f"{observation_dim} observations and {action_dim} actions"
            )
        self.policy_preset = policy_preset
        self.enforce_elevator_symmetry = (
            observation_dim == 6 and action_dim == 1
        )
        mirror_sign = (
            [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0]
            if self.enforce_elevator_symmetry
            else [1.0] * observation_dim
        )
        self.register_buffer(
            "_observation_mirror_sign",
            torch.tensor(mirror_sign, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "policy_prior_weight",
            prior_weight_tensor,
        )
        self.register_buffer(
            "_policy_prior_limit",
            torch.tensor(prior_limit, dtype=torch.float32),
        )
        self.register_buffer(
            "_policy_residual_limit",
            torch.tensor(residual_limit, dtype=torch.float32),
        )
        self.register_buffer("action_scale", (action_high_tensor - action_low_tensor) / 2.0)
        self.register_buffer("action_bias", (action_high_tensor + action_low_tensor) / 2.0)
        use_linear_elevator_policy = (
            self.enforce_elevator_symmetry
            and policy_preset == POLICY_PRESET_NONE
        )
        if use_linear_elevator_policy:
            initial_gain = torch.as_tensor(
                _ELEVATOR_PPO_INITIAL_GAIN,
                dtype=torch.float32,
            )
            self.elevator_policy_raw_gain = nn.Parameter(
                torch.log(torch.expm1(initial_gain))
            )
        else:
            self.register_parameter("elevator_policy_raw_gain", None)
        hidden_sizes = [128, 128]
        layers = []
        input_size = observation_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden))
            layers.append(nn.ReLU(inplace=True))
            input_size = hidden
        self.shared = nn.Sequential(*layers)
        self.policy_mean = (
            None
            if use_linear_elevator_policy
            else nn.Linear(hidden_sizes[-1], action_dim)
        )
        self.policy_log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        self.value_head = nn.Linear(hidden_sizes[-1], 1)
        with torch.no_grad():
            for module in self.shared:
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                    module.bias.zero_()
            if self.policy_mean is not None:
                nn.init.orthogonal_(self.policy_mean.weight, gain=0.01)
                self.policy_mean.bias.zero_()
            nn.init.orthogonal_(self.value_head.weight, gain=1.0)
            self.value_head.bias.zero_()
            self.policy_log_std.fill_(math.log(initial_policy_std))
            if action_dim >= 3:
                # Hover training needs non-zero throttle from the first step.
                normalized_throttle = 2.0 * float(DEFAULT_INITIAL_ACTION[2]) - 1.0
                assert self.policy_mean is not None
                self.policy_mean.bias[2] = math.atanh(normalized_throttle)
                self.policy_log_std[2] = math.log(0.15)

    @property
    def policy_prior_limit(self) -> Optional[float]:
        if self.policy_preset == POLICY_PRESET_NONE:
            return None
        return float(self._policy_prior_limit.item())

    @property
    def policy_residual_limit(self) -> Optional[float]:
        if self.policy_preset == POLICY_PRESET_NONE:
            return None
        return float(self._policy_residual_limit.item())

    @property
    def elevator_policy_gain(self) -> Optional[torch.Tensor]:
        if self.elevator_policy_raw_gain is None:
            return None
        return F.softplus(self.elevator_policy_raw_gain)

    def _compute_policy_mean(
        self,
        obs: torch.Tensor,
        *,
        hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        elevator_gain = self.elevator_policy_gain
        if elevator_gain is not None:
            return (
                -elevator_gain[0] * obs[..., 0:1]
                + elevator_gain[1] * obs[..., 1:2]
            )

        if hidden is None:
            hidden = self.shared(obs)
        assert self.policy_mean is not None
        residual = self.policy_mean(hidden)
        if self.enforce_elevator_symmetry:
            mirrored_hidden = self.shared(
                obs * self._observation_mirror_sign
            )
            mirrored_residual = self.policy_mean(mirrored_hidden)
            residual = 0.5 * (residual - mirrored_residual)
        if self.policy_preset == POLICY_PRESET_ELEVATOR_PD:
            residual = self._policy_residual_limit * torch.tanh(residual)
            prior = obs @ self.policy_prior_weight.T
            prior = torch.minimum(
                torch.maximum(prior, -self._policy_prior_limit),
                self._policy_prior_limit,
            )
            return residual + prior
        return residual

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.shared(obs)
        mean = self._compute_policy_mean(obs, hidden=hidden)
        value = self.value_head(hidden).squeeze(-1)
        log_std = self.policy_log_std.clamp(-5.0, 1.0).expand_as(mean)
        return mean, log_std, value

    def _squash(self, latent_action: torch.Tensor) -> torch.Tensor:
        return self.action_bias + self.action_scale * torch.tanh(latent_action)

    def _unsquash(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = (action - self.action_bias) / self.action_scale
        normalized = normalized.clamp(-1.0 + self._SQUASH_EPSILON, 1.0 - self._SQUASH_EPSILON)
        return torch.atanh(normalized), normalized

    def _squashed_log_prob(
        self,
        dist: Normal,
        latent_action: torch.Tensor,
        normalized_action: torch.Tensor,
    ) -> torch.Tensor:
        correction = torch.log(
            self.action_scale * (1.0 - normalized_action.pow(2)) + self._SQUASH_EPSILON
        )
        return (dist.log_prob(latent_action) - correction).sum(-1)

    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        return self._squash(self._compute_policy_mean(obs))

    def get_action(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, value = self(obs)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        sampled_latent_action = dist.rsample()
        normalized_action = torch.tanh(sampled_latent_action).clamp(
            -1.0 + self._SQUASH_EPSILON,
            1.0 - self._SQUASH_EPSILON,
        )
        # Reconstruct the latent value after the numerical clamp so that the
        # stored log probability and evaluate_actions() use the same action.
        latent_action = torch.atanh(normalized_action)
        action = self.action_bias + self.action_scale * normalized_action
        log_prob = self._squashed_log_prob(dist, latent_action, normalized_action)
        return action, log_prob, value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, value = self(obs)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        latent_actions, normalized_actions = self._unsquash(actions)
        log_probs = self._squashed_log_prob(dist, latent_actions, normalized_actions)
        # A transformed Normal has no simple analytic entropy. The sampled
        # negative log probability is the appropriate Monte Carlo estimate.
        entropy = -log_probs
        return log_probs, entropy, value, mean


class RolloutBuffer:
    def __init__(self, capacity: int, observation_dim: int, action_dim: int, device: torch.device):
        self.device = device
        self.observations = torch.zeros((capacity, observation_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.values = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.advantages = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.returns = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.index = 0
        self.capacity = capacity

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ):
        if self.index >= self.capacity:
            raise IndexError("RolloutBuffer is full")
        self.observations[self.index].copy_(torch.as_tensor(observation, dtype=torch.float32, device=self.device))
        self.actions[self.index].copy_(torch.as_tensor(action, dtype=torch.float32, device=self.device))
        self.rewards[self.index] = reward
        self.dones[self.index] = 0.0 if done else 1.0
        self.values[self.index] = value
        self.log_probs[self.index] = log_prob
        self.index += 1

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float,
        lam: float,
    ):
        gae = 0.0
        last_value_tensor = torch.tensor(last_value, dtype=torch.float32, device=self.device)
        for step in reversed(range(self.index)):
            next_value = last_value_tensor if step == self.index - 1 else self.values[step + 1]
            delta = self.rewards[step] + gamma * next_value * self.dones[step] - self.values[step]
            gae = delta + gamma * lam * self.dones[step] * gae
            self.advantages[step] = gae
        self.returns[: self.index] = self.advantages[: self.index] + self.values[: self.index]

    def normalize_advantages(self) -> torch.Tensor:
        advantages = self.advantages[: self.index]
        normalized = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        advantages.copy_(normalized)
        return advantages

    def get_batches(self, batch_size: int):
        indices = torch.randperm(self.index, device=self.device)
        for start in range(0, self.index, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            yield (
                self.observations[batch_idx],
                self.actions[batch_idx],
                self.log_probs[batch_idx],
                self.advantages[batch_idx],
                self.returns[batch_idx],
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
    if control_mode == CONTROL_MODE_ELEVATOR:
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
    if control_mode == CONTROL_MODE_ELEVATOR:
        return np.asarray(
            [0.0, action[0], elevator_fixed_throttle, 0.0],
            dtype=np.float32,
        )
    return action


def _initial_env_action(
    control_mode: str,
    elevator_fixed_throttle: float,
    default_action: Tuple[float, float, float, float],
) -> np.ndarray:
    if control_mode == CONTROL_MODE_ELEVATOR:
        return np.asarray(
            [0.0, 0.0, elevator_fixed_throttle, 0.0],
            dtype=np.float32,
        )
    return np.asarray(default_action, dtype=np.float32)


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


def _observation_config_from_reward_config(
    reward_config: RewardConfig,
) -> Dict[str, float]:
    return {
        name: float(getattr(reward_config, name))
        for name in _ELEVATOR_OBSERVATION_CONFIG_FIELDS
    }


def _validate_observation_config(value: object) -> Dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError("PPO checkpoint observation_config must be a mapping")
    expected_fields = set(_ELEVATOR_OBSERVATION_CONFIG_FIELDS)
    if set(value) != expected_fields:
        raise ValueError(
            "PPO checkpoint observation_config fields do not match the "
            "current elevator observation"
        )
    observation_config: Dict[str, float] = {}
    for name in _ELEVATOR_OBSERVATION_CONFIG_FIELDS:
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
        or format_version != PPO_CHECKPOINT_VERSION
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
            checkpoint.get("observation_config")
        ),
    )


def build_policy_checkpoint(
    model: ActorCritic,
    *,
    control_mode: str,
    elevator_fixed_throttle: float,
    reward_config: RewardConfig,
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
            reward_config
        ),
    }
    return checkpoint


class PPOTrainer:
    def __init__(self, config: PPOConfig):
        config = _resolve_training_defaults(config)
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
        if config.telemetry_log_interval_steps < 0:
            raise ValueError("telemetry_log_interval_steps must be non-negative")
        resume_checkpoint = (
            load_policy_checkpoint(config.resume_from)
            if config.resume_from is not None
            else None
        )
        if resume_checkpoint is not None:
            config = replace(
                config,
                reward_config=_apply_observation_config(
                    config.reward_config,
                    resume_checkpoint.observation_config,
                ),
            )
            if resume_checkpoint.control_mode != config.control_mode:
                raise ValueError(
                    "Resume checkpoint uses control mode "
                    f"{resume_checkpoint.control_mode!r}, but "
                    f"{config.control_mode!r} was requested"
                )
            self.policy_preset = resume_checkpoint.policy_preset
            self.elevator_fixed_throttle = (
                resume_checkpoint.elevator_fixed_throttle
            )
        else:
            self.policy_preset = _validate_policy_preset(
                config.policy_preset,
                config.control_mode,
            )
            self.elevator_fixed_throttle = _validate_fixed_throttle(
                config.elevator_fixed_throttle,
                "PPO config",
            )
        self.config = config
        self.entropy_coef = (
            config.entropy_coef
            if config.entropy_coef is not None
            else (
                DEFAULT_ELEVATOR_ENTROPY_COEF
                if config.control_mode == CONTROL_MODE_ELEVATOR
                else 0.01
            )
        )
        self.policy_initial_std = (
            config.policy_initial_std
            if config.policy_initial_std is not None
            else (
                DEFAULT_ELEVATOR_POLICY_STD
                if config.control_mode == CONTROL_MODE_ELEVATOR
                else 0.25
            )
        )
        self.device = resolve_device(config.device)
        if config.seed is not None:
            self.seed(config.seed)
        self.env = self._build_env()
        self.policy_action_space = _policy_action_space(
            config.control_mode,
            self.env.action_space,
        )
        observation_dim = int(np.prod(self.env.observation_space.shape))
        self.model = ActorCritic(
            observation_dim,
            self.policy_action_space.low,
            self.policy_action_space.high,
            initial_policy_std=self.policy_initial_std,
            policy_preset=self.policy_preset,
        ).to(self.device)
        if resume_checkpoint is not None:
            self._load_resume_checkpoint(resume_checkpoint, config.resume_from)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.writer = self._build_writer()

    def _load_resume_checkpoint(
        self,
        checkpoint: PPOCheckpoint,
        checkpoint_path: str,
    ):
        try:
            self.model.load_state_dict(checkpoint.model_state_dict, strict=True)
        except RuntimeError as exc:
            raise ValueError(f"Resume checkpoint is incompatible: {exc}") from exc
        print(
            f"[PPO] Resumed policy weights from {checkpoint_path} "
            f"policy_preset={self.policy_preset}"
        )

    def _build_writer(self):
        if self.config.tensorboard_log_dir is None:
            return None
        if SummaryWriter is None:
            raise ImportError(
                "TensorBoard logging requires `tensorboard`. Install the RL extra with "
                "`uv sync --extra rl`."
            )
        return SummaryWriter(log_dir=self.config.tensorboard_log_dir)

    def _wait_action(self) -> np.ndarray:
        return np.asarray(self.config.wait_action, dtype=np.float32)

    def _initial_action(self) -> np.ndarray:
        return _initial_env_action(
            self.config.control_mode,
            self.elevator_fixed_throttle,
            self.config.initial_action,
        )

    def _policy_action_labels(self) -> tuple[str, ...]:
        if self.config.control_mode == CONTROL_MODE_ELEVATOR:
            return ("elevator",)
        return ("aileron", "elevator", "throttle", "rudder")

    def _to_env_action(self, policy_action: np.ndarray) -> np.ndarray:
        return _expand_policy_action(
            policy_action,
            self.policy_action_space,
            self.config.control_mode,
            self.elevator_fixed_throttle,
        )

    def _format_action_stats(self, actions: np.ndarray) -> str:
        short_labels = {
            "aileron": "ail",
            "elevator": "ele",
            "throttle": "thr",
            "rudder": "rud",
        }
        parts = []
        for index, action_label in enumerate(self._policy_action_labels()):
            column = actions[:, index]
            label = short_labels[action_label]
            parts.append(f"{label}=mean:{column.mean():+.3f} std:{column.std():.3f}")
        return " ".join(parts)

    def _write_scalar(self, tag: str, value: float, step: int):
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def _write_action_metrics(self, actions: np.ndarray, step: int):
        labels = self._policy_action_labels()
        low = self.policy_action_space.low
        high = self.policy_action_space.high
        normalized = 2.0 * (actions - low) / (high - low) - 1.0
        for index, label in enumerate(labels):
            column = actions[:, index]
            self._write_scalar(f"train/action/{label}_mean", float(column.mean()), step)
            self._write_scalar(f"train/action/{label}_std", float(column.std()), step)
            self._write_scalar(
                f"train/action/{label}_saturation_fraction",
                float(np.mean(np.abs(normalized[:, index]) >= 0.98)),
                step,
            )
            self._write_scalar(
                f"train/action/{label}_positive_fraction",
                float(np.mean(normalized[:, index] > 0.05)),
                step,
            )
            self._write_scalar(
                f"train/action/{label}_negative_fraction",
                float(np.mean(normalized[:, index] < -0.05)),
                step,
            )

    def _write_elevator_recovery_probe(self, step: int):
        if (
            self.config.control_mode != CONTROL_MODE_ELEVATOR
            or self.env.observation_space.shape != (6,)
            or not hasattr(self.env, "reward_config")
        ):
            return

        scenarios = (
            (
                "attitude",
                ElevatorHoverFeatures(15.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                -1.0,
            ),
            (
                "pitch_rate",
                ElevatorHoverFeatures(0.0, 30.0, 0.0, 0.0, 0.0, 0.0),
                1.0,
            ),
            (
                "position",
                ElevatorHoverFeatures(0.0, 0.0, 4.0, 0.0, 0.0, 0.0),
                -1.0,
            ),
            (
                "velocity",
                ElevatorHoverFeatures(0.0, 0.0, 0.0, 5.0, 0.0, 0.0),
                -1.0,
            ),
            (
                "outward_drift",
                ElevatorHoverFeatures(0.0, 0.0, 4.0, 2.0, 0.0, 0.0),
                -1.0,
            ),
        )
        observations = []
        for _, features, _ in scenarios:
            observations.append(self._elevator_probe_observation(features))
            observations.append(
                self._elevator_probe_observation(
                    ElevatorHoverFeatures(
                        -features.inclination_error_deg,
                        -features.pitch_rate_deg_s,
                        -features.longitudinal_position_error_m,
                        -features.longitudinal_velocity_mps,
                        features.altitude_error_m,
                        features.vertical_velocity_mps,
                    )
                )
            )
        probe = torch.tensor(
            np.asarray(observations),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            actions = self.model.deterministic_action(probe)
        action_values = actions[:, 0].detach().cpu().numpy()
        symmetry_errors = []
        restoring_margins = []
        summaries = []
        for index, (name, _, positive_expected_sign) in enumerate(scenarios):
            positive_action = float(action_values[index * 2])
            negative_action = float(action_values[index * 2 + 1])
            symmetry_errors.append(abs(positive_action + negative_action))
            scenario_margins = (
                positive_action * positive_expected_sign,
                negative_action * -positive_expected_sign,
            )
            restoring_margins.extend(scenario_margins)
            summaries.append(
                (
                    f"{name}={positive_action:+.3f}/{negative_action:+.3f}"
                    f"(margin={min(scenario_margins):.3f})"
                )
            )
            self._write_scalar(
                f"train/recovery_probe/{name}_positive_action",
                positive_action,
                step,
            )
            self._write_scalar(
                f"train/recovery_probe/{name}_negative_action",
                negative_action,
                step,
            )
        symmetry_error = float(np.mean(symmetry_errors))
        minimum_restoring_margin = float(np.min(restoring_margins))
        restoring_fraction = float(
            np.mean(
                np.asarray(restoring_margins)
                >= _ELEVATOR_EFFECTIVE_RESTORING_ACTION
            )
        )
        print(
            f"[PPO] recovery probe {' '.join(summaries)} "
            f"symmetry_error={symmetry_error:.3f} "
            f"minimum_margin={minimum_restoring_margin:.3f} "
            f"effective_restoring_fraction={restoring_fraction:.2f} "
            f"threshold={_ELEVATOR_EFFECTIVE_RESTORING_ACTION:.2f}"
        )
        self._write_scalar(
            "train/recovery_probe/symmetry_error",
            symmetry_error,
            step,
        )
        self._write_scalar(
            "train/recovery_probe/minimum_restoring_margin",
            minimum_restoring_margin,
            step,
        )
        self._write_scalar(
            "train/recovery_probe/effective_restoring_fraction",
            restoring_fraction,
            step,
        )

    def _elevator_probe_observation(
        self,
        features: ElevatorHoverFeatures,
    ) -> np.ndarray:
        config = self.env.reward_config
        return elevator_features_to_observation(
            features,
            config=config,
        )

    def _write_termination_metrics(self, termination_reasons: list[str], step: int):
        counts = Counter(termination_reasons)
        total = max(1, len(termination_reasons))
        for reason, count in counts.items():
            self._write_scalar(f"train/termination/{reason}", float(count), step)
            self._write_scalar(f"train/termination_rate/{reason}", float(count) / total, step)

    def _format_reward_breakdown(self, info: Optional[Dict]) -> str:
        if not info:
            return ""
        breakdown = info.get("reward_breakdown")
        if not breakdown:
            return ""
        return (
            " "
            f"reward_terms(pos=-{breakdown.get('position_penalty', 0.0):.3f} "
            f"alt=-{breakdown.get('altitude_penalty', 0.0):.3f} "
            f"att_track=-{breakdown.get('attitude_penalty', 0.0):.3f} "
            f"rate=-{breakdown.get('angular_rate_penalty', 0.0):.3f} "
            f"vel=-{breakdown.get('velocity_penalty', 0.0):.3f} "
            f"smooth=-{breakdown.get('action_smoothness_penalty', 0.0):.3f} "
            f"boundary=-{breakdown.get('boundary_proximity_penalty', 0.0):.3f} "
            f"alive=+{breakdown.get('survival_reward', 0.0):.3f} "
            f"terminal={breakdown.get('terminal_penalty', 0.0):+.3f})"
        )

    def _log_episode_start(self, info: dict[str, object]):
        debug_state = info.get("debug_state") if isinstance(info, dict) else None
        print(
            f"[PPO] episode start reason={info.get('episode_start_reason')} "
            f"waiting={info.get('waiting_for_reset')}"
        )
        if debug_state:
            print(f"[PPO] start state {format_debug_state(debug_state)}")

    def _log_episode_end(
        self,
        *,
        episode_length: int,
        episode_reward: float,
        info: dict[str, object],
    ):
        debug_state = info.get("debug_state") if isinstance(info, dict) else None
        print(
            f"[PPO] episode end steps={episode_length} reward={episode_reward:.3f} "
            f"reason={info.get('termination_reason')}"
            f"{self._format_reward_breakdown(info)}"
        )
        if debug_state:
            print(f"[PPO] end state {format_debug_state(debug_state)}")

    def _log_control_telemetry(
        self,
        *,
        total_steps: int,
        env_action: np.ndarray,
        reward: float,
        info: Dict,
    ):
        features = info.get("elevator_hover_features", {})
        if not features:
            return
        inclination_error = float(features.get("inclination_error_deg", 0.0))
        pitch_rate = float(features.get("pitch_rate_deg_s", 0.0))
        longitudinal_error = float(
            features.get("longitudinal_position_error_m", 0.0)
        )
        longitudinal_velocity = float(
            features.get("longitudinal_velocity_mps", 0.0)
        )
        target_inclination_error = float(
            info.get("elevator_recovery_target_deg", 0.0)
        )
        inclination_tracking_error = (
            inclination_error - target_inclination_error
        )
        radial_distance = float(
            info.get("debug_state", {}).get(
                "distance_from_cylinder_axis_m",
                0.0,
            )
        )
        print(
            f"[PPO] control step={total_steps} reward={reward:+.3f} "
            f"elevator={float(env_action[1]):+.3f} "
            f"inc_error={inclination_error:+.2f}deg "
            f"target_inc={target_inclination_error:+.2f}deg "
            f"pitch_rate={pitch_rate:+.2f}deg/s "
            f"long_error={longitudinal_error:+.2f}m "
            f"long_velocity={longitudinal_velocity:+.2f}m/s "
            f"radial={radial_distance:.2f}m"
        )
        self._write_scalar(
            "train/control/elevator_action",
            float(env_action[1]),
            total_steps,
        )
        self._write_scalar("train/control/reward", reward, total_steps)
        self._write_scalar(
            "train/state/inclination_error_deg",
            inclination_error,
            total_steps,
        )
        self._write_scalar(
            "train/state/abs_inclination_error_deg",
            abs(inclination_error),
            total_steps,
        )
        self._write_scalar(
            "train/state/target_inclination_error_deg",
            target_inclination_error,
            total_steps,
        )
        self._write_scalar(
            "train/state/inclination_tracking_error_deg",
            inclination_tracking_error,
            total_steps,
        )
        self._write_scalar("train/state/pitch_rate_deg_s", pitch_rate, total_steps)
        self._write_scalar(
            "train/state/abs_pitch_rate_deg_s",
            abs(pitch_rate),
            total_steps,
        )
        self._write_scalar(
            "train/state/longitudinal_error_m",
            longitudinal_error,
            total_steps,
        )
        self._write_scalar(
            "train/state/longitudinal_velocity_mps",
            longitudinal_velocity,
            total_steps,
        )
        self._write_scalar(
            "train/state/radial_distance_m",
            radial_distance,
            total_steps,
        )

    def _log_rollout_summary(
        self,
        *,
        total_steps: int,
        rollout: RolloutBuffer,
        actions: list[np.ndarray],
        rewards: list[float],
        termination_reasons: list[str],
        elapsed_s: float,
    ):
        if not rewards:
            return
        action_array = np.asarray(actions, dtype=np.float32)
        termination_counts = Counter(termination_reasons)
        reason_summary = ", ".join(
            f"{reason}:{count}" for reason, count in sorted(termination_counts.items())
        ) or "none"
        print(
            f"[PPO] rollout steps={total_steps}/{self.config.timesteps} "
            f"samples={rollout.index} reward_mean={np.mean(rewards):+.3f} "
            f"reward_min={np.min(rewards):+.3f} reward_max={np.max(rewards):+.3f} "
            f"done_rate={sum(1 for reason in termination_reasons if reason != 'incomplete') / max(1, rollout.index):.3f} "
            f"elapsed={elapsed_s:.1f}s"
        )
        print(f"[PPO] rollout actions {self._format_action_stats(action_array)}")
        print(f"[PPO] rollout terminations {reason_summary}")

    def _log_update_summary(
        self,
        *,
        total_steps: int,
        policy_losses: list[float],
        value_losses: list[float],
        entropy_values: list[float],
        ratio_values: list[float],
        approx_kl_values: list[float],
        clip_fraction_values: list[float],
        epochs_completed: int,
        stopped_for_kl: bool,
        returns: torch.Tensor,
        advantages: torch.Tensor,
    ):
        print(
            f"[PPO] update steps={total_steps}/{self.config.timesteps} "
            f"policy_loss={np.mean(policy_losses):+.4f} "
            f"value_loss={np.mean(value_losses):+.4f} "
            f"entropy={np.mean(entropy_values):.4f} "
            f"ratio={np.mean(ratio_values):.4f} "
            f"approx_kl={np.mean(approx_kl_values):.5f} "
            f"clip_fraction={np.mean(clip_fraction_values):.3f} "
            f"epochs={epochs_completed}/{self.config.epochs} "
            f"kl_stop={stopped_for_kl} "
            f"return_mean={returns.mean().item():+.3f} "
            f"adv_mean={advantages.mean().item():+.3f} adv_std={advantages.std(unbiased=False).item():.3f}"
        )

    def _build_env(self) -> gym.Env:
        return _build_hover_env(
            self.config,
            self.config.control_mode,
            reward_config=self.config.reward_config,
        )

    def seed(self, seed: int):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if hasattr(torch, "cuda"):
            torch.cuda.manual_seed_all(seed)

    def _normalize_action(self, raw_action: np.ndarray) -> np.ndarray:
        low = self.policy_action_space.low
        high = self.policy_action_space.high
        return np.clip(raw_action, low, high)

    def _save_model(self, *, step: int, reason: str):
        save_path = os.path.abspath(self.config.save_path)
        save_directory = os.path.dirname(save_path)
        os.makedirs(save_directory, exist_ok=True)
        temporary_path = f"{save_path}.tmp-{os.getpid()}"
        try:
            torch.save(
                build_policy_checkpoint(
                    self.model,
                    control_mode=self.config.control_mode,
                    elevator_fixed_throttle=self.elevator_fixed_throttle,
                    reward_config=self.config.reward_config,
                ),
                temporary_path,
            )
            os.replace(temporary_path, save_path)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
        print(f"[PPO] Saved {reason} model at step={step} to {self.config.save_path}")
        self._write_scalar("train/checkpoint_step", float(step), step)

    def train(self):
        total_steps = 0
        last_completed_update_steps = 0
        last_saved_steps = 0
        next_checkpoint_step = self.config.checkpoint_interval_steps
        report_every = max(1, self.config.log_interval)
        training_start = time.time()
        episode_rewards = []
        episode_lengths = []
        if self.writer is not None:
            self.writer.add_text(
                "run/config",
                "\n".join(
                    [
                        f"timesteps={self.config.timesteps}",
                        f"n_steps={self.config.n_steps}",
                        f"batch_size={self.config.batch_size}",
                        f"epochs={self.config.epochs}",
                        f"learning_rate={self.config.learning_rate}",
                        f"target_kl={self.config.target_kl}",
                        f"reward_scale={self.config.reward_scale}",
                        f"reward_profile={self.env.reward_config.profile}",
                        f"task_profile={self.env.task_profile.value}",
                        f"control_mode={self.config.control_mode}",
                        f"policy_preset={self.policy_preset}",
                        f"entropy_coef={self.entropy_coef}",
                        f"policy_initial_std={self.policy_initial_std}",
                        f"elevator_fixed_throttle={self.elevator_fixed_throttle}",
                        "elevator_recovery_position_gain_deg_per_m="
                        f"{self.env.reward_config.elevator_recovery_position_gain_deg_per_m}",
                        "elevator_recovery_velocity_gain_deg_per_mps="
                        f"{self.env.reward_config.elevator_recovery_velocity_gain_deg_per_mps}",
                        "elevator_recovery_inclination_limit_deg="
                        f"{self.config.reward_config.elevator_recovery_inclination_limit_deg}",
                        f"rflink_socket_timeout_s={self.config.rflink_socket_timeout_s}",
                        f"rflink_request_attempts={self.config.rflink_request_attempts}",
                        f"rflink_retry_backoff_s={self.config.rflink_retry_backoff_s}",
                        f"checkpoint_interval_steps={self.config.checkpoint_interval_steps}",
                        f"telemetry_log_interval_steps={self.config.telemetry_log_interval_steps}",
                        f"max_episode_steps={self.config.max_episode_steps}",
                        f"seed={self.config.seed}",
                        f"resume_from={self.config.resume_from}",
                    ]
                ),
                0,
            )
        try:
            observation, info = reset_env_with_wait(
                self.env,
                action=self._wait_action(),
                initial_action=self._initial_action(),
            )
            self._log_episode_start(info)
            episode_reward = 0.0
            episode_length = 0

            while total_steps < self.config.timesteps:
                rollout = RolloutBuffer(
                    self.config.n_steps,
                    *self.env.observation_space.shape,
                    self.policy_action_space.shape[0],
                    self.device,
                )
                rollout_actions: list[np.ndarray] = []
                rollout_rewards: list[float] = []
                rollout_termination_reasons: list[str] = []
                for _ in range(self.config.n_steps):
                    obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                    action_tensor, log_prob_tensor, value_tensor = self.model.get_action(obs_tensor)
                    action = action_tensor.squeeze(0).detach().cpu().numpy()
                    executed_action = self._normalize_action(action)
                    env_action = self._to_env_action(executed_action)
                    next_obs, reward, terminated, truncated, info = self.env.step(env_action)
                    episode_boundary = bool(terminated or truncated)
                    rollout_actions.append(executed_action.copy())
                    rollout_rewards.append(float(reward))
                    rollout_termination_reasons.append(info.get("termination_reason") or ("truncated" if truncated else "incomplete"))
                    rollout.add(
                        observation,
                        executed_action,
                        reward * self.config.reward_scale,
                        bool(terminated),
                        float(value_tensor.item()),
                        float(log_prob_tensor.item()),
                    )
                    episode_reward += reward
                    episode_length += 1
                    observation = next_obs
                    total_steps += 1
                    if (
                        self.config.control_mode == CONTROL_MODE_ELEVATOR
                        and
                        self.config.telemetry_log_interval_steps > 0
                        and total_steps % self.config.telemetry_log_interval_steps == 0
                    ):
                        self._log_control_telemetry(
                            total_steps=total_steps,
                            env_action=env_action,
                            reward=float(reward),
                            info=info,
                        )
                    if episode_boundary:
                        episode_info = dict(info)
                        if truncated and not episode_info.get("termination_reason"):
                            episode_info["termination_reason"] = "truncated"
                        self._log_episode_end(
                            episode_length=episode_length,
                            episode_reward=episode_reward,
                            info=episode_info,
                        )
                        self._write_scalar("train/episode_reward", float(episode_reward), total_steps)
                        self._write_scalar("train/episode_length", float(episode_length), total_steps)
                        episode_rewards.append(episode_reward)
                        episode_lengths.append(episode_length)
                        if terminated:
                            observation, info = reset_env_with_wait(
                                self.env,
                                action=self._wait_action(),
                                initial_action=self._initial_action(),
                            )
                        else:
                            observation, info = continue_env_after_truncation(self.env)
                        self._log_episode_start(info)
                        episode_reward = 0.0
                        episode_length = 0
                    if total_steps >= self.config.timesteps:
                        break

                last_value = self.model(torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0))[2].item()
                rollout.compute_returns_and_advantages(last_value, self.config.gamma, self.config.gae_lambda)
                advantages = rollout.normalize_advantages()
                returns = rollout.returns[: rollout.index]

                self._log_rollout_summary(
                    total_steps=total_steps,
                    rollout=rollout,
                    actions=rollout_actions,
                    rewards=rollout_rewards,
                    termination_reasons=rollout_termination_reasons,
                    elapsed_s=time.time() - training_start,
                )

                action_array = np.asarray(rollout_actions, dtype=np.float32)
                self._write_scalar("train/reward_mean", float(np.mean(rollout_rewards)), total_steps)
                self._write_scalar("train/reward_min", float(np.min(rollout_rewards)), total_steps)
                self._write_scalar("train/reward_max", float(np.max(rollout_rewards)), total_steps)
                self._write_scalar(
                    "train/optimization_reward_mean",
                    float(np.mean(rollout_rewards) * self.config.reward_scale),
                    total_steps,
                )
                self._write_scalar(
                    "train/done_rate",
                    float(sum(1 for reason in rollout_termination_reasons if reason != "incomplete") / max(1, rollout.index)),
                    total_steps,
                )
                self._write_scalar("train/return_mean", float(returns.mean().item()), total_steps)
                self._write_scalar("train/return_std", float(returns.std(unbiased=False).item()), total_steps)
                self._write_scalar("train/advantage_mean", float(advantages.mean().item()), total_steps)
                self._write_scalar("train/advantage_std", float(advantages.std(unbiased=False).item()), total_steps)
                self._write_action_metrics(action_array, total_steps)
                self._write_termination_metrics(rollout_termination_reasons, total_steps)

                policy_losses = []
                value_losses = []
                entropy_values = []
                ratio_values = []
                approx_kl_values = []
                clip_fraction_values = []
                epochs_completed = 0
                stopped_for_kl = False
                for epoch in range(self.config.epochs):
                    epoch_kl_values = []
                    for batch_obs, batch_actions, batch_old_log_probs, batch_advantages, batch_returns in rollout.get_batches(self.config.batch_size):
                        batch_log_probs, batch_entropy, batch_values, _ = self.model.evaluate_actions(batch_obs, batch_actions)
                        ratio = torch.exp(batch_log_probs - batch_old_log_probs)
                        log_ratio = batch_log_probs - batch_old_log_probs
                        if epoch == 0 and not policy_losses:
                            initial_ratio_deviation = float(torch.max(torch.abs(ratio - 1.0)).item())
                            if not math.isfinite(initial_ratio_deviation) or initial_ratio_deviation > 1.0e-3:
                                raise RuntimeError(
                                    "PPO action/log-prob mismatch before the first optimizer step: "
                                    f"max ratio deviation={initial_ratio_deviation:.6g}"
                                )
                            self._write_scalar(
                                "train/initial_ratio_deviation",
                                initial_ratio_deviation,
                                total_steps,
                            )
                        surrogate1 = ratio * batch_advantages
                        surrogate2 = torch.clamp(ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon) * batch_advantages
                        policy_loss = -torch.min(surrogate1, surrogate2).mean()
                        value_loss = self.config.value_coef * (batch_returns - batch_values).pow(2).mean()
                        entropy_loss = -self.entropy_coef * batch_entropy.mean()
                        loss = policy_loss + value_loss + entropy_loss
                        self.optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        self.optimizer.step()
                        policy_losses.append(float(policy_loss.item()))
                        value_losses.append(float(value_loss.item()))
                        entropy_values.append(float(batch_entropy.mean().item()))
                        ratio_values.append(float(ratio.mean().item()))
                        approx_kl = float(((ratio - 1.0) - log_ratio).mean().item())
                        approx_kl_values.append(approx_kl)
                        epoch_kl_values.append(approx_kl)
                        clip_fraction_values.append(
                            float((torch.abs(ratio - 1.0) > self.config.clip_epsilon).float().mean().item())
                        )
                    epochs_completed = epoch + 1
                    if (
                        self.config.target_kl is not None
                        and epoch_kl_values
                        and float(np.mean(epoch_kl_values)) > self.config.target_kl
                    ):
                        stopped_for_kl = True
                        break

                self._log_update_summary(
                    total_steps=total_steps,
                    policy_losses=policy_losses,
                    value_losses=value_losses,
                    entropy_values=entropy_values,
                    ratio_values=ratio_values,
                    approx_kl_values=approx_kl_values,
                    clip_fraction_values=clip_fraction_values,
                    epochs_completed=epochs_completed,
                    stopped_for_kl=stopped_for_kl,
                    returns=returns,
                    advantages=advantages,
                )
                self._write_scalar("train/policy_loss", float(np.mean(policy_losses)), total_steps)
                self._write_scalar("train/value_loss", float(np.mean(value_losses)), total_steps)
                self._write_scalar("train/entropy", float(np.mean(entropy_values)), total_steps)
                self._write_scalar("train/ratio", float(np.mean(ratio_values)), total_steps)
                self._write_scalar("train/ratio_min", float(np.min(ratio_values)), total_steps)
                self._write_scalar("train/ratio_max", float(np.max(ratio_values)), total_steps)
                self._write_scalar("train/approx_kl", float(np.mean(approx_kl_values)), total_steps)
                self._write_scalar("train/clip_fraction", float(np.mean(clip_fraction_values)), total_steps)
                self._write_scalar("train/update_epochs", float(epochs_completed), total_steps)
                self._write_scalar("train/kl_early_stop", float(stopped_for_kl), total_steps)
                self._write_elevator_recovery_probe(total_steps)
                with torch.no_grad():
                    post_update_values = self.model(
                        rollout.observations[: rollout.index]
                    )[2]
                return_variance = returns.var(unbiased=False)
                if float(return_variance.item()) > 1.0e-8:
                    explained_variance = 1.0 - (
                        (returns - post_update_values).var(unbiased=False) / return_variance
                    )
                    explained_variance_value = float(explained_variance.item())
                else:
                    explained_variance_value = 0.0
                self._write_scalar("train/value_mean", float(post_update_values.mean().item()), total_steps)
                self._write_scalar(
                    "train/value_std",
                    float(post_update_values.std(unbiased=False).item()),
                    total_steps,
                )
                self._write_scalar("train/explained_variance", explained_variance_value, total_steps)
                last_completed_update_steps = total_steps

                if (
                    self.config.checkpoint_interval_steps > 0
                    and total_steps >= next_checkpoint_step
                    and total_steps < self.config.timesteps
                ):
                    self._save_model(step=total_steps, reason="periodic checkpoint")
                    last_saved_steps = total_steps
                    while next_checkpoint_step <= total_steps:
                        next_checkpoint_step += self.config.checkpoint_interval_steps

                if len(episode_rewards) >= report_every:
                    avg_reward = float(np.mean(episode_rewards[-report_every:]))
                    avg_length = float(np.mean(episode_lengths[-report_every:]))
                    elapsed = time.time() - training_start
                    print(
                        f"[PPO] steps={total_steps}/{self.config.timesteps} "
                        f"avg_reward={avg_reward:.3f} avg_length={avg_length:.1f} elapsed={elapsed:.1f}s"
                    )
                    self._write_scalar("train/avg_reward", avg_reward, total_steps)
                    self._write_scalar("train/avg_length", avg_length, total_steps)

                if self.writer is not None:
                    self.writer.flush()

            self._save_model(step=total_steps, reason="final")
            last_saved_steps = total_steps
            self._evaluate_policy()
        except (Exception, KeyboardInterrupt):
            if last_completed_update_steps > last_saved_steps:
                try:
                    self._save_model(
                        step=last_completed_update_steps,
                        reason="emergency checkpoint",
                    )
                except Exception as checkpoint_error:
                    print(f"[PPO] Emergency checkpoint failed: {checkpoint_error}")
            raise
        finally:
            try:
                self.env.close()
            except Exception as close_error:
                print(f"[PPO] Environment close failed: {close_error}")
            if self.writer is not None:
                self.writer.close()

    def _evaluate_policy(self):
        rewards = []
        lengths = []
        termination_counts = Counter()
        position_errors = []
        altitude_errors = []
        attitude_errors = []
        observation = None
        for _ in range(self.config.eval_episodes):
            if observation is None:
                observation, _ = reset_env_with_wait(
                    self.env,
                    action=self._wait_action(),
                    initial_action=self._initial_action(),
                )
            episode_reward = 0.0
            episode_length = 0
            while True:
                obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    action_tensor = self.model.deterministic_action(obs_tensor)
                action = action_tensor.squeeze(0).cpu().numpy()
                action = self._normalize_action(action)
                observation, reward, terminated, truncated, info = self.env.step(
                    self._to_env_action(action)
                )
                episode_reward += reward
                episode_length += 1
                debug_state = info.get("debug_state", {})
                target_hover = info.get("target_hover", {})
                if debug_state and target_hover:
                    dx = float(debug_state.get("x_m", 0.0)) - float(target_hover.get("x_m", 0.0))
                    dy = float(debug_state.get("y_m", 0.0)) - float(target_hover.get("y_m", 0.0))
                    position_errors.append(math.hypot(dx, dy))
                    altitude_errors.append(
                        abs(
                            float(debug_state.get("altitude_agl_m", 0.0))
                            - float(target_hover.get("altitude_agl_m", 0.0))
                        )
                    )
                    if self.config.control_mode == CONTROL_MODE_ELEVATOR:
                        elevator_features = info.get(
                            "elevator_hover_features",
                            {},
                        )
                        attitude_errors.append(
                            abs(
                                float(
                                    elevator_features.get(
                                        "inclination_error_deg",
                                        0.0,
                                    )
                                )
                            )
                        )
                    else:
                        inclination_error = abs(
                            angular_error_deg(
                                float(debug_state.get("inclination_deg", 0.0)),
                                float(target_hover.get("inclination_deg", 0.0)),
                            )
                        )
                        attitude_errors.append(
                            inclination_error
                            + abs(
                                angular_error_deg(
                                    float(debug_state.get("roll_deg", 0.0)),
                                    float(target_hover.get("roll_deg", 0.0)),
                                )
                            )
                        )
                if terminated or truncated:
                    reason = info.get("termination_reason") or ("truncated" if truncated else "unknown")
                    termination_counts[reason] += 1
                    if truncated:
                        observation, _ = continue_env_after_truncation(self.env)
                    else:
                        observation = None
                    break
            rewards.append(episode_reward)
            lengths.append(episode_length)
        avg_reward = float(np.mean(rewards))
        avg_length = float(np.mean(lengths))
        reward_per_step = float(np.sum(rewards) / max(1, np.sum(lengths)))
        print(
            f"Evaluation: avg_reward={avg_reward:.3f}, avg_length={avg_length:.1f}, "
            f"reward_per_step={reward_per_step:.3f}"
        )
        self._write_scalar("eval/avg_reward", avg_reward, self.config.timesteps)
        self._write_scalar("eval/avg_length", avg_length, self.config.timesteps)
        self._write_scalar("eval/reward_per_step", reward_per_step, self.config.timesteps)
        self._write_scalar(
            "eval/position_error_m",
            float(np.mean(position_errors)) if position_errors else 0.0,
            self.config.timesteps,
        )
        self._write_scalar(
            "eval/altitude_error_m",
            float(np.mean(altitude_errors)) if altitude_errors else 0.0,
            self.config.timesteps,
        )
        self._write_scalar(
            "eval/attitude_error_deg",
            float(np.mean(attitude_errors)) if attitude_errors else 0.0,
            self.config.timesteps,
        )
        for reason, count in termination_counts.items():
            self._write_scalar(f"eval/termination/{reason}", float(count), self.config.timesteps)
            self._write_scalar(
                f"eval/termination_rate/{reason}",
                float(count) / max(1, self.config.eval_episodes),
                self.config.timesteps,
            )


def reset_env_with_wait(
    env: gym.Env,
    *,
    action: Optional[Union[np.ndarray, list, tuple]] = None,
    initial_action: Optional[Union[np.ndarray, list, tuple]] = None,
):
    if getattr(env, "_waiting_for_reset", False):
        poll_wait = getattr(env, "poll_wait_for_next_episode", None)
        if not callable(poll_wait):
            raise RuntimeError("environment reports waiting-for-reset but does not expose poll_wait_for_next_episode()")
        return _wait_for_episode_start(env, poll_wait=poll_wait, action=action)

    try:
        reset_options = None
        if initial_action is not None:
            reset_options = {"initial_action": initial_action}
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


def _add_rflink_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--rflink-socket-timeout-s",
        type=float,
        default=3.0,
        help="Seconds to wait for each RealFlight Link socket operation.",
    )
    parser.add_argument(
        "--rflink-request-attempts",
        type=int,
        default=4,
        help="Maximum RFLink connect or ExchangeData attempts before aborting.",
    )
    parser.add_argument(
        "--rflink-retry-backoff-s",
        type=float,
        default=0.1,
        help="Initial exponential backoff between RFLink retries.",
    )
    parser.add_argument("--host", type=str, default=HOST)
    parser.add_argument("--port", type=int, default=PORT)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or diagnose PPO policies on the HoverPilot Hover Env."
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    train_parser = subparsers.add_parser("train", help="Train a PPO policy")
    train_parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Training steps. Defaults to 300000 for elevator and 50000 otherwise.",
    )
    train_parser.add_argument("--save-path", type=str, default="ppo_hoverpilot.pt")
    train_parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Continue training from a structured HoverPilot PPO checkpoint.",
    )
    train_parser.add_argument("--seed", type=int, default=None)
    train_parser.add_argument("--max-episode-steps", type=int, default=300)
    train_parser.add_argument("--sleep-interval-s", type=float, default=0.0)
    train_parser.add_argument("--n-steps", type=int, default=1024)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Optimizer learning rate. Defaults to 1e-4 for elevator and 3e-4 otherwise.",
    )
    train_parser.add_argument("--gamma", type=float, default=0.99)
    train_parser.add_argument("--gae-lambda", type=float, default=0.95)
    train_parser.add_argument("--clip-epsilon", type=float, default=0.2)
    train_parser.add_argument(
        "--target-kl",
        type=float,
        default=0.02,
        help="Stop a PPO update early when mean approximate KL exceeds this value; use 0 to disable.",
    )
    train_parser.add_argument(
        "--reward-scale",
        type=float,
        default=0.1,
        help="Scale rewards used for PPO returns while keeping displayed episode rewards unscaled.",
    )
    train_parser.add_argument("--value-coef", type=float, default=0.5)
    train_parser.add_argument(
        "--entropy-coef",
        type=float,
        default=None,
        help="Entropy coefficient. Defaults to 0.0001 for elevator and 0.01 for all controls.",
    )
    train_parser.add_argument(
        "--policy-initial-std",
        type=float,
        default=None,
        help="Initial policy exploration standard deviation. Defaults to 0.08 for elevator and 0.25 for all controls.",
    )
    train_parser.add_argument("--max-grad-norm", type=float, default=0.5)
    train_parser.add_argument("--log-interval", type=int, default=1)
    train_parser.add_argument(
        "--telemetry-log-interval-steps",
        type=int,
        default=25,
        help="Print and record inclination, pitch rate, elevator, and longitudinal error every N steps; 0 disables.",
    )
    train_parser.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help="Final evaluation episodes. Defaults to 10 for elevator and 3 otherwise.",
    )
    train_parser.add_argument("--tensorboard-log-dir", type=str, default="runs/hoverpilot-ppo")
    train_parser.add_argument("--disable-tensorboard", action="store_true")
    train_parser.add_argument(
        "--control-mode",
        choices=CONTROL_MODES,
        default=CONTROL_MODE_ALL,
        help="Policy-controlled channels. elevator uses a one-dimensional elevator policy.",
    )
    train_parser.add_argument(
        "--policy-preset",
        choices=POLICY_PRESETS,
        default=POLICY_PRESET_NONE,
        help=(
            "Optional fixed policy controller. none uses a PPO-only action "
            "policy with trainable measured-sign initialization; elevator-pd "
            "adds the measured elevator PD prior and limits the learned "
            "residual. Resumed checkpoints always restore their saved preset."
        ),
    )
    train_parser.add_argument(
        "--elevator-fixed-throttle",
        type=float,
        default=0.55,
        help=(
            "Throttle sent with elevator-only control. A resumed checkpoint "
            "restores its saved value."
        ),
    )
    train_parser.add_argument(
        "--checkpoint-interval-steps",
        type=int,
        default=1024,
        help="Save the current model after this many completed training steps; 0 disables.",
    )
    train_parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Training device. auto selects CUDA when available and otherwise CPU; MPS is opt-in.",
    )
    _add_rflink_args(train_parser)

    diagnose_parser = subparsers.add_parser(
        "diagnose-elevator",
        help="Measure RealFlight pitch response to conservative elevator pulses",
    )
    diagnose_parser.add_argument("--elevator-fixed-throttle", type=float, default=0.55)
    diagnose_parser.add_argument("--pulse", type=float, default=0.1)
    diagnose_parser.add_argument("--pulse-steps", type=int, default=8)
    diagnose_parser.add_argument("--settle-steps", type=int, default=8)
    _add_rflink_args(diagnose_parser)

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    if args.command == "train":
        config = PPOConfig(
            host=args.host,
            port=args.port,
            timesteps=args.timesteps,
            max_episode_steps=args.max_episode_steps,
            sleep_interval_s=args.sleep_interval_s,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_epsilon=args.clip_epsilon,
            target_kl=None if args.target_kl <= 0.0 else args.target_kl,
            reward_scale=args.reward_scale,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            policy_initial_std=args.policy_initial_std,
            max_grad_norm=args.max_grad_norm,
            save_path=args.save_path,
            resume_from=args.resume_from,
            seed=args.seed,
            eval_episodes=args.eval_episodes,
            log_interval=args.log_interval,
            telemetry_log_interval_steps=args.telemetry_log_interval_steps,
            tensorboard_log_dir=None if args.disable_tensorboard else args.tensorboard_log_dir,
            device=args.device,
            control_mode=args.control_mode,
            policy_preset=args.policy_preset,
            elevator_fixed_throttle=args.elevator_fixed_throttle,
            rflink_socket_timeout_s=args.rflink_socket_timeout_s,
            rflink_request_attempts=args.rflink_request_attempts,
            rflink_retry_backoff_s=args.rflink_retry_backoff_s,
            checkpoint_interval_steps=args.checkpoint_interval_steps,
        )
        trainer = PPOTrainer(config)
        trainer.train()
    elif args.command == "diagnose-elevator":
        diagnose_elevator_response(
            args.host,
            args.port,
            elevator_fixed_throttle=args.elevator_fixed_throttle,
            pulse=args.pulse,
            pulse_steps=args.pulse_steps,
            settle_steps=args.settle_steps,
            rflink_socket_timeout_s=args.rflink_socket_timeout_s,
            rflink_request_attempts=args.rflink_request_attempts,
            rflink_retry_backoff_s=args.rflink_retry_backoff_s,
        )
if __name__ == "__main__":
    main()
