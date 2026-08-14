"""Actor-critic networks and structured hover-control policies."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from .constants import (
    CONTROL_MODE_AILERON,
    CONTROL_MODE_AILERON_THROTTLE,
    CONTROL_MODE_ALL,
    CONTROL_MODE_ELEVATOR,
    CONTROL_MODE_ELEVATOR_THROTTLE,
    CONTROL_MODE_RUDDER,
    CONTROL_MODE_RUDDER_THROTTLE,
    CONTROL_MODE_THROTTLE,
    CONTROL_MODES,
    DEFAULT_INITIAL_ACTION,
    POLICY_PRESET_ELEVATOR_PD,
    POLICY_PRESET_NONE,
    _AILERON_PPO_INITIAL_GAIN,
    _AILERON_PPO_INITIAL_TRIM,
    _ALL_CONTROLS_RESIDUAL_SCALE,
    _ELEVATOR_PD_PRIOR_LIMIT,
    _ELEVATOR_PD_PRIOR_WEIGHT,
    _ELEVATOR_PD_RESIDUAL_LIMIT,
    _ELEVATOR_PPO_INITIAL_GAIN,
    _RUDDER_PPO_INITIAL_GAIN,
    _THROTTLE_PPO_INITIAL_GAIN,
    _THROTTLE_PPO_INITIAL_TRIM,
)

# Actor-critic architecture (Sutton & Barto, 2018, Sections 13.2 and 13.5).
# http://incompleteideas.net/book/the-book-2nd.html
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
        control_mode: str,
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
        if control_mode not in CONTROL_MODES:
            raise ValueError(
                f"Unsupported control mode {control_mode!r}; "
                f"choose one of {CONTROL_MODES}."
            )
        self.policy_preset = policy_preset
        enforce_all_controls_structure = (
            control_mode == CONTROL_MODE_ALL
            and observation_dim == 14
            and action_dim == 4
        )
        self.enforce_elevator_symmetry = (
            enforce_all_controls_structure
            or (
                control_mode in {
                    CONTROL_MODE_ELEVATOR,
                    CONTROL_MODE_ELEVATOR_THROTTLE,
                }
                and observation_dim == 6
                and action_dim == (
                    2
                    if control_mode == CONTROL_MODE_ELEVATOR_THROTTLE
                    else 1
                )
            )
        )
        self.enforce_aileron_symmetry = (
            enforce_all_controls_structure
            or (
                control_mode == CONTROL_MODE_AILERON
                and observation_dim == 2
                and action_dim == 1
            )
            or (
                control_mode == CONTROL_MODE_AILERON_THROTTLE
                and observation_dim == 4
                and action_dim == 2
            )
        )
        self.enforce_rudder_symmetry = (
            enforce_all_controls_structure
            or (
                control_mode == CONTROL_MODE_RUDDER
                and observation_dim == 2
                and action_dim == 1
            )
            or (
                control_mode == CONTROL_MODE_RUDDER_THROTTLE
                and observation_dim == 4
                and action_dim == 2
            )
        )
        self.enforce_throttle_structure = (
            enforce_all_controls_structure
            or (
                control_mode == CONTROL_MODE_THROTTLE
                and observation_dim == 2
                and action_dim == 1
            )
            or (
                control_mode == CONTROL_MODE_ELEVATOR_THROTTLE
                and observation_dim == 6
                and action_dim == 2
            )
            or (
                control_mode == CONTROL_MODE_AILERON_THROTTLE
                and observation_dim == 4
                and action_dim == 2
            )
            or (
                control_mode == CONTROL_MODE_RUDDER_THROTTLE
                and observation_dim == 4
                and action_dim == 2
            )
        )
        mirror_sign = (
            [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0]
            if self.enforce_elevator_symmetry and observation_dim == 6
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
        use_linear_aileron_policy = (
            self.enforce_aileron_symmetry
            and policy_preset == POLICY_PRESET_NONE
        )
        use_linear_rudder_policy = (
            self.enforce_rudder_symmetry
            and policy_preset == POLICY_PRESET_NONE
        )
        use_linear_throttle_policy = (
            self.enforce_throttle_structure
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
        if use_linear_aileron_policy:
            initial_gain = torch.as_tensor(
                _AILERON_PPO_INITIAL_GAIN,
                dtype=torch.float32,
            )
            self.aileron_policy_raw_gain = nn.Parameter(
                torch.log(torch.expm1(initial_gain))
            )
            self.aileron_policy_trim_latent = nn.Parameter(
                torch.tensor(
                    math.atanh(_AILERON_PPO_INITIAL_TRIM),
                    dtype=torch.float32,
                )
            )
        else:
            self.register_parameter("aileron_policy_raw_gain", None)
            self.register_parameter("aileron_policy_trim_latent", None)
        if use_linear_rudder_policy:
            initial_gain = torch.as_tensor(
                _RUDDER_PPO_INITIAL_GAIN,
                dtype=torch.float32,
            )
            self.rudder_policy_raw_gain = nn.Parameter(
                torch.log(torch.expm1(initial_gain))
            )
        else:
            self.register_parameter("rudder_policy_raw_gain", None)
        if use_linear_throttle_policy:
            initial_gain = torch.as_tensor(
                _THROTTLE_PPO_INITIAL_GAIN,
                dtype=torch.float32,
            )
            self.throttle_policy_raw_gain = nn.Parameter(
                torch.log(torch.expm1(initial_gain))
            )
            self.throttle_policy_trim_latent = nn.Parameter(
                torch.tensor(
                    math.atanh(
                        2.0 * _THROTTLE_PPO_INITIAL_TRIM - 1.0
                    ),
                    dtype=torch.float32,
                )
            )
        else:
            self.register_parameter("throttle_policy_raw_gain", None)
            self.register_parameter("throttle_policy_trim_latent", None)
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
            if (
                use_linear_elevator_policy
                or use_linear_aileron_policy
                or use_linear_rudder_policy
                or use_linear_throttle_policy
            ) and not enforce_all_controls_structure
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
                if enforce_all_controls_structure:
                    self.policy_mean.weight.zero_()
                else:
                    nn.init.orthogonal_(self.policy_mean.weight, gain=0.01)
                self.policy_mean.bias.zero_()
            nn.init.orthogonal_(self.value_head.weight, gain=1.0)
            self.value_head.bias.zero_()
            self.policy_log_std.fill_(math.log(initial_policy_std))
            if (
                action_dim >= 3
                and self.policy_mean is not None
                and not enforce_all_controls_structure
            ):
                # Hover training needs non-zero throttle from the first step.
                normalized_throttle = 2.0 * float(DEFAULT_INITIAL_ACTION[2]) - 1.0
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

    @property
    def aileron_policy_gain(self) -> Optional[torch.Tensor]:
        if self.aileron_policy_raw_gain is None:
            return None
        return F.softplus(self.aileron_policy_raw_gain)

    @property
    def aileron_policy_trim(self) -> Optional[torch.Tensor]:
        if self.aileron_policy_trim_latent is None:
            return None
        return torch.tanh(self.aileron_policy_trim_latent)

    @property
    def rudder_policy_gain(self) -> Optional[torch.Tensor]:
        if self.rudder_policy_raw_gain is None:
            return None
        return F.softplus(self.rudder_policy_raw_gain)

    @property
    def throttle_policy_gain(self) -> Optional[torch.Tensor]:
        if self.throttle_policy_raw_gain is None:
            return None
        return F.softplus(self.throttle_policy_raw_gain)

    @property
    def throttle_policy_trim(self) -> Optional[torch.Tensor]:
        if self.throttle_policy_trim_latent is None:
            return None
        return 0.5 * (
            torch.tanh(self.throttle_policy_trim_latent) + 1.0
        )

    def _compute_policy_mean(
        self,
        obs: torch.Tensor,
        *,
        hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        elevator_gain = self.elevator_policy_gain
        aileron_gain = self.aileron_policy_gain
        rudder_gain = self.rudder_policy_gain
        throttle_gain = self.throttle_policy_gain
        if (
            elevator_gain is not None
            and aileron_gain is not None
            and rudder_gain is not None
            and throttle_gain is not None
        ):
            assert self.aileron_policy_trim_latent is not None
            assert self.throttle_policy_trim_latent is not None
            aileron_mean = (
                self.aileron_policy_trim_latent
                - aileron_gain[0] * obs[..., 0:1]
                - aileron_gain[1] * obs[..., 1:2]
            )
            elevator_mean = (
                -elevator_gain[0] * obs[..., 2:3]
                + elevator_gain[1] * obs[..., 3:4]
            )
            throttle_mean = (
                self.throttle_policy_trim_latent
                - throttle_gain[0] * obs[..., 8:9]
                - throttle_gain[1] * obs[..., 9:10]
            )
            rudder_mean = (
                rudder_gain[0] * obs[..., 10:11]
                + rudder_gain[1] * obs[..., 11:12]
            )
            structured_mean = torch.cat(
                (
                    aileron_mean,
                    elevator_mean,
                    throttle_mean,
                    rudder_mean,
                ),
                dim=-1,
            )
            assert self.policy_mean is not None
            if hidden is None:
                hidden = self.shared(obs)
            residual = (
                _ALL_CONTROLS_RESIDUAL_SCALE
                * self.policy_mean(hidden)
            )
            return structured_mean + residual
        if elevator_gain is not None and throttle_gain is not None:
            assert self.throttle_policy_trim_latent is not None
            elevator_mean = (
                -elevator_gain[0] * obs[..., 0:1]
                + elevator_gain[1] * obs[..., 1:2]
            )
            throttle_mean = (
                self.throttle_policy_trim_latent
                - throttle_gain[0] * obs[..., 4:5]
                - throttle_gain[1] * obs[..., 5:6]
            )
            return torch.cat((elevator_mean, throttle_mean), dim=-1)
        if aileron_gain is not None and throttle_gain is not None:
            assert self.aileron_policy_trim_latent is not None
            assert self.throttle_policy_trim_latent is not None
            aileron_mean = (
                self.aileron_policy_trim_latent
                - aileron_gain[0] * obs[..., 0:1]
                - aileron_gain[1] * obs[..., 1:2]
            )
            throttle_mean = (
                self.throttle_policy_trim_latent
                - throttle_gain[0] * obs[..., 2:3]
                - throttle_gain[1] * obs[..., 3:4]
            )
            return torch.cat((aileron_mean, throttle_mean), dim=-1)
        if rudder_gain is not None and throttle_gain is not None:
            assert self.throttle_policy_trim_latent is not None
            rudder_mean = (
                rudder_gain[0] * obs[..., 0:1]
                + rudder_gain[1] * obs[..., 1:2]
            )
            throttle_mean = (
                self.throttle_policy_trim_latent
                - throttle_gain[0] * obs[..., 2:3]
                - throttle_gain[1] * obs[..., 3:4]
            )
            return torch.cat((rudder_mean, throttle_mean), dim=-1)
        if elevator_gain is not None:
            return (
                -elevator_gain[0] * obs[..., 0:1]
                + elevator_gain[1] * obs[..., 1:2]
            )
        if aileron_gain is not None:
            assert self.aileron_policy_trim_latent is not None
            return (
                self.aileron_policy_trim_latent
                - aileron_gain[0] * obs[..., 0:1]
                - aileron_gain[1] * obs[..., 1:2]
            )
        if rudder_gain is not None:
            return (
                rudder_gain[0] * obs[..., 0:1]
                + rudder_gain[1] * obs[..., 1:2]
            )
        if throttle_gain is not None:
            assert self.throttle_policy_trim_latent is not None
            return (
                self.throttle_policy_trim_latent
                - throttle_gain[0] * obs[..., 0:1]
                - throttle_gain[1] * obs[..., 1:2]
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
