"""RealFlight rollout collection, PPO optimization, and evaluation."""

from __future__ import annotations

import math
import os
import random
import time
from collections import Counter
from dataclasses import replace
from typing import Dict, Mapping, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch import nn

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from hoverpilot.envs import (
    elevator_features_to_observation,
    elevator_throttle_features_to_observation,
)
from hoverpilot.training.hover import (
    HOVER_TARGET_INCLINATION_DEG,
    ElevatorHoverFeatures,
)
from hoverpilot.utils.logger import format_debug_state

from .buffer import RolloutBuffer
from .checkpoints import (
    _apply_observation_config,
    _validate_fixed_throttle,
    _validate_policy_preset,
    build_policy_checkpoint,
    load_policy_checkpoint,
)
from .config import PPOCheckpoint, PPOConfig
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
    CONTROL_MODES,
    DEFAULT_ENTROPY_COEF,
    DEFAULT_POLICY_STD,
    _ELEVATOR_EFFECTIVE_RESTORING_ACTION,
    _THROTTLE_PPO_INITIAL_TRIM,
)
from .models import ActorCritic
from .runtime import (
    _build_hover_env,
    _expand_policy_action,
    _initial_env_action,
    _policy_action_space,
    _resolve_training_defaults,
    _validate_rflink_settings,
    continue_env_after_truncation,
    reset_env_with_wait,
    resolve_device,
)

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
        if (
            not math.isfinite(config.episode_start_idle_seconds)
            or config.episode_start_idle_seconds < 0.0
        ):
            raise ValueError(
                "episode_start_idle_seconds must be a finite non-negative value"
            )
        if config.episode_start_idle_curriculum_steps < 0:
            raise ValueError(
                "episode_start_idle_curriculum_steps must be non-negative"
            )
        if (
            not math.isfinite(
                config.episode_start_idle_curriculum_start_seconds
            )
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
        if resume_checkpoint is not None:
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
            else DEFAULT_ENTROPY_COEF
        )
        self.policy_initial_std = (
            config.policy_initial_std
            if config.policy_initial_std is not None
            else DEFAULT_POLICY_STD
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
            control_mode=config.control_mode,
        ).to(self.device)
        if resume_checkpoint is not None:
            self._load_resume_checkpoint(resume_checkpoint, config.resume_from)
            if config.policy_initial_std is not None:
                with torch.no_grad():
                    self.model.policy_log_std.fill_(
                        math.log(config.policy_initial_std)
                    )
                print(
                    "[PPO] Overrode resumed policy exploration std with "
                    f"{config.policy_initial_std}"
                )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.writer = self._build_writer()
        self._curriculum_exposure_steps = 0
        self._evaluating = False

    def _load_resume_checkpoint(
        self,
        checkpoint: PPOCheckpoint,
        checkpoint_path: str,
    ):
        try:
            incompatible = self.model.load_state_dict(
                checkpoint.model_state_dict,
                strict=False,
            )
        except RuntimeError as exc:
            raise ValueError(f"Resume checkpoint is incompatible: {exc}") from exc
        allowed_missing = {"policy_mean.weight", "policy_mean.bias"}
        if (
            set(incompatible.missing_keys) - allowed_missing
            or incompatible.unexpected_keys
        ):
            raise ValueError(
                "Resume checkpoint is incompatible: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
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
        if self.config.control_mode in {
            CONTROL_MODE_ALL,
            CONTROL_MODE_THROTTLE,
            CONTROL_MODE_ELEVATOR_THROTTLE,
            CONTROL_MODE_AILERON_THROTTLE,
            CONTROL_MODE_RUDDER_THROTTLE,
        }:
            return self._initial_action()
        return np.asarray(self.config.wait_action, dtype=np.float32)

    def _initial_action(self) -> np.ndarray:
        return _initial_env_action(
            self.config.control_mode,
            self.elevator_fixed_throttle,
            self.config.initial_action,
        )

    def _episode_start_idle_action(self) -> np.ndarray:
        return np.asarray(
            [0.0, 0.0, self.config.episode_start_idle_throttle, 0.0],
            dtype=np.float32,
        )

    def _episode_start_idle_duration_s(self) -> float:
        if self._evaluating or self.config.episode_start_idle_curriculum_steps == 0:
            return self.config.episode_start_idle_seconds
        progress = min(
            1.0,
            self._curriculum_exposure_steps
            / self.config.episode_start_idle_curriculum_steps,
        )
        start_seconds = (
            self.config.episode_start_idle_curriculum_start_seconds
        )
        return start_seconds + (
            self.config.episode_start_idle_seconds - start_seconds
        ) * progress

    def _deterministic_env_action(self, observation: np.ndarray) -> np.ndarray:
        observation_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            policy_action = self.model.deterministic_action(
                observation_tensor
            ).squeeze(0)
        normalized_action = self._normalize_action(
            policy_action.cpu().numpy()
        )
        return self._to_env_action(normalized_action)

    def _apply_episode_start_idle(
        self,
        observation: np.ndarray,
        info: dict[str, object],
    ) -> Optional[Tuple[np.ndarray, dict[str, object]]]:
        idle_duration_s = self._episode_start_idle_duration_s()
        if idle_duration_s == 0.0:
            return observation, info
        run_idle = getattr(self.env, "run_episode_start_idle", None)
        if not callable(run_idle):
            raise RuntimeError(
                "environment does not support an episode start idle period"
            )
        started, observation, info = run_idle(
            duration_s=idle_duration_s,
            action=self._episode_start_idle_action(),
        )
        if not started:
            details = info.get("episode_start_idle", {})
            print(
                "[PPO] episode start idle ended before policy handoff "
                f"reason={info.get('termination_reason')} details={details}"
            )
            return None

        idle_details = info.get("episode_start_idle", {})
        if self.config.episode_start_handoff_seconds == 0.0:
            return observation, info
        handoff_duration_s = self.config.episode_start_handoff_seconds
        run_handoff = getattr(self.env, "run_episode_start_handoff", None)
        if not callable(run_handoff):
            raise RuntimeError(
                "environment does not support an episode start handoff"
            )
        started, observation, info = run_handoff(
            duration_s=handoff_duration_s,
            start_action=self._episode_start_idle_action(),
            action_provider=self._deterministic_env_action,
        )
        info["episode_start_idle"] = idle_details
        if started:
            return observation, info
        details = info.get("episode_start_handoff", {})
        print(
            "[PPO] episode start handoff ended before policy control "
            f"reason={info.get('termination_reason')} details={details}"
        )
        return None

    def _reset_episode(
        self,
        *,
        require_reset_boundary: bool = False,
    ) -> Tuple[np.ndarray, dict[str, object]]:
        require_reset_boundary = (
            require_reset_boundary
            or self.config.episode_start_idle_seconds > 0.0
        )
        while True:
            observation, info = reset_env_with_wait(
                self.env,
                action=self._wait_action(),
                initial_action=self._initial_action(),
                require_reset_boundary=require_reset_boundary,
            )
            prepared = self._apply_episode_start_idle(observation, info)
            if prepared is not None:
                return prepared

    def _continue_episode(self) -> Tuple[np.ndarray, dict[str, object]]:
        observation, info = continue_env_after_truncation(self.env)
        prepared = self._apply_episode_start_idle(observation, info)
        if prepared is not None:
            return prepared
        return self._reset_episode()

    def _start_after_truncation(
        self,
    ) -> Tuple[np.ndarray, dict[str, object]]:
        if (
            self.config.control_mode in CONNECTION_EPISODE_CONTROL_MODES
            or self.config.episode_start_idle_seconds > 0.0
        ):
            return self._reset_episode(require_reset_boundary=True)
        return self._continue_episode()

    def _advance_episode_start_idle_curriculum(
        self,
        successful_steps: int,
    ) -> None:
        if (
            self._evaluating
            or self.config.episode_start_idle_curriculum_steps == 0
        ):
            return
        self._curriculum_exposure_steps += successful_steps

    def _episode_qualifies_for_curriculum_progress(
        self,
        info: Mapping[str, object],
    ) -> bool:
        debug_state = info.get("debug_state")
        if not isinstance(debug_state, Mapping):
            return False
        try:
            radial_distance_m = float(
                debug_state["distance_from_cylinder_axis_m"]
            )
            horizontal_speed_mps = math.hypot(
                float(debug_state["velocity_world_u_mps"]),
                float(debug_state["velocity_world_v_mps"]),
            )
            altitude_error_m = abs(
                float(debug_state["altitude_agl_m"])
                - self.env.reward_config.target_altitude_agl_m
            )
            tilt_error_deg = abs(
                float(debug_state["inclination_deg"])
                - HOVER_TARGET_INCLINATION_DEG
            )
        except (KeyError, TypeError, ValueError):
            return False
        reward_config = self.env.reward_config
        return (
            radial_distance_m
            <= reward_config.trainer_cylinder_radius_m / 3.0
            and horizontal_speed_mps
            <= reward_config.velocity_error_scale_mps * 0.3
            and altitude_error_m
            <= reward_config.altitude_error_scale_m / 3.0
            and tilt_error_deg
            <= reward_config.inclination_error_scale_deg
        )

    def _policy_action_labels(self) -> tuple[str, ...]:
        if self.config.control_mode == CONTROL_MODE_AILERON:
            return ("aileron",)
        if self.config.control_mode == CONTROL_MODE_RUDDER:
            return ("rudder",)
        if self.config.control_mode == CONTROL_MODE_THROTTLE:
            return ("throttle",)
        if self.config.control_mode == CONTROL_MODE_ELEVATOR_THROTTLE:
            return ("elevator", "throttle")
        if self.config.control_mode == CONTROL_MODE_AILERON_THROTTLE:
            return ("aileron", "throttle")
        if self.config.control_mode == CONTROL_MODE_RUDDER_THROTTLE:
            return ("rudder", "throttle")
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
            self.config.control_mode not in {
                CONTROL_MODE_ELEVATOR,
                CONTROL_MODE_ELEVATOR_THROTTLE,
            }
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

    def _write_aileron_recovery_probe(self, step: int):
        if self.config.control_mode not in {
            CONTROL_MODE_AILERON,
            CONTROL_MODE_AILERON_THROTTLE,
        }:
            return
        expected_shape = (
            (4,)
            if self.config.control_mode
            == CONTROL_MODE_AILERON_THROTTLE
            else (2,)
        )
        if self.env.observation_space.shape != expected_shape:
            return
        probe_array = np.zeros((4, expected_shape[0]))
        probe_array[0, 0] = 1.0
        probe_array[1, 0] = -1.0
        probe_array[2, 1] = 1.0
        probe_array[3, 1] = -1.0
        probe = torch.tensor(
            probe_array,
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            latent_means = (
                self.model._compute_policy_mean(probe)[:, 0]
                .detach()
                .cpu()
                .numpy()
            )
        assert self.model.aileron_policy_trim_latent is not None
        trim_latent = float(
            self.model.aileron_policy_trim_latent.detach().cpu().item()
        )
        corrections = latent_means - trim_latent
        restoring_margins = np.asarray(
            [
                -corrections[0],
                corrections[1],
                -corrections[2],
                corrections[3],
            ],
            dtype=np.float32,
        )
        symmetry_error = float(
            0.5
            * (
                abs(float(corrections[0] + corrections[1]))
                + abs(float(corrections[2] + corrections[3]))
            )
        )
        minimum_margin = float(restoring_margins.min())
        trim = float(self.model.aileron_policy_trim.detach().cpu().item())
        print(
            "[PPO] aileron recovery probe "
            f"trim={trim:+.3f} "
            f"roll_correction={corrections[0]:+.3f}/{corrections[1]:+.3f} "
            f"rate_correction={corrections[2]:+.3f}/{corrections[3]:+.3f} "
            f"symmetry_error={symmetry_error:.3f} "
            f"minimum_margin={minimum_margin:.3f}"
        )
        self._write_scalar(
            "train/control/aileron_trim",
            trim,
            step,
        )
        self._write_scalar(
            "train/recovery_probe/symmetry_error",
            symmetry_error,
            step,
        )
        self._write_scalar(
            "train/recovery_probe/minimum_restoring_margin",
            minimum_margin,
            step,
        )

    def _write_rudder_recovery_probe(self, step: int):
        if self.config.control_mode not in {
            CONTROL_MODE_RUDDER,
            CONTROL_MODE_RUDDER_THROTTLE,
        }:
            return
        expected_shape = (
            (4,)
            if self.config.control_mode == CONTROL_MODE_RUDDER_THROTTLE
            else (2,)
        )
        if self.env.observation_space.shape != expected_shape:
            return
        probe_array = np.zeros((4, expected_shape[0]))
        probe_array[0, 0] = 1.0
        probe_array[1, 0] = -1.0
        probe_array[2, 1] = 1.0
        probe_array[3, 1] = -1.0
        probe = torch.tensor(
            probe_array,
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            corrections = (
                self.model._compute_policy_mean(probe)[:, 0]
                .detach()
                .cpu()
                .numpy()
            )
        restoring_margins = np.asarray(
            [
                corrections[0],
                -corrections[1],
                corrections[2],
                -corrections[3],
            ],
            dtype=np.float32,
        )
        symmetry_error = float(
            0.5
            * (
                abs(float(corrections[0] + corrections[1]))
                + abs(float(corrections[2] + corrections[3]))
            )
        )
        minimum_margin = float(restoring_margins.min())
        print(
            "[PPO] rudder recovery probe "
            f"angle_correction={corrections[0]:+.3f}/{corrections[1]:+.3f} "
            f"rate_correction={corrections[2]:+.3f}/{corrections[3]:+.3f} "
            f"symmetry_error={symmetry_error:.3f} "
            f"minimum_margin={minimum_margin:.3f}"
        )
        self._write_scalar(
            "train/recovery_probe/symmetry_error",
            symmetry_error,
            step,
        )
        self._write_scalar(
            "train/recovery_probe/minimum_restoring_margin",
            minimum_margin,
            step,
        )

    def _write_throttle_recovery_probe(self, step: int):
        if self.config.control_mode not in {
            CONTROL_MODE_THROTTLE,
            CONTROL_MODE_ELEVATOR_THROTTLE,
            CONTROL_MODE_AILERON_THROTTLE,
            CONTROL_MODE_RUDDER_THROTTLE,
        }:
            return
        throttle_probe_layout = {
            CONTROL_MODE_THROTTLE: ((2,), 0, 1, 0),
            CONTROL_MODE_ELEVATOR_THROTTLE: ((6,), 4, 5, 1),
            CONTROL_MODE_AILERON_THROTTLE: ((4,), 2, 3, 1),
            CONTROL_MODE_RUDDER_THROTTLE: ((4,), 2, 3, 1),
        }
        (
            expected_shape,
            altitude_index,
            vertical_velocity_index,
            throttle_action_index,
        ) = throttle_probe_layout[self.config.control_mode]
        if self.env.observation_space.shape != expected_shape:
            return
        probe_array = np.zeros((4, self.env.observation_space.shape[0]))
        probe_array[0, altitude_index] = 1.0
        probe_array[1, altitude_index] = -1.0
        probe_array[2, vertical_velocity_index] = 1.0
        probe_array[3, vertical_velocity_index] = -1.0
        probe = torch.tensor(
            probe_array,
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            latent_means = (
                self.model._compute_policy_mean(probe)[
                    :, throttle_action_index
                ]
                .detach()
                .cpu()
                .numpy()
            )
        assert self.model.throttle_policy_trim_latent is not None
        trim_latent = float(
            self.model.throttle_policy_trim_latent.detach().cpu().item()
        )
        corrections = latent_means - trim_latent
        restoring_margins = np.asarray(
            [
                -corrections[0],
                corrections[1],
                -corrections[2],
                corrections[3],
            ],
            dtype=np.float32,
        )
        symmetry_error = float(
            0.5
            * (
                abs(float(corrections[0] + corrections[1]))
                + abs(float(corrections[2] + corrections[3]))
            )
        )
        minimum_margin = float(restoring_margins.min())
        assert self.model.throttle_policy_trim is not None
        trim = float(
            self.model.throttle_policy_trim.detach().cpu().item()
        )
        print(
            "[PPO] throttle recovery probe "
            f"trim={trim:.3f} "
            f"altitude_correction={corrections[0]:+.3f}/{corrections[1]:+.3f} "
            f"velocity_correction={corrections[2]:+.3f}/{corrections[3]:+.3f} "
            f"symmetry_error={symmetry_error:.3f} "
            f"minimum_margin={minimum_margin:.3f}"
        )
        self._write_scalar(
            "train/control/throttle_trim",
            trim,
            step,
        )
        self._write_scalar(
            "train/recovery_probe/symmetry_error",
            symmetry_error,
            step,
        )
        self._write_scalar(
            "train/recovery_probe/minimum_restoring_margin",
            minimum_margin,
            step,
        )

    def _elevator_probe_observation(
        self,
        features: ElevatorHoverFeatures,
    ) -> np.ndarray:
        config = self.env.reward_config
        if self.config.control_mode == CONTROL_MODE_ELEVATOR_THROTTLE:
            return elevator_throttle_features_to_observation(
                features,
                config=config,
            )
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
        episode_start_idle = (
            info.get("episode_start_idle") if isinstance(info, dict) else None
        )
        episode_start_handoff = (
            info.get("episode_start_handoff")
            if isinstance(info, dict)
            else None
        )
        print(
            f"[PPO] episode start reason={info.get('episode_start_reason')} "
            f"waiting={info.get('waiting_for_reset')}"
        )
        if isinstance(episode_start_idle, dict):
            print(
                "[PPO] episode start idle "
                f"duration={float(episode_start_idle.get('elapsed_physics_s', 0.0)):.2f}s "
                f"steps={int(episode_start_idle.get('hold_steps', 0))} "
                f"throttle={float(episode_start_idle.get('throttle', 0.0)):.2f} "
                f"tilt={float(episode_start_idle.get('initial_tilt_deg', 0.0)):.1f}"
                "->"
                f"{float(episode_start_idle.get('control_start_tilt_deg', 0.0)):.1f}deg "
                f"altitude={float(episode_start_idle.get('initial_altitude_agl_m', 0.0)):.2f}"
                "->"
                f"{float(episode_start_idle.get('control_start_altitude_agl_m', 0.0)):.2f}m"
            )
        if isinstance(episode_start_handoff, dict):
            print(
                "[PPO] episode start handoff "
                f"duration={float(episode_start_handoff.get('elapsed_physics_s', 0.0)):.2f}s "
                f"steps={int(episode_start_handoff.get('handoff_steps', 0))} "
                f"max_delta={float(episode_start_handoff.get('max_action_delta', 0.0)):.3f} "
                f"max_step={float(episode_start_handoff.get('max_action_step', 0.0)):.3f} "
                f"action={episode_start_handoff.get('start_action')}"
                "->"
                f"{episode_start_handoff.get('end_action')}"
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
        if self.config.control_mode == CONTROL_MODE_THROTTLE:
            features = info.get("throttle_hover_features", {})
            if not features:
                return
            altitude_error = float(
                features.get("altitude_error_m", 0.0)
            )
            vertical_velocity = float(
                features.get("vertical_velocity_mps", 0.0)
            )
            throttle = float(env_action[2])
            gains_tensor = self.model.throttle_policy_gain
            gains = (
                gains_tensor.detach().cpu().numpy()
                if gains_tensor is not None
                else np.ones(2, dtype=np.float32)
            )
            trim_tensor = self.model.throttle_policy_trim
            trim = (
                float(trim_tensor.detach().cpu().item())
                if trim_tensor is not None
                else _THROTTLE_PPO_INITIAL_TRIM
            )
            weighted_error = (
                float(gains[0])
                * altitude_error
                / self.env.reward_config.altitude_error_scale_m
                + float(gains[1])
                * vertical_velocity
                / self.env.reward_config.velocity_error_scale_mps
            )
            restoring = (
                abs(weighted_error) < 1.0e-3
                or (throttle - trim) * weighted_error < 0.0
            )
            print(
                f"[PPO] control step={total_steps} reward={reward:+.3f} "
                f"throttle={throttle:.3f} "
                f"alt_error={altitude_error:+.3f}m "
                f"vertical_velocity={vertical_velocity:+.3f}m/s "
                f"restoring={restoring}"
            )
            self._write_scalar(
                "train/control/throttle_action",
                throttle,
                total_steps,
            )
            self._write_scalar(
                "train/control/restoring_action",
                float(restoring),
                total_steps,
            )
            self._write_scalar(
                "train/state/altitude_error_m",
                altitude_error,
                total_steps,
            )
            self._write_scalar(
                "train/state/vertical_velocity_mps",
                vertical_velocity,
                total_steps,
            )
            self._write_scalar(
                "train/control/reward",
                reward,
                total_steps,
            )
            return
        if self.config.control_mode in {
            CONTROL_MODE_RUDDER,
            CONTROL_MODE_RUDDER_THROTTLE,
        }:
            features = info.get("rudder_hover_features", {})
            if not features:
                return
            rudder_angle_error = float(
                features.get("rudder_angle_error_deg", 0.0)
            )
            yaw_rate = float(features.get("yaw_rate_deg_s", 0.0))
            rudder = float(env_action[3])
            gains_tensor = self.model.rudder_policy_gain
            gains = (
                gains_tensor.detach().cpu().numpy()
                if gains_tensor is not None
                else np.ones(2, dtype=np.float32)
            )
            weighted_error = (
                float(gains[0])
                * rudder_angle_error
                / self.env.reward_config.rudder_angle_error_scale_deg
                + float(gains[1])
                * yaw_rate
                / self.env.reward_config.yaw_rate_scale_deg_s
            )
            restoring = (
                abs(weighted_error) < 1.0e-3
                or rudder * weighted_error > 0.0
            )
            combined_mode = (
                self.config.control_mode
                == CONTROL_MODE_RUDDER_THROTTLE
            )
            throttle_description = ""
            throttle_restoring = True
            if combined_mode:
                throttle_features = info.get(
                    "throttle_hover_features",
                    {},
                )
                altitude_error = float(
                    throttle_features.get("altitude_error_m", 0.0)
                )
                vertical_velocity = float(
                    throttle_features.get("vertical_velocity_mps", 0.0)
                )
                throttle = float(env_action[2])
                throttle_gains = self.model.throttle_policy_gain
                throttle_trim = self.model.throttle_policy_trim
                assert throttle_gains is not None
                assert throttle_trim is not None
                throttle_gain_values = (
                    throttle_gains.detach().cpu().numpy()
                )
                throttle_trim_value = float(
                    throttle_trim.detach().cpu().item()
                )
                throttle_weighted_error = (
                    float(throttle_gain_values[0])
                    * altitude_error
                    / self.env.reward_config.altitude_error_scale_m
                    + float(throttle_gain_values[1])
                    * vertical_velocity
                    / self.env.reward_config.velocity_error_scale_mps
                )
                throttle_restoring = (
                    abs(throttle_weighted_error) < 1.0e-3
                    or (
                        throttle - throttle_trim_value
                    )
                    * throttle_weighted_error
                    < 0.0
                )
                throttle_description = (
                    f" throttle={throttle:.3f}"
                    f" alt_error={altitude_error:+.2f}m"
                    f" vertical_velocity={vertical_velocity:+.2f}m/s"
                    f" throttle_restoring={throttle_restoring}"
                )
            print(
                f"[PPO] control step={total_steps} reward={reward:+.3f} "
                f"rudder={rudder:+.3f} "
                f"angle_error={rudder_angle_error:+.2f}deg "
                f"yaw_rate={yaw_rate:+.2f}deg/s "
                f"rudder_restoring={restoring}"
                f"{throttle_description}"
            )
            self._write_scalar(
                "train/control/rudder_action",
                rudder,
                total_steps,
            )
            self._write_scalar(
                "train/control/restoring_action",
                float(restoring and throttle_restoring),
                total_steps,
            )
            self._write_scalar("train/control/reward", reward, total_steps)
            self._write_scalar(
                "train/state/rudder_angle_error_deg",
                rudder_angle_error,
                total_steps,
            )
            self._write_scalar(
                "train/state/abs_rudder_angle_error_deg",
                abs(rudder_angle_error),
                total_steps,
            )
            self._write_scalar(
                "train/state/yaw_rate_deg_s",
                yaw_rate,
                total_steps,
            )
            self._write_scalar(
                "train/state/abs_yaw_rate_deg_s",
                abs(yaw_rate),
                total_steps,
            )
            if combined_mode:
                self._write_scalar(
                    "train/control/throttle_action",
                    throttle,
                    total_steps,
                )
                self._write_scalar(
                    "train/state/altitude_error_m",
                    altitude_error,
                    total_steps,
                )
                self._write_scalar(
                    "train/state/vertical_velocity_mps",
                    vertical_velocity,
                    total_steps,
                )
            return
        if self.config.control_mode in {
            CONTROL_MODE_AILERON,
            CONTROL_MODE_AILERON_THROTTLE,
        }:
            features = info.get("aileron_hover_features", {})
            if not features:
                return
            combined_mode = (
                self.config.control_mode
                == CONTROL_MODE_AILERON_THROTTLE
            )
            roll_error = float(features.get("roll_error_deg", 0.0))
            roll_rate = float(features.get("roll_rate_deg_s", 0.0))
            aileron = float(env_action[0])
            trim_tensor = self.model.aileron_policy_trim
            trim = (
                float(trim_tensor.detach().cpu().item())
                if trim_tensor is not None
                else 0.0
            )
            correction = aileron - trim
            gains_tensor = self.model.aileron_policy_gain
            gains = (
                gains_tensor.detach().cpu().numpy()
                if gains_tensor is not None
                else np.ones(2, dtype=np.float32)
            )
            weighted_error = (
                float(gains[0])
                * roll_error
                / self.env.reward_config.roll_error_scale_deg
                + float(gains[1])
                * roll_rate
                / self.env.reward_config.roll_rate_scale_deg_s
            )
            aileron_restoring = (
                abs(weighted_error) < 1.0e-3
                or correction * weighted_error < 0.0
            )
            throttle_description = ""
            throttle_restoring = True
            if combined_mode:
                throttle_features = info.get(
                    "throttle_hover_features",
                    {},
                )
                altitude_error = float(
                    throttle_features.get("altitude_error_m", 0.0)
                )
                vertical_velocity = float(
                    throttle_features.get(
                        "vertical_velocity_mps",
                        0.0,
                    )
                )
                throttle = float(env_action[2])
                throttle_gains = self.model.throttle_policy_gain
                throttle_trim = self.model.throttle_policy_trim
                assert throttle_gains is not None
                assert throttle_trim is not None
                throttle_gain_values = (
                    throttle_gains.detach().cpu().numpy()
                )
                throttle_trim_value = float(
                    throttle_trim.detach().cpu().item()
                )
                throttle_weighted_error = (
                    float(throttle_gain_values[0])
                    * altitude_error
                    / self.env.reward_config.altitude_error_scale_m
                    + float(throttle_gain_values[1])
                    * vertical_velocity
                    / self.env.reward_config.velocity_error_scale_mps
                )
                throttle_restoring = (
                    abs(throttle_weighted_error) < 1.0e-3
                    or (
                        throttle - throttle_trim_value
                    )
                    * throttle_weighted_error
                    < 0.0
                )
                throttle_description = (
                    f" throttle={throttle:.3f}"
                    f" alt_error={altitude_error:+.2f}m"
                    f" vertical_velocity={vertical_velocity:+.2f}m/s"
                    f" throttle_restoring={throttle_restoring}"
                )
            print(
                f"[PPO] control step={total_steps} reward={reward:+.3f} "
                f"aileron={aileron:+.3f} trim={trim:+.3f} "
                f"correction={correction:+.3f} "
                f"roll_error={roll_error:+.2f}deg "
                f"roll_rate={roll_rate:+.2f}deg/s "
                f"aileron_restoring={aileron_restoring}"
                f"{throttle_description}"
            )
            self._write_scalar(
                "train/control/aileron_action",
                aileron,
                total_steps,
            )
            self._write_scalar(
                "train/control/restoring_action",
                float(aileron_restoring and throttle_restoring),
                total_steps,
            )
            self._write_scalar("train/control/reward", reward, total_steps)
            self._write_scalar(
                "train/state/roll_error_deg",
                roll_error,
                total_steps,
            )
            self._write_scalar(
                "train/state/abs_roll_error_deg",
                abs(roll_error),
                total_steps,
            )
            self._write_scalar(
                "train/state/roll_rate_deg_s",
                roll_rate,
                total_steps,
            )
            self._write_scalar(
                "train/state/abs_roll_rate_deg_s",
                abs(roll_rate),
                total_steps,
            )
            if combined_mode:
                self._write_scalar(
                    "train/control/throttle_action",
                    throttle,
                    total_steps,
                )
                self._write_scalar(
                    "train/state/altitude_error_m",
                    altitude_error,
                    total_steps,
                )
                self._write_scalar(
                    "train/state/vertical_velocity_mps",
                    vertical_velocity,
                    total_steps,
                )
            return
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
        combined_mode = (
            self.config.control_mode
            == CONTROL_MODE_ELEVATOR_THROTTLE
        )
        altitude_error = float(features.get("altitude_error_m", 0.0))
        upward_velocity = -float(
            features.get("vertical_velocity_mps", 0.0)
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
        lateral_error = float(info.get("lateral_position_error_m", 0.0))
        lateral_velocity = float(info.get("lateral_velocity_mps", 0.0))
        target_rudder = float(info.get("rudder_recovery_target_deg", 0.0))
        rudder_features = info.get("rudder_hover_features", {})
        rudder_error = (
            float(rudder_features.get("rudder_angle_error_deg", 0.0))
            if isinstance(rudder_features, dict)
            else 0.0
        )
        yaw_rate = (
            float(rudder_features.get("yaw_rate_deg_s", 0.0))
            if isinstance(rudder_features, dict)
            else 0.0
        )
        combined_description = (
            f"alt_error={altitude_error:+.2f}m "
            f"up_velocity={upward_velocity:+.2f}m/s "
            f"throttle={float(env_action[2]):.3f} "
            if combined_mode
            else ""
        )
        print(
            f"[PPO] control step={total_steps} reward={reward:+.3f} "
            f"elevator={float(env_action[1]):+.3f} "
            f"inc_error={inclination_error:+.2f}deg "
            f"target_inc={target_inclination_error:+.2f}deg "
            f"pitch_rate={pitch_rate:+.2f}deg/s "
            f"long_error={longitudinal_error:+.2f}m "
            f"long_velocity={longitudinal_velocity:+.2f}m/s "
            f"rudder={float(env_action[3]):+.3f} "
            f"rudder_error={rudder_error:+.2f}deg "
            f"target_rudder={target_rudder:+.2f}deg "
            f"yaw_rate={yaw_rate:+.2f}deg/s "
            f"lat_error={lateral_error:+.2f}m "
            f"lat_velocity={lateral_velocity:+.2f}m/s "
            f"{combined_description}"
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
        if combined_mode:
            self._write_scalar(
                "train/control/throttle_action",
                float(env_action[2]),
                total_steps,
            )
            self._write_scalar(
                "train/state/altitude_error_m",
                altitude_error,
                total_steps,
            )
            self._write_scalar(
                "train/state/vertical_velocity_mps",
                upward_velocity,
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
                        "episode_start_idle_seconds="
                        f"{self.config.episode_start_idle_seconds}",
                        "episode_start_idle_throttle="
                        f"{self.config.episode_start_idle_throttle}",
                        "episode_start_idle_curriculum_steps="
                        f"{self.config.episode_start_idle_curriculum_steps}",
                        "episode_start_idle_curriculum_start_seconds="
                        f"{self.config.episode_start_idle_curriculum_start_seconds}",
                        "episode_start_handoff_seconds="
                        f"{self.config.episode_start_handoff_seconds}",
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
            observation, info = self._reset_episode()
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
                        self.config.control_mode in {
                            CONTROL_MODE_ALL,
                            CONTROL_MODE_AILERON,
                            CONTROL_MODE_ELEVATOR,
                            CONTROL_MODE_RUDDER,
                            CONTROL_MODE_THROTTLE,
                            CONTROL_MODE_ELEVATOR_THROTTLE,
                            CONTROL_MODE_AILERON_THROTTLE,
                            CONTROL_MODE_RUDDER_THROTTLE,
                        }
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
                        if (
                            truncated
                            and self._episode_qualifies_for_curriculum_progress(
                                info
                            )
                        ):
                            self._advance_episode_start_idle_curriculum(
                                episode_length
                            )
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
                            observation, info = self._reset_episode()
                        else:
                            observation, info = self._start_after_truncation()
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
                self._write_aileron_recovery_probe(total_steps)
                self._write_elevator_recovery_probe(total_steps)
                self._write_rudder_recovery_probe(total_steps)
                self._write_throttle_recovery_probe(total_steps)
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
        self._evaluating = True
        rewards = []
        lengths = []
        termination_counts = Counter()
        position_errors = []
        altitude_errors = []
        attitude_errors = []
        idle_end_tilts = []
        observation = None
        for evaluation_index in range(self.config.eval_episodes):
            if observation is None:
                observation, start_info = self._reset_episode()
                self._log_episode_start(start_info)
                episode_start_idle = start_info.get("episode_start_idle", {})
                if isinstance(episode_start_idle, dict):
                    idle_end_tilts.append(
                        float(
                            episode_start_idle.get(
                                "control_start_tilt_deg",
                                0.0,
                            )
                        )
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
                    if self.config.control_mode in {
                        CONTROL_MODE_AILERON,
                        CONTROL_MODE_AILERON_THROTTLE,
                    }:
                        aileron_features = info.get(
                            "aileron_hover_features",
                            {},
                        )
                        attitude_errors.append(
                            abs(
                                float(
                                    aileron_features.get(
                                        "roll_error_deg",
                                        0.0,
                                    )
                                )
                            )
                        )
                    elif self.config.control_mode in {
                        CONTROL_MODE_RUDDER,
                        CONTROL_MODE_RUDDER_THROTTLE,
                    }:
                        rudder_features = info.get(
                            "rudder_hover_features",
                            {},
                        )
                        attitude_errors.append(
                            abs(
                                float(
                                    rudder_features.get(
                                        "rudder_angle_error_deg",
                                        0.0,
                                    )
                                )
                            )
                        )
                    elif self.config.control_mode == CONTROL_MODE_THROTTLE:
                        pass
                    elif self.config.control_mode in {
                        CONTROL_MODE_ELEVATOR,
                        CONTROL_MODE_ELEVATOR_THROTTLE,
                    }:
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
                        elevator_features = info.get(
                            "elevator_hover_features",
                            {},
                        )
                        aileron_features = info.get(
                            "aileron_hover_features",
                            {},
                        )
                        rudder_features = info.get(
                            "rudder_hover_features",
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
                            + abs(
                                float(
                                    aileron_features.get(
                                        "roll_error_deg",
                                        0.0,
                                    )
                                )
                            )
                            + abs(
                                float(
                                    rudder_features.get(
                                        "rudder_angle_error_deg",
                                        0.0,
                                    )
                                )
                            )
                        )
                if terminated or truncated:
                    reason = info.get("termination_reason") or ("truncated" if truncated else "unknown")
                    termination_counts[reason] += 1
                    if (
                        truncated
                        and evaluation_index + 1 < self.config.eval_episodes
                    ):
                        observation, start_info = (
                            self._start_after_truncation()
                        )
                        self._log_episode_start(start_info)
                        episode_start_idle = start_info.get(
                            "episode_start_idle",
                            {},
                        )
                        if isinstance(episode_start_idle, dict):
                            idle_end_tilts.append(
                                float(
                                    episode_start_idle.get(
                                        "control_start_tilt_deg",
                                        0.0,
                                    )
                                )
                            )
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
            f"reward_per_step={reward_per_step:.3f}, "
            f"position_error={float(np.mean(position_errors)) if position_errors else 0.0:.3f}m, "
            f"altitude_error={float(np.mean(altitude_errors)) if altitude_errors else 0.0:.3f}m, "
            f"attitude_error={float(np.mean(attitude_errors)) if attitude_errors else 0.0:.3f}deg, "
            f"idle_end_tilt={float(np.mean(idle_end_tilts)) if idle_end_tilts else 0.0:.3f}deg, "
            f"terminations={dict(termination_counts)}"
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
        self._evaluating = False
