"""Deterministic execution of saved PPO policies in RealFlight."""

from __future__ import annotations

import os
from typing import Dict

import gymnasium as gym
import numpy as np
import torch

from hoverpilot.training.hover import RewardConfig
from hoverpilot.utils.logger import format_debug_state

from .checkpoints import _apply_observation_config, load_policy_checkpoint
from .config import PPOPlayConfig
from .constants import (
    CONNECTION_EPISODE_CONTROL_MODES,
    CONTROL_MODE_AILERON_THROTTLE,
    CONTROL_MODE_ALL,
    CONTROL_MODE_ELEVATOR_THROTTLE,
    CONTROL_MODE_RUDDER_THROTTLE,
    CONTROL_MODE_THROTTLE,
)
from .evaluation import EvaluationAccumulator, EvaluationResult, format_evaluation
from .models import ActorCritic
from .runtime import (
    _build_hover_env,
    _expand_policy_action,
    _initial_env_action,
    _policy_action_space,
    _validate_rflink_settings,
    continue_env_after_truncation,
    reset_env_with_wait,
    resolve_device,
)


class PPOPlayer:
    def __init__(self, config: PPOPlayConfig):
        if config.episodes < 0:
            raise ValueError(
                "episodes must be non-negative; use 0 to run until interrupted"
            )
        if config.log_interval_steps < 0:
            raise ValueError("log_interval_steps must be non-negative")
        _validate_rflink_settings(config)

        self.config = config
        self.device = resolve_device(config.device)
        checkpoint = load_policy_checkpoint(config.checkpoint_path)
        self.control_mode = checkpoint.control_mode
        self.policy_preset = checkpoint.policy_preset
        self.elevator_fixed_throttle = checkpoint.elevator_fixed_throttle
        self.reward_config = _apply_observation_config(
            RewardConfig(),
            checkpoint.observation_config,
        )
        self.env = self._build_env()
        self.policy_action_space = _policy_action_space(
            self.control_mode,
            self.env.action_space,
        )
        observation_dim = int(np.prod(self.env.observation_space.shape))
        self.model = ActorCritic(
            observation_dim,
            self.policy_action_space.low,
            self.policy_action_space.high,
            policy_preset=self.policy_preset,
            control_mode=self.control_mode,
        ).to(self.device)
        try:
            incompatible = self.model.load_state_dict(
                checkpoint.model_state_dict,
                strict=False,
            )
        except RuntimeError as exc:
            self.env.close()
            raise ValueError(
                "PPO checkpoint is incompatible with the current HoverPilot policy architecture: "
                f"{exc}"
            ) from exc
        allowed_missing = {"policy_mean.weight", "policy_mean.bias"}
        if (
            set(incompatible.missing_keys) - allowed_missing
            or incompatible.unexpected_keys
        ):
            self.env.close()
            raise ValueError(
                "PPO checkpoint is incompatible with the current HoverPilot "
                "policy architecture: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        self.model.eval()

    def _build_env(self) -> gym.Env:
        return _build_hover_env(
            self.config,
            self.control_mode,
            reward_config=self.reward_config,
        )

    def _to_env_action(self, policy_action: np.ndarray) -> np.ndarray:
        return _expand_policy_action(
            policy_action,
            self.policy_action_space,
            self.control_mode,
            self.elevator_fixed_throttle,
        )

    def _initial_action(self) -> np.ndarray:
        return _initial_env_action(
            self.control_mode,
            self.elevator_fixed_throttle,
            self.config.initial_action,
        )

    def _wait_action(self) -> np.ndarray:
        if self.control_mode in {
            CONTROL_MODE_ALL,
            CONTROL_MODE_THROTTLE,
            CONTROL_MODE_ELEVATOR_THROTTLE,
            CONTROL_MODE_AILERON_THROTTLE,
            CONTROL_MODE_RUDDER_THROTTLE,
        }:
            return self._initial_action()
        return np.asarray(self.config.wait_action, dtype=np.float32)

    def _format_control_state(self, info: Dict) -> str:
        if self.control_mode != CONTROL_MODE_ALL:
            return ""
        pitch = info.get("elevator_hover_features", {})
        roll = info.get("aileron_hover_features", {})
        yaw = info.get("rudder_hover_features", {})
        height = info.get("throttle_hover_features", {})
        return (
            " "
            f"long=({float(pitch.get('longitudinal_position_error_m', 0.0)):+.2f}m,"
            f"{float(pitch.get('longitudinal_velocity_mps', 0.0)):+.2f}m/s) "
            f"inc=({float(pitch.get('inclination_error_deg', 0.0)):+.1f},"
            f"{float(info.get('elevator_recovery_target_deg', 0.0)):+.1f})deg "
            f"lat=({float(info.get('lateral_position_error_m', 0.0)):+.2f}m,"
            f"{float(info.get('lateral_velocity_mps', 0.0)):+.2f}m/s) "
            f"roll=({float(roll.get('roll_error_deg', 0.0)):+.1f},"
            f"{float(roll.get('roll_rate_deg_s', 0.0)):+.1f}) "
            f"yaw=({float(yaw.get('rudder_angle_error_deg', 0.0)):+.1f},"
            f"{float(info.get('rudder_recovery_target_deg', 0.0)):+.1f},"
            f"{float(yaw.get('yaw_rate_deg_s', 0.0)):+.1f}) "
            f"alt=({float(height.get('altitude_error_m', 0.0)):+.2f}m,"
            f"{float(height.get('vertical_velocity_mps', 0.0)):+.2f}m/s)"
        )

    def play(self):
        completed_episodes = 0
        total_steps = 0
        checkpoint_path = os.path.abspath(
            os.path.expanduser(self.config.checkpoint_path)
        )
        episode_limit = (
            "unlimited" if self.config.episodes == 0 else str(self.config.episodes)
        )
        throttle_description = (
            "throttle=policy"
            if self.control_mode
            in {
                CONTROL_MODE_ALL,
                CONTROL_MODE_THROTTLE,
                CONTROL_MODE_ELEVATOR_THROTTLE,
                CONTROL_MODE_AILERON_THROTTLE,
                CONTROL_MODE_RUDDER_THROTTLE,
            }
            else f"fixed_throttle={self.elevator_fixed_throttle:.3f}"
        )
        print(
            f"[PLAY] Loaded checkpoint={checkpoint_path} device={self.device.type} "
            f"control_mode={self.control_mode} policy_preset={self.policy_preset} "
            f"{throttle_description} "
            f"episodes={episode_limit}"
        )

        try:
            next_segment = None
            while (
                self.config.episodes == 0 or completed_episodes < self.config.episodes
            ):
                if next_segment is None:
                    observation, info = reset_env_with_wait(
                        self.env,
                        action=self._wait_action(),
                        initial_action=self._initial_action(),
                    )
                else:
                    observation, info = next_segment
                    next_segment = None
                episode_reward = 0.0
                episode_steps = 0
                print(
                    f"[PLAY] episode={completed_episodes + 1} start "
                    f"reason={info.get('episode_start_reason')}"
                )

                while True:
                    obs_tensor = torch.as_tensor(
                        observation,
                        dtype=torch.float32,
                        device=self.device,
                    ).unsqueeze(0)
                    with torch.inference_mode():
                        action_tensor = self.model.deterministic_action(obs_tensor)
                    policy_action = action_tensor.squeeze(0).cpu().numpy()
                    env_action = self._to_env_action(policy_action)
                    observation, reward, terminated, truncated, info = self.env.step(
                        env_action
                    )
                    total_steps += 1
                    episode_steps += 1
                    episode_reward += float(reward)

                    if (
                        self.config.log_interval_steps > 0
                        and total_steps % self.config.log_interval_steps == 0
                    ):
                        print(
                            f"[PLAY] step={total_steps} episode_step={episode_steps} "
                            f"reward={float(reward):+.3f} action={env_action.tolist()} "
                            f"state={format_debug_state(info.get('debug_state'))}"
                            f"{self._format_control_state(info)}"
                        )

                    if terminated or truncated:
                        reason = info.get("termination_reason") or (
                            "truncated" if truncated else "unknown"
                        )
                        completed_episodes += 1
                        print(
                            f"[PLAY] episode={completed_episodes} end steps={episode_steps} "
                            f"reward={episode_reward:+.3f} reason={reason}"
                        )
                        if (
                            truncated
                            and self.control_mode
                            not in CONNECTION_EPISODE_CONTROL_MODES
                        ):
                            next_segment = continue_env_after_truncation(self.env)
                        break
        except KeyboardInterrupt:
            print(
                f"\n[PLAY] Interrupted after episodes={completed_episodes} steps={total_steps}"
            )
        finally:
            self.env.close()

    def evaluate(self) -> EvaluationResult:
        """Run deterministic episodes and return comparable hover metrics."""

        if self.config.episodes <= 0:
            raise ValueError("evaluation requires a positive episode count")
        accumulator = EvaluationAccumulator(self.control_mode)
        next_segment = None
        try:
            for _ in range(self.config.episodes):
                if next_segment is None:
                    observation, _ = reset_env_with_wait(
                        self.env,
                        action=self._wait_action(),
                        initial_action=self._initial_action(),
                    )
                else:
                    observation, _ = next_segment
                    next_segment = None
                episode_reward = 0.0
                episode_steps = 0
                while True:
                    observation_tensor = torch.as_tensor(
                        observation, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                    with torch.inference_mode():
                        policy_action = (
                            self.model.deterministic_action(observation_tensor)
                            .squeeze(0)
                            .cpu()
                            .numpy()
                        )
                    observation, reward, terminated, truncated, info = self.env.step(
                        self._to_env_action(policy_action)
                    )
                    episode_reward += float(reward)
                    episode_steps += 1
                    accumulator.record_step(info)
                    if terminated or truncated:
                        reason = info.get("termination_reason") or (
                            "truncated" if truncated else "unknown"
                        )
                        accumulator.finish_episode(
                            episode_reward, episode_steps, str(reason)
                        )
                        if (
                            truncated
                            and self.control_mode
                            not in CONNECTION_EPISODE_CONTROL_MODES
                        ):
                            next_segment = continue_env_after_truncation(self.env)
                        break
            result = accumulator.result()
            print(f"[EVAL] {format_evaluation(result)}")
            return result
        finally:
            self.env.close()
