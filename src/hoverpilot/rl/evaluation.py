"""Shared deterministic policy evaluation metrics and reporting."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Mapping

import numpy as np

from .constants import (
    CONTROL_MODE_AILERON,
    CONTROL_MODE_AILERON_THROTTLE,
    CONTROL_MODE_ELEVATOR,
    CONTROL_MODE_ELEVATOR_THROTTLE,
    CONTROL_MODE_RUDDER,
    CONTROL_MODE_RUDDER_THROTTLE,
    CONTROL_MODE_THROTTLE,
)


@dataclass(frozen=True)
class EvaluationResult:
    episodes: int
    mean_reward: float
    mean_episode_steps: float
    mean_survival_time_s: float
    reward_per_step: float
    mean_position_error_m: float
    mean_altitude_error_m: float
    mean_attitude_error_deg: float
    termination_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class EvaluationAccumulator:
    control_mode: str
    rewards: list[float] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    survival_times_s: list[float] = field(default_factory=list)
    position_errors: list[float] = field(default_factory=list)
    altitude_errors: list[float] = field(default_factory=list)
    attitude_errors: list[float] = field(default_factory=list)
    termination_counts: Counter[str] = field(default_factory=Counter)
    _episode_start_physics_time_s: float | None = None
    _episode_last_physics_time_s: float | None = None

    def record_step(self, info: Mapping[str, object]) -> None:
        debug = info.get("debug_state", {})
        target = info.get("target_hover", {})
        if not isinstance(debug, Mapping):
            return
        physics_time = debug.get("physics_time_s")
        if isinstance(physics_time, (int, float)) and math.isfinite(physics_time):
            if self._episode_start_physics_time_s is None:
                self._episode_start_physics_time_s = float(physics_time)
            self._episode_last_physics_time_s = float(physics_time)
        if isinstance(target, Mapping) and debug:
            dx = float(debug.get("x_m", 0.0)) - float(target.get("x_m", 0.0))
            dy = float(debug.get("y_m", 0.0)) - float(target.get("y_m", 0.0))
            self.position_errors.append(math.hypot(dx, dy))
            self.altitude_errors.append(
                abs(
                    float(debug.get("altitude_agl_m", 0.0))
                    - float(target.get("altitude_agl_m", 0.0))
                )
            )
        if self.control_mode in {CONTROL_MODE_AILERON, CONTROL_MODE_AILERON_THROTTLE}:
            features = info.get("aileron_hover_features", {})
            if isinstance(features, Mapping):
                self.attitude_errors.append(
                    abs(float(features.get("roll_error_deg", 0.0)))
                )
        elif self.control_mode in {CONTROL_MODE_RUDDER, CONTROL_MODE_RUDDER_THROTTLE}:
            features = info.get("rudder_hover_features", {})
            if isinstance(features, Mapping):
                self.attitude_errors.append(
                    abs(float(features.get("rudder_angle_error_deg", 0.0)))
                )
        elif self.control_mode in {
            CONTROL_MODE_ELEVATOR,
            CONTROL_MODE_ELEVATOR_THROTTLE,
        }:
            features = info.get("elevator_hover_features", {})
            if isinstance(features, Mapping):
                self.attitude_errors.append(
                    abs(float(features.get("inclination_error_deg", 0.0)))
                )
        elif self.control_mode != CONTROL_MODE_THROTTLE:
            pitch = info.get("elevator_hover_features", {})
            roll = info.get("aileron_hover_features", {})
            yaw = info.get("rudder_hover_features", {})
            if all(isinstance(value, Mapping) for value in (pitch, roll, yaw)):
                self.attitude_errors.append(
                    abs(float(pitch.get("inclination_error_deg", 0.0)))
                    + abs(float(roll.get("roll_error_deg", 0.0)))
                    + abs(float(yaw.get("rudder_angle_error_deg", 0.0)))
                )

    def finish_episode(self, reward: float, steps: int, reason: str) -> None:
        self.rewards.append(float(reward))
        self.lengths.append(steps)
        self.termination_counts[reason] += 1
        if (
            self._episode_start_physics_time_s is not None
            and self._episode_last_physics_time_s is not None
        ):
            self.survival_times_s.append(
                max(
                    0.0,
                    self._episode_last_physics_time_s
                    - self._episode_start_physics_time_s,
                )
            )
        else:
            self.survival_times_s.append(0.0)
        self._episode_start_physics_time_s = None
        self._episode_last_physics_time_s = None

    def result(self) -> EvaluationResult:
        total_steps = sum(self.lengths)
        return EvaluationResult(
            episodes=len(self.rewards),
            mean_reward=float(np.mean(self.rewards)) if self.rewards else 0.0,
            mean_episode_steps=float(np.mean(self.lengths)) if self.lengths else 0.0,
            mean_survival_time_s=float(np.mean(self.survival_times_s))
            if self.survival_times_s
            else 0.0,
            reward_per_step=float(sum(self.rewards) / max(1, total_steps)),
            mean_position_error_m=float(np.mean(self.position_errors))
            if self.position_errors
            else 0.0,
            mean_altitude_error_m=float(np.mean(self.altitude_errors))
            if self.altitude_errors
            else 0.0,
            mean_attitude_error_deg=float(np.mean(self.attitude_errors))
            if self.attitude_errors
            else 0.0,
            termination_counts=dict(self.termination_counts),
        )


def format_evaluation(result: EvaluationResult) -> str:
    return (
        f"mean_reward={result.mean_reward:.3f} "
        f"mean_steps={result.mean_episode_steps:.1f} "
        f"survival_time={result.mean_survival_time_s:.3f}s "
        f"reward_per_step={result.reward_per_step:.3f} "
        f"position_error={result.mean_position_error_m:.3f}m "
        f"altitude_error={result.mean_altitude_error_m:.3f}m "
        f"attitude_error={result.mean_attitude_error_deg:.3f}deg "
        f"terminations={result.termination_counts}"
    )
