from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

from hoverpilot.envs import ELEVATOR_HOVER_TASK, HoverPilotHoverEnv
from hoverpilot.rflink.client import RFLinkClient
from hoverpilot.utils.logger import format_debug_state


WAITING_LOG_INTERVAL_S = 0.75
MAX_CONSERVATIVE_PULSE = 0.25
INITIAL_PITCH_RATE_TOLERANCE_DEG_S = 5.0
INITIAL_INCLINATION_TOLERANCE_DEG = 2.0
Sample = Tuple[float, float, float]  # physics time, pitch rate, inclination


def diagnose_elevator_response(
    host: str,
    port: int,
    *,
    elevator_fixed_throttle: float = 0.55,
    pulse: float = 0.1,
    pulse_steps: int = 8,
    settle_steps: int = 8,
    rflink_socket_timeout_s: float = 3.0,
    rflink_request_attempts: int = 4,
    rflink_retry_backoff_s: float = 0.1,
    env_factory: Optional[Callable[[], Any]] = None,
    output: Callable[[str], None] = print,
) -> Dict[str, float]:
    """Measure pitch response to small positive and negative elevator pulses."""

    if (
        not math.isfinite(elevator_fixed_throttle)
        or not 0.0 <= elevator_fixed_throttle <= 1.0
    ):
        raise ValueError("elevator_fixed_throttle must be in [0, 1]")
    if not math.isfinite(pulse) or not 0.0 < pulse <= MAX_CONSERVATIVE_PULSE:
        raise ValueError(
            f"pulse must be in (0, {MAX_CONSERVATIVE_PULSE}] "
            "for a conservative diagnostic"
        )
    if pulse_steps < 2:
        raise ValueError("pulse_steps must be at least 2")
    if settle_steps < 0:
        raise ValueError("settle_steps must be non-negative")

    if env_factory is None:
        env_factory = lambda: _make_env(
            host,
            port,
            rflink_socket_timeout_s,
            rflink_request_attempts,
            rflink_retry_backoff_s,
        )

    neutral = _action(0.0, elevator_fixed_throttle)
    env = env_factory()
    try:
        _, start_info = _reset_env(env, neutral, output)
        reset_sample = _sample(start_info, "diagnostic reset")

        positive_settle = _run(
            env, neutral, settle_steps, "before positive pulse"
        )
        positive_baseline = positive_settle[-1] if positive_settle else reset_sample
        positive_samples = [positive_baseline] + _run(
            env,
            _action(pulse, elevator_fixed_throttle),
            pulse_steps,
            "positive pulse",
        )

        negative_settle = _run(
            env, neutral, settle_steps, "before negative pulse"
        )
        negative_baseline = (
            negative_settle[-1] if negative_settle else positive_samples[-1]
        )
        negative_samples = [negative_baseline] + _run(
            env,
            _action(-pulse, elevator_fixed_throttle),
            pulse_steps,
            "negative pulse",
        )

        positive_acceleration = _slope(positive_samples)
        negative_acceleration = _slope(negative_samples)
        pitch_rate_delta = abs(positive_baseline[1] - negative_baseline[1])
        inclination_delta = abs(positive_baseline[2] - negative_baseline[2])
        comparable = (
            pitch_rate_delta <= INITIAL_PITCH_RATE_TOLERANCE_DEG_S
            and inclination_delta <= INITIAL_INCLINATION_TOLERANCE_DEG
        )
        opposite = positive_acceleration * negative_acceleration < 0.0

        result = {
            "pulse": pulse,
            "positive_pitch_acceleration_deg_s2": positive_acceleration,
            "negative_pitch_acceleration_deg_s2": negative_acceleration,
            "positive_final_inclination_deg": positive_samples[-1][2],
            "negative_final_inclination_deg": negative_samples[-1][2],
            "initial_pitch_rate_difference_deg_s": pitch_rate_delta,
            "initial_inclination_difference_deg": inclination_delta,
            "initial_conditions_comparable": float(comparable),
            "opposite_sign_response": float(opposite),
        }

        output(
            f"[DIAG] start {format_debug_state(start_info.get('debug_state'))}"
        )
        output(
            f"[DIAG] elevator=+{pulse:.3f} "
            f"pitch_accel={positive_acceleration:+.2f}deg/s^2 "
            f"final_inc={positive_samples[-1][2]:+.2f}deg"
        )
        output(
            f"[DIAG] elevator=-{pulse:.3f} "
            f"pitch_accel={negative_acceleration:+.2f}deg/s^2 "
            f"final_inc={negative_samples[-1][2]:+.2f}deg"
        )
        if not comparable:
            output(
                "[DIAG] WARNING: pulse initial conditions differ "
                f"(pitch_rate_delta={pitch_rate_delta:.2f}deg/s, "
                f"inclination_delta={inclination_delta:.2f}deg); "
                "reset the trainer and repeat before relying on polarity."
            )
        if not opposite:
            output(
                "[DIAG] WARNING: pulse responses do not have opposite signs; "
                "repeat after a trainer reset before training."
            )
        else:
            direction = "increases" if positive_acceleration > 0.0 else "decreases"
            output(
                f"[DIAG] Polarity verified: positive elevator "
                f"{direction} pitch rate."
            )
        return result
    finally:
        # The connection may already be broken, so neutral hand-off is best effort.
        try:
            env.step(neutral)
        except Exception:
            pass
        env.close()


def _make_env(
    host: str,
    port: int,
    socket_timeout_s: float,
    request_attempts: int,
    retry_backoff_s: float,
) -> HoverPilotHoverEnv:
    return HoverPilotHoverEnv(
        host=host,
        port=port,
        max_episode_steps=None,
        task_profile=ELEVATOR_HOVER_TASK,
        client_factory=lambda: RFLinkClient(
            host,
            port,
            socket_timeout_s=socket_timeout_s,
            request_attempts=request_attempts,
            retry_backoff_s=retry_backoff_s,
        ),
    )


def _action(elevator: float, throttle: float) -> np.ndarray:
    return np.asarray([0.0, elevator, throttle, 0.0], dtype=np.float32)


def _run(
    env: Any,
    action: np.ndarray,
    steps: int,
    phase: str,
) -> List[Sample]:
    samples = []
    for _ in range(steps):
        _, _, terminated, truncated, info = env.step(action)
        samples.append(_sample(info, phase))
        if terminated or truncated:
            reason = info.get("termination_reason") or (
                "time_limit" if truncated else "unknown"
            )
            raise RuntimeError(
                f"elevator diagnostic ended early during {phase} ({reason}); "
                "reset the trainer and retry"
            )
    return samples


def _sample(info: Mapping[str, Any], context: str) -> Sample:
    try:
        debug = info["debug_state"]
        sample = (
            float(debug["physics_time_s"]),
            float(debug["pitch_rate_deg_s"]),
            float(debug["inclination_deg"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{context} is missing numeric elevator telemetry"
        ) from exc
    return sample


def _slope(samples: List[Sample]) -> float:
    times = np.asarray([sample[0] for sample in samples], dtype=np.float64)
    rates = np.asarray([sample[1] for sample in samples], dtype=np.float64)
    elapsed = times - times[0]
    if elapsed[-1] <= 1.0e-6:
        raise RuntimeError("physics time did not advance during elevator diagnostic")
    return float(np.polyfit(elapsed, rates, 1)[0])


def _reset_env(
    env: Any,
    neutral: np.ndarray,
    output: Callable[[str], None],
) -> Tuple[Any, Mapping[str, Any]]:
    if not getattr(env, "_waiting_for_reset", False):
        try:
            return env.reset(options={"initial_action": neutral})
        except TimeoutError as exc:
            output(f"waiting for trainer reset before diagnostic | {exc}")

    poll = getattr(env, "poll_wait_for_next_episode", None)
    if not callable(poll):
        raise RuntimeError(
            "environment cannot poll while waiting for trainer reset"
        )
    last_log_at = 0.0
    while True:
        started, observation, info = poll(action=neutral)
        if started:
            return observation, info
        now = time.monotonic()
        if now - last_log_at >= WAITING_LOG_INTERVAL_S:
            output(
                "waiting for trainer reset | "
                f"{format_debug_state(info.get('debug_state'))}"
            )
            last_log_at = now
