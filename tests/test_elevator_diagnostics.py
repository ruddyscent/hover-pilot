import unittest

import numpy as np

from hoverpilot.rl.elevator_diagnostics import diagnose_elevator_response


class FakeDiagnosticEnv:
    def __init__(
        self,
        *,
        mismatched_second_baseline=False,
        terminate_on_positive=False,
        truncate_on_positive=False,
        advance_physics_time=True,
    ):
        self.mismatched_second_baseline = mismatched_second_baseline
        self.terminate_on_positive = terminate_on_positive
        self.truncate_on_positive = truncate_on_positive
        self.advance_physics_time = advance_physics_time
        self.actions = []
        self.reset_options = None
        self.closed = False
        self.physics_time_s = 10.0
        self._last_pulse_sign = 0
        self._pulse_step = 0
        self._positive_seen = False

    def reset(self, *, options=None):
        self.reset_options = options
        return np.zeros(6, dtype=np.float32), self._info(
            pitch_rate_deg_s=0.0,
            inclination_deg=90.0,
        )

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).copy()
        self.actions.append(action)
        if self.advance_physics_time:
            self.physics_time_s += 0.1

        elevator = float(action[1])
        sign = 1 if elevator > 0.0 else -1 if elevator < 0.0 else 0
        if sign == 0:
            self._last_pulse_sign = 0
            self._pulse_step = 0
            if self.mismatched_second_baseline and self._positive_seen:
                pitch_rate = 12.0
                inclination = 86.0
            else:
                pitch_rate = 0.0
                inclination = 90.0
        else:
            if sign != self._last_pulse_sign:
                self._pulse_step = 0
            self._last_pulse_sign = sign
            self._pulse_step += 1
            if sign > 0:
                self._positive_seen = True
                baseline_rate = 0.0
                baseline_inclination = 90.0
            elif self.mismatched_second_baseline:
                baseline_rate = 12.0
                baseline_inclination = 86.0
            else:
                baseline_rate = 0.0
                baseline_inclination = 90.0
            pitch_rate = baseline_rate + sign * self._pulse_step
            inclination = baseline_inclination + sign * 0.1 * self._pulse_step

        terminated = (
            self.terminate_on_positive
            and sign > 0
            and self._pulse_step == 1
        )
        truncated = (
            self.truncate_on_positive
            and sign > 0
            and self._pulse_step == 1
        )
        info = self._info(
            pitch_rate_deg_s=pitch_rate,
            inclination_deg=inclination,
        )
        if terminated:
            info["termination_reason"] = "altitude_too_low"
        return (
            np.zeros(6, dtype=np.float32),
            0.0,
            terminated,
            truncated,
            info,
        )

    def close(self):
        self.closed = True

    def _info(self, *, pitch_rate_deg_s, inclination_deg):
        return {
            "debug_state": {
                "physics_time_s": self.physics_time_s,
                "pitch_rate_deg_s": pitch_rate_deg_s,
                "inclination_deg": inclination_deg,
            }
        }


class ElevatorDiagnosticsTests(unittest.TestCase):
    def test_rejects_unsafe_or_invalid_parameters_before_creating_env(self):
        def unexpected_factory():
            raise AssertionError("environment should not be created")

        invalid_cases = (
            {"elevator_fixed_throttle": -0.1},
            {"elevator_fixed_throttle": 1.1},
            {"pulse": 0.0},
            {"pulse": 0.251},
            {"pulse_steps": 1},
            {"settle_steps": -1},
        )
        for overrides in invalid_cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    diagnose_elevator_response(
                        "unused",
                        0,
                        env_factory=unexpected_factory,
                        output=lambda _: None,
                        **overrides,
                    )

    def test_measures_opposite_pulse_polarity_from_neutral_baselines(self):
        env = FakeDiagnosticEnv()
        messages = []

        result = diagnose_elevator_response(
            "unused",
            0,
            elevator_fixed_throttle=0.6,
            pulse=0.1,
            pulse_steps=3,
            settle_steps=2,
            env_factory=lambda: env,
            output=messages.append,
        )

        self.assertAlmostEqual(
            result["positive_pitch_acceleration_deg_s2"],
            10.0,
        )
        self.assertAlmostEqual(
            result["negative_pitch_acceleration_deg_s2"],
            -10.0,
        )
        self.assertEqual(result["opposite_sign_response"], 1.0)
        self.assertEqual(result["initial_conditions_comparable"], 1.0)
        self.assertEqual(result["initial_pitch_rate_difference_deg_s"], 0.0)
        self.assertEqual(result["initial_inclination_difference_deg"], 0.0)
        np.testing.assert_allclose(
            env.reset_options["initial_action"],
            np.asarray([0.0, 0.0, 0.6, 0.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            [action[1] for action in env.actions],
            [0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, -0.1, -0.1, -0.1, 0.0],
            atol=1.0e-6,
        )
        self.assertTrue(
            any("Polarity verified" in message for message in messages)
        )
        self.assertTrue(env.closed)

    def test_reports_when_pulses_start_from_different_conditions(self):
        env = FakeDiagnosticEnv(mismatched_second_baseline=True)
        messages = []

        result = diagnose_elevator_response(
            "unused",
            0,
            pulse_steps=3,
            settle_steps=2,
            env_factory=lambda: env,
            output=messages.append,
        )

        self.assertEqual(result["initial_conditions_comparable"], 0.0)
        self.assertEqual(
            result["initial_pitch_rate_difference_deg_s"],
            12.0,
        )
        self.assertEqual(
            result["initial_inclination_difference_deg"],
            4.0,
        )
        self.assertTrue(
            any(
                "initial conditions differ" in message
                for message in messages
            )
        )
        self.assertTrue(env.closed)

    def test_early_episode_end_still_sends_neutral_and_closes(self):
        for end_kind in ("terminated", "truncated"):
            with self.subTest(end_kind=end_kind):
                env = FakeDiagnosticEnv(
                    terminate_on_positive=end_kind == "terminated",
                    truncate_on_positive=end_kind == "truncated",
                )

                expected_reason = (
                    "altitude_too_low"
                    if end_kind == "terminated"
                    else "time_limit"
                )
                with self.assertRaisesRegex(RuntimeError, expected_reason):
                    diagnose_elevator_response(
                        "unused",
                        0,
                        pulse_steps=2,
                        settle_steps=1,
                        env_factory=lambda: env,
                        output=lambda _: None,
                    )

                self.assertEqual(float(env.actions[-1][1]), 0.0)
                self.assertTrue(env.closed)

    def test_static_physics_time_fails_and_cleans_up(self):
        env = FakeDiagnosticEnv(advance_physics_time=False)

        with self.assertRaisesRegex(
            RuntimeError,
            "physics time did not advance",
        ):
            diagnose_elevator_response(
                "unused",
                0,
                pulse_steps=2,
                settle_steps=1,
                env_factory=lambda: env,
                output=lambda _: None,
            )

        self.assertEqual(float(env.actions[-1][1]), 0.0)
        self.assertTrue(env.closed)


if __name__ == "__main__":
    unittest.main()
