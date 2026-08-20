from unittest.mock import Mock

import numpy as np

from hoverpilot.validate_env import parse_args, validate_environment, validation_action


def test_validation_defaults_to_safe_mode():
    assert parse_args([]).control_test is False


def test_safe_validation_action_uses_zero_throttle_idle_controls():
    env = Mock()

    action = validation_action(env, control_test=False)

    np.testing.assert_array_equal(action, np.zeros(4, dtype=np.float32))
    env.action_space.sample.assert_not_called()


def test_control_test_samples_the_environment_action_space():
    env = Mock()
    expected = np.asarray([0.2, -0.1, 0.5, 0.3], dtype=np.float32)
    env.action_space.sample.return_value = expected

    assert validation_action(env, control_test=True) is expected


def test_validation_returns_failure_and_closes_environment(monkeypatch):
    env = Mock()
    env.reset.side_effect = TimeoutError("not ready")
    monkeypatch.setattr("hoverpilot.validate_env.HoverPilotHoverEnv", lambda **_: env)

    result = validate_environment("127.0.0.1", 18083, episodes=1)

    assert result == 1
    env.close.assert_called_once_with()
