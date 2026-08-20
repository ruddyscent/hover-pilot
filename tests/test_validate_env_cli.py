from unittest.mock import Mock

import numpy as np

from hoverpilot.validate_env import parse_args, validation_action


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
