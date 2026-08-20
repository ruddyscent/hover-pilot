import random

import numpy as np
import torch

from hoverpilot.rl.config import PPOConfig
from hoverpilot.rl.training_state import (
    capture_rng_state,
    checkpoint_environment_config,
    restore_rng_state,
)


def test_rng_state_round_trip_restores_all_cpu_generators():
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    state = capture_rng_state()

    expected = (random.random(), np.random.random(), torch.rand(1).item())
    random.random()
    np.random.random()
    torch.rand(1)

    restore_rng_state(state)

    actual = (random.random(), np.random.random(), torch.rand(1).item())
    assert actual == expected


def test_checkpoint_environment_config_contains_only_resume_sensitive_settings():
    config = PPOConfig(
        max_episode_steps=123,
        sleep_interval_s=0.25,
        episode_start_idle_seconds=2.0,
        episode_start_idle_throttle=0.61,
        episode_start_idle_curriculum_steps=500,
        episode_start_idle_curriculum_start_seconds=0.5,
        episode_start_handoff_seconds=0.2,
    )

    assert checkpoint_environment_config(config) == {
        "max_episode_steps": 123,
        "sleep_interval_s": 0.25,
        "episode_start_idle_seconds": 2.0,
        "episode_start_idle_throttle": 0.61,
        "episode_start_idle_curriculum_steps": 500,
        "episode_start_idle_curriculum_start_seconds": 0.5,
        "episode_start_handoff_seconds": 0.2,
    }
