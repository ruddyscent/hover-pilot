from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hoverpilot.rl.config import PPOConfig
from hoverpilot.rl.trainer import PPOTrainer
from hoverpilot.training.hover import RewardConfig


def test_full_training_checkpoint_restores_run_state(tmp_path: Path):
    checkpoint_path = tmp_path / "resume.pt"
    original = PPOTrainer(
        PPOConfig(
            save_path=str(checkpoint_path),
            max_episode_steps=77,
            reward_config=RewardConfig(target_x_m=3.5),
            seed=123,
            tensorboard_log_dir=None,
        )
    )
    loss = sum(parameter.square().sum() for parameter in original.model.parameters())
    original.optimizer.zero_grad()
    loss.backward()
    original.optimizer.step()
    original.scheduler.step()
    original.evaluation_history = [
        {"step": 100, "mean_reward": 12.5, "termination_counts": {"time_limit": 2}}
    ]
    original.best_mean_reward = 12.5
    original._save_model(step=128, reason="test resume")

    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    resumed = PPOTrainer(
        PPOConfig(
            resume_from=str(checkpoint_path),
            save_path=str(tmp_path / "continued.pt"),
            max_episode_steps=300,
            reward_config=RewardConfig(),
            seed=999,
            tensorboard_log_dir=None,
        )
    )

    assert resumed.training_step == 128
    assert resumed.scheduler.state_dict() == saved["scheduler_state_dict"]
    assert resumed.evaluation_history == saved["evaluation_history"]
    assert resumed.best_mean_reward == 12.5
    assert resumed.config.max_episode_steps == 77
    assert resumed.config.reward_config.target_x_m == 3.5
    assert torch.equal(torch.get_rng_state(), saved["rng_state"]["torch"])
    numpy_state = np.random.get_state()
    assert numpy_state[0] == saved["rng_state"]["numpy"]["bit_generator"]
    assert np.array_equal(
        numpy_state[1], saved["rng_state"]["numpy"]["state"].numpy()
    )
    assert resumed.optimizer.state_dict()["state"]

    original.env.close()
    resumed.env.close()
