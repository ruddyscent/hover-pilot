from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hoverpilot.rl.config import PPOConfig
from hoverpilot.rl.constants import CONTROL_MODE_ELEVATOR
from hoverpilot.rl.trainer import PPOTrainer


class EndlessElevatorEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=np.asarray([-1.0, -1.0, 0.0, -1.0], dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
            dtype=np.float32,
        )
        self.reset_calls = 0
        self.closed = False

    def reset(self, *, seed=None, options=None):
        del seed, options
        self.reset_calls += 1
        return np.zeros(6, dtype=np.float32), {
            "episode_start_reason": "test_reset",
            "waiting_for_reset": False,
        }

    def step(self, action):
        del action
        return np.zeros(6, dtype=np.float32), 1.0, False, False, {
            "termination_reason": None
        }

    def close(self):
        self.closed = True


class WorkflowTrainer(PPOTrainer):
    def __init__(self, config, evaluation_scores):
        self.evaluation_scores = iter(evaluation_scores)
        self.saved = []
        super().__init__(config)

    def _build_env(self):
        return EndlessElevatorEnv()

    def _evaluate_policy(self, step=None):
        del step
        return {
            "mean_reward": float(next(self.evaluation_scores)),
            "mean_episode_steps": 5.0,
            "mean_survival_time_s": 1.0,
            "reward_per_step": 0.2,
            "mean_position_error_m": 1.0,
            "mean_altitude_error_m": 0.5,
            "mean_attitude_error_deg": 2.0,
            "termination_counts": {"time_limit": 1},
        }

    def _save_model(self, *, step, reason, save_path=None):
        self.saved.append((step, reason, save_path or self.config.save_path))


def test_periodic_evaluation_tracks_history_and_updates_only_better_model(
    tmp_path: Path,
):
    latest = tmp_path / "policy.pt"
    trainer = WorkflowTrainer(
        PPOConfig(
            control_mode=CONTROL_MODE_ELEVATOR,
            timesteps=3,
            n_steps=1,
            batch_size=1,
            epochs=1,
            target_kl=None,
            checkpoint_interval_steps=0,
            eval_interval_steps=1,
            eval_episodes=1,
            save_path=str(latest),
            tensorboard_log_dir=None,
            seed=42,
        ),
        evaluation_scores=(1.0, 0.5, 2.0),
    )

    trainer.train()

    best = str(tmp_path / "policy.best.pt")
    assert [item["step"] for item in trainer.evaluation_history] == [1, 2, 3]
    assert trainer.best_mean_reward == 2.0
    assert trainer.saved == [
        (1, "best evaluation", best),
        (3, "best evaluation", best),
        (3, "final", str(latest)),
    ]
    assert trainer.env.reset_calls == 3
    assert trainer.env.closed is True


def test_training_step_continues_from_restored_global_step(tmp_path: Path):
    trainer = WorkflowTrainer(
        PPOConfig(
            control_mode=CONTROL_MODE_ELEVATOR,
            timesteps=2,
            n_steps=1,
            batch_size=1,
            epochs=1,
            target_kl=None,
            checkpoint_interval_steps=0,
            eval_interval_steps=0,
            eval_episodes=1,
            save_path=str(tmp_path / "continued.pt"),
            tensorboard_log_dir=None,
            seed=42,
        ),
        evaluation_scores=(1.0,),
    )
    trainer.training_step = 128

    trainer.train()

    assert trainer.evaluation_history[-1]["step"] == 130
    assert trainer.saved[-1][:2] == (130, "final")
