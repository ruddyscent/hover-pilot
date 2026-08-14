import pytest

torch = pytest.importorskip("torch")
import gymnasium as gym
import numpy as np

from hoverpilot.rl.checkpoints import build_policy_checkpoint
from hoverpilot.rl.config import PPOPlayConfig
from hoverpilot.rl.constants import CONTROL_MODE_ELEVATOR
from hoverpilot.rl.evaluation import EvaluationAccumulator, format_evaluation
from hoverpilot.rl.models import ActorCritic
from hoverpilot.rl.player import PPOPlayer
from hoverpilot.training.hover import RewardConfig


def test_evaluation_accumulator_records_hover_metrics_and_survival_time():
    accumulator = EvaluationAccumulator(CONTROL_MODE_ELEVATOR)
    for physics_time, x_m, inclination_error in (
        (10.0, 3.0, -4.0),
        (12.5, 4.0, 2.0),
    ):
        accumulator.record_step(
            {
                "debug_state": {
                    "physics_time_s": physics_time,
                    "x_m": x_m,
                    "y_m": 0.0,
                    "altitude_agl_m": 2.0,
                },
                "target_hover": {
                    "x_m": 0.0,
                    "y_m": 0.0,
                    "altitude_agl_m": 1.0,
                },
                "elevator_hover_features": {
                    "inclination_error_deg": inclination_error,
                },
            }
        )
    accumulator.finish_episode(8.0, 2, "time_limit")

    result = accumulator.result()

    assert result.mean_reward == 8.0
    assert result.mean_episode_steps == 2.0
    assert result.mean_survival_time_s == 2.5
    assert result.reward_per_step == 4.0
    assert result.mean_position_error_m == 3.5
    assert result.mean_altitude_error_m == 1.0
    assert result.mean_attitude_error_deg == 3.0
    assert result.termination_counts == {"time_limit": 1}
    assert "survival_time=2.500s" in format_evaluation(result)


def test_evaluate_cli_accepts_checkpoint_comparison():
    from hoverpilot.rl.cli import parse_args

    args = parse_args(
        [
            "evaluate",
            "--checkpoint",
            "current.pt",
            "--compare-to",
            "previous.pt",
            "--episodes",
            "5",
        ]
    )

    assert args.command == "evaluate"
    assert args.checkpoint == "current.pt"
    assert args.compare_to == "previous.pt"
    assert args.episodes == 5


def test_player_evaluate_runs_deterministic_episodes_and_aggregates_metrics(tmp_path):
    class EvaluationEnv(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(6,), dtype=np.float32
            )
            self.action_space = gym.spaces.Box(
                low=np.asarray([-1.0, -1.0, 0.0, -1.0], dtype=np.float32),
                high=np.ones(4, dtype=np.float32),
                dtype=np.float32,
            )
            self.steps = 0
            self.closed = False

        def reset(self, *, seed=None, options=None):
            del seed, options
            self.steps = 0
            return np.zeros(6, dtype=np.float32), {
                "episode_start_reason": "test"
            }

        def step(self, action):
            assert np.all(np.isfinite(action))
            self.steps += 1
            terminated = self.steps == 2
            return np.zeros(6, dtype=np.float32), 1.0, terminated, False, {
                "termination_reason": "time_limit" if terminated else None,
                "debug_state": {
                    "physics_time_s": float(self.steps),
                    "x_m": 1.0,
                    "y_m": 0.0,
                    "altitude_agl_m": 2.0,
                },
                "target_hover": {"x_m": 0.0, "y_m": 0.0, "altitude_agl_m": 2.0},
                "elevator_hover_features": {"inclination_error_deg": 3.0},
            }

        def close(self):
            self.closed = True

    class TestPlayer(PPOPlayer):
        def _build_env(self):
            return EvaluationEnv()

    checkpoint_path = tmp_path / "policy.pt"
    model = ActorCritic(
        6,
        np.asarray([-1.0], dtype=np.float32),
        np.asarray([1.0], dtype=np.float32),
        control_mode=CONTROL_MODE_ELEVATOR,
    )
    torch.save(
        build_policy_checkpoint(
            model,
            control_mode=CONTROL_MODE_ELEVATOR,
            elevator_fixed_throttle=0.55,
            reward_config=RewardConfig(),
        ),
        checkpoint_path,
    )
    player = TestPlayer(
        PPOPlayConfig(checkpoint_path=str(checkpoint_path), episodes=2, device="cpu")
    )

    result = player.evaluate()

    assert result.episodes == 2
    assert result.mean_reward == 2.0
    assert result.mean_episode_steps == 2.0
    assert result.mean_survival_time_s == 1.0
    assert result.mean_position_error_m == 1.0
    assert result.mean_attitude_error_deg == 3.0
    assert result.termination_counts == {"time_limit": 2}
    assert player.env.closed is True
