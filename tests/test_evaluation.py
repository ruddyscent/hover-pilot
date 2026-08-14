import pytest

pytest.importorskip("torch")

from hoverpilot.rl.constants import CONTROL_MODE_ELEVATOR
from hoverpilot.rl.evaluation import EvaluationAccumulator, format_evaluation


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
