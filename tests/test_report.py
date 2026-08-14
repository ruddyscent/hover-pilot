from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tensorboard")

from torch.utils.tensorboard import SummaryWriter

from hoverpilot.rl.checkpoints import build_policy_checkpoint
from hoverpilot.rl.cli import parse_args
from hoverpilot.rl.constants import CONTROL_MODE_ELEVATOR
from hoverpilot.rl.models import ActorCritic
from hoverpilot.rl.report import generate_training_report
from hoverpilot.training.hover import RewardConfig


def _write_evaluated_checkpoint(path: Path, *, reward: float, position_error: float):
    model = ActorCritic(
        6,
        np.asarray([-1.0], dtype=np.float32),
        np.asarray([1.0], dtype=np.float32),
        control_mode=CONTROL_MODE_ELEVATOR,
    )
    checkpoint = build_policy_checkpoint(
        model,
        control_mode=CONTROL_MODE_ELEVATOR,
        elevator_fixed_throttle=0.55,
        reward_config=RewardConfig(),
        training_step=200,
        evaluation_history=(
            {
                "step": 200,
                "mean_reward": reward,
                "mean_survival_time_s": 20.0,
                "mean_position_error_m": position_error,
                "mean_altitude_error_m": 0.5,
                "mean_attitude_error_deg": 2.0,
                "termination_counts": {"time_limit": 3, "altitude_too_low": 1},
            },
        ),
        best_mean_reward=reward,
        experiment_metadata={"seed": 42, "git_commit": "a" * 40},
    )
    torch.save(checkpoint, path)


def test_report_generates_reward_error_termination_and_trajectory_sections(
    tmp_path: Path,
):
    run_dir = tmp_path / "experiment-001"
    writer = SummaryWriter(log_dir=run_dir)
    for step in (10, 20, 30):
        writer.add_scalar("train/episode_reward", step / 10, step)
        writer.add_scalar("train/state/abs_roll_error_deg", 10 - step / 10, step)
        writer.add_scalar("train/state/x_m", step / 10, step)
        writer.add_scalar("train/state/y_m", step / 20, step)
    writer.add_scalar("eval/termination/time_limit", 4, 30)
    writer.close()

    output = generate_training_report(str(run_dir))
    document = output.read_text(encoding="utf-8")

    assert output == run_dir / "report.html"
    assert "Reward trend" in document
    assert "Control errors" in document
    assert "Flight trajectory (X/Y)" in document
    assert "Termination reasons" in document
    assert "time_limit" in document
    assert "Reproducibility metadata" in document


def test_report_cli_accepts_run_and_artifact_paths():
    args = parse_args(
        [
            "report",
            "runs/experiment-001",
            "--checkpoint",
            "checkpoints/best.pt",
            "--compare-to",
            "checkpoints/previous.pt",
            "--output",
            "reports/result.html",
            "--video",
            "evaluation.mp4",
        ]
    )

    assert args.command == "report"
    assert args.run_dir == "runs/experiment-001"
    assert args.checkpoint == "checkpoints/best.pt"
    assert args.compare_to == "checkpoints/previous.pt"
    assert args.output == "reports/result.html"
    assert args.video == "evaluation.mp4"


def test_report_discovers_best_checkpoint_and_compares_evaluations(tmp_path: Path):
    run_dir = tmp_path / "experiment-002"
    writer = SummaryWriter(log_dir=run_dir)
    writer.add_scalar("train/episode_reward", 2.0, 200)
    writer.close()
    current_path = run_dir / "policy.best.pt"
    baseline_path = tmp_path / "baseline.pt"
    _write_evaluated_checkpoint(current_path, reward=12.0, position_error=1.0)
    _write_evaluated_checkpoint(baseline_path, reward=8.0, position_error=2.0)
    output = tmp_path / "nested" / "comparison.html"

    generated = generate_training_report(
        str(run_dir), compare_to=str(baseline_path), output_path=str(output)
    )
    document = generated.read_text(encoding="utf-8")

    assert generated == output
    assert "policy.best.pt" in document
    assert "Checkpoint comparison" in document
    assert "Mean reward" in document
    assert "+4.000" in document
    assert "-1.000" in document
    assert "altitude_too_low" in document
    assert "&quot;seed&quot;: 42" in document
    assert "Flight trajectory (X/Y)" not in document
