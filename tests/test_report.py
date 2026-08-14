from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("tensorboard")

from torch.utils.tensorboard import SummaryWriter

from hoverpilot.rl.cli import parse_args
from hoverpilot.rl.report import generate_training_report


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
