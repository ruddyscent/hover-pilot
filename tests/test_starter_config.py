from pathlib import Path

import pytest

from hoverpilot.rl.experiment_config import load_experiment_config
from hoverpilot.rl.starter_config import write_starter_config


def test_packaged_starter_config_can_be_written_and_loaded(tmp_path: Path):
    output = write_starter_config(str(tmp_path / "starter.toml"))

    defaults = load_experiment_config(str(output))
    assert defaults["control_mode"] == "elevator"
    assert defaults["seed"] == 42
    assert defaults["save_path"] == "checkpoints/elevator.pt"


def test_starter_config_does_not_overwrite_without_force(tmp_path: Path):
    output = tmp_path / "starter.toml"
    output.write_text("custom", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--force"):
        write_starter_config(str(output))

    write_starter_config(str(output), force=True)
    assert 'control_mode = "elevator"' in output.read_text(encoding="utf-8")
