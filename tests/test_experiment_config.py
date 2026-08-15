from pathlib import Path

import pytest

pytest.importorskip("torch")

from hoverpilot.rl.cli import parse_args
from hoverpilot.rl.config import PPOConfig
from hoverpilot.rl.experiment_config import (
    build_experiment_metadata,
    load_experiment_config,
)


def test_toml_values_become_defaults_and_cli_overrides_them(tmp_path: Path):
    config_path = tmp_path / "experiment.toml"
    config_path.write_text(
        """
[environment]
host = "simulator.local"
port = 19000

[training]
timesteps = 1234
seed = 7

[policy]
control_mode = "elevator"

[logging]
enabled = false
""".strip(),
        encoding="utf-8",
    )

    args = parse_args(
        [
            "train",
            "--config",
            str(config_path),
            "--timesteps",
            "4321",
            "--host",
            "cli.local",
        ]
    )

    assert args.timesteps == 4321
    assert args.seed == 7
    assert args.host == "cli.local"
    assert args.port == 19000
    assert args.control_mode == "elevator"
    assert args.disable_tensorboard is True

    enabled_args = parse_args(
        ["train", "--config", str(config_path), "--enable-tensorboard"]
    )
    assert enabled_args.disable_tensorboard is False


def test_unknown_toml_key_is_rejected(tmp_path: Path):
    config_path = tmp_path / "invalid.toml"
    config_path.write_text("[training]\ntimestep = 10\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown keys.*timestep"):
        load_experiment_config(str(config_path))


def test_unknown_toml_section_is_rejected(tmp_path: Path):
    config_path = tmp_path / "invalid-section.toml"
    config_path.write_text("[curriculum]\nstages = 3\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown experiment config sections"):
        load_experiment_config(str(config_path))


def test_invalid_toml_and_missing_file_report_the_source(tmp_path: Path):
    invalid_path = tmp_path / "broken.toml"
    invalid_path.write_text("[training\nseed = 1", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid experiment TOML.*broken.toml"):
        load_experiment_config(str(invalid_path))
    with pytest.raises(ValueError, match="Cannot read experiment config.*missing.toml"):
        load_experiment_config(str(tmp_path / "missing.toml"))


def test_logging_enabled_requires_boolean(tmp_path: Path):
    config_path = tmp_path / "invalid-logging.toml"
    config_path.write_text('[logging]\nenabled = "yes"\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"\[logging\]\.enabled must be boolean"):
        load_experiment_config(str(config_path))


def test_experiment_metadata_contains_source_and_resolved_config(tmp_path: Path):
    config_path = tmp_path / "experiment.toml"
    config_path.write_text("[training]\nseed = 42\n", encoding="utf-8")
    config = PPOConfig(seed=42, config_path=str(config_path))

    metadata = build_experiment_metadata(config, config_path=str(config_path))

    assert metadata["seed"] == 42
    assert metadata["package_version"] == "2.1.0"
    assert metadata["config_path"] == str(config_path.resolve())
    assert metadata["config_file"] == {"training": {"seed": 42}}
    assert metadata["resolved_config"]["seed"] == 42
    assert metadata["git_commit"] is None or len(metadata["git_commit"]) == 40
