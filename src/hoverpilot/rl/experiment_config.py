"""TOML experiment configuration loading and reproducibility metadata."""

from __future__ import annotations

import subprocess
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

from .config import PPOConfig


_SECTION_FIELDS = {
    "environment": {
        "host",
        "port",
        "max_episode_steps",
        "sleep_interval_s",
    },
    "training": {
        "timesteps",
        "seed",
        "n_steps",
        "batch_size",
        "epochs",
        "learning_rate",
        "gamma",
        "gae_lambda",
        "clip_epsilon",
        "target_kl",
        "reward_scale",
        "value_coef",
        "entropy_coef",
        "policy_initial_std",
        "max_grad_norm",
        "log_interval",
        "telemetry_log_interval_steps",
        "device",
    },
    "policy": {
        "control_mode",
        "policy_preset",
        "elevator_fixed_throttle",
    },
    "episode_start": {
        "episode_start_idle_seconds",
        "episode_start_idle_throttle",
        "episode_start_idle_curriculum_steps",
        "episode_start_idle_curriculum_start_seconds",
        "episode_start_handoff_seconds",
    },
    "checkpoint": {
        "save_path",
        "resume_from",
        "checkpoint_interval_steps",
        "best_save_path",
    },
    "evaluation": {"eval_episodes", "eval_interval_steps"},
    "logging": {"tensorboard_log_dir", "enabled"},
    "rflink": {
        "rflink_socket_timeout_s",
        "rflink_request_attempts",
        "rflink_retry_backoff_s",
    },
}


def load_experiment_config(path: str) -> dict[str, Any]:
    """Return argparse-compatible defaults from a validated TOML file."""

    resolved = Path(path).expanduser().resolve()
    try:
        with resolved.open("rb") as stream:
            document = tomllib.load(stream)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid experiment TOML {resolved}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Cannot read experiment config {resolved}: {exc}") from exc

    unknown_sections = set(document) - set(_SECTION_FIELDS)
    if unknown_sections:
        raise ValueError(
            f"Unknown experiment config sections: {sorted(unknown_sections)}"
        )

    defaults: dict[str, Any] = {}
    for section, values in document.items():
        if not isinstance(values, Mapping):
            raise ValueError(f"Experiment config [{section}] must be a table")
        unknown_fields = set(values) - _SECTION_FIELDS[section]
        if unknown_fields:
            raise ValueError(
                f"Unknown keys in experiment config [{section}]: "
                f"{sorted(unknown_fields)}"
            )
        for key, value in values.items():
            if section == "logging" and key == "enabled":
                if not isinstance(value, bool):
                    raise ValueError(
                        "Experiment config [logging].enabled must be boolean"
                    )
                defaults["disable_tensorboard"] = not value
            else:
                defaults[key] = value
    return defaults


def build_experiment_metadata(
    config: PPOConfig,
    *,
    config_path: str | None,
) -> dict[str, object]:
    """Capture the resolved run configuration and source revision."""

    try:
        package_version = version("hover-pilot")
    except PackageNotFoundError:
        package_version = "unknown"
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        git_commit = completed.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        git_commit = None

    resolved_path = (
        str(Path(config_path).expanduser().resolve()) if config_path else None
    )
    config_file: Mapping[str, object] | None = None
    if resolved_path:
        try:
            with Path(resolved_path).open("rb") as stream:
                config_file = tomllib.load(stream)
        except (OSError, tomllib.TOMLDecodeError):
            config_file = None
    return {
        "seed": config.seed,
        "package_version": package_version,
        "git_commit": git_commit,
        "config_path": resolved_path,
        "config_file": dict(config_file) if config_file is not None else None,
        "resolved_config": asdict(config),
    }
