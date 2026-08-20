"""Materialize packaged experiment configurations for local customization."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def write_starter_config(output_path: str, *, force: bool = False) -> Path:
    """Write the maintained elevator starter config without silent overwrites."""
    output = Path(output_path).expanduser()
    if output.exists() and not force:
        raise FileExistsError(
            f"{output} already exists; choose another path or pass --force"
        )
    source = files("hoverpilot.configs").joinpath("elevator.toml")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return output.resolve()
