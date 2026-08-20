"""Helpers for capturing and restoring reproducible PPO training state."""

from __future__ import annotations

import random
from typing import Mapping

import numpy as np
import torch

from .config import PPOConfig


def capture_rng_state() -> dict[str, object]:
    """Capture every random-number generator used by the trainer."""
    numpy_state = np.random.get_state()
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": torch.as_tensor(numpy_state[1].copy()),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, object]) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""
    python_state = state.get("python")
    if isinstance(python_state, tuple):
        random.setstate(python_state)

    numpy_state = state.get("numpy")
    if isinstance(numpy_state, Mapping) and isinstance(
        numpy_state.get("state"), torch.Tensor
    ):
        np.random.set_state(
            (
                str(numpy_state["bit_generator"]),
                numpy_state["state"].cpu().numpy().astype(np.uint32),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )

    torch_state = state.get("torch")
    if isinstance(torch_state, torch.Tensor):
        torch.set_rng_state(torch_state.cpu())

    cuda_state = state.get("torch_cuda")
    if torch.cuda.is_available() and isinstance(cuda_state, list):
        torch.cuda.set_rng_state_all(cuda_state)


def checkpoint_environment_config(config: PPOConfig) -> dict[str, object]:
    """Return environment settings whose values must survive a resume."""
    return {
        "max_episode_steps": config.max_episode_steps,
        "sleep_interval_s": config.sleep_interval_s,
        "episode_start_idle_seconds": config.episode_start_idle_seconds,
        "episode_start_idle_throttle": config.episode_start_idle_throttle,
        "episode_start_idle_curriculum_steps": config.episode_start_idle_curriculum_steps,
        "episode_start_idle_curriculum_start_seconds": config.episode_start_idle_curriculum_start_seconds,
        "episode_start_handoff_seconds": config.episode_start_handoff_seconds,
    }
