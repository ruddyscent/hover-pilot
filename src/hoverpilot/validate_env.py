#!/usr/bin/env python3
"""
Environment validation script that works without PyTorch.
Use this to check environment quality before installing RL dependencies.
"""

import argparse
from typing import List, Optional

import numpy as np

from hoverpilot.config import HOST, PORT
from hoverpilot.envs import HoverPilotHoverEnv


def validation_action(env: HoverPilotHoverEnv, *, control_test: bool) -> np.ndarray:
    """Return safe idle controls unless an active control test was requested."""
    if control_test:
        return env.action_space.sample()
    return np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def validate_environment(
    host: str,
    port: int,
    episodes: int = 2,
    max_episode_steps: Optional[int] = 100,
    *,
    control_test: bool = False,
):
    env = HoverPilotHoverEnv(
        host=host,
        port=port,
        max_episode_steps=max_episode_steps,
    )
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)
    if control_test:
        print(
            "[VALIDATE] CONTROL TEST ENABLED: random flight controls will be sent."
        )
    else:
        print("[VALIDATE] Safe mode: only zero-throttle idle controls will be sent.")

    for episode in range(episodes):
        try:
            observation, info = env.reset()
            print(f"Episode {episode + 1} reset observation shape={observation.shape}, reason={info.get('episode_start_reason')}")
            if info.get("episode_readiness"):
                print("  readiness:", info["episode_readiness"])
            step = 0
            while step < max_episode_steps:
                action = validation_action(env, control_test=control_test)
                next_obs, reward, terminated, truncated, info = env.step(action)
                print(
                    f"  step={step} reward={reward:.3f} terminated={terminated} truncated={truncated} "
                    f"reason={info.get('termination_reason')}"
                )
                if terminated or truncated:
                    break
                step += 1
            if step == max_episode_steps:
                print("  reached max_episode_steps without termination")
        except Exception as e:
            print(f"Episode {episode + 1} failed: {e}")
            break
    env.close()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate environment quality without RL dependencies.")
    parser.add_argument("--episodes", type=int, default=2, help="Number of episodes to validate")
    parser.add_argument("--max-episode-steps", type=int, default=100, help="Max steps per episode")
    parser.add_argument("--host", type=str, default=HOST, help="RealFlight Link host")
    parser.add_argument("--port", type=int, default=PORT, help="RealFlight Link port")
    parser.add_argument(
        "--control-test",
        action="store_true",
        help="Send random controls to exercise action scaling (moves the aircraft).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    validate_environment(
        args.host,
        args.port,
        episodes=args.episodes,
        max_episode_steps=args.max_episode_steps,
        control_test=args.control_test,
    )


if __name__ == "__main__":
    main()
