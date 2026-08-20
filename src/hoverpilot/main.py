import argparse
import time
from typing import List, Optional

import numpy as np

from hoverpilot.config import HOST, PORT
from hoverpilot.envs import HoverPilotHoverEnv
from hoverpilot.rflink.client import RFLinkConnectionError
from hoverpilot.training.hover import RewardConfig
from hoverpilot.utils.logger import format_action, format_debug_state, format_step_log


WAITING_LOG_INTERVAL_S = 0.75


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a bounded RealFlight Link hover-control demonstration."
    )
    parser.add_argument("--host", default=HOST, help="RealFlight Link host")
    parser.add_argument("--port", type=int, default=PORT, help="RealFlight Link port")
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Control steps to run before stopping (default: 100).",
    )
    duration.add_argument(
        "--forever",
        action="store_true",
        help="Run across episodes until interrupted.",
    )
    parser.add_argument(
        "--throttle",
        type=float,
        default=0.55,
        help="Fixed throttle command in [0, 1] (default: 0.55).",
    )
    args = parser.parse_args(argv)
    if args.steps is not None and args.steps <= 0:
        parser.error("--steps must be greater than zero")
    if not 0.0 <= args.throttle <= 1.0:
        parser.error("--throttle must be between 0 and 1")
    return args


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    demo_reward_config = RewardConfig(
        min_altitude_agl_m=-1.0,
        controller_active_threshold=None,
    )
    env = HoverPilotHoverEnv(
        host=args.host,
        port=args.port,
        reward_config=demo_reward_config,
        max_episode_steps=None,
    )

    hover_test_action = np.asarray(
        [0.0, 0.0, args.throttle, 0.0], dtype=np.float32
    )
    wait_action = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    try:
        limit = None if args.forever else args.steps
        completed_steps = 0
        print(
            "[DEMO] This command sends neutral flight controls with "
            f"throttle={args.throttle:.2f} for "
            f"{'an unlimited number of steps' if limit is None else f'{limit} steps'}."
        )
        last_wait_log_at = 0.0
        while True:
            try:
                observation, info = env.reset()
                break
            except TimeoutError as exc:
                print(f"waiting for trainer reset before first episode | {exc}")
                while True:
                    started, observation, info = env.poll_wait_for_next_episode(action=wait_action)
                    if started:
                        break
                    now = time.monotonic()
                    if now - last_wait_log_at >= WAITING_LOG_INTERVAL_S:
                        print(f"waiting for trainer reset | {format_debug_state(info.get('debug_state'))}")
                        last_wait_log_at = now
                break
        print(f"episode start shape={observation.shape} reason={info['episode_start_reason']}")
        print(info["state_summary"])
        print(format_debug_state(info.get("debug_state")))
        print(format_action(hover_test_action))

        while True:
            observation, reward, terminated, truncated, info = env.step(hover_test_action)
            completed_steps += 1
            print(format_step_log(
                action=hover_test_action,
                info=info,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
            ))
            if limit is not None and completed_steps >= limit:
                print(f"[DEMO] Completed {completed_steps} control steps.")
                break
            if terminated or truncated:
                print(
                    f"episode ended reason={info.get('termination_reason')} "
                    f"waiting_for_reset={info.get('waiting_for_reset')}"
                )
                print(format_debug_state(info.get("debug_state")))
                last_wait_log_at = 0.0
                while True:
                    started, observation, info = env.poll_wait_for_next_episode(action=wait_action)
                    if started:
                        print(f"episode start shape={observation.shape} reason={info['episode_start_reason']}")
                        print(info["state_summary"])
                        print(format_debug_state(info.get("debug_state")))
                        print(format_action(hover_test_action))
                        break
                    now = __import__("time").monotonic()
                    if now - last_wait_log_at >= WAITING_LOG_INTERVAL_S:
                        print(f"waiting for trainer reset | {format_debug_state(info.get('debug_state'))}")
                        last_wait_log_at = now
    except RFLinkConnectionError as exc:
        print(f"[RFLINK] {exc}")
        print(
            "[RFLINK] Check that RealFlight is running with RealFlight Link enabled, "
            "and set RFLINK_HOST to the host/IP reachable from this shell."
        )
        return 2
    except KeyboardInterrupt:
        print("Stopping...")
        return 130
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
