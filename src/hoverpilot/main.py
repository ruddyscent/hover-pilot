import time

import numpy as np

from hoverpilot.config import HOST, PORT
from hoverpilot.envs import HoverPilotHoverEnv
from hoverpilot.rflink.client import RFLinkConnectionError
from hoverpilot.training.hover import RewardConfig
from hoverpilot.utils.logger import format_action, format_debug_state, format_step_log


WAITING_LOG_INTERVAL_S = 0.75


def main():
    demo_reward_config = RewardConfig(
        min_altitude_agl_m=-1.0,
        controller_active_threshold=None,
    )
    env = HoverPilotHoverEnv(
        host=HOST,
        port=PORT,
        reward_config=demo_reward_config,
        max_episode_steps=None,
    )

    hover_test_action = np.asarray([0.0, 0.0, 0.55, 0.0], dtype=np.float32)
    wait_action = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    try:
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
            print(format_step_log(
                action=hover_test_action,
                info=info,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
            ))
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
    finally:
        env.close()


if __name__ == "__main__":
    main()
