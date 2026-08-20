"""Command-line interface for PPO training, playback, and diagnostics."""

from __future__ import annotations

import argparse
from typing import List, Optional

from hoverpilot.config import HOST, PORT
from hoverpilot.rl.elevator_diagnostics import diagnose_elevator_response

from .config import PPOConfig, PPOPlayConfig
from .constants import (
    CONTROL_MODE_ALL,
    CONTROL_MODES,
    POLICY_PRESET_NONE,
    POLICY_PRESETS,
)
from .experiment_config import load_experiment_config
from .player import PPOPlayer
from .report import generate_training_report
from .starter_config import write_starter_config
from .trainer import PPOTrainer


def _add_rflink_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--rflink-socket-timeout-s",
        type=float,
        default=3.0,
        help="Seconds to wait for each RealFlight Link socket operation.",
    )
    parser.add_argument(
        "--rflink-request-attempts",
        type=int,
        default=4,
        help="Maximum RFLink connect or ExchangeData attempts before aborting.",
    )
    parser.add_argument(
        "--rflink-retry-backoff-s",
        type=float,
        default=0.1,
        help="Initial exponential backoff between RFLink retries.",
    )
    parser.add_argument("--host", type=str, default=HOST)
    parser.add_argument("--port", type=int, default=PORT)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train, play, evaluate, report, or diagnose PPO policies on the "
            "HoverPilot Hover Env."
        )
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    init_parser = subparsers.add_parser(
        "init-config",
        help="Write a maintained elevator starter configuration",
    )
    init_parser.add_argument(
        "output",
        nargs="?",
        default="hoverpilot-elevator.toml",
        help="Output path (default: hoverpilot-elevator.toml).",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output file.",
    )

    train_parser = subparsers.add_parser("train", help="Train a PPO policy")
    train_parser.add_argument(
        "--config",
        help="Load experiment defaults from a TOML file. CLI options override it.",
    )
    train_parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help=(
            "Training steps. Defaults to 300000 for elevator-based "
            "modes and 50000 otherwise."
        ),
    )
    train_parser.add_argument("--save-path", type=str, default="ppo_hoverpilot.pt")
    train_parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Continue training from a structured HoverPilot PPO checkpoint.",
    )
    train_parser.add_argument("--seed", type=int, default=None)
    train_parser.add_argument("--max-episode-steps", type=int, default=300)
    train_parser.add_argument("--sleep-interval-s", type=float, default=0.0)
    train_parser.add_argument("--n-steps", type=int, default=1024)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help=("Optimizer learning rate. Defaults to 1e-4."),
    )
    train_parser.add_argument("--gamma", type=float, default=0.99)
    train_parser.add_argument("--gae-lambda", type=float, default=0.95)
    train_parser.add_argument("--clip-epsilon", type=float, default=0.2)
    train_parser.add_argument(
        "--target-kl",
        type=float,
        default=0.02,
        help="Stop a PPO update early when mean approximate KL exceeds this value; use 0 to disable.",
    )
    train_parser.add_argument(
        "--reward-scale",
        type=float,
        default=0.1,
        help="Scale rewards used for PPO returns while keeping displayed episode rewards unscaled.",
    )
    train_parser.add_argument("--value-coef", type=float, default=0.5)
    train_parser.add_argument(
        "--entropy-coef",
        type=float,
        default=None,
        help=("Entropy coefficient. Defaults to 0.0001."),
    )
    train_parser.add_argument(
        "--policy-initial-std",
        type=float,
        default=None,
        help=("Initial policy exploration standard deviation. Defaults to 0.08."),
    )
    train_parser.add_argument("--max-grad-norm", type=float, default=0.5)
    train_parser.add_argument("--log-interval", type=int, default=1)
    train_parser.add_argument(
        "--telemetry-log-interval-steps",
        type=int,
        default=25,
        help=(
            "Print and record task-specific control telemetry "
            "every N steps; 0 disables."
        ),
    )
    train_parser.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help=(
            "Episodes per periodic and final deterministic evaluation. "
            "Defaults to 10."
        ),
    )
    train_parser.add_argument(
        "--tensorboard-log-dir", type=str, default="runs/hoverpilot-ppo"
    )
    train_parser.add_argument("--disable-tensorboard", action="store_true")
    train_parser.add_argument(
        "--enable-tensorboard",
        action="store_false",
        dest="disable_tensorboard",
        help="Enable TensorBoard even when the TOML configuration disables it.",
    )
    train_parser.add_argument(
        "--control-mode",
        choices=CONTROL_MODES,
        default=CONTROL_MODE_ALL,
        help=(
            "Policy-controlled channels. aileron, elevator, rudder, and "
            "throttle use one-dimensional policies; elevator-throttle "
            "aileron-throttle, and rudder-throttle use two-dimensional "
            "policies."
        ),
    )
    train_parser.add_argument(
        "--policy-preset",
        choices=POLICY_PRESETS,
        default=POLICY_PRESET_NONE,
        help=(
            "Optional fixed policy controller. none uses a PPO-only action "
            "policy with trainable measured-sign initialization; elevator-pd "
            "adds the measured elevator PD prior and limits the learned "
            "residual. Resumed checkpoints always restore their saved preset."
        ),
    )
    train_parser.add_argument(
        "--elevator-fixed-throttle",
        type=float,
        default=0.55,
        help=(
            "Fixed throttle sent in aileron, elevator, and rudder modes. "
            "This option does not affect modes where throttle is controlled."
        ),
    )
    train_parser.add_argument(
        "--episode-start-idle-seconds",
        type=float,
        default=0.0,
        help=(
            "Simulator-physics seconds to wait before policy control starts. "
            "Only the configured idle throttle is applied during this period."
        ),
    )
    train_parser.add_argument(
        "--episode-start-idle-throttle",
        type=float,
        default=0.66,
        help="Fixed throttle used during the episode start idle period.",
    )
    train_parser.add_argument(
        "--episode-start-idle-curriculum-steps",
        type=int,
        default=0,
        help=(
            "Linearly increase episode start idle duration from zero to "
            "--episode-start-idle-seconds over this many control-step "
            "equivalents. 0 disables the curriculum."
        ),
    )
    train_parser.add_argument(
        "--episode-start-idle-curriculum-start-seconds",
        type=float,
        default=0.0,
        help=(
            "Idle duration at the beginning of a curriculum run. Use this "
            "to continue a later curriculum stage from a resumed checkpoint."
        ),
    )
    train_parser.add_argument(
        "--episode-start-handoff-seconds",
        type=float,
        default=0.1,
        help=(
            "Simulator-physics seconds used to blend idle controls into the "
            "initial deterministic policy controls."
        ),
    )
    train_parser.add_argument(
        "--checkpoint-interval-steps",
        type=int,
        default=1024,
        help="Save the current model after this many completed training steps; 0 disables.",
    )
    train_parser.add_argument(
        "--eval-interval-steps",
        type=int,
        default=10240,
        help="Run deterministic evaluation every N training steps; 0 disables periodic evaluation.",
    )
    train_parser.add_argument(
        "--best-save-path",
        default=None,
        help="Best checkpoint path. Defaults to <save-path stem>.best.pt.",
    )
    train_parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Training device. auto selects CUDA when available and otherwise CPU; MPS is opt-in.",
    )
    _add_rflink_args(train_parser)

    play_parser = subparsers.add_parser(
        "play",
        help="Control Airplane Hover Trainer with a saved PPO checkpoint",
    )
    play_parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a HoverPilot PPO .pt checkpoint.",
    )
    play_parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="Number of episodes to run; 0 runs until interrupted.",
    )
    play_parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help=(
            "Maximum control steps per episode. When omitted, an episode "
            "continues until a simulator termination condition is reached."
        ),
    )
    play_parser.add_argument("--sleep-interval-s", type=float, default=0.0)
    play_parser.add_argument(
        "--log-interval-steps",
        type=int,
        default=25,
        help="Print policy action and state every N steps; 0 disables step logs.",
    )
    play_parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Inference device. auto selects CUDA when available and otherwise CPU; MPS is opt-in.",
    )
    _add_rflink_args(play_parser)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a checkpoint and optionally compare an earlier checkpoint",
    )
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--compare-to", default=None)
    evaluate_parser.add_argument("--episodes", type=int, default=10)
    evaluate_parser.add_argument("--max-episode-steps", type=int, default=300)
    evaluate_parser.add_argument("--sleep-interval-s", type=float, default=0.0)
    evaluate_parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    _add_rflink_args(evaluate_parser)

    report_parser = subparsers.add_parser(
        "report", help="Generate an HTML report from a training run"
    )
    report_parser.add_argument("run_dir", help="TensorBoard run directory")
    report_parser.add_argument("--checkpoint", default=None)
    report_parser.add_argument("--compare-to", default=None)
    report_parser.add_argument("--output", default=None)
    report_parser.add_argument(
        "--video", default=None, help="Optional evaluation video to link in the report"
    )

    diagnose_parser = subparsers.add_parser(
        "diagnose-elevator",
        help="Measure RealFlight pitch response to conservative elevator pulses",
    )
    diagnose_parser.add_argument("--elevator-fixed-throttle", type=float, default=0.55)
    diagnose_parser.add_argument("--pulse", type=float, default=0.1)
    diagnose_parser.add_argument("--pulse-steps", type=int, default=8)
    diagnose_parser.add_argument("--settle-steps", type=int, default=8)
    _add_rflink_args(diagnose_parser)

    config_probe = argparse.ArgumentParser(add_help=False)
    config_probe.add_argument("command", nargs="?")
    config_probe.add_argument("--config")
    probed, _ = config_probe.parse_known_args(argv)
    if probed.command == "train" and probed.config:
        train_parser.set_defaults(**load_experiment_config(probed.config))

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    if args.command == "init-config":
        try:
            output = write_starter_config(args.output, force=args.force)
        except FileExistsError as exc:
            print(f"[CONFIG] FAILED: {exc}")
            return 1
        print(f"[CONFIG] Wrote {output}")
        print(f"[CONFIG] Start training with: hoverpilot-ppo train --config {output}")
        return 0
    if args.command == "train":
        config = PPOConfig(
            host=args.host,
            port=args.port,
            timesteps=args.timesteps,
            max_episode_steps=args.max_episode_steps,
            sleep_interval_s=args.sleep_interval_s,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_epsilon=args.clip_epsilon,
            target_kl=None if args.target_kl <= 0.0 else args.target_kl,
            reward_scale=args.reward_scale,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            policy_initial_std=args.policy_initial_std,
            max_grad_norm=args.max_grad_norm,
            save_path=args.save_path,
            resume_from=args.resume_from,
            seed=args.seed,
            eval_episodes=args.eval_episodes,
            log_interval=args.log_interval,
            telemetry_log_interval_steps=args.telemetry_log_interval_steps,
            tensorboard_log_dir=None
            if args.disable_tensorboard
            else args.tensorboard_log_dir,
            device=args.device,
            control_mode=args.control_mode,
            policy_preset=args.policy_preset,
            elevator_fixed_throttle=args.elevator_fixed_throttle,
            episode_start_idle_seconds=args.episode_start_idle_seconds,
            episode_start_idle_throttle=args.episode_start_idle_throttle,
            episode_start_idle_curriculum_steps=(
                args.episode_start_idle_curriculum_steps
            ),
            episode_start_idle_curriculum_start_seconds=(
                args.episode_start_idle_curriculum_start_seconds
            ),
            episode_start_handoff_seconds=args.episode_start_handoff_seconds,
            rflink_socket_timeout_s=args.rflink_socket_timeout_s,
            rflink_request_attempts=args.rflink_request_attempts,
            rflink_retry_backoff_s=args.rflink_retry_backoff_s,
            checkpoint_interval_steps=args.checkpoint_interval_steps,
            eval_interval_steps=args.eval_interval_steps,
            best_save_path=args.best_save_path,
            config_path=args.config,
        )
        trainer = PPOTrainer(config)
        trainer.train()
    elif args.command == "play":
        player = PPOPlayer(
            PPOPlayConfig(
                checkpoint_path=args.checkpoint,
                host=args.host,
                port=args.port,
                max_episode_steps=args.max_episode_steps,
                sleep_interval_s=args.sleep_interval_s,
                device=args.device,
                episodes=args.episodes,
                log_interval_steps=args.log_interval_steps,
                rflink_socket_timeout_s=args.rflink_socket_timeout_s,
                rflink_request_attempts=args.rflink_request_attempts,
                rflink_retry_backoff_s=args.rflink_retry_backoff_s,
            )
        )
        player.play()
    elif args.command == "evaluate":
        if args.episodes <= 0:
            raise ValueError("--episodes must be greater than zero")

        def evaluate_checkpoint(path: str):
            evaluator = PPOPlayer(
                PPOPlayConfig(
                    checkpoint_path=path,
                    host=args.host,
                    port=args.port,
                    max_episode_steps=args.max_episode_steps,
                    sleep_interval_s=args.sleep_interval_s,
                    device=args.device,
                    episodes=args.episodes,
                    log_interval_steps=0,
                    rflink_socket_timeout_s=args.rflink_socket_timeout_s,
                    rflink_request_attempts=args.rflink_request_attempts,
                    rflink_retry_backoff_s=args.rflink_retry_backoff_s,
                )
            )
            print(f"[EVAL] checkpoint={path}")
            return evaluator.evaluate()

        baseline = evaluate_checkpoint(args.compare_to) if args.compare_to else None
        current = evaluate_checkpoint(args.checkpoint)
        if baseline is not None:
            print(
                "[COMPARE] current-baseline "
                f"mean_reward={current.mean_reward - baseline.mean_reward:+.3f} "
                f"survival_time={current.mean_survival_time_s - baseline.mean_survival_time_s:+.3f}s "
                f"position_error={current.mean_position_error_m - baseline.mean_position_error_m:+.3f}m "
                f"attitude_error={current.mean_attitude_error_deg - baseline.mean_attitude_error_deg:+.3f}deg"
            )
    elif args.command == "report":
        output = generate_training_report(
            args.run_dir,
            checkpoint_path=args.checkpoint,
            compare_to=args.compare_to,
            output_path=args.output,
            video_path=args.video,
        )
        print(f"[REPORT] Wrote {output}")
    elif args.command == "diagnose-elevator":
        diagnose_elevator_response(
            args.host,
            args.port,
            elevator_fixed_throttle=args.elevator_fixed_throttle,
            pulse=args.pulse,
            pulse_steps=args.pulse_steps,
            settle_steps=args.settle_steps,
            rflink_socket_timeout_s=args.rflink_socket_timeout_s,
            rflink_request_attempts=args.rflink_request_attempts,
            rflink_retry_backoff_s=args.rflink_retry_backoff_s,
        )


if __name__ == "__main__":
    main()
