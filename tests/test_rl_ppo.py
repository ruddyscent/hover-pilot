import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import gymnasium as gym
import numpy as np

from hoverpilot.training.hover import ElevatorHoverFeatures, RewardConfig

try:
    import torch
except ImportError as exc:
    torch = None
    PPOConfig = None
    PPOTrainer = None
    ActorCritic = None
    RolloutBuffer = None
    build_policy_checkpoint = None
    POLICY_PRESET_ELEVATOR_PD = None
    POLICY_PRESET_NONE = None
    PPO_CHECKPOINT_FORMAT = None
    PPO_CHECKPOINT_VERSION = None
    load_policy_checkpoint = None
    parse_args = None
    reset_env_with_wait = None
    resolve_device = None
    IMPORT_ERROR = exc
else:
    from hoverpilot.rl.ppo import (
        ActorCritic,
        CONTROL_MODE_ELEVATOR,
        POLICY_PRESET_ELEVATOR_PD,
        POLICY_PRESET_NONE,
        PPO_CHECKPOINT_FORMAT,
        PPO_CHECKPOINT_VERSION,
        PPOConfig,
        PPOTrainer,
        RolloutBuffer,
        build_policy_checkpoint,
        load_policy_checkpoint,
        parse_args,
        reset_env_with_wait,
        resolve_device,
    )
    IMPORT_ERROR = None


class ResetWaitEnv:
    def __init__(self):
        self.reset_calls = 0
        self.poll_calls = 0
        self.reset_options = None
        self._waiting_for_reset = False

    def reset(self, options=None):
        self.reset_calls += 1
        self.reset_options = options
        raise TimeoutError("trainer reset pending")

    def poll_wait_for_next_episode(self, action=None):
        self.poll_calls += 1
        observation = np.zeros(12, dtype=np.float32)
        info = {
            "debug_state": {
                "x_m": 0.0,
                "y_m": 0.0,
                "altitude_agl_m": 1.5,
            },
            "episode_start_reason": "trainer_repositioned",
        }
        return True, observation, info


class PPOTrainingModuleTests(unittest.TestCase):
    @unittest.skipIf(IMPORT_ERROR is not None, f"RL dependencies unavailable: {IMPORT_ERROR}")
    def test_rollout_advantages_are_normalized_in_place_for_minibatches(self):
        buffer = RolloutBuffer(capacity=4, observation_dim=2, action_dim=1, device=torch.device("cpu"))
        buffer.index = 4
        buffer.advantages[:] = torch.tensor([-4.0, -2.0, 2.0, 8.0])

        normalized = buffer.normalize_advantages()
        minibatch_advantages = torch.cat([batch[3] for batch in buffer.get_batches(batch_size=2)])

        self.assertAlmostEqual(float(normalized.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(normalized.std(unbiased=False)), 1.0, places=6)
        self.assertAlmostEqual(float(minibatch_advantages.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(minibatch_advantages.std(unbiased=False)), 1.0, places=6)

    @unittest.skipIf(IMPORT_ERROR is not None, f"RL dependencies unavailable: {IMPORT_ERROR}")
    def test_squashed_policy_actions_respect_bounds_and_log_probs_match(self):
        low = np.asarray([-1.0, -1.0, 0.0, -1.0], dtype=np.float32)
        high = np.ones(4, dtype=np.float32)
        model = ActorCritic(13, low, high)
        observation = torch.zeros((64, 13), dtype=torch.float32)

        action, old_log_prob, _ = model.get_action(observation)
        new_log_prob, _, _, _ = model.evaluate_actions(observation, action)
        ratio = torch.exp(new_log_prob - old_log_prob)

        self.assertTrue(torch.all(action >= torch.as_tensor(low)))
        self.assertTrue(torch.all(action <= torch.as_tensor(high)))
        torch.testing.assert_close(ratio, torch.ones_like(ratio), atol=1.0e-5, rtol=1.0e-5)

    @unittest.skipIf(IMPORT_ERROR is not None, f"RL dependencies unavailable: {IMPORT_ERROR}")
    def test_squashed_policy_log_probs_match_even_at_numerical_saturation(self):
        model = ActorCritic(
            13,
            np.asarray([-1.0], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
        )
        with torch.no_grad():
            model.policy_mean.weight.zero_()
            model.policy_mean.bias.fill_(20.0)
            model.policy_log_std.fill_(-5.0)
        observation = torch.zeros((8, 13), dtype=torch.float32)

        action, old_log_prob, _ = model.get_action(observation)
        new_log_prob, _, _, _ = model.evaluate_actions(observation, action)

        self.assertTrue(torch.all(action < 1.0))
        torch.testing.assert_close(new_log_prob, old_log_prob, atol=1.0e-5, rtol=1.0e-5)

    @unittest.skipIf(IMPORT_ERROR is not None, f"RL dependencies unavailable: {IMPORT_ERROR}")
    def test_deterministic_initial_policy_is_centered_near_hover_action(self):
        low = np.asarray([-1.0, -1.0, 0.0, -1.0], dtype=np.float32)
        high = np.ones(4, dtype=np.float32)
        model = ActorCritic(13, low, high)

        action = model.deterministic_action(torch.zeros((1, 13), dtype=torch.float32))

        np.testing.assert_allclose(
            action.detach().numpy()[0],
            np.asarray([0.0, 0.0, 0.55, 0.0], dtype=np.float32),
            atol=1.0e-5,
        )

    @unittest.skipIf(IMPORT_ERROR is not None, f"RL dependencies unavailable: {IMPORT_ERROR}")
    def test_compact_policy_residual_cannot_override_large_restoring_prior(self):
        model = ActorCritic(
            6,
            np.asarray([-1.0], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
            policy_preset=POLICY_PRESET_ELEVATOR_PD,
        )
        with torch.no_grad():
            model.policy_mean.weight.zero_()
            model.policy_mean.bias.fill_(20.0)

        action = model.deterministic_action(
            torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        )

        self.assertAlmostEqual(
            float(action.detach()[0, 0]),
            math.tanh(-0.5),
            places=6,
        )

    @unittest.skipIf(IMPORT_ERROR is not None, f"RL dependencies unavailable: {IMPORT_ERROR}")
    def test_compact_pure_ppo_policy_is_antisymmetric(self):
        model = ActorCritic(
            6,
            np.asarray([-1.0], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
            policy_preset=POLICY_PRESET_NONE,
        )
        observation = torch.tensor(
            [[1.4, -0.3, 0.8, 0.6, 0.2, -0.1]],
            dtype=torch.float32,
        )
        mirrored = observation * torch.tensor(
            [[-1.0, -1.0, -1.0, -1.0, 1.0, 1.0]],
            dtype=torch.float32,
        )

        action = model.deterministic_action(observation)
        mirrored_action = model.deterministic_action(mirrored)

        torch.testing.assert_close(mirrored_action, -action)

    @unittest.skipIf(IMPORT_ERROR is not None, f"RL dependencies unavailable: {IMPORT_ERROR}")
    def test_compact_pure_ppo_actor_cannot_cancel_linear_recovery_with_critic_features(self):
        model = ActorCritic(
            6,
            np.asarray([-1.0], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
            policy_preset=POLICY_PRESET_NONE,
        )
        observation = torch.tensor(
            [[1.0, -0.5, 0.8, 0.3, 0.2, -0.1]],
            dtype=torch.float32,
        )
        before = model.deterministic_action(observation)

        with torch.no_grad():
            for parameter in model.shared.parameters():
                parameter.fill_(10.0)
        shared_calls = []
        hook = model.shared.register_forward_hook(
            lambda *_: shared_calls.append(True)
        )
        try:
            after = model.deterministic_action(observation)
        finally:
            hook.remove()

        self.assertIsNone(model.policy_mean)
        self.assertEqual(shared_calls, [])
        torch.testing.assert_close(after, before)

    @unittest.skipIf(IMPORT_ERROR is not None, f"RL dependencies unavailable: {IMPORT_ERROR}")
    def test_compact_pure_ppo_gain_parameterization_preserves_restoring_signs(self):
        model = ActorCritic(
            6,
            np.asarray([-1.0], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
            policy_preset=POLICY_PRESET_NONE,
        )
        with torch.no_grad():
            model.elevator_policy_raw_gain.fill_(-20.0)

        attitude_action = model.deterministic_action(
            torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        )
        rate_action = model.deterministic_action(
            torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
        )

        self.assertLess(float(attitude_action.detach()[0, 0]), 0.0)
        self.assertGreater(float(rate_action.detach()[0, 0]), 0.0)

    @unittest.skipIf(IMPORT_ERROR is not None, f"RL dependencies unavailable: {IMPORT_ERROR}")
    def test_compact_pure_ppo_position_response_is_tied_to_attitude_gain(self):
        model = ActorCritic(
            6,
            np.asarray([-1.0], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
            policy_preset=POLICY_PRESET_NONE,
        )
        with torch.no_grad():
            model.elevator_policy_raw_gain.copy_(
                torch.tensor([0.3, -0.7])
            )

        attitude_action = model.deterministic_action(
            torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        )
        position_action = model.deterministic_action(
            torch.tensor([[8.0 / 15.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
        )
        latent_ratio = torch.atanh(position_action.abs()) / torch.atanh(
            attitude_action.abs()
        )

        torch.testing.assert_close(
            latent_ratio,
            torch.tensor([[8.0 / 15.0]]),
        )

    @unittest.skipIf(IMPORT_ERROR is not None, f"RL dependencies unavailable: {IMPORT_ERROR}")
    def test_compact_pure_ppo_policy_and_value_have_separate_gradients(self):
        model = ActorCritic(
            6,
            np.asarray([-1.0], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
            policy_preset=POLICY_PRESET_NONE,
        )
        observation = torch.tensor(
            [[1.0, 0.5, 0.2, -0.1, 0.0, 0.0]],
            dtype=torch.float32,
        )

        mean, _, value = model(observation)
        model.zero_grad(set_to_none=True)
        mean.sum().backward(retain_graph=True)
        self.assertIsNotNone(model.elevator_policy_raw_gain.grad)
        self.assertTrue(
            all(parameter.grad is None for parameter in model.shared.parameters())
        )

        model.zero_grad(set_to_none=True)
        value.sum().backward()
        self.assertIsNone(model.elevator_policy_raw_gain.grad)
        self.assertTrue(
            any(parameter.grad is not None for parameter in model.shared.parameters())
        )

    @unittest.skipIf(IMPORT_ERROR is not None, f"RL dependencies unavailable: {IMPORT_ERROR}")
    def test_compact_elevator_prior_damps_positive_pitch_rate(self):
        model = ActorCritic(
            6,
            np.asarray([-1.0], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
            policy_preset=POLICY_PRESET_ELEVATOR_PD,
        )
        with torch.no_grad():
            model.policy_mean.weight.zero_()
            model.policy_mean.bias.zero_()

        action = model.deterministic_action(
            torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        )

        self.assertAlmostEqual(
            float(action.detach()[0, 0]),
            math.tanh(0.5),  # the prior is clamped before tanh
            places=6,
        )

    @unittest.skipIf(IMPORT_ERROR is not None, f"RL dependencies unavailable: {IMPORT_ERROR}")
    def test_compact_elevator_prior_tracks_recovery_inclination(self):
        model = ActorCritic(
            6,
            np.asarray([-1.0], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
            policy_preset=POLICY_PRESET_ELEVATOR_PD,
        )
        with torch.no_grad():
            model.policy_mean.weight.zero_()
            model.policy_mean.bias.zero_()

        action = model.deterministic_action(
            torch.tensor([[1.5, 0.0, 1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
        )

        self.assertLess(float(action.detach()[0, 0]), 0.0)

    @unittest.skipIf(IMPORT_ERROR is not None, f"RL dependencies unavailable: {IMPORT_ERROR}")
    def test_ppo_config_and_trainer_import(self):
        config = PPOConfig(timesteps=1, max_episode_steps=1, tensorboard_log_dir=None)
        trainer = PPOTrainer(config)
        self.assertEqual(trainer.config.timesteps, 1)
        self.assertEqual(trainer.config.max_episode_steps, 1)

    @unittest.skipIf(IMPORT_ERROR is not None, f"RL dependencies unavailable: {IMPORT_ERROR}")
    def test_elevator_control_mode_uses_one_policy_action_and_fixed_env_channels(self):
        trainer = PPOTrainer(
            PPOConfig(
                control_mode=CONTROL_MODE_ELEVATOR,
                elevator_fixed_throttle=0.6,
                tensorboard_log_dir=None,
            )
        )

        self.assertEqual(trainer.policy_action_space.shape, (1,))
        self.assertEqual(trainer.env.observation_space.shape, (6,))
        self.assertEqual(trainer.policy_preset, POLICY_PRESET_NONE)
        self.assertEqual(torch.count_nonzero(trainer.model.policy_prior_weight), 0)
        self.assertIsNone(trainer.model.policy_prior_limit)
        self.assertIsNone(trainer.model.policy_residual_limit)
        self.assertIsNotNone(trainer.model.elevator_policy_raw_gain)
        self.assertTrue(trainer.model.elevator_policy_raw_gain.requires_grad)
        torch.testing.assert_close(
            trainer.model.elevator_policy_gain,
            torch.tensor([0.55, 0.45]),
        )
        self.assertIsNone(trainer.model.policy_mean)
        self.assertAlmostEqual(
            float(torch.exp(trainer.model.policy_log_std.detach())[0]),
            0.08,
            places=6,
        )
        self.assertEqual(trainer.entropy_coef, 0.0001)
        np.testing.assert_allclose(
            trainer._to_env_action(np.asarray([-0.25], dtype=np.float32)),
            np.asarray([0.0, -0.25, 0.6, 0.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            trainer._normalize_action(np.asarray([2.0], dtype=np.float32)),
            np.asarray([1.0], dtype=np.float32),
        )

    @unittest.skipIf(IMPORT_ERROR is not None, f"RL dependencies unavailable: {IMPORT_ERROR}")
    def test_elevator_pd_prior_is_explicit_and_changes_initial_action(self):
        pure_ppo = PPOTrainer(
            PPOConfig(
                control_mode=CONTROL_MODE_ELEVATOR,
                policy_preset=POLICY_PRESET_NONE,
                tensorboard_log_dir=None,
            )
        )
        pd_assisted = PPOTrainer(
            PPOConfig(
                control_mode=CONTROL_MODE_ELEVATOR,
                policy_preset=POLICY_PRESET_ELEVATOR_PD,
                tensorboard_log_dir=None,
            )
        )
        with torch.no_grad():
            pd_assisted.model.policy_mean.weight.zero_()
            pd_assisted.model.policy_mean.bias.zero_()
        observation = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        )

        pure_action = pure_ppo.model.deterministic_action(observation)
        pd_action = pd_assisted.model.deterministic_action(observation)

        self.assertAlmostEqual(
            float(pure_action.detach()[0, 0]),
            math.tanh(-0.55),
            places=6,
        )
        self.assertAlmostEqual(
            float(pd_action.detach()[0, 0]),
            math.tanh(-0.5),
            places=6,
        )
        self.assertEqual(pd_assisted.policy_preset, POLICY_PRESET_ELEVATOR_PD)
        self.assertIsNone(pure_ppo.model.policy_mean)
        self.assertIsNone(pd_assisted.model.elevator_policy_raw_gain)
        self.assertIsNone(pd_assisted.model.elevator_policy_gain)
        self.assertEqual(pd_assisted.model.policy_prior_limit, 0.5)
        self.assertAlmostEqual(pd_assisted.model.policy_residual_limit, 0.2)

    @unittest.skipIf(IMPORT_ERROR is not None, f"RL dependencies unavailable: {IMPORT_ERROR}")
    def test_pure_ppo_initial_policy_restores_each_elevator_feature(self):
        trainer = PPOTrainer(
            PPOConfig(
                control_mode=CONTROL_MODE_ELEVATOR,
                tensorboard_log_dir=None,
                seed=42,
            )
        )
        scenarios = (
            (
                ElevatorHoverFeatures(15.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                math.tanh(-0.55),
            ),
            (
                ElevatorHoverFeatures(0.0, 30.0, 0.0, 0.0, 0.0, 0.0),
                math.tanh(0.45),
            ),
            (
                ElevatorHoverFeatures(0.0, 0.0, 4.0, 0.0, 0.0, 0.0),
                math.tanh(-0.55 * 8.0 / 15.0),
            ),
            (
                ElevatorHoverFeatures(0.0, 0.0, 0.0, 5.0, 0.0, 0.0),
                math.tanh(-0.55),
            ),
            (
                ElevatorHoverFeatures(0.0, 0.0, 4.0, 2.0, 0.0, 0.0),
                math.tanh(-0.55 * 14.0 / 15.0),
            ),
        )

        for features, expected_action in scenarios:
            observation = trainer._elevator_probe_observation(features)
            mirrored = observation * np.asarray(
                [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0],
                dtype=np.float32,
            )
            with torch.no_grad():
                action = trainer.model.deterministic_action(
                    torch.as_tensor(observation).unsqueeze(0)
                )
                mirrored_action = trainer.model.deterministic_action(
                    torch.as_tensor(mirrored).unsqueeze(0)
                )

            self.assertAlmostEqual(
                float(action[0, 0]),
                expected_action,
                places=6,
            )
            torch.testing.assert_close(mirrored_action, -action)

    @unittest.skipIf(IMPORT_ERROR is not None, f"RL dependencies unavailable: {IMPORT_ERROR}")
    def test_seed_is_applied_before_model_initialization(self):
        config = PPOConfig(seed=42, tensorboard_log_dir=None)

        first = PPOTrainer(config)
        second = PPOTrainer(config)

        for first_parameter, second_parameter in zip(
            first.model.parameters(), second.model.parameters()
        ):
            torch.testing.assert_close(first_parameter, second_parameter)

    @unittest.skipIf(IMPORT_ERROR is not None, f"RL dependencies unavailable: {IMPORT_ERROR}")
    def test_auto_device_is_cuda_or_cpu_and_never_implicitly_mps(self):
        device = resolve_device("auto")

        self.assertIn(device.type, {"cuda", "cpu"})

    @unittest.skipIf(IMPORT_ERROR is not None, f"RL dependencies unavailable: {IMPORT_ERROR}")
    def test_train_cli_accepts_explicit_mps_device(self):
        args = parse_args(["train", "--device", "mps"])

        self.assertEqual(args.device, "mps")

    @unittest.skipIf(IMPORT_ERROR is not None, f"RL dependencies unavailable: {IMPORT_ERROR}")
    def test_train_cli_uses_conservative_update_defaults(self):
        args = parse_args(["train"])

        self.assertIsNone(args.timesteps)
        self.assertEqual(args.epochs, 5)
        self.assertIsNone(args.learning_rate)
        self.assertEqual(args.target_kl, 0.02)
        self.assertEqual(args.reward_scale, 0.1)
        self.assertIsNone(args.eval_episodes)
        self.assertEqual(args.rflink_socket_timeout_s, 3.0)
        self.assertEqual(args.rflink_request_attempts, 4)
        self.assertEqual(args.rflink_retry_backoff_s, 0.1)
        self.assertEqual(args.checkpoint_interval_steps, 1024)
        self.assertEqual(args.policy_preset, POLICY_PRESET_NONE)

    @unittest.skipIf(IMPORT_ERROR is not None, f"RL dependencies unavailable: {IMPORT_ERROR}")
    def test_elevator_mode_resolves_longer_safer_training_defaults(self):
        standard = PPOTrainer(PPOConfig(tensorboard_log_dir=None))
        elevator = PPOTrainer(
            PPOConfig(
                control_mode=CONTROL_MODE_ELEVATOR,
                tensorboard_log_dir=None,
            )
        )

        self.assertEqual(standard.config.timesteps, 50_000)
        self.assertEqual(standard.config.learning_rate, 3e-4)
        self.assertEqual(standard.config.eval_episodes, 3)
        self.assertEqual(elevator.config.timesteps, 300_000)
        self.assertEqual(elevator.config.learning_rate, 1e-4)
        self.assertEqual(elevator.config.eval_episodes, 10)

    @unittest.skipIf(IMPORT_ERROR is not None, f"RL dependencies unavailable: {IMPORT_ERROR}")
    def test_train_cli_accepts_elevator_control_mode(self):
        args = parse_args(
            ["train", "--control-mode", "elevator", "--elevator-fixed-throttle", "0.6"]
        )

        self.assertEqual(args.control_mode, "elevator")
        self.assertEqual(args.elevator_fixed_throttle, 0.6)

    @unittest.skipIf(IMPORT_ERROR is not None, f"RL dependencies unavailable: {IMPORT_ERROR}")
    def test_train_cli_accepts_explicit_elevator_pd_policy_preset(self):
        args = parse_args(
            [
                "train",
                "--control-mode",
                "elevator",
                "--policy-preset",
                "elevator-pd",
            ]
        )

        self.assertEqual(args.policy_preset, POLICY_PRESET_ELEVATOR_PD)

    @unittest.skipIf(IMPORT_ERROR is not None, f"RL dependencies unavailable: {IMPORT_ERROR}")
    def test_diagnose_elevator_cli_uses_conservative_pulse_defaults(self):
        args = parse_args(["diagnose-elevator"])

        self.assertEqual(args.pulse, 0.1)
        self.assertEqual(args.pulse_steps, 8)
        self.assertEqual(args.settle_steps, 8)

    @unittest.skipIf(IMPORT_ERROR is not None, f"RL dependencies unavailable: {IMPORT_ERROR}")
    def test_checkpoint_loader_rejects_legacy_and_unknown_formats(self):
        model = ActorCritic(
            6,
            np.asarray([-1.0], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
        )
        with TemporaryDirectory() as directory:
            unsupported_checkpoints = [
                model.state_dict(),
                {
                    **build_policy_checkpoint(
                        model,
                        control_mode=CONTROL_MODE_ELEVATOR,
                        elevator_fixed_throttle=0.55,
                        reward_config=RewardConfig(),
                    ),
                    "format_version": PPO_CHECKPOINT_VERSION - 1,
                },
                {
                    **build_policy_checkpoint(
                        model,
                        control_mode=CONTROL_MODE_ELEVATOR,
                        elevator_fixed_throttle=0.55,
                        reward_config=RewardConfig(),
                    ),
                    "format_version": PPO_CHECKPOINT_VERSION + 1,
                },
            ]
            for index, checkpoint in enumerate(unsupported_checkpoints):
                with self.subTest(index=index):
                    checkpoint_path = f"{directory}/unsupported-{index}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    with self.assertRaisesRegex(
                        ValueError,
                        "Unsupported PPO checkpoint",
                    ):
                        load_policy_checkpoint(checkpoint_path)

    @unittest.skipIf(IMPORT_ERROR is not None, f"RL dependencies unavailable: {IMPORT_ERROR}")
    def test_checkpoint_loader_rejects_invalid_observation_config(self):
        model = ActorCritic(
            6,
            np.asarray([-1.0], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
        )
        base = build_policy_checkpoint(
            model,
            control_mode=CONTROL_MODE_ELEVATOR,
            elevator_fixed_throttle=0.55,
            reward_config=RewardConfig(),
        )
        invalid_configs = []
        missing_field = dict(base["observation_config"])
        missing_field.pop("inclination_error_scale_deg")
        invalid_configs.append(missing_field)
        extra_field = dict(base["observation_config"])
        extra_field["unknown"] = 1.0
        invalid_configs.append(extra_field)
        zero_scale = dict(base["observation_config"])
        zero_scale["velocity_error_scale_mps"] = 0.0
        invalid_configs.append(zero_scale)
        negative_gain = dict(base["observation_config"])
        negative_gain["elevator_recovery_position_gain_deg_per_m"] = -1.0
        invalid_configs.append(negative_gain)
        non_finite = dict(base["observation_config"])
        non_finite["pitch_rate_scale_deg_s"] = float("nan")
        invalid_configs.append(non_finite)

        with TemporaryDirectory() as directory:
            for index, observation_config in enumerate(invalid_configs):
                with self.subTest(index=index):
                    checkpoint_path = f"{directory}/invalid-{index}.pt"
                    torch.save(
                        {
                            **base,
                            "observation_config": observation_config,
                        },
                        checkpoint_path,
                    )
                    with self.assertRaisesRegex(
                        ValueError,
                        "observation_config",
                    ):
                        load_policy_checkpoint(checkpoint_path)

    @unittest.skipIf(IMPORT_ERROR is not None, f"RL dependencies unavailable: {IMPORT_ERROR}")
    def test_trainer_can_resume_structured_elevator_checkpoint(self):
        checkpoint_reward_config = RewardConfig(
            inclination_error_scale_deg=12.0,
            elevator_recovery_position_gain_deg_per_m=2.5,
        )
        source_model = ActorCritic(
            6,
            np.asarray([-1.0], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
            policy_preset=POLICY_PRESET_ELEVATOR_PD,
        )
        with torch.no_grad():
            source_model.policy_mean.bias.fill_(-0.25)

        with TemporaryDirectory() as directory:
            checkpoint_path = f"{directory}/compact.pt"
            torch.save(
                build_policy_checkpoint(
                    source_model,
                    control_mode=CONTROL_MODE_ELEVATOR,
                    elevator_fixed_throttle=0.63,
                    reward_config=checkpoint_reward_config,
                ),
                checkpoint_path,
            )
            trainer = PPOTrainer(
                PPOConfig(
                    control_mode=CONTROL_MODE_ELEVATOR,
                    resume_from=checkpoint_path,
                    reward_config=RewardConfig(
                        inclination_error_scale_deg=99.0,
                        elevator_recovery_position_gain_deg_per_m=9.0,
                    ),
                    tensorboard_log_dir=None,
                )
            )

        torch.testing.assert_close(
            trainer.model.policy_mean.bias,
            source_model.policy_mean.bias,
        )
        self.assertEqual(trainer.policy_preset, POLICY_PRESET_ELEVATOR_PD)
        self.assertEqual(trainer.elevator_fixed_throttle, 0.63)
        self.assertEqual(
            trainer.config.reward_config.inclination_error_scale_deg,
            12.0,
        )
        self.assertEqual(
            trainer.env.reward_config.elevator_recovery_position_gain_deg_per_m,
            2.5,
        )

    @unittest.skipIf(IMPORT_ERROR is not None, f"RL dependencies unavailable: {IMPORT_ERROR}")
    def test_reward_scale_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "reward_scale must be greater than zero"):
            PPOTrainer(PPOConfig(reward_scale=0.0, tensorboard_log_dir=None))

    @unittest.skipIf(IMPORT_ERROR is not None, f"RL dependencies unavailable: {IMPORT_ERROR}")
    def test_elevator_fixed_throttle_must_be_bounded(self):
        with self.assertRaisesRegex(ValueError, "elevator_fixed_throttle"):
            PPOTrainer(
                PPOConfig(
                    control_mode=CONTROL_MODE_ELEVATOR,
                    elevator_fixed_throttle=1.1,
                    tensorboard_log_dir=None,
                )
            )

    @unittest.skipIf(IMPORT_ERROR is not None, f"RL dependencies unavailable: {IMPORT_ERROR}")
    def test_checkpoint_interval_must_be_non_negative(self):
        with self.assertRaisesRegex(ValueError, "checkpoint_interval_steps"):
            PPOTrainer(
                PPOConfig(
                    checkpoint_interval_steps=-1,
                    tensorboard_log_dir=None,
                )
            )

    @unittest.skipIf(IMPORT_ERROR is not None, f"RL dependencies unavailable: {IMPORT_ERROR}")
    def test_atomic_model_save_replaces_target_without_leaving_temporary_file(self):
        with TemporaryDirectory() as directory:
            save_path = f"{directory}/policy.pt"
            trainer = PPOTrainer(
                PPOConfig(
                    save_path=save_path,
                    tensorboard_log_dir=None,
                )
            )

            trainer._save_model(step=128, reason="test")

            saved_checkpoint = torch.load(save_path, map_location="cpu", weights_only=True)
            self.assertEqual(
                set(saved_checkpoint),
                {
                    "checkpoint_format",
                    "format_version",
                    "model_state_dict",
                    "control_mode",
                    "policy_preset",
                    "elevator_fixed_throttle",
                    "observation_config",
                },
            )
            self.assertEqual(saved_checkpoint["checkpoint_format"], PPO_CHECKPOINT_FORMAT)
            self.assertEqual(saved_checkpoint["format_version"], PPO_CHECKPOINT_VERSION)
            self.assertIn("policy_mean.weight", saved_checkpoint["model_state_dict"])
            self.assertIn("policy_prior_weight", saved_checkpoint["model_state_dict"])
            self.assertIn("_policy_prior_limit", saved_checkpoint["model_state_dict"])
            self.assertIn("_policy_residual_limit", saved_checkpoint["model_state_dict"])
            self.assertEqual(
                list(Path(directory).glob("policy.pt.tmp-*")),
                [],
            )

    @unittest.skipIf(IMPORT_ERROR is not None, f"RL dependencies unavailable: {IMPORT_ERROR}")
    def test_episode_metrics_span_rollout_boundaries(self):
        class ThreeStepEnv(gym.Env):
            def __init__(self):
                self.observation_space = gym.spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(13,),
                    dtype=np.float32,
                )
                self.action_space = gym.spaces.Box(
                    low=np.asarray([-1.0, -1.0, 0.0, -1.0], dtype=np.float32),
                    high=np.ones(4, dtype=np.float32),
                    dtype=np.float32,
                )
                self.steps = 0

            def reset(self, *, seed=None, options=None):
                del seed, options
                self.steps = 0
                return np.zeros(13, dtype=np.float32), {
                    "episode_start_reason": "test_reset",
                    "waiting_for_reset": False,
                }

            def step(self, action):
                self.steps += 1
                terminated = self.steps == 3
                info = {"termination_reason": "test_end" if terminated else None}
                return np.zeros(13, dtype=np.float32), 1.0, terminated, False, info

        class TestTrainer(PPOTrainer):
            def __init__(self, config):
                self.scalars = []
                super().__init__(config)

            def _build_env(self):
                return ThreeStepEnv()

            def _write_scalar(self, tag, value, step):
                self.scalars.append((tag, value, step))

            def _evaluate_policy(self):
                return None

        with TemporaryDirectory() as directory:
            trainer = TestTrainer(
                PPOConfig(
                    control_mode=CONTROL_MODE_ELEVATOR,
                    timesteps=3,
                    n_steps=2,
                    batch_size=2,
                    epochs=1,
                    target_kl=None,
                    save_path=f"{directory}/policy.pt",
                    tensorboard_log_dir=None,
                    seed=42,
                )
            )
            trainer.train()

        episode_lengths = [value for tag, value, _ in trainer.scalars if tag == "train/episode_length"]
        episode_rewards = [value for tag, value, _ in trainer.scalars if tag == "train/episode_reward"]
        self.assertEqual(episode_lengths, [3.0])
        self.assertEqual(episode_rewards, [3.0])

    @unittest.skipIf(IMPORT_ERROR is not None, f"RL dependencies unavailable: {IMPORT_ERROR}")
    def test_training_continues_time_limit_without_resetting_live_aircraft(self):
        class TruncatingEnv(gym.Env):
            def __init__(self):
                self.observation_space = gym.spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(6,),
                    dtype=np.float32,
                )
                self.action_space = gym.spaces.Box(
                    low=np.asarray([-1.0, -1.0, 0.0, -1.0], dtype=np.float32),
                    high=np.ones(4, dtype=np.float32),
                    dtype=np.float32,
                )
                self.reset_calls = 0
                self.continue_calls = 0

            def reset(self, *, seed=None, options=None):
                del seed, options
                self.reset_calls += 1
                return np.zeros(6, dtype=np.float32), {
                    "episode_start_reason": "test_reset",
                    "waiting_for_reset": False,
                }

            def step(self, action):
                del action
                return np.zeros(6, dtype=np.float32), 1.0, False, True, {
                    "termination_reason": None,
                    "debug_state": {},
                }

            def continue_after_truncation(self):
                self.continue_calls += 1
                return np.zeros(6, dtype=np.float32), {
                    "episode_start_reason": "time_limit_continuation",
                    "waiting_for_reset": False,
                }

        class TestTrainer(PPOTrainer):
            def __init__(self, config):
                self.saved_models = []
                super().__init__(config)

            def _build_env(self):
                return TruncatingEnv()

            def _save_model(self, *, step, reason):
                self.saved_models.append((step, reason))

            def _evaluate_policy(self):
                return None

        with TemporaryDirectory() as directory:
            trainer = TestTrainer(
                PPOConfig(
                    control_mode=CONTROL_MODE_ELEVATOR,
                    timesteps=2,
                    n_steps=2,
                    batch_size=2,
                    epochs=1,
                    target_kl=None,
                    checkpoint_interval_steps=1,
                    save_path=f"{directory}/policy.pt",
                    tensorboard_log_dir=None,
                    seed=42,
                )
            )
            trainer.train()

        self.assertEqual(trainer.env.reset_calls, 1)
        self.assertEqual(trainer.env.continue_calls, 2)
        self.assertEqual(trainer.saved_models, [(2, "final")])

    @unittest.skipIf(IMPORT_ERROR is not None, f"RL dependencies unavailable: {IMPORT_ERROR}")
    def test_training_failure_saves_latest_checkpoint_once_and_closes_environment(self):
        class FailingEnv(gym.Env):
            def __init__(self):
                self.observation_space = gym.spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(13,),
                    dtype=np.float32,
                )
                self.action_space = gym.spaces.Box(
                    low=np.asarray([-1.0, -1.0, 0.0, -1.0], dtype=np.float32),
                    high=np.ones(4, dtype=np.float32),
                    dtype=np.float32,
                )
                self.steps = 0
                self.closed = False

            def reset(self, *, seed=None, options=None):
                del seed, options
                return np.zeros(13, dtype=np.float32), {
                    "episode_start_reason": "test_reset",
                    "waiting_for_reset": False,
                }

            def step(self, action):
                del action
                self.steps += 1
                if self.steps == 2:
                    raise TimeoutError("simulated RFLink timeout")
                return np.zeros(13, dtype=np.float32), 1.0, False, False, {
                    "termination_reason": None
                }

            def close(self):
                self.closed = True

        class FailingTrainer(PPOTrainer):
            def __init__(self, config):
                self.saved_models = []
                super().__init__(config)

            def _build_env(self):
                return FailingEnv()

            def _save_model(self, *, step, reason):
                self.saved_models.append((step, reason))

            def _evaluate_policy(self):
                return None

        cases = (
            (1, [(1, "periodic checkpoint")]),
            (0, [(1, "emergency checkpoint")]),
        )
        for checkpoint_interval_steps, expected_saves in cases:
            with self.subTest(
                checkpoint_interval_steps=checkpoint_interval_steps
            ):
                trainer = FailingTrainer(
                    PPOConfig(
                        control_mode=CONTROL_MODE_ELEVATOR,
                        timesteps=3,
                        n_steps=1,
                        batch_size=1,
                        epochs=1,
                        target_kl=None,
                        checkpoint_interval_steps=checkpoint_interval_steps,
                        tensorboard_log_dir=None,
                        seed=42,
                    )
                )

                with self.assertRaisesRegex(
                    TimeoutError,
                    "simulated RFLink timeout",
                ):
                    trainer.train()

                self.assertEqual(trainer.saved_models, expected_saves)
                self.assertTrue(trainer.env.closed)

    @unittest.skipIf(IMPORT_ERROR is not None, f"RL dependencies unavailable: {IMPORT_ERROR}")
    def test_unknown_device_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported device"):
            resolve_device("tpu")

    @unittest.skipIf(IMPORT_ERROR is not None, f"RL dependencies unavailable: {IMPORT_ERROR}")
    def test_reset_wait_helper_recovers_via_polling(self):
        env = ResetWaitEnv()

        observation, info = reset_env_with_wait(env, action=np.zeros(4, dtype=np.float32))

        self.assertEqual(env.reset_calls, 1)
        self.assertEqual(env.poll_calls, 1)
        self.assertEqual(observation.shape, (12,))
        self.assertEqual(info["episode_start_reason"], "trainer_repositioned")

    @unittest.skipIf(IMPORT_ERROR is not None, f"RL dependencies unavailable: {IMPORT_ERROR}")
    def test_reset_wait_helper_reuses_existing_wait_state_without_reset(self):
        env = ResetWaitEnv()
        env._waiting_for_reset = True

        observation, info = reset_env_with_wait(env, action=np.zeros(4, dtype=np.float32))

        self.assertEqual(env.reset_calls, 0)
        self.assertEqual(env.poll_calls, 1)
        self.assertEqual(observation.shape, (12,))
        self.assertEqual(info["episode_start_reason"], "trainer_repositioned")

    @unittest.skipIf(IMPORT_ERROR is not None, f"RL dependencies unavailable: {IMPORT_ERROR}")
    def test_reset_wait_helper_passes_initial_action_to_reset(self):
        class ReadyResetEnv(ResetWaitEnv):
            def reset(self, options=None):
                self.reset_calls += 1
                self.reset_options = options
                return np.zeros(12, dtype=np.float32), {"episode_start_reason": "reset_ready"}

        env = ReadyResetEnv()
        initial_action = np.asarray([0.0, 0.0, 0.55, 0.0], dtype=np.float32)

        observation, info = reset_env_with_wait(env, initial_action=initial_action)

        self.assertEqual(env.reset_calls, 1)
        self.assertEqual(env.poll_calls, 0)
        self.assertEqual(observation.shape, (12,))
        self.assertEqual(info["episode_start_reason"], "reset_ready")
        self.assertIsNotNone(env.reset_options)
        self.assertIn("initial_action", env.reset_options)
