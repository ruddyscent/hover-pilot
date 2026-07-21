import unittest

try:
    import numpy as np
    from hoverpilot.rl.ppo import PPOConfig, PPOTrainer, parse_args, reset_env_with_wait, resolve_device
    IMPORT_ERROR = None
except Exception as exc:
    np = None
    PPOConfig = None
    PPOTrainer = None
    parse_args = None
    reset_env_with_wait = None
    resolve_device = None
    IMPORT_ERROR = exc


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
    def test_ppo_config_and_trainer_import(self):
        config = PPOConfig(timesteps=1, max_episode_steps=1, tensorboard_log_dir=None)
        trainer = PPOTrainer(config)
        self.assertEqual(trainer.config.timesteps, 1)
        self.assertEqual(trainer.config.max_episode_steps, 1)

    @unittest.skipIf(IMPORT_ERROR is not None, f"RL dependencies unavailable: {IMPORT_ERROR}")
    def test_auto_device_is_cuda_or_cpu_and_never_implicitly_mps(self):
        device = resolve_device("auto")

        self.assertIn(device.type, {"cuda", "cpu"})

    @unittest.skipIf(IMPORT_ERROR is not None, f"RL dependencies unavailable: {IMPORT_ERROR}")
    def test_train_cli_accepts_explicit_mps_device(self):
        args = parse_args(["train", "--device", "mps"])

        self.assertEqual(args.device, "mps")

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
