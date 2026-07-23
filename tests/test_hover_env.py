import unittest

import numpy as np

from hoverpilot.envs import (
    ELEVATOR_HOVER_TASK,
    HoverPilotHoverEnv,
    STANDARD_HOVER_TASK,
    elevator_features_to_observation,
    gym_action_to_rf_action,
    state_to_observation,
)
from hoverpilot.envs.hover_env import EpisodeLifecycleResult
from hoverpilot.rflink.models import FlightAxisState, RFControlAction
from hoverpilot.training.hover import (
    RewardConfig,
    compute_elevator_hover_features,
)


class StubRFLinkClient:
    def __init__(self, states):
        self._states = list(states)
        self.connected = False
        self.closed = False
        self.actions = []

    def connect(self):
        self.connected = True

    def request_state(self, action=None):
        self.actions.append(action)
        if not self._states:
            raise RuntimeError("no more stub states")
        return self._states.pop(0)

    def step(self, action=None):
        self.actions.append(action)
        if not self._states:
            raise RuntimeError("no more stub states")
        return self._states.pop(0)

    def close(self):
        self.closed = True


class HoverEnvTests(unittest.TestCase):
    def _state(self, **overrides):
        state = FlightAxisState(
            m_aircraftPositionX_MTR=0.0,
            m_aircraftPositionY_MTR=0.0,
            m_altitudeAGL_MTR=1.5,
            m_roll_DEG=0.0,
            m_inclination_DEG=0.0,
            m_azimuth_DEG=0.0,
            m_velocityWorldU_MPS=0.0,
            m_velocityWorldV_MPS=0.0,
            m_velocityWorldW_MPS=0.0,
            m_pitchRate_DEGpSEC=0.0,
            m_rollRate_DEGpSEC=0.0,
            m_yawRate_DEGpSEC=0.0,
            m_groundspeed_MPS=0.0,
            m_flightAxisControllerIsActive=1.0,
            m_hasLostComponents=0.0,
            m_currentPhysicsTime_SEC=10.0,
            m_resetButtonHasBeenPressed=0.0,
            m_anEngineIsRunning=1.0,
            m_isLocked=0.0,
            m_isTouchingGround=0.0,
            m_currentAircraftStatus=0.0,
        )
        for name, value in overrides.items():
            setattr(state, name, value)
        return state

    def test_action_array_is_converted_to_rf_action(self):
        action = gym_action_to_rf_action(np.asarray([2.0, -2.0, 3.0, -0.25], dtype=np.float32))

        self.assertEqual(action.aileron, 1.0)
        self.assertEqual(action.elevator, -1.0)
        self.assertEqual(action.throttle, 1.0)
        self.assertEqual(action.rudder, -0.25)

    def test_observation_vector_shape_and_dtype(self):
        observation = state_to_observation(self._state())

        self.assertEqual(observation.shape, (13,))
        self.assertEqual(observation.dtype, np.float32)
        np.testing.assert_allclose(observation, np.asarray([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32))

    def test_observation_is_relative_to_hover_target_and_scaled(self):
        observation = state_to_observation(
            self._state(
                m_aircraftPositionX_MTR=10.0,
                m_aircraftPositionY_MTR=-4.0,
                m_altitudeAGL_MTR=4.5,
                m_roll_DEG=45.0,
                m_inclination_DEG=-45.0,
                m_azimuth_DEG=90.0,
                m_velocityWorldU_MPS=5.0,
                m_pitchRate_DEGpSEC=90.0,
            ),
            target_x_m=2.0,
            target_y_m=4.0,
            target_altitude_agl_m=1.5,
        )

        np.testing.assert_allclose(
            observation,
            np.asarray([1, -1, 1, 1, -1, 1, 0, 0.5, 0, 0, 0.5, 0, 0], dtype=np.float32),
            atol=1.0e-6,
        )

    def test_elevator_observation_contains_only_longitudinal_control_state(self):
        config = RewardConfig(
            target_x_m=2.0,
            target_y_m=4.0,
            target_altitude_agl_m=1.5,
            target_azimuth_deg=90.0,
        )
        features = compute_elevator_hover_features(
            self._state(
                m_aircraftPositionX_MTR=10.0,
                m_aircraftPositionY_MTR=12.0,
                m_altitudeAGL_MTR=4.5,
                m_azimuth_DEG=90.0,
                m_inclination_DEG=15.0,
                m_pitchRate_DEGpSEC=-90.0,
                m_velocityWorldU_MPS=5.0,
                m_velocityWorldV_MPS=5.0,
                m_velocityWorldW_MPS=-2.0,
            ),
            target_x_m=config.target_x_m,
            target_y_m=config.target_y_m,
            target_altitude_agl_m=config.target_altitude_agl_m,
            target_azimuth_deg=config.target_azimuth_deg,
        )
        observation = elevator_features_to_observation(
            features,
            config=config,
        )

        np.testing.assert_allclose(
            observation,
            np.asarray(
                [-3.0, -3.0, 2.0, 1.0, 2.0, -0.4],
                dtype=np.float32,
            ),
            atol=1.0e-6,
        )

    def test_elevator_profile_anchors_heading_and_zeroes_nose_up_error(self):
        client = StubRFLinkClient([
            self._state(
                m_aircraftPositionX_MTR=12.5,
                m_aircraftPositionY_MTR=-3.0,
                m_altitudeAGL_MTR=4.2,
                m_azimuth_DEG=37.0,
                m_inclination_DEG=90.0,
            ),
        ])
        env = HoverPilotHoverEnv(
            host="127.0.0.1",
            port=18083,
            task_profile=ELEVATOR_HOVER_TASK,
            client_factory=lambda: client,
        )

        observation, info = env.reset()

        self.assertEqual(observation.shape, (6,))
        self.assertAlmostEqual(observation[0], 0.0)
        self.assertTrue(env.task_profile.anchor_heading_to_reset_state)
        self.assertEqual(info["target_hover"]["inclination_deg"], 90.0)
        self.assertEqual(info["target_hover"]["azimuth_deg"], 37.0)
        self.assertEqual(env.reward_config.profile, "elevator")
        self.assertEqual(
            info["elevator_hover_features"]["inclination_error_deg"],
            0.0,
        )
        self.assertEqual(info["elevator_recovery_target_deg"], 0.0)
        self.assertEqual(info["target_hover"]["x_m"], 12.5)
        self.assertEqual(info["target_hover"]["y_m"], -3.0)
        self.assertEqual(info["target_hover"]["altitude_agl_m"], 4.2)
        self.assertEqual(env.reward_config.trainer_cylinder_radius_m, 6.0)
        env.close()

    def test_elevator_observation_uses_recovery_tracking_error(self):
        config = RewardConfig(target_azimuth_deg=0.0)
        features = compute_elevator_hover_features(
            self._state(
                m_aircraftPositionY_MTR=-4.0,
                m_inclination_DEG=90.0,
            ),
            target_x_m=config.target_x_m,
            target_y_m=config.target_y_m,
            target_altitude_agl_m=config.target_altitude_agl_m,
            target_azimuth_deg=config.target_azimuth_deg,
        )
        observation = elevator_features_to_observation(
            features,
            config=config,
        )

        self.assertAlmostEqual(observation[0], 8.0 / 15.0)
        self.assertAlmostEqual(observation[2], 1.0)

    def test_elevator_position_rate_comes_from_consecutive_positions(self):
        env = HoverPilotHoverEnv(
            host="127.0.0.1",
            port=18083,
            task_profile=ELEVATOR_HOVER_TASK,
            client_factory=lambda: StubRFLinkClient([]),
        )
        env.reward_config = RewardConfig(target_azimuth_deg=0.0)
        env._last_state = self._state(
            m_aircraftPositionY_MTR=0.0,
            m_currentPhysicsTime_SEC=10.0,
        )
        current = self._state(
            m_aircraftPositionY_MTR=-0.5,
            m_velocityWorldV_MPS=99.0,
            m_currentPhysicsTime_SEC=10.5,
        )

        features = env._compute_elevator_features(current)

        self.assertEqual(features.longitudinal_position_error_m, 0.5)
        self.assertEqual(features.longitudinal_velocity_mps, 1.0)

    def test_elevator_step_reuses_measured_rate_for_observation_reward_and_info(self):
        client = StubRFLinkClient(
            [
                self._state(m_inclination_DEG=90.0),
                self._state(
                    m_aircraftPositionY_MTR=-0.5,
                    m_inclination_DEG=90.0,
                    m_velocityWorldV_MPS=99.0,
                    m_currentPhysicsTime_SEC=10.5,
                ),
            ]
        )
        env = HoverPilotHoverEnv(
            host="127.0.0.1",
            port=18083,
            task_profile=ELEVATOR_HOVER_TASK,
            client_factory=lambda: client,
        )
        env.reset()

        observation, _, terminated, _, info = env.step(
            np.asarray([0.0, 0.0, 0.55, 0.0], dtype=np.float32)
        )

        self.assertFalse(terminated)
        self.assertAlmostEqual(observation[0], 4.0 / 15.0)
        self.assertAlmostEqual(observation[2], 0.5 / 4.0)
        self.assertAlmostEqual(observation[3], 1.0 / 5.0)
        self.assertEqual(
            info["elevator_hover_features"]["longitudinal_velocity_mps"],
            1.0,
        )
        self.assertEqual(info["elevator_recovery_target_deg"], -4.0)
        self.assertEqual(
            info["reward_breakdown"]["target_inclination_error_deg"],
            -4.0,
        )
        self.assertEqual(
            info["reward_breakdown"]["inclination_tracking_error_deg"],
            4.0,
        )
        env.close()

    def test_elevator_position_rate_handles_time_reset_teleport_and_repeated_state(self):
        env = HoverPilotHoverEnv(
            host="127.0.0.1",
            port=18083,
            task_profile=ELEVATOR_HOVER_TASK,
            client_factory=lambda: StubRFLinkClient([]),
        )
        env.reward_config = RewardConfig(target_azimuth_deg=0.0)
        env._last_state = self._state(
            m_aircraftPositionY_MTR=0.0,
            m_currentPhysicsTime_SEC=10.0,
        )
        cases = (
            (
                "physics_time_reset",
                self._state(
                    m_aircraftPositionY_MTR=-0.5,
                    m_currentPhysicsTime_SEC=8.0,
                ),
                0.0,
            ),
            (
                "trainer_teleport",
                self._state(
                    m_aircraftPositionY_MTR=-3.0,
                    m_currentPhysicsTime_SEC=10.5,
                ),
                0.0,
            ),
            (
                "repeated_physics_time",
                self._state(
                    m_aircraftPositionY_MTR=-0.5,
                    m_currentPhysicsTime_SEC=10.0,
                ),
                0.75,
            ),
        )

        for name, current, expected_rate in cases:
            with self.subTest(name=name):
                env._last_longitudinal_position_rate_mps = 0.75
                features = env._compute_elevator_features(current)
                self.assertEqual(
                    features.longitudinal_velocity_mps,
                    expected_rate,
                )

    def test_elevator_profile_rejects_midflight_tilt_as_episode_start(self):
        env = HoverPilotHoverEnv(
            host="127.0.0.1",
            port=18083,
            task_profile=ELEVATOR_HOVER_TASK,
            client_factory=lambda: StubRFLinkClient([]),
        )

        self.assertFalse(
            env._is_start_stable_state(self._state(m_inclination_DEG=81.0))
        )
        self.assertTrue(
            env._is_start_stable_state(self._state(m_inclination_DEG=89.75))
        )

    def test_elevator_reset_signal_waits_for_vertical_attitude(self):
        env = HoverPilotHoverEnv(
            host="127.0.0.1",
            port=18083,
            task_profile=ELEVATOR_HOVER_TASK,
            client_factory=lambda: StubRFLinkClient([]),
        )

        tilted = env._assess_episode_boundary(
            self._state(
                m_inclination_DEG=80.0,
                m_resetButtonHasBeenPressed=1.0,
            ),
            require_reset_boundary=True,
            pending_reset_reason=None,
        )
        vertical = env._assess_episode_boundary(
            self._state(
                m_inclination_DEG=90.0,
                m_resetButtonHasBeenPressed=1.0,
            ),
            require_reset_boundary=True,
            pending_reset_reason=None,
        )

        self.assertFalse(tilted.can_start)
        self.assertTrue(vertical.can_start)

    def test_vertical_reset_teleport_does_not_require_xy_motion(self):
        env = HoverPilotHoverEnv(
            host="127.0.0.1",
            port=18083,
            client_factory=lambda: StubRFLinkClient([]),
        )

        detected = env._looks_like_reset_teleport(
            self._state(m_altitudeAGL_MTR=0.1),
            self._state(m_altitudeAGL_MTR=1.5),
        )

        self.assertTrue(detected)

    def test_small_low_altitude_recovery_is_not_a_reset_teleport(self):
        env = HoverPilotHoverEnv(
            host="127.0.0.1",
            port=18083,
            client_factory=lambda: StubRFLinkClient([]),
        )

        detected = env._looks_like_reset_teleport(
            self._state(m_altitudeAGL_MTR=0.24),
            self._state(m_altitudeAGL_MTR=0.26),
        )

        self.assertFalse(detected)

    def test_reset_waits_for_ready_state(self):
        client = StubRFLinkClient([
            self._state(m_isLocked=1.0),
            self._state(m_isLocked=0.0),
        ])
        env = HoverPilotHoverEnv(
            host="127.0.0.1",
            port=18083,
            ready_controller_active_threshold=None,
            ready_running_threshold=None,
            client_factory=lambda: client,
        )

        observation, info = env.reset()

        self.assertEqual(observation.shape, (13,))
        self.assertEqual(info["episode_start_reason"], "reset_ready")
        self.assertTrue(info["episode_readiness"]["ready"])
        self.assertEqual(len(client.actions), 2)
        env.close()

    def test_reset_skips_inactive_stationary_reset_like_state(self):
        client = StubRFLinkClient([
            self._state(
                m_aircraftPositionX_MTR=12.0,
                m_aircraftPositionY_MTR=8.0,
                m_altitudeAGL_MTR=0.15,
                m_flightAxisControllerIsActive=0.0,
                m_anEngineIsRunning=0.0,
                m_groundspeed_MPS=0.0,
                m_airspeed_MPS=0.0,
                m_pitchRate_DEGpSEC=0.0,
                m_rollRate_DEGpSEC=0.0,
                m_yawRate_DEGpSEC=0.0,
            ),
            self._state(
                m_aircraftPositionX_MTR=0.0,
                m_aircraftPositionY_MTR=0.0,
                m_altitudeAGL_MTR=1.6,
                m_flightAxisControllerIsActive=0.0,
                m_anEngineIsRunning=0.0,
                m_groundspeed_MPS=0.0,
                m_airspeed_MPS=0.0,
                m_pitchRate_DEGpSEC=0.0,
                m_rollRate_DEGpSEC=0.0,
                m_yawRate_DEGpSEC=0.0,
            ),
            self._state(
                m_flightAxisControllerIsActive=1.0,
                m_anEngineIsRunning=1.0,
                m_groundspeed_MPS=0.3,
                m_airspeed_MPS=0.4,
                m_rollRate_DEGpSEC=8.0,
            ),
        ])
        env = HoverPilotHoverEnv(
            host="127.0.0.1",
            port=18083,
            client_factory=lambda: client,
        )

        observation, info = env.reset()

        self.assertEqual(observation.shape, (13,))
        self.assertTrue(info["episode_readiness"]["ready"])
        self.assertEqual(info["episode_start_reason"], "trainer_repositioned")
        self.assertEqual(len(client.actions), 2)
        env.close()

    def test_reset_skips_low_altitude_stationary_crash_wait_state(self):
        client = StubRFLinkClient([
            self._state(
                m_aircraftPositionX_MTR=12.0,
                m_aircraftPositionY_MTR=8.0,
                m_altitudeAGL_MTR=0.14,
                m_groundspeed_MPS=0.0,
                m_airspeed_MPS=0.0,
                m_pitchRate_DEGpSEC=0.0,
                m_rollRate_DEGpSEC=0.0,
                m_yawRate_DEGpSEC=0.0,
            ),
            self._state(
                m_aircraftPositionX_MTR=0.0,
                m_aircraftPositionY_MTR=0.0,
                m_altitudeAGL_MTR=1.6,
                m_flightAxisControllerIsActive=0.0,
                m_anEngineIsRunning=0.0,
                m_groundspeed_MPS=0.0,
                m_airspeed_MPS=0.0,
                m_pitchRate_DEGpSEC=0.0,
                m_rollRate_DEGpSEC=0.0,
                m_yawRate_DEGpSEC=0.0,
            ),
            self._state(
                m_altitudeAGL_MTR=1.8,
                m_groundspeed_MPS=0.2,
                m_airspeed_MPS=0.3,
                m_rollRate_DEGpSEC=7.0,
            ),
        ])
        env = HoverPilotHoverEnv(
            host="127.0.0.1",
            port=18083,
            client_factory=lambda: client,
            minimum_start_altitude_agl_m=0.25,
        )

        observation, info = env.reset()

        self.assertEqual(observation.shape, (13,))
        self.assertTrue(info["episode_readiness"]["ready"])
        self.assertEqual(info["episode_start_reason"], "trainer_repositioned")
        self.assertEqual(len(client.actions), 2)
        env.close()

    def test_reset_skips_low_altitude_moving_crash_wait_state(self):
        client = StubRFLinkClient([
            self._state(
                m_aircraftPositionX_MTR=12.0,
                m_aircraftPositionY_MTR=8.0,
                m_altitudeAGL_MTR=0.14,
                m_groundspeed_MPS=0.2,
                m_airspeed_MPS=0.3,
                m_pitchRate_DEGpSEC=8.0,
                m_rollRate_DEGpSEC=20.0,
                m_yawRate_DEGpSEC=5.0,
            ),
            self._state(
                m_aircraftPositionX_MTR=0.0,
                m_aircraftPositionY_MTR=0.0,
                m_altitudeAGL_MTR=1.6,
                m_flightAxisControllerIsActive=0.0,
                m_anEngineIsRunning=0.0,
                m_groundspeed_MPS=0.0,
                m_airspeed_MPS=0.0,
                m_pitchRate_DEGpSEC=0.0,
                m_rollRate_DEGpSEC=0.0,
                m_yawRate_DEGpSEC=0.0,
            ),
            self._state(
                m_altitudeAGL_MTR=1.8,
                m_groundspeed_MPS=0.2,
                m_airspeed_MPS=0.3,
                m_rollRate_DEGpSEC=7.0,
            ),
        ])
        env = HoverPilotHoverEnv(
            host="127.0.0.1",
            port=18083,
            client_factory=lambda: client,
            minimum_start_altitude_agl_m=0.25,
        )

        observation, info = env.reset()

        self.assertEqual(observation.shape, (13,))
        self.assertEqual(info["episode_start_reason"], "trainer_repositioned")
        self.assertEqual(len(client.actions), 2)
        env.close()

    def test_reset_skips_fast_falling_crash_state(self):
        client = StubRFLinkClient([
            self._state(
                m_aircraftPositionX_MTR=12.0,
                m_aircraftPositionY_MTR=8.0,
                m_altitudeAGL_MTR=1.2,
                m_flightAxisControllerIsActive=0.0,
                m_anEngineIsRunning=0.0,
                m_groundspeed_MPS=3.5,
                m_airspeed_MPS=4.0,
                m_pitchRate_DEGpSEC=85.0,
                m_rollRate_DEGpSEC=120.0,
                m_yawRate_DEGpSEC=70.0,
            ),
            self._state(
                m_aircraftPositionX_MTR=0.0,
                m_aircraftPositionY_MTR=0.0,
                m_altitudeAGL_MTR=1.6,
                m_flightAxisControllerIsActive=0.0,
                m_anEngineIsRunning=0.0,
                m_groundspeed_MPS=0.0,
                m_airspeed_MPS=0.0,
                m_pitchRate_DEGpSEC=0.0,
                m_rollRate_DEGpSEC=0.0,
                m_yawRate_DEGpSEC=0.0,
            ),
            self._state(
                m_altitudeAGL_MTR=1.8,
                m_groundspeed_MPS=0.2,
                m_airspeed_MPS=0.3,
                m_pitchRate_DEGpSEC=4.0,
                m_rollRate_DEGpSEC=8.0,
                m_yawRate_DEGpSEC=3.0,
            ),
        ])
        env = HoverPilotHoverEnv(
            host="127.0.0.1",
            port=18083,
            client_factory=lambda: client,
        )

        observation, info = env.reset()

        self.assertEqual(observation.shape, (13,))
        self.assertTrue(info["episode_readiness"]["ready"])
        self.assertEqual(info["episode_start_reason"], "trainer_repositioned")
        self.assertEqual(len(client.actions), 2)
        env.close()

    def test_reset_detects_vertical_reposition_from_crash_wait_state(self):
        client = StubRFLinkClient([
            self._state(
                m_aircraftPositionX_MTR=12.0,
                m_aircraftPositionY_MTR=8.0,
                m_altitudeAGL_MTR=0.14,
                m_groundspeed_MPS=0.0,
                m_airspeed_MPS=0.0,
                m_pitchRate_DEGpSEC=0.0,
                m_rollRate_DEGpSEC=0.0,
                m_yawRate_DEGpSEC=0.0,
            ),
            self._state(
                m_aircraftPositionX_MTR=12.0,
                m_aircraftPositionY_MTR=8.0,
                m_altitudeAGL_MTR=1.8,
                m_groundspeed_MPS=0.2,
                m_airspeed_MPS=0.3,
                m_pitchRate_DEGpSEC=3.0,
                m_rollRate_DEGpSEC=4.0,
                m_yawRate_DEGpSEC=2.0,
            ),
        ])
        env = HoverPilotHoverEnv(
            host="127.0.0.1",
            port=18083,
            client_factory=lambda: client,
        )

        observation, info = env.reset()

        self.assertEqual(observation.shape, (13,))
        self.assertEqual(info["episode_start_reason"], "trainer_repositioned")
        self.assertEqual(len(client.actions), 2)
        env.close()

    def test_wait_for_next_episode_starts_from_repositioned_reset_signal_even_if_inactive(self):
        client = StubRFLinkClient([
            self._state(),
            self._state(m_currentPhysicsTime_SEC=10.2, m_aircraftPositionX_MTR=11.0),
            self._state(
                m_currentPhysicsTime_SEC=10.4,
                m_aircraftPositionX_MTR=0.0,
                m_aircraftPositionY_MTR=0.0,
                m_altitudeAGL_MTR=1.6,
                m_flightAxisControllerIsActive=0.0,
                m_anEngineIsRunning=0.0,
                m_groundspeed_MPS=0.2,
                m_airspeed_MPS=0.3,
                m_pitchRate_DEGpSEC=3.0,
                m_rollRate_DEGpSEC=4.0,
                m_yawRate_DEGpSEC=2.0,
            ),
        ])
        env = HoverPilotHoverEnv(
            host="127.0.0.1",
            port=18083,
            client_factory=lambda: client,
            anchor_target_to_reset_state=False,
        )
        env.reset()

        _, _, terminated, _, info = env.step(np.asarray([0.0, 0.0, 0.5, 0.0], dtype=np.float32))
        self.assertTrue(terminated)
        self.assertEqual(
            info["termination_reason"],
            "outside_trainer_cylinder",
        )

        observation, next_info = env.wait_for_next_episode(action=np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32))

        self.assertEqual(observation.shape, (13,))
        self.assertEqual(next_info["episode_start_reason"], "trainer_repositioned")
        env.close()

    def test_wait_for_next_episode_detects_low_agl_to_reset_jump(self):
        client = StubRFLinkClient([
            self._state(),
            self._state(
                m_currentPhysicsTime_SEC=10.2,
                m_aircraftPositionX_MTR=12.0,
                m_aircraftPositionY_MTR=8.0,
                m_altitudeAGL_MTR=0.12,
                m_groundspeed_MPS=0.0,
                m_airspeed_MPS=0.0,
            ),
            self._state(
                m_currentPhysicsTime_SEC=10.4,
                m_aircraftPositionX_MTR=0.0,
                m_aircraftPositionY_MTR=0.0,
                m_altitudeAGL_MTR=1.8,
                m_flightAxisControllerIsActive=0.0,
                m_anEngineIsRunning=0.0,
                m_groundspeed_MPS=1.2,
                m_airspeed_MPS=1.3,
                m_pitchRate_DEGpSEC=12.0,
                m_rollRate_DEGpSEC=15.0,
                m_yawRate_DEGpSEC=8.0,
            ),
            self._state(
                m_currentPhysicsTime_SEC=10.6,
                m_aircraftPositionX_MTR=0.0,
                m_aircraftPositionY_MTR=0.0,
                m_altitudeAGL_MTR=1.8,
                m_flightAxisControllerIsActive=0.0,
                m_anEngineIsRunning=0.0,
                m_groundspeed_MPS=0.2,
                m_airspeed_MPS=0.3,
                m_pitchRate_DEGpSEC=3.0,
                m_rollRate_DEGpSEC=4.0,
                m_yawRate_DEGpSEC=2.0,
            ),
        ])
        env = HoverPilotHoverEnv(
            host="127.0.0.1",
            port=18083,
            client_factory=lambda: client,
            anchor_target_to_reset_state=False,
        )
        env.reset()

        _, _, terminated, _, info = env.step(np.asarray([0.0, 0.0, 0.5, 0.0], dtype=np.float32))
        self.assertTrue(terminated)
        self.assertEqual(
            info["termination_reason"],
            "outside_trainer_cylinder",
        )

        observation, next_info = env.wait_for_next_episode(action=np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32))

        self.assertEqual(observation.shape, (13,))
        self.assertEqual(next_info["episode_start_reason"], "trainer_repositioned")
        env.close()

    def test_reset_returns_observation_and_info(self):
        client = StubRFLinkClient([self._state()])
        env = HoverPilotHoverEnv(
            host="127.0.0.1",
            port=18083,
            client_factory=lambda: client,
        )

        observation, info = env.reset()

        self.assertEqual(observation.shape, (13,))
        self.assertIs(env.task_profile, STANDARD_HOVER_TASK)
        self.assertIsInstance(info, dict)
        self.assertIn("state_summary", info)
        self.assertEqual(info["episode_start_reason"], "reset_ready")
        self.assertTrue(client.connected)
        self.assertIsInstance(client.actions[0], RFControlAction)
        env.close()

    def test_reset_anchors_target_to_current_state_by_default(self):
        client = StubRFLinkClient([
            self._state(m_aircraftPositionX_MTR=12.5, m_aircraftPositionY_MTR=-3.0, m_altitudeAGL_MTR=4.2)
        ])
        env = HoverPilotHoverEnv(
            host="127.0.0.1",
            port=18083,
            client_factory=lambda: client,
        )

        _, info = env.reset()

        self.assertEqual(info["target_hover"]["x_m"], 12.5)
        self.assertEqual(info["target_hover"]["y_m"], -3.0)
        self.assertEqual(info["target_hover"]["altitude_agl_m"], 4.2)
        env.close()

    def test_locked_state_is_not_ready(self):
        env = HoverPilotHoverEnv(host="127.0.0.1", port=18083, client_factory=lambda: StubRFLinkClient([]))
        readiness = env.compute_episode_start_status(self._state(m_isLocked=1.0))

        self.assertIsInstance(readiness, EpisodeLifecycleResult)
        self.assertFalse(readiness.ready)
        self.assertEqual(readiness.reason, "vehicle_locked")

    def test_inactive_controller_state_is_not_ready_when_required(self):
        env = HoverPilotHoverEnv(
            host="127.0.0.1",
            port=18083,
            ready_controller_active_threshold=0.5,
            client_factory=lambda: StubRFLinkClient([]),
        )
        readiness = env.compute_episode_start_status(self._state(m_flightAxisControllerIsActive=0.0))

        self.assertFalse(readiness.ready)
        self.assertEqual(readiness.reason, "controller_inactive")

    def test_engine_stopped_state_is_not_ready_when_required(self):
        env = HoverPilotHoverEnv(
            host="127.0.0.1",
            port=18083,
            ready_running_threshold=0.5,
            client_factory=lambda: StubRFLinkClient([]),
        )
        readiness = env.compute_episode_start_status(self._state(m_anEngineIsRunning=0.0))

        self.assertFalse(readiness.ready)
        self.assertEqual(readiness.reason, "engine_stopped")

    def test_reset_timeout_raises_clearly(self):
        client = StubRFLinkClient([self._state(m_isLocked=1.0) for _ in range(4)])
        env = HoverPilotHoverEnv(
            host="127.0.0.1",
            port=18083,
            max_reset_wait_seconds=0.0,
            reset_poll_interval_seconds=0.0,
            client_factory=lambda: client,
        )

        with self.assertRaises(TimeoutError):
            env.reset()

        env.close()

    def test_step_returns_gymnasium_tuple(self):
        client = StubRFLinkClient([
            self._state(),
            self._state(m_aircraftPositionX_MTR=1.0, m_currentPhysicsTime_SEC=10.1),
        ])
        env = HoverPilotHoverEnv(
            host="127.0.0.1",
            port=18083,
            reward_config=RewardConfig(),
            client_factory=lambda: client,
        )
        env.reset()

        result = env.step(np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32))

        self.assertEqual(len(result), 5)
        observation, reward, terminated, truncated, info = result
        self.assertEqual(observation.shape, (13,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)
        self.assertIn("reward_breakdown", info)
        self.assertIn("episode_lifecycle", info)
        env.close()

    def test_episode_truncation_logic(self):
        client = StubRFLinkClient([
            self._state(),
            self._state(m_currentPhysicsTime_SEC=10.1),
            self._state(m_currentPhysicsTime_SEC=10.2),
        ])
        env = HoverPilotHoverEnv(
            host="127.0.0.1",
            port=18083,
            max_episode_steps=1,
            client_factory=lambda: client,
        )
        env.reset()

        _, _, terminated, truncated, info = env.step(
            np.asarray([0.0, 0.0, 0.5, 0.0], dtype=np.float32)
        )

        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["episode_step"], 1)
        continued_observation, continued_info = env.continue_after_truncation()
        self.assertEqual(continued_observation.shape, (13,))
        self.assertEqual(
            continued_info["episode_start_reason"],
            "time_limit_continuation",
        )
        self.assertEqual(continued_info["episode_step"], 0)
        self.assertFalse(client.closed)

        _, _, terminated, truncated, _ = env.step(
            np.asarray([0.0, 0.0, 0.5, 0.0], dtype=np.float32)
        )
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        env.close()

    def test_close_resets_client_instance(self):
        client = StubRFLinkClient([self._state()])
        env = HoverPilotHoverEnv(
            host="127.0.0.1",
            port=18083,
            client_factory=lambda: client,
        )

        env.reset()
        env.close()

        self.assertTrue(client.closed)

    def test_lost_components_terminates_episode(self):
        client = StubRFLinkClient([
            self._state(),
            self._state(m_currentPhysicsTime_SEC=10.1, m_hasLostComponents=1.0),
        ])
        env = HoverPilotHoverEnv(host="127.0.0.1", port=18083, client_factory=lambda: client)
        env.reset()

        _, _, terminated, _, info = env.step(np.asarray([0.0, 0.0, 0.5, 0.0], dtype=np.float32))

        self.assertTrue(terminated)
        self.assertEqual(info["termination_reason"], "lost_components")
        env.close()

    def test_controller_inactive_terminates_when_threshold_configured(self):
        client = StubRFLinkClient([
            self._state(),
            self._state(m_currentPhysicsTime_SEC=10.1, m_flightAxisControllerIsActive=0.0),
        ])
        env = HoverPilotHoverEnv(
            host="127.0.0.1",
            port=18083,
            reward_config=RewardConfig(controller_active_threshold=0.5),
            client_factory=lambda: client,
        )
        env.reset()

        _, _, terminated, _, info = env.step(np.asarray([0.0, 0.0, 0.5, 0.0], dtype=np.float32))

        self.assertTrue(terminated)
        self.assertEqual(info["termination_reason"], "controller_inactive")
        env.close()

    def test_engine_stopped_terminates_when_configured(self):
        client = StubRFLinkClient([
            self._state(),
            self._state(m_currentPhysicsTime_SEC=10.1, m_anEngineIsRunning=0.0),
        ])
        env = HoverPilotHoverEnv(
            host="127.0.0.1",
            port=18083,
            reward_config=RewardConfig(terminate_on_engine_stopped=True),
            client_factory=lambda: client,
        )
        env.reset()

        _, _, terminated, _, info = env.step(np.asarray([0.0, 0.0, 0.5, 0.0], dtype=np.float32))

        self.assertTrue(terminated)
        self.assertEqual(info["termination_reason"], "engine_stopped")
        env.close()

    def test_touching_ground_before_start_is_allowed_when_configured(self):
        client = StubRFLinkClient([
            self._state(m_isTouchingGround=1.0),
        ])
        env = HoverPilotHoverEnv(
            host="127.0.0.1",
            port=18083,
            allow_ground_contact_at_ready=True,
            client_factory=lambda: client,
        )

        observation, info = env.reset()

        self.assertEqual(observation.shape, (13,))
        self.assertTrue(info["episode_readiness"]["ready"])
        env.close()

    def test_touching_ground_after_start_does_terminate(self):
        client = StubRFLinkClient([
            self._state(),
            self._state(m_currentPhysicsTime_SEC=10.1, m_isTouchingGround=1.0),
        ])
        env = HoverPilotHoverEnv(
            host="127.0.0.1",
            port=18083,
            reward_config=RewardConfig(ground_contact_grace_seconds=0.0),
            client_factory=lambda: client,
        )
        env.reset()

        _, _, terminated, _, info = env.step(np.asarray([0.0, 0.0, 0.5, 0.0], dtype=np.float32))

        self.assertTrue(terminated)
        self.assertEqual(info["termination_reason"], "touching_ground")
        env.close()

    def test_parked_on_ground_state_terminates_episode_and_waits_for_reset(self):
        client = StubRFLinkClient([
            self._state(
                m_altitudeAGL_MTR=1.6,
                m_groundspeed_MPS=0.2,
                m_airspeed_MPS=0.2,
                m_pitchRate_DEGpSEC=2.0,
                m_rollRate_DEGpSEC=3.0,
                m_yawRate_DEGpSEC=1.0,
            ),
            self._state(
                m_currentPhysicsTime_SEC=10.2,
                m_altitudeAGL_MTR=0.14,
                m_groundspeed_MPS=0.0,
                m_airspeed_MPS=0.0,
                m_pitchRate_DEGpSEC=0.0,
                m_rollRate_DEGpSEC=0.0,
                m_yawRate_DEGpSEC=0.0,
            ),
            self._state(
                m_currentPhysicsTime_SEC=10.4,
                m_aircraftPositionX_MTR=0.5,
                m_aircraftPositionY_MTR=-0.2,
                m_altitudeAGL_MTR=1.7,
                m_groundspeed_MPS=0.3,
                m_airspeed_MPS=0.4,
                m_pitchRate_DEGpSEC=4.0,
                m_rollRate_DEGpSEC=5.0,
                m_yawRate_DEGpSEC=2.0,
            ),
        ])
        env = HoverPilotHoverEnv(
            host="127.0.0.1",
            port=18083,
            client_factory=lambda: client,
            minimum_start_altitude_agl_m=0.25,
        )
        env.reset()

        _, _, terminated, truncated, info = env.step(
            np.asarray([0.0, 0.0, 0.5, 0.0], dtype=np.float32)
        )

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["termination_reason"], "parked_on_ground")
        self.assertEqual(
            info["reward_breakdown"]["terminal_penalty"],
            env.reward_config.terminal_failure_reward,
        )
        self.assertTrue(info["waiting_for_reset"])

        started, observation, next_info = env.poll_wait_for_next_episode(
            action=np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        )

        self.assertTrue(started)
        self.assertEqual(observation.shape, (13,))
        self.assertEqual(
            next_info["episode_start_reason"],
            "trainer_repositioned",
        )
        env.close()

    def test_boundary_logic_still_works(self):
        client = StubRFLinkClient([
            self._state(),
            self._state(m_currentPhysicsTime_SEC=10.1, m_aircraftPositionX_MTR=20.0),
        ])
        env = HoverPilotHoverEnv(
            host="127.0.0.1",
            port=18083,
            client_factory=lambda: client,
            anchor_target_to_reset_state=False,
        )
        env.reset()

        _, _, terminated, _, info = env.step(np.asarray([0.0, 0.0, 0.5, 0.0], dtype=np.float32))

        self.assertTrue(terminated)
        self.assertEqual(
            info["termination_reason"],
            "outside_trainer_cylinder",
        )
        env.close()

    def test_wait_for_next_episode_uses_pending_start_immediately(self):
        client = StubRFLinkClient([
            self._state(),
            self._state(m_currentPhysicsTime_SEC=1.0),
        ])
        env = HoverPilotHoverEnv(
            host="127.0.0.1",
            port=18083,
            client_factory=lambda: client,
        )
        env.reset()

        env.step(np.asarray([0.0, 0.0, 0.5, 0.0], dtype=np.float32))
        observation, info = env.wait_for_next_episode()

        self.assertEqual(observation.shape, (13,))
        self.assertEqual(info["episode_start_reason"], "trainer_reset")
        env.close()

    def test_reset_button_during_episode_starts_new_episode(self):
        client = StubRFLinkClient([
            self._state(),
            self._state(
                m_currentPhysicsTime_SEC=10.2,
                m_resetButtonHasBeenPressed=1.0,
                m_altitudeAGL_MTR=1.8,
                m_groundspeed_MPS=0.2,
                m_airspeed_MPS=0.3,
                m_pitchRate_DEGpSEC=3.0,
                m_rollRate_DEGpSEC=4.0,
                m_yawRate_DEGpSEC=2.0,
            ),
        ])
        env = HoverPilotHoverEnv(
            host="127.0.0.1",
            port=18083,
            client_factory=lambda: client,
        )
        env.reset()

        _, _, terminated, truncated, info = env.step(np.asarray([0.0, 0.0, 0.5, 0.0], dtype=np.float32))

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["termination_reason"], "trainer_reset_button")

        started, observation, next_info = env.poll_wait_for_next_episode(
            action=np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        )

        self.assertTrue(started)
        self.assertEqual(observation.shape, (13,))
        self.assertEqual(next_info["episode_start_reason"], "trainer_reset_button")
        env.close()

    def test_step_detects_reset_teleport_before_boundary_failure(self):
        client = StubRFLinkClient([
            self._state(),
            self._state(
                m_currentPhysicsTime_SEC=10.2,
                m_aircraftPositionX_MTR=20.0,
                m_aircraftPositionY_MTR=20.0,
                m_altitudeAGL_MTR=1.5,
                m_flightAxisControllerIsActive=0.0,
                m_anEngineIsRunning=0.0,
                m_groundspeed_MPS=0.0,
                m_airspeed_MPS=0.0,
                m_pitchRate_DEGpSEC=0.0,
                m_rollRate_DEGpSEC=0.0,
                m_yawRate_DEGpSEC=0.0,
            ),
        ])
        env = HoverPilotHoverEnv(
            host="127.0.0.1",
            port=18083,
            client_factory=lambda: client,
            anchor_target_to_reset_state=False,
            reset_teleport_distance_m=2.0,
        )
        env.reset()
        env.reward_config = RewardConfig(target_x_m=20.0, target_y_m=20.0, target_altitude_agl_m=1.5)

        _, _, terminated, _, info = env.step(np.asarray([0.0, 0.0, 0.5, 0.0], dtype=np.float32))

        self.assertTrue(terminated)
        self.assertEqual(info["termination_reason"], "trainer_repositioned")
        self.assertEqual(info["episode_lifecycle"]["reason"], "trainer_repositioned")
        env.close()

    def test_step_detects_reset_teleport_even_when_far_from_current_target(self):
        client = StubRFLinkClient([
            self._state(m_aircraftPositionX_MTR=50.0, m_aircraftPositionY_MTR=50.0),
            self._state(
                m_currentPhysicsTime_SEC=10.2,
                m_aircraftPositionX_MTR=0.0,
                m_aircraftPositionY_MTR=0.0,
                m_altitudeAGL_MTR=1.5,
                m_flightAxisControllerIsActive=0.0,
                m_anEngineIsRunning=0.0,
                m_groundspeed_MPS=0.0,
                m_airspeed_MPS=0.0,
                m_pitchRate_DEGpSEC=0.0,
                m_rollRate_DEGpSEC=0.0,
                m_yawRate_DEGpSEC=0.0,
            ),
        ])
        env = HoverPilotHoverEnv(
            host="127.0.0.1",
            port=18083,
            client_factory=lambda: client,
            reset_teleport_distance_m=2.0,
        )
        env.reset()

        _, _, terminated, _, info = env.step(np.asarray([0.0, 0.0, 0.5, 0.0], dtype=np.float32))

        self.assertTrue(terminated)
        self.assertEqual(info["termination_reason"], "trainer_repositioned")
        self.assertEqual(
            info["reward_breakdown"]["terminal_penalty"],
            env.reward_config.terminal_failure_reward,
        )
        env.close()

    def test_wait_for_next_episode_detects_repositioned_ready_state(self):
        client = StubRFLinkClient([
            self._state(),
            self._state(m_currentPhysicsTime_SEC=10.2, m_aircraftPositionX_MTR=11.0),
            self._state(
                m_currentPhysicsTime_SEC=10.4,
                m_aircraftPositionX_MTR=0.2,
                m_aircraftPositionY_MTR=-0.2,
                m_altitudeAGL_MTR=1.6,
                m_flightAxisControllerIsActive=0.0,
                m_anEngineIsRunning=0.0,
                m_groundspeed_MPS=0.0,
                m_airspeed_MPS=0.0,
                m_pitchRate_DEGpSEC=0.0,
                m_rollRate_DEGpSEC=0.0,
                m_yawRate_DEGpSEC=0.0,
            ),
            self._state(
                m_currentPhysicsTime_SEC=10.6,
                m_aircraftPositionX_MTR=0.2,
                m_aircraftPositionY_MTR=-0.2,
                m_altitudeAGL_MTR=1.6,
                m_flightAxisControllerIsActive=0.0,
                m_anEngineIsRunning=0.0,
                m_groundspeed_MPS=0.3,
                m_airspeed_MPS=0.4,
                m_rollRate_DEGpSEC=8.0,
            ),
        ])
        env = HoverPilotHoverEnv(
            host="127.0.0.1",
            port=18083,
            client_factory=lambda: client,
            anchor_target_to_reset_state=False,
        )
        env.reset()

        _, _, terminated, _, info = env.step(np.asarray([0.0, 0.0, 0.5, 0.0], dtype=np.float32))
        self.assertTrue(terminated)
        self.assertEqual(
            info["termination_reason"],
            "outside_trainer_cylinder",
        )
        self.assertTrue(info["waiting_for_reset"])

        observation, next_info = env.wait_for_next_episode(action=np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32))

        self.assertEqual(observation.shape, (13,))
        self.assertEqual(next_info["episode_start_reason"], "trainer_repositioned")
        env.close()


if __name__ == "__main__":
    unittest.main()
