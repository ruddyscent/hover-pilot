import unittest

from hoverpilot.rflink.models import FlightAxisState
from hoverpilot.training.hover import (
    REALFLIGHT_VERTICAL_HOVER_INCLINATION_DEG,
    REWARD_PROFILE_ELEVATOR,
    RewardConfig,
    angular_error_deg,
    compute_elevator_hover_features,
    compute_elevator_recovery_target_deg,
    compute_reward,
    compute_termination,
    project_onto_target_heading,
    signed_vertical_inclination_error_deg,
)


class HoverTrainingTests(unittest.TestCase):
    def setUp(self):
        self.config = RewardConfig(
            target_x_m=0.0,
            target_y_m=0.0,
            target_altitude_agl_m=1.5,
            trainer_cylinder_radius_m=10.0,
            min_altitude_agl_m=0.5,
            boundary_proximity_weight=2.0,
            terminal_failure_reward=-50.0,
        )

    def _state(self, **overrides):
        state = FlightAxisState(
            m_aircraftPositionX_MTR=0.0,
            m_aircraftPositionY_MTR=0.0,
            m_altitudeAGL_MTR=1.5,
            m_roll_DEG=0.0,
            m_inclination_DEG=0.0,
            m_flightAxisControllerIsActive=1.0,
            m_hasLostComponents=0.0,
        )
        for name, value in overrides.items():
            setattr(state, name, value)
        return state

    def test_inside_boundary_state_is_not_terminated(self):
        result = compute_termination(self._state(), self.config)

        self.assertFalse(result.terminated)
        self.assertIsNone(result.termination_reason)

    def test_near_boundary_state_gets_higher_penalty(self):
        centered = compute_reward(self._state(m_aircraftPositionX_MTR=0.0), self.config)
        near_x_edge = compute_reward(self._state(m_aircraftPositionX_MTR=9.4), self.config)

        self.assertGreater(
            near_x_edge.boundary_proximity_penalty,
            centered.boundary_proximity_penalty,
        )
        self.assertLess(near_x_edge.reward, centered.reward)
        self.assertFalse(near_x_edge.terminated)

    def test_outside_boundary_state_is_terminated(self):
        result = compute_termination(self._state(m_aircraftPositionX_MTR=10.1), self.config)
        reward = compute_reward(self._state(m_aircraftPositionX_MTR=10.1), self.config)

        self.assertTrue(result.terminated)
        self.assertEqual(result.termination_reason, "outside_trainer_cylinder")
        self.assertTrue(reward.terminated)
        self.assertEqual(
            reward.termination_reason,
            "outside_trainer_cylinder",
        )
        self.assertLess(reward.reward, -40.0)

    def test_cylinder_boundary_is_centered_on_reset_xy(self):
        config = RewardConfig(
            target_x_m=100.0,
            target_y_m=-50.0,
            trainer_cylinder_radius_m=5.0,
        )

        inside = compute_termination(
            self._state(
                m_aircraftPositionX_MTR=103.0,
                m_aircraftPositionY_MTR=-46.0,
                m_altitudeAGL_MTR=20.0,
            ),
            config,
        )
        outside = compute_termination(
            self._state(
                m_aircraftPositionX_MTR=103.1,
                m_aircraftPositionY_MTR=-46.0,
                m_altitudeAGL_MTR=20.0,
            ),
            config,
        )

        self.assertFalse(inside.terminated)
        self.assertTrue(outside.terminated)
        self.assertEqual(
            outside.termination_reason,
            "outside_trainer_cylinder",
        )

    def test_altitude_does_not_reduce_cylinder_radius(self):
        config = RewardConfig(
            trainer_cylinder_radius_m=5.0,
            min_altitude_agl_m=-10.0,
            boundary_proximity_weight=1.0,
            proximity_penalty_margin_ratio=0.2,
        )
        low = self._state(
            m_aircraftPositionX_MTR=3.0,
            m_altitudeAGL_MTR=1.0,
        )
        high = self._state(
            m_aircraftPositionX_MTR=3.0,
            m_altitudeAGL_MTR=20.0,
        )

        low_termination = compute_termination(low, config)
        high_termination = compute_termination(high, config)
        low_reward = compute_reward(low, config)
        high_reward = compute_reward(high, config)

        self.assertFalse(low_termination.terminated)
        self.assertFalse(high_termination.terminated)
        self.assertEqual(
            low_reward.boundary_proximity_penalty,
            high_reward.boundary_proximity_penalty,
        )

    def test_cylinder_radius_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "trainer_cylinder_radius_m"):
            RewardConfig(trainer_cylinder_radius_m=0.0)

    def test_elevator_observation_scales_must_be_finite_and_positive(self):
        invalid_scales = {
            "inclination_error_scale_deg": 0.0,
            "pitch_rate_scale_deg_s": -1.0,
            "longitudinal_position_scale_m": float("nan"),
            "altitude_error_scale_m": float("inf"),
            "velocity_error_scale_mps": 0.0,
        }

        for name, value in invalid_scales.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, name):
                    RewardConfig(**{name: value})

    def test_low_altitude_boundary_termination(self):
        result = compute_termination(self._state(m_altitudeAGL_MTR=0.2), self.config)

        self.assertTrue(result.terminated)
        self.assertEqual(result.termination_reason, "altitude_too_low")

    def test_lost_components_termination_uses_state_flag(self):
        result = compute_termination(self._state(m_hasLostComponents=1.0), self.config)

        self.assertTrue(result.terminated)
        self.assertEqual(result.termination_reason, "lost_components")

    def test_vehicle_locked_is_terminal(self):
        result = compute_termination(self._state(m_isLocked=1.0), self.config)

        self.assertTrue(result.terminated)
        self.assertEqual(result.termination_reason, "vehicle_locked")

    def test_touching_ground_only_terminates_after_episode_start(self):
        before_start = compute_termination(
            self._state(m_isTouchingGround=1.0),
            RewardConfig(ground_contact_grace_seconds=0.0),
            episode_started=False,
        )
        after_start = compute_termination(
            self._state(m_isTouchingGround=1.0),
            RewardConfig(ground_contact_grace_seconds=0.0),
            episode_started=True,
            ground_contact_duration_s=0.0,
        )

        self.assertFalse(before_start.terminated)
        self.assertTrue(after_start.terminated)
        self.assertEqual(after_start.termination_reason, "touching_ground")

    def test_engine_stopped_only_terminates_when_enabled(self):
        disabled = compute_termination(self._state(m_anEngineIsRunning=0.0), self.config)
        enabled = compute_termination(
            self._state(m_anEngineIsRunning=0.0),
            RewardConfig(terminate_on_engine_stopped=True),
        )

        self.assertFalse(disabled.terminated)
        self.assertTrue(enabled.terminated)
        self.assertEqual(enabled.termination_reason, "engine_stopped")

    def test_controller_inactive_only_terminates_when_threshold_is_enabled(self):
        disabled = compute_termination(self._state(m_flightAxisControllerIsActive=0.0), self.config)
        enabled = compute_termination(
            self._state(m_flightAxisControllerIsActive=0.0),
            RewardConfig(controller_active_threshold=0.5),
        )

        self.assertFalse(disabled.terminated)
        self.assertTrue(enabled.terminated)
        self.assertEqual(enabled.termination_reason, "controller_inactive")

    def test_elevator_reward_maps_vertical_realflight_attitude_to_zero_error(self):
        config = RewardConfig(
            profile=REWARD_PROFILE_ELEVATOR,
            boundary_proximity_weight=0.0,
        )

        balanced = compute_reward(self._state(m_inclination_DEG=90.0), config)
        tilted = compute_reward(self._state(m_inclination_DEG=112.5), config)
        rotating = compute_reward(
            self._state(
                m_inclination_DEG=90.0,
                m_pitchRate_DEGpSEC=90.0,
            ),
            config,
        )

        self.assertEqual(balanced.attitude_penalty, 0.0)
        self.assertEqual(balanced.angular_rate_penalty, 0.0)
        self.assertAlmostEqual(balanced.reward, config.survival_reward)
        self.assertAlmostEqual(tilted.attitude_penalty, 2.25)
        self.assertAlmostEqual(rotating.angular_rate_penalty, 4.5)
        self.assertLess(tilted.reward, balanced.reward)
        self.assertLess(rotating.reward, balanced.reward)

    def test_elevator_recovery_target_is_symmetric_and_bounded(self):
        self.assertEqual(
            compute_elevator_recovery_target_deg(4.0, 2.0),
            -14.0,
        )
        self.assertEqual(
            compute_elevator_recovery_target_deg(-4.0, -2.0),
            14.0,
        )
        self.assertEqual(
            compute_elevator_recovery_target_deg(10.0, 5.0),
            -30.0,
        )

    def test_elevator_reward_prefers_opposite_tilt_during_drift(self):
        config = RewardConfig(
            profile=REWARD_PROFILE_ELEVATOR,
            boundary_proximity_weight=0.0,
        )
        vertical = compute_reward(
            self._state(
                m_aircraftPositionY_MTR=-4.0,
                m_velocityWorldV_MPS=-2.0,
                m_inclination_DEG=90.0,
            ),
            config,
        )
        recovering = compute_reward(
            self._state(
                m_aircraftPositionY_MTR=-4.0,
                m_velocityWorldV_MPS=-2.0,
                m_inclination_DEG=76.0,
            ),
            config,
        )

        self.assertEqual(vertical.target_inclination_error_deg, -14.0)
        self.assertEqual(recovering.target_inclination_error_deg, -14.0)
        self.assertAlmostEqual(vertical.attitude_penalty, (14.0 / 15.0) ** 2)
        self.assertAlmostEqual(recovering.attitude_penalty, 0.0)
        self.assertGreater(recovering.reward, vertical.reward)

    def test_vertical_reference_is_fixed_instead_of_reward_configurable(self):
        with self.assertRaises(TypeError):
            RewardConfig(target_inclination_deg=45.0)

        features = compute_elevator_hover_features(
            self._state(
                m_inclination_DEG=REALFLIGHT_VERTICAL_HOVER_INCLINATION_DEG,
                m_azimuth_DEG=215.0,
            ),
            target_x_m=0.0,
            target_y_m=0.0,
            target_altitude_agl_m=1.5,
            target_azimuth_deg=35.0,
        )

        self.assertAlmostEqual(features.inclination_error_deg, 0.0)

    def test_elevator_reward_penalizes_longitudinal_not_lateral_error(self):
        config = RewardConfig(
            profile=REWARD_PROFILE_ELEVATOR,
            target_azimuth_deg=90.0,
            boundary_proximity_weight=0.0,
        )

        longitudinal = compute_reward(self._state(m_aircraftPositionX_MTR=4.0), config)
        lateral = compute_reward(self._state(m_aircraftPositionY_MTR=4.0), config)

        self.assertGreater(longitudinal.position_penalty, 0.0)
        self.assertAlmostEqual(lateral.position_penalty, 0.0)

    def test_world_projection_uses_realflight_azimuth_convention(self):
        self.assertAlmostEqual(project_onto_target_heading(4.0, 0.0, 90.0), 4.0)
        self.assertAlmostEqual(project_onto_target_heading(0.0, -4.0, 0.0), 4.0)

    def test_longitudinal_velocity_can_use_measured_position_rate(self):
        features = compute_elevator_hover_features(
            self._state(m_velocityWorldV_MPS=-2.0),
            target_x_m=0.0,
            target_y_m=0.0,
            target_altitude_agl_m=1.5,
            target_azimuth_deg=0.0,
            longitudinal_position_rate_mps=-3.5,
        )

        self.assertEqual(features.longitudinal_velocity_mps, -3.5)

    def test_elevator_reward_penalizes_velocity_and_action_changes(self):
        config = RewardConfig(
            profile=REWARD_PROFILE_ELEVATOR,
            boundary_proximity_weight=0.0,
        )

        moving = compute_reward(
            self._state(
                m_velocityWorldV_MPS=-5.0,
                m_velocityWorldW_MPS=-5.0,
            ),
            config,
        )
        abrupt = compute_reward(self._state(), config, elevator_delta=1.0)

        self.assertAlmostEqual(moving.velocity_penalty, 0.4)
        self.assertAlmostEqual(abrupt.action_smoothness_penalty, 0.0025)

    def test_angular_error_wraps_across_360_degrees(self):
        self.assertAlmostEqual(angular_error_deg(179.0, -179.0), -2.0)

    def test_vertical_inclination_error_distinguishes_opposite_tilts(self):
        self.assertAlmostEqual(
            signed_vertical_inclination_error_deg(80.0, 30.0, 30.0),
            -10.0,
        )
        self.assertAlmostEqual(
            signed_vertical_inclination_error_deg(80.0, 210.0, 30.0),
            10.0,
        )
        self.assertAlmostEqual(
            signed_vertical_inclination_error_deg(90.0, 210.0, 30.0),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
