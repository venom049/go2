"""Gait evaluation mathematics without an Isaac Sim application."""
import json
import math
from types import SimpleNamespace
import unittest

try:
    import torch
    from go2_him.gait_metrics import EpisodeCounter, GaitMetrics
except ImportError:
    torch = None


class MockScene(dict):
    """Match InteractiveScene's entity mapping plus per-environment origins."""


@unittest.skipIf(torch is None, "PyTorch is required; use the training environment Python")
class GaitMetricsTests(unittest.TestCase):
    feet = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

    def make_env(self, count=2):
        robot_names = ["base", "RR_foot", "FR_foot", "FL_foot", "RL_foot"]
        sensor_names = ["FL_foot", "base", "RL_foot", "FR_foot", "RR_foot"]
        joint_names = ["FR_hip_joint", "RL_hip_joint", "RR_hip_joint", "FL_hip_joint"]
        data = SimpleNamespace(
            root_lin_vel_b=torch.zeros(count, 3),
            root_lin_vel_w=torch.zeros(count, 3),
            projected_gravity_b=torch.tensor([[0., 0., -1.]]).repeat(count, 1),
            root_ang_vel_b=torch.zeros(count, 3),
            joint_vel=torch.zeros(count, 4),
            applied_torque=torch.zeros(count, 4),
            joint_pos=torch.tensor([[-.1, .1, -.1, .1]]).repeat(count, 1),
            root_pos_w=torch.zeros(count, 3),
            root_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(count, 1),
            body_pos_w=torch.tensor([[[0., 0., 0.], [-.2, -.15, -.3], [.2, -.15, -.3], [.2, .15, -.3], [-.2, .15, -.3]]]).repeat(count, 1, 1),
            body_lin_vel_w=torch.zeros(count, 5, 3),
        )
        sensor = SimpleNamespace(body_names=sensor_names,
            data=SimpleNamespace(net_forces_w=torch.zeros(count, 5, 3)))
        commands = torch.zeros(count, 3)
        scene = MockScene(robot=SimpleNamespace(data=data, body_names=robot_names, joint_names=joint_names), contact_forces=sensor)
        scene.env_origins = torch.zeros(count, 3)
        return SimpleNamespace(
            scene=scene,
            device="cpu", step_dt=.02,
            cfg=SimpleNamespace(go2_settings={"targets": {
                "hip_abduction_max": .3, "feet_lateral_min": .1, "feet_lateral_max": .22}}),
            episode_length_buf=torch.full((count,), 2),
            command_manager=SimpleNamespace(get_command=lambda _: commands),
            action_manager=SimpleNamespace(action=torch.zeros(count, 2)),
        )

    def test_first_episode_limit_ignores_finished_rows_and_accounts_fail_timeout(self):
        c = EpisodeCounter(3, "cpu", 1)
        c.update(torch.tensor([1, 0, 0]), torch.tensor([1, 0, 0]), torch.tensor([0, 0, 0]), torch.tensor([1, 0, 0]))
        self.assertEqual(c.active.tolist(), [False, True, True])
        self.assertFalse(c.all_finished())
        # Env 0 completes again but cannot inflate the first-episode test.
        c.update(torch.tensor([1, 1, 1]), torch.tensor([1, 0, 0]), torch.tensor([0, 1, 1]), torch.tensor([1, 0, 1]))
        r = c.report()
        self.assertTrue(c.all_finished())
        self.assertEqual((r["episodes"], r["successes"], r["timeouts"], r["failures"]), (3, 1, 1, 1))
        self.assertEqual(r["completed_episodes_per_env"], [1, 1, 1])
        self.assertEqual(r["incomplete_env_ids"], [])

    def test_incomplete_and_unlimited_episode_accounting(self):
        c = EpisodeCounter(2, "cpu", 2)
        c.update(torch.tensor([1, 0]), torch.tensor([0, 0]), torch.tensor([0, 0]), torch.tensor([1, 0]))
        self.assertEqual(c.report()["envs_incomplete"], 2)
        self.assertEqual(c.report()["completed_episodes_per_env"], [1, 0])
        unlimited = EpisodeCounter(1, "cpu")
        for _ in range(3):
            unlimited.update(torch.tensor([1]), torch.tensor([1]), torch.tensor([0]), torch.tensor([1]))
        self.assertEqual(unlimited.report()["episodes"], 3)
        self.assertFalse(unlimited.all_finished())

    def test_expected_values_ordering_masks_and_pre_step_raw_applied_alignment(self):
        e = self.make_env()
        d = e.scene["robot"].data
        d.root_lin_vel_b[:] = torch.tensor([[3., 1., 0.], [1000., 1000., 0.]])
        d.projected_gravity_b[0] = torch.tensor([0., -math.sin(.2), -math.cos(.2)])
        d.root_ang_vel_b[0] = torch.tensor([1., 2., 3.])
        d.joint_vel[0] = torch.tensor([3., 4., 3., 4.])
        d.applied_torque[0] = torch.tensor([-2., 2., 0., 0.])
        # Only FL has contact and moves at 5 m/s. Sensor index 0 != robot 3.
        e.scene["contact_forces"].data.net_forces_w[0, 0, 2] = 10
        d.body_lin_vel_w[0, 3, :2] = torch.tensor([3., 4.])
        d.body_lin_vel_w[0, 0, :2] = 999
        e.action_manager.action[0] = torch.tensor([2., 1.])
        m = GaitMetrics(e, self.feet, .25, 2)
        m.update_pre_step(torch.tensor([[4., 1.], [1000., 1000.]]), torch.tensor([True, False]))
        r = m.report()
        self.assertAlmostEqual(r["planar_velocity_mae_m_s"], 2.)
        self.assertAlmostEqual(r["body_roll_rms_rad"], .2, places=6)
        self.assertAlmostEqual(r["body_pitch_rms_rad"], 0.)
        self.assertAlmostEqual(r["body_yaw_rate_rms_rad_s"], 3.)
        self.assertAlmostEqual(r["joint_velocity_rms_rad_s"], math.sqrt(12.5))
        self.assertAlmostEqual(r["mechanical_power_mean_w"], 14.)
        self.assertAlmostEqual(r["stance_foot_slip_rms_m_s"], 5.)
        self.assertEqual(r["contact_1_feet_fraction"], 1.)
        self.assertEqual(r["raw_action_clip_fraction"], .5)
        self.assertEqual(r["raw_action_abs_mean"], 2.5)
        self.assertEqual(r["raw_action_abs_max"], 4.)
        self.assertEqual(r["applied_vs_clipped_raw_action_rms"], 0.)
        self.assertEqual(r["valid_env_samples"], 1)
        json.dumps(r, allow_nan=False)

    def test_differences_require_consecutive_valid_frames_and_skip_resets(self):
        e = self.make_env(1)
        m = GaitMetrics(e, self.feet, .25, 4)
        active = torch.tensor([True])
        for length, action, velocity in [(0, 999., 999.), (1, -999., -999.), (2, 0., 0.), (3, 2., 1.), (4, 6., 3.),
                                          (0, 888., 888.), (1, -888., -888.), (2, 0., 0.)]:
            e.episode_length_buf[:] = length
            e.action_manager.action[:] = action
            e.scene["robot"].data.joint_vel[:] = velocity
            m.update_pre_step(torch.full((1, 2), action), active)
        r = m.report()
        self.assertEqual(r["valid_env_samples"], 4)
        self.assertAlmostEqual(r["applied_action_delta_rms_rad"], math.sqrt((.5**2 + 1.**2) / 2))
        self.assertAlmostEqual(r["applied_action_second_delta_rms_rad"], .5)
        self.assertAlmostEqual(r["joint_acceleration_rms_rad_s2"], math.sqrt((50**2 + 100**2) / 2))
        self.assertEqual(r["metric_element_counts"]["applied_action_delta_rms_rad"], 4)
        self.assertEqual(r["metric_element_counts"]["applied_action_second_delta_rms_rad"], 2)

    def test_no_valid_samples_and_nonfinite_values_yield_serializable_json(self):
        e = self.make_env(1)
        m = GaitMetrics(e, self.feet, .25, 4)
        e.episode_length_buf[:] = 0
        m.update_pre_step(torch.zeros(1, 2), torch.tensor([True]))
        self.assertIsNone(m.report()["raw_action_abs_max"])
        e.episode_length_buf[:] = 2
        e.scene["robot"].data.root_lin_vel_b[:] = float("nan")
        m.update_pre_step(torch.full((1, 2), float("inf")), torch.tensor([True]))
        r = m.report()
        self.assertIsNone(r["planar_velocity_mae_m_s"])
        self.assertGreater(r["nonfinite_metric_elements"], 0)
        json.dumps(r, allow_nan=False)

    def test_nonfinite_inward_angle_does_not_become_zero_outward_component(self):
        e = self.make_env(1)
        # FL is positive-left, so -inf must remain invalid after clamping.
        e.scene["robot"].data.joint_pos[0, 3] = float("-inf")
        m = GaitMetrics(e, self.feet, .25, 4)
        m.update_pre_step(torch.zeros(1, 2), torch.tensor([True]))
        r = m.report()
        self.assertIsNone(r["FL_hip_outward_component_mean_rad"])
        self.assertEqual(r["metric_element_counts"]["hip_outward_component_mean_rad"], 3)
        self.assertAlmostEqual(r["hip_outward_component_mean_rad"], .1, places=6)
        json.dumps(r, allow_nan=False)

    def test_successful_subset_excludes_other_env_values_counts_maxima_and_nonfinites(self):
        e = self.make_env(2)
        d = e.scene["robot"].data
        d.root_lin_vel_b[:] = torch.tensor([[1., 1., 0.], [100., 100., 0.]])
        d.joint_vel[1, 0] = float("inf")
        m = GaitMetrics(e, self.feet, .25, 4)
        active = torch.tensor([True, True])
        m.update_pre_step(torch.tensor([[1., 1.], [100., 100.]]), active)
        e.episode_length_buf += 1
        e.action_manager.action[:] = torch.tensor([[2., 2.], [200., 200.]])
        m.update_pre_step(torch.tensor([[2., 2.], [200., 200.]]), active)
        global_report = m.report()
        selected = m.report(torch.tensor([True, False]))
        self.assertEqual(global_report["raw_action_abs_max"], 200.)
        self.assertEqual(global_report["valid_env_samples"], 4)
        self.assertGreater(global_report["nonfinite_metric_elements"], 0)
        self.assertEqual(selected["raw_action_abs_max"], 2.)
        self.assertEqual(selected["valid_env_samples"], 2)
        self.assertEqual(selected["nonfinite_metric_elements"], 0)
        self.assertEqual(selected["selected_env_count"], 1)
        self.assertEqual(selected["planar_velocity_mae_m_s"], 1.)
        self.assertAlmostEqual(selected["applied_action_delta_rms_rad"], .5)
        self.assertEqual(selected["metric_element_counts"]["applied_action_delta_rms_rad"], 2)
        empty = m.report(torch.tensor([False, False]))
        self.assertEqual(empty["valid_env_samples"], 0)
        self.assertIsNone(empty["raw_action_abs_max"])
        self.assertIsNone(empty["applied_action_delta_rms_rad"])
        json.dumps(empty, allow_nan=False)

    def test_lateral_metrics_resolve_joint_and_body_order_and_keep_each_leg_visible(self):
        e = self.make_env(1)
        d = e.scene["robot"].data
        # Joint order is FR, RL, RR, FL; signed outward angles are
        # FL=.4, FR=.2, RL=-.6, RR=.5. Large inward RL is not outward excess.
        d.joint_pos[0] = torch.tensor([-.2, -.6, -.5, .4])
        # Body order is base, RR, FR, FL, RL.
        d.body_pos_w[0, :, 1] = torch.tensor([999., -.35, -.05, .25, .15])
        m = GaitMetrics(e, self.feet, .25, 4)
        m.update_pre_step(torch.zeros(1, 2), torch.tensor([True]))
        r = m.report()
        for leg, angle, lateral in zip(self.feet, [.4, .2, -.6, .5], [.25, .05, .15, .35]):
            prefix = leg[:2]
            self.assertAlmostEqual(r[f"{prefix}_hip_abduction_mean_rad"], angle, places=6)
            self.assertAlmostEqual(r[f"{prefix}_hip_abduction_rms_rad"], abs(angle), places=6)
            self.assertAlmostEqual(r[f"{prefix}_foot_lateral_mean_m"], lateral, places=6)
        self.assertEqual(r["hip_abduction_over_limit_fraction"], .5)
        self.assertEqual(r["RL_hip_abduction_over_limit_fraction"], 0.)
        # A strongly inward rear leg cannot cancel the outward-component mean.
        self.assertAlmostEqual(r["hip_outward_component_mean_rad"], (.4 + .2 + .5) / 4, places=6)
        self.assertEqual(r["hip_adduction_fraction"], .25)
        self.assertEqual(r["RL_hip_adduction_fraction"], 1.)
        self.assertEqual(r["RL_hip_outward_component_mean_rad"], 0.)
        self.assertEqual(r["feet_lateral_outer_fraction"], .5)
        self.assertEqual(r["feet_lateral_inner_fraction"], .25)
        self.assertEqual(r["FR_foot_lateral_inner_fraction"], 1.)
        self.assertEqual(r["feet_cross_body_center_fraction"], 0.)
        self.assertAlmostEqual(r["front_foot_lateral_width_mean_m"], .30, places=6)
        self.assertAlmostEqual(r["rear_foot_lateral_width_mean_m"], .50, places=6)
        self.assertEqual(r["metric_element_counts"]["feet_lateral_outer_fraction"], 4)
        self.assertEqual(r["metric_element_counts"]["FL_foot_lateral_outer_fraction"], 1)
        # These positions are entirely on the approach, so no stairs-only
        # observation exists; reporting 0 would falsely imply perfect posture.
        self.assertEqual(r["stairs_valid_env_samples"], 0)
        self.assertEqual(r["metric_element_counts"]["stairs_feet_lateral_outer_fraction"], 0)
        self.assertEqual(r["metric_element_counts"]["stairs_FL_hip_abduction_mean_rad"], 0)
        self.assertIsNone(r["stairs_feet_lateral_outer_fraction"])
        self.assertIsNone(r["stairs_FL_hip_abduction_mean_rad"])

    def test_lateral_metrics_use_full_body_rotation_and_configurable_thresholds(self):
        e = self.make_env(1)
        d = e.scene["robot"].data
        # Rotate and translate the complete robot by roll=60 degrees. A
        # yaw-only or world-frame implementation would see the wrong widths.
        angle = math.pi / 3
        c, s = math.cos(angle), math.sin(angle)
        rotation = torch.tensor([[1., 0., 0.], [0., c, -s], [0., s, c]])
        d.root_pos_w[:] = torch.tensor([2., -5., 3.])
        d.root_quat_w[:] = torch.tensor([math.cos(angle / 2), math.sin(angle / 2), 0., 0.])
        d.body_pos_w[:] = d.body_pos_w @ rotation.T + d.root_pos_w[:, None]
        e.cfg.go2_settings["targets"] = {"hip_abduction_max": .05,
            "feet_lateral_min": .16, "feet_lateral_max": .20}
        m = GaitMetrics(e, self.feet, .25, 4)
        m.update_pre_step(torch.zeros(1, 2), torch.tensor([True]))
        r = m.report()
        for leg in ("FL", "FR", "RL", "RR"):
            self.assertAlmostEqual(r[f"{leg}_foot_lateral_mean_m"], .15, places=5)
        self.assertAlmostEqual(r["front_foot_lateral_width_mean_m"], .30, places=5)
        self.assertEqual(r["hip_abduction_over_limit_fraction"], 1.)
        self.assertEqual(r["feet_lateral_inner_fraction"], 1.)
        self.assertEqual(r["feet_lateral_outer_fraction"], 0.)
        self.assertEqual(r["lateral_metric_thresholds"], {
            "hip_abduction_max_rad": .05, "feet_lateral_min_m": .16, "feet_lateral_max_m": .20})

    def test_lateral_subset_excludes_extreme_rows_and_reset_frames(self):
        e = self.make_env(2)
        d = e.scene["robot"].data
        d.joint_pos[1] = torch.tensor([-10., 10., -10., 10.])
        d.body_pos_w[1, :, 1] *= 10
        m = GaitMetrics(e, self.feet, .25, 4)
        m.update_pre_step(torch.zeros(2, 2), torch.tensor([True, True]))
        # A reset sample with extreme values must not enter either subset.
        e.episode_length_buf[:] = 0
        d.joint_pos[:] = float("inf")
        m.update_pre_step(torch.zeros(2, 2), torch.tensor([True, True]))
        all_envs = m.report()
        selected = m.report(torch.tensor([True, False]))
        self.assertEqual(all_envs["hip_abduction_over_limit_fraction"], .5)
        self.assertEqual(all_envs["feet_lateral_outer_fraction"], .5)
        self.assertEqual(selected["hip_abduction_over_limit_fraction"], 0.)
        self.assertEqual(selected["feet_lateral_outer_fraction"], 0.)
        self.assertAlmostEqual(selected["RR_hip_abduction_mean_rad"], .1, places=6)
        self.assertEqual(selected["metric_element_counts"]["hip_abduction_mean_rad"], 4)
        self.assertEqual(selected["metric_element_counts"]["FL_hip_abduction_mean_rad"], 1)
        self.assertEqual(selected["nonfinite_metric_elements"], 0)
        json.dumps(selected, allow_nan=False)

    def test_stairs_velocity_and_stability_exclude_flat_and_keep_course_progress_distinct(self):
        e = self.make_env(2)
        d = e.scene["robot"].data
        commands = e.command_manager.get_command("base_velocity")
        commands[:, 0] = .6
        m = GaitMetrics(e, self.feet, .25, 4)
        # Both environments have large flat-ground movement; it must not
        # improve the stairs-only speed or conceal the stairs-only roll.
        d.root_lin_vel_b[:, 0] = 3
        d.root_lin_vel_w[:, 0] = 3
        m.update_pre_step(torch.zeros(2, 2), torch.tensor([True, True]))
        e.episode_length_buf += 1
        d.root_pos_w[:, 0] = 1.5
        d.root_lin_vel_b[:, 0] = torch.tensor([.5, 5.])
        d.root_lin_vel_w[:, 0] = torch.tensor([.3, 4.])
        d.projected_gravity_b[0] = torch.tensor([0., -math.sin(.2), -math.cos(.2)])
        d.root_ang_vel_b[0] = torch.tensor([.4, .5, .6])
        e.scene["contact_forces"].data.net_forces_w[0, 0, 2] = 10
        d.body_lin_vel_w[0, 3, 0] = .2
        m.update_pre_step(torch.zeros(2, 2), torch.tensor([True, True]))
        r = m.report(torch.tensor([True, False]))
        self.assertAlmostEqual(r["stairs_course_forward_velocity_mean_m_s"], .3, places=6)
        self.assertAlmostEqual(r["stairs_body_forward_velocity_mean_m_s"], .5, places=6)
        self.assertAlmostEqual(r["stairs_forward_command_mean_m_s"], .6, places=6)
        self.assertAlmostEqual(r["stairs_forward_velocity_mae_m_s"], .1, places=6)
        self.assertAlmostEqual(r["stairs_body_roll_rms_rad"], .2, places=6)
        self.assertAlmostEqual(r["body_roll_rms_rad"], .2 / math.sqrt(2), places=6)
        self.assertAlmostEqual(r["stairs_body_roll_rate_rms_rad_s"], .4, places=6)
        self.assertAlmostEqual(r["stairs_stance_foot_slip_rms_m_s"], .2, places=6)
        self.assertEqual(r["metric_element_counts"]["stairs_course_forward_velocity_mean_m_s"], 1)
        json.dumps(r, allow_nan=False)

    def test_stair_crossing_interpolation_retreat_and_successful_subset(self):
        e = self.make_env(2)
        e.cfg.go2_settings["terrain"] = {"num_steps": 2, "tread": .4}
        e.step_dt = .1
        e.scene.env_origins[1, 0] = 10
        d = e.scene["robot"].data
        m = GaitMetrics(e, self.feet, .25, 4)
        # Env 0 crosses 1.2 at t=.25 and 2.0 at t=.65; the retreat
        # at t=.4 remains counted. Env 1 starts inside, so entry is unknown.
        for length, x in enumerate((1.1, 1.3, 1.1, 1.5, 1.9, 2.1, 1.9, 2.1), start=2):
            e.episode_length_buf[:] = length
            d.root_pos_w[:, 0] = torch.tensor([x, 11.5 + .2 * (length - 2)])
            m.update_pre_step(torch.zeros(2, 2), torch.tensor([True, True]))
        r = m.report(torch.tensor([True, False]))
        self.assertEqual(r["stairs_completed_traversals"], 1)
        self.assertAlmostEqual(r["stairs_traversal_time_mean_s"], .4, places=6)
        self.assertAlmostEqual(r["stairs_traversal_speed_mean_m_s"], 2., places=5)
        inside = m.report(torch.tensor([False, True]))
        self.assertEqual(inside["stairs_completed_traversals"], 0)
        self.assertIsNone(inside["stairs_traversal_time_mean_s"])
        self.assertIsNone(inside["stairs_traversal_speed_mean_m_s"])
        json.dumps(inside, allow_nan=False)

    def test_stair_crossing_does_not_bridge_reset_missing_samples_or_incomplete_flight(self):
        e = self.make_env(1)
        e.cfg.go2_settings["terrain"] = {"num_steps": 2, "tread": .4}
        d = e.scene["robot"].data
        for samples in (
            [(2, 1.1), (3, 1.3), (0, 2.1), (2, 2.2)],  # reset before exit
            [(2, 1.1), (3, 1.3), (5, 1.9), (6, 2.1)],  # missing time sample
            [(2, 1.1), (3, 1.3), (4, 1.9)],  # never exits
            [(2, 1.1), (3, float("nan")), (4, 1.9), (5, 2.1)],
            [(2, 1.1), (3, 1.3), (4, float("nan")), (5, 1.9), (6, 2.1)],
        ):
            m = GaitMetrics(e, self.feet, .25, 4)
            for length, x in samples:
                e.episode_length_buf[:] = length
                d.root_pos_w[:, 0] = x
                m.update_pre_step(torch.zeros(1, 2), torch.tensor([True]))
            r = m.report()
            self.assertEqual(r["stairs_completed_traversals"], 0)
            self.assertIsNone(r["stairs_traversal_time_mean_s"])

    def test_stair_crossings_count_once_per_episode_and_reset_per_episode(self):
        e = self.make_env(1)
        e.cfg.go2_settings["terrain"] = {"num_steps": 2, "tread": .4}
        d = e.scene["robot"].data
        m = GaitMetrics(e, self.feet, .25, 4)
        for length, x in [(2, 1.), (3, 1.5), (4, 2.1), (5, 1.1), (6, 2.1),
                          (0, 0.), (1, .5), (2, 1.), (3, 1.5), (4, 2.1)]:
            e.episode_length_buf[:] = length
            d.root_pos_w[:, 0] = x
            m.update_pre_step(torch.zeros(1, 2), torch.tensor([True]))
        self.assertEqual(m.report()["stairs_completed_traversals"], 2)

    def test_inner_width_and_crossed_center_are_distinct(self):
        e = self.make_env(1)
        # Robot body order base, RR, FR, FL, RL. FL has crossed to y<0;
        # FR is on its proper side but too close to the body center.
        e.scene["robot"].data.body_pos_w[0, :, 1] = torch.tensor([0., -.15, -.05, -.02, .15])
        m = GaitMetrics(e, self.feet, .25, 4)
        m.update_pre_step(torch.zeros(1, 2), torch.tensor([True]))
        r = m.report()
        self.assertEqual(r["feet_lateral_inner_fraction"], .5)
        self.assertEqual(r["feet_cross_body_center_fraction"], .25)
        self.assertEqual(r["FL_foot_cross_body_center_fraction"], 1.)
        self.assertEqual(r["FR_foot_cross_body_center_fraction"], 0.)

    def test_stairs_phase_uses_relative_position_excludes_approach_platform_and_reset_in_subset(self):
        e = self.make_env(2)
        e.cfg.go2_settings["terrain"] = {"num_steps": 2, "tread": .4}
        e.scene.env_origins[1, 0] = 10.
        d = e.scene["robot"].data
        m = GaitMetrics(e, self.feet, .25, 4)
        # Env 0 is selected as the successful episode. Env 1's different
        # origin ensures filtering uses relative x rather than world x.
        for x, length, hip, foot_y, other_active in (
            (1.199, 2, .8, .4, True),  # approach excluded
            (1.2, 3, .2, .15, True),  # inclusive first riser
            (1.5, 4, .4, .25, False),
            (2.0, 5, .9, .4, False),  # exclusive platform boundary
            (1.5, 1, 99., 99., False),  # reset frame excluded
        ):
            d.root_pos_w[:, 0] = torch.tensor([x, 11.5])
            d.body_pos_w[:] = d.root_pos_w[:, None, :]
            d.body_pos_w[0, :, 1] += torch.tensor([0., -foot_y, -foot_y, foot_y, foot_y])
            d.body_pos_w[1, :, 1] += torch.tensor([0., -.4, -.4, .4, .4])
            d.joint_pos[0] = torch.tensor([-hip, hip, -hip, hip])
            d.joint_pos[1] = torch.tensor([-.8, .8, -.8, .8])
            e.episode_length_buf[:] = length
            m.update_pre_step(torch.zeros(2, 2), torch.tensor([True, other_active]))
        all_envs = m.report()
        selected = m.report(torch.tensor([True, False]))
        self.assertEqual(all_envs["stairs_valid_env_samples"], 4)
        self.assertEqual(selected["stairs_valid_env_samples"], 2)
        self.assertEqual(selected["valid_env_samples"], 4)
        self.assertEqual(selected["hip_abduction_over_limit_fraction"], .75)
        self.assertEqual(selected["stairs_hip_abduction_over_limit_fraction"], .5)
        self.assertEqual(selected["stairs_feet_lateral_outer_fraction"], .5)
        self.assertAlmostEqual(selected["stairs_FL_hip_abduction_mean_rad"], .3, places=6)
        self.assertAlmostEqual(selected["stairs_RR_foot_lateral_mean_m"], .2, places=6)
        self.assertAlmostEqual(selected["stairs_front_foot_lateral_width_mean_m"], .4, places=6)
        self.assertEqual(selected["metric_element_counts"]["stairs_hip_abduction_mean_rad"], 8)
        self.assertEqual(selected["metric_element_counts"]["stairs_FL_hip_abduction_mean_rad"], 2)
        self.assertEqual(selected["stairs_lateral_sampling"]["start_x_inclusive_m"], 1.2)
        self.assertEqual(selected["stairs_lateral_sampling"]["end_x_exclusive_m"], 2.)
        json.dumps(selected, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
