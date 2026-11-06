"""Evaluation guard stops preserve boundary evidence without executing a request.

The deterministic physics stand-in advances only when ``mj_step`` is called.
Guard evaluation, audit aggregation, environment control flow and reward
accounting are the production code; this suite needs neither ONNX nor a robot.
"""
from __future__ import annotations

from collections import deque
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from go2_him.moe_stair_env import JointGuardStop, MoEStairEnv
from go2_him.official_guard import OfficialGuardAudit, load_official_profile


class ControlledPhysics:
    def __init__(self, after_step=None):
        self.steps = 0
        self.after_step = after_step

    def mj_step(self, model, data):
        self.steps += 1
        data.time += .002
        data.actuator_force[:] = data.ctrl
        if self.after_step is not None:
            self.after_step(self.steps, data)

    def mj_forward(self, model, data):
        pass


def environment(mode="strict", *, decimation=4, after_step=None):
    """A stationary valid pose, with complete fields consumed by ``step``."""
    env = MoEStairEnv.__new__(MoEStairEnv)
    env.profile = load_official_profile()
    env.guard_mode = mode
    env.guard_objective = "balanced"
    env.evaluation = True
    env.calf_mask = np.array([name.endswith("_calf") for name in env.profile.names])
    q0 = np.array([.1, .8, -1.5, -.1, .8, -1.5,
                   .1, 1., -1.5, -.1, 1., -1.5], dtype=np.float32)
    feet = np.array([[-.5, .1, .02], [-.5, -.1, .02],
                     [-.9, .1, .02], [-.9, -.1, .02]])
    data = SimpleNamespace(
        qpos=np.r_[[-.7, 0., .3, 1., 0., 0., 0.], q0],
        qvel=np.zeros(18), ctrl=np.zeros(12), actuator_force=np.zeros(12),
        time=0., warning=[], xmat=np.eye(3).reshape(1, 9),
        geom_xpos=feet, sensordata=np.zeros(3),
    )
    physics = ControlledPhysics(after_step)
    env.exp = SimpleNamespace(
        data=data, model=object(), mj=physics,
        policy=SimpleNamespace(scale=np.full(12, .25), offset=q0.copy(),
                               clip=np.tile([-100., 100.], (12, 1)), q0=q0.copy()),
        qadr=np.arange(7, 19), vadr=np.arange(6, 18), aids=np.arange(12),
        kp=np.full(12, 20.), kd=np.full(12, .5),
        physical_effort=env.profile.effort, decimation=decimation,
        numeric_warnings={}, base=0, feet=np.arange(4), radii=np.full(4, .02),
        gyro_adr=0, contacts=lambda: ({0, 1, 2, 3}, set(), False, False),
        wall_clearance=lambda: {"signed_clearance_m": 1.5},
        wall_contacts=lambda elapsed: ([], 0.),
    )
    env.physics_callback = None
    env.stop_phase = env.platform_seen = False
    env.vx, env.height, env.top, env.ramp = .6, .15, .3, 1.
    env.start_x = env.last_x = -.7
    env.elapsed = env.episode_return = env.roll_sq = 0.
    env.control_samples = 0
    env.stable_time = env.trunk_time = env.side_time = env.fall_time = 0.
    env.min_wall_clearance = float("inf")
    env.last_action = np.zeros(12, dtype=np.float32)
    env.prev_action = env.last_action.copy()
    env.last_foot = feet.copy()
    env.recent = deque()
    env.audit = OfficialGuardAudit(env.profile)
    env.observation = lambda: np.zeros(225, dtype=np.float32)
    return env


def illegal_calf_target(env):
    action = np.zeros(12, dtype=np.float32)
    index = env.profile.names.index("RL_calf")
    action[index] = (env.profile.upper[index] + .05 - env.exp.policy.offset[index]) / .25
    return action


class StrictGuardTest(unittest.TestCase):
    def test_target_rejection_precedes_physics_and_applied_torque_is_unknown(self):
        env = environment()
        q_before, dq_before = env.exp.data.qpos.copy(), env.exp.data.qvel.copy()
        with self.assertRaises(JointGuardStop) as caught:
            env.step(illegal_calf_target(env))
        self.assertEqual(caught.exception.details["boundary"], "pre_command")
        self.assertIn("RL_calf", caught.exception.details["violations"]["target_position"])
        self.assertEqual(env.exp.mj.steps, 0)
        self.assertEqual(env.elapsed, 0.)
        self.assertEqual(env.exp.data.time, 0.)
        np.testing.assert_array_equal(env.exp.data.qpos, q_before)
        np.testing.assert_array_equal(env.exp.data.qvel, dq_before)
        np.testing.assert_array_equal(env.exp.data.ctrl, np.zeros(12))
        audit = env.audit.summary()
        self.assertEqual(audit["physics_state_samples"], 1)
        self.assertEqual(audit["terminal_physics_state_samples"], 1)
        self.assertFalse(audit["all_metrics_observed"])
        self.assertFalse(audit["joint_and_torque_acceptance_passed"])
        target = audit["metrics"]["target_position"]
        self.assertEqual(target["per_joint"]["RL_calf"]["excess_count"], 1)
        self.assertEqual(audit["metrics"]["position"]["excess_count"], 0)
        applied = audit["metrics"]["applied_torque"]
        self.assertEqual(applied["observed_state_samples"], 0)
        self.assertEqual(applied["unobserved_state_samples"], 1)
        self.assertEqual(applied["sample_count"], 0)
        self.assertIsNone(applied["excess_fraction"])
        self.assertIsNone(applied["peak_absolute_value"])
        self.assertIsNone(applied["per_joint"]["RL_calf"]["excess_fraction"])
        self.assertEqual(audit["metrics"]["requested_torque"]["observed_state_samples"], 1)

    def test_post_failure_keeps_prior_substeps_and_terminal_post_boundary(self):
        def violate_on_third_step(number, data):
            if number == 3:
                data.qvel[6 + 2] = 16.

        env = environment(decimation=5, after_step=violate_on_third_step)
        with self.assertRaises(JointGuardStop) as caught:
            env.step(np.zeros(12))
        self.assertEqual(caught.exception.details["boundary"], "post_integration")
        self.assertEqual(caught.exception.details["violations"]["velocity"], ["FL_calf"])
        self.assertEqual(env.exp.mj.steps, 3)
        self.assertAlmostEqual(env.elapsed, .006)
        self.assertAlmostEqual(env.exp.data.time, .006)
        audit = env.audit.summary()
        self.assertEqual(audit["physics_state_samples"], 6)
        self.assertEqual(audit["terminal_physics_state_samples"], 1)
        for metric in audit["metrics"].values():
            self.assertEqual(metric["observed_state_samples"], 6)
            self.assertEqual(metric["unobserved_state_samples"], 0)
            self.assertEqual(metric["sample_count"], 72)
        velocity = audit["metrics"]["velocity"]
        self.assertEqual(velocity["excess_count"], 1)
        self.assertEqual(velocity["per_joint"]["FL_calf"]["sample_count"], 6)
        self.assertEqual(velocity["per_joint"]["FL_calf"]["peak_absolute_value"], 16.)
        self.assertAlmostEqual(velocity["excess_fraction"], 1 / 72)
        self.assertFalse(audit["joint_and_torque_acceptance_passed"])

    def test_later_pre_rejection_retains_known_applied_and_new_unknown_boundary(self):
        env = environment(decimation=4)
        calls = 0

        def contacts_with_new_measurement():
            # Inject a fresh state between the completed post check and the
            # next command pre check; the earlier physical evidence stays valid.
            nonlocal calls
            calls += 1
            if calls == 2:
                env.exp.data.qvel[6 + 2] = 16.
            return {0, 1, 2, 3}, set(), False, False

        env.exp.contacts = contacts_with_new_measurement
        with self.assertRaises(JointGuardStop) as caught:
            env.step(np.zeros(12))
        self.assertEqual(caught.exception.details["boundary"], "pre_command")
        self.assertEqual(env.exp.mj.steps, 2)
        self.assertAlmostEqual(env.elapsed, .004)
        self.assertAlmostEqual(env.exp.data.time, .004)
        audit = env.audit.summary()
        self.assertEqual(audit["physics_state_samples"], 5)
        self.assertEqual(audit["terminal_physics_state_samples"], 1)
        self.assertEqual(audit["metrics"]["velocity"]["sample_count"], 60)
        self.assertEqual(audit["metrics"]["velocity"]["excess_count"], 1)
        applied = audit["metrics"]["applied_torque"]
        self.assertEqual(applied["sample_count"], 48)
        self.assertEqual(applied["observed_state_samples"], 4)
        self.assertEqual(applied["unobserved_state_samples"], 1)
        self.assertFalse(audit["all_metrics_observed"])
        self.assertFalse(audit["joint_and_torque_acceptance_passed"])

    def test_successful_strict_steps_audit_each_pre_post_boundary_once(self):
        env = environment(decimation=4)
        for control_steps in (1, 2):
            _, reward, done, _ = env.step(np.zeros(12))
            self.assertFalse(done)
            self.assertTrue(np.isfinite(reward))
            self.assertEqual(env.exp.mj.steps, 4 * control_steps)
            audit = env.audit.summary()
            self.assertEqual(audit["physics_state_samples"], 8 * control_steps)
            self.assertEqual(audit["terminal_physics_state_samples"], 0)
            for metric in audit["metrics"].values():
                self.assertEqual(metric["sample_count"], 96 * control_steps)
                self.assertEqual(metric["excess_count"], 0)
            self.assertTrue(audit["all_metrics_observed"])
            self.assertTrue(audit["joint_and_torque_acceptance_passed"])

    def test_numeric_failure_preserves_known_boundaries_without_recording_nan(self):
        def corrupt_third_step(number, data):
            if number == 3:
                data.qvel[6 + 2] = np.nan
                data.actuator_force[:] = np.nan

        for mode in ("strict", "audit"):
            with self.subTest(mode=mode):
                env = environment(mode=mode, decimation=5, after_step=corrupt_third_step)
                with self.assertRaisesRegex(FloatingPointError, "MuJoCo numeric failure"):
                    env.step(np.zeros(12))
                self.assertEqual(env.exp.mj.steps, 3)
                # Only two integrations completed with valid state evidence.
                self.assertAlmostEqual(env.elapsed, .004)
                self.assertAlmostEqual(env.exp.data.time, .006)
                audit = env.audit.summary()
                self.assertEqual(audit["physics_state_samples"], 5)
                self.assertEqual(audit["terminal_physics_state_samples"], 1)
                for name, metric in audit["metrics"].items():
                    if name != "applied_torque":
                        self.assertEqual(metric["observed_state_samples"], 5)
                        self.assertEqual(metric["sample_count"], 60)
                        self.assertEqual(metric["excess_count"], 0)
                applied = audit["metrics"]["applied_torque"]
                self.assertEqual(applied["observed_state_samples"], 4)
                self.assertEqual(applied["unobserved_state_samples"], 1)
                self.assertEqual(applied["sample_count"], 48)
                self.assertFalse(audit["all_metrics_observed"])
                self.assertIsNone(audit["joint_and_torque_acceptance_passed"])
                # Fail rather than serialize invalid simulator state as NaN.
                json.dumps(audit, allow_nan=False)

    def test_audit_mode_continues_physics_but_records_illegal_target(self):
        env = environment(mode="audit", decimation=4)
        _, reward, done, _ = env.step(illegal_calf_target(env))
        self.assertFalse(done)
        self.assertTrue(np.isfinite(reward))
        self.assertEqual(env.exp.mj.steps, 4)
        self.assertAlmostEqual(env.elapsed, .008)
        self.assertAlmostEqual(env.exp.data.time, .008)
        audit = env.audit.summary()
        self.assertEqual(audit["physics_state_samples"], 8)
        self.assertEqual(audit["terminal_physics_state_samples"], 0)
        target = audit["metrics"]["target_position"]
        self.assertEqual(target["per_joint"]["RL_calf"]["excess_count"], 8)
        self.assertEqual(audit["metrics"]["applied_torque"]["observed_state_samples"], 8)
        self.assertFalse(audit["joint_and_torque_acceptance_passed"])

    def test_strict_training_is_rejected_before_simulator_initialization(self):
        with patch("go2_him.moe_stair_env.wall.Experiment") as experiment:
            with self.assertRaisesRegex(ValueError, "evaluation-only"):
                MoEStairEnv(evaluation=False, guard_mode="strict")
            experiment.assert_not_called()


if __name__ == "__main__":
    unittest.main()
