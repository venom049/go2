"""Velocity-only refinement keeps useful motion and preserves limit evidence.

These simulator-free checks validate the experiment objective and audit behavior;
they do not establish that a learned policy will remain below the speed limit.
"""
from pathlib import Path
import unittest

from go2_him.settings import load_settings

try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    from go2_him.joint_constraints import JointConstraintAudit, JointConstraintWindow


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stairs_velocity_limited.yaml"
NAMES = [f"{leg}_{kind}_joint" for leg in ("FL", "FR", "RL", "RR")
         for kind in ("hip", "thigh", "calf")]


class VelocityTrainingConfigTests(unittest.TestCase):
    def test_refinement_removes_position_objectives_and_all_kind_termination(self):
        settings = load_settings(CONFIG)
        baseline = load_settings(ROOT / "configs/stairs_torque_limited.yaml")
        joint = settings["joint_constraints"]
        self.assertTrue(joint["enabled"])
        self.assertFalse(joint["terminate_on_violation"])
        self.assertEqual(joint["velocity_limit"], 13.0)
        self.assertEqual(joint["velocity_soft_ratio"], .9)
        self.assertEqual(joint["startup_multiplier"], 1.0)
        self.assertEqual(settings["rewards"]["velocity_margin"], -.2)
        self.assertEqual(settings["rewards"]["velocity_excess"], -2.0)
        for name in ("position_margin", "position_excess", "target_position_excess",
                     "tracking_error", "joint_limits", "joint_deviation",
                     "hip_deviation", "hip_abduction", "joint_vel"):
            with self.subTest(removed_objective=name):
                self.assertEqual(settings["rewards"][name], 0.0)
        # Refinement must retain the source policy's task, stability and torque
        # objectives instead of accidentally becoming an exclusively safety task.
        for name in ("tracking_lin_vel", "forward_progress", "success", "failure",
                     "roll_orientation", "roll_rate", "feet_lateral",
                     "torque_margin", "torque_excess"):
            with self.subTest(retained_objective=name):
                self.assertEqual(settings["rewards"][name], baseline["rewards"][name])
        self.assertEqual(settings["torque"], baseline["torque"])
        self.assertEqual(settings["control"], baseline["control"])
        self.assertEqual(settings["training"]["init_noise_std"], .15)
        self.assertEqual(settings["training"]["min_noise_std"], .05)
        self.assertEqual(settings["training"]["max_noise_std"], .25)


@unittest.skipIf(torch is None, "PyTorch required; use .venv-deploy/bin/python")
class VelocityTrainingAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = load_settings(CONFIG)

    def window(self, steps=1, envs=1):
        joint = self.settings["joint_constraints"]
        window = JointConstraintWindow(
            envs, NAMES, velocity_limit=joint["velocity_limit"],
            velocity_soft_ratio=joint["velocity_soft_ratio"],
            position_soft_ratio=joint["position_soft_ratio"],
            tracking_error_limit=joint["tracking_error_limit"],
            tracking_soft_ratio=joint["tracking_soft_ratio"])
        startup = torch.zeros(envs, dtype=torch.bool)
        startup[0] = True
        window.begin(steps, startup)
        return window

    def pose(self, window):
        return window.center.expand(window.startup.numel(), -1).clone()

    def audit(self, window):
        audit = JointConstraintAudit(window)
        audit.update(window, torch.ones_like(window.startup))
        return audit.report()

    def weighted_constraint_reward(self, window):
        return sum(self.settings["rewards"][name] * window.penalty(
            name, self.settings["joint_constraints"]["startup_multiplier"])
            for name in window.PENALTIES)

    def test_useful_velocity_is_unpenalized_and_threshold_is_not_a_clamp(self):
        for speed in (11.6, 12.0, 13.0, 13.313, -13.313):
            with self.subTest(speed=speed):
                window = self.window()
                position = self.pose(window)
                velocity = torch.zeros_like(position)
                velocity[0, 8] = speed
                window.record_command(position, position)
                window.record_state(position, velocity)
                margin = max(abs(speed) / 13.0 - .9, 0.0) ** 2
                excess = max(abs(speed) / 13.0 - 1.0, 0.0) ** 2
                self.assertAlmostEqual(float(window.penalty("velocity_margin")), margin, places=7)
                self.assertAlmostEqual(float(window.penalty("velocity_excess")), excess, places=7)
                self.assertAlmostEqual(float(self.weighted_constraint_reward(window)),
                                       -.2 * margin - 2.0 * excess, places=7)
                report = self.audit(window)
                self.assertEqual(report["velocity_excess_joint_samples"], int(abs(speed) > 13.0))
                self.assertEqual(report["joint_limits_passed"], abs(speed) <= 13.0)
                self.assertAlmostEqual(report["joints"]["RL_calf_joint"]["velocity_peak_rad_s"],
                                       abs(speed), places=5)
                self.assertAlmostEqual(float(velocity[0, 8]), speed, places=5)

    def test_startup_no_longer_doubles_speed_penalty(self):
        window = self.window(envs=2)
        position = self.pose(window)
        velocity = torch.zeros_like(position)
        velocity[:, 0] = 13.0
        window.record_command(position, position)
        window.record_state(position, velocity)
        reward = self.weighted_constraint_reward(window)
        self.assertTrue(torch.allclose(reward, torch.tensor([-.002, -.002]), atol=1e-7))

    def test_actual_position_violation_remains_audited_without_a_training_cost(self):
        window = self.window()
        before = self.pose(window)
        before[0, 1] = window.upper[1] - .001
        after = before.clone()
        after[0, 1] = window.upper[1] + .01
        velocity = torch.zeros_like(before)
        velocity[0, 1] = 2.2  # 0.011 rad over 5 ms, comfortably below 11.7.
        window.record_command(before, before)
        window.record_state(after, velocity)
        self.assertGreater(float(window.penalty("position_excess")), 0.0)
        self.assertEqual(float(self.weighted_constraint_reward(window)), 0.0)
        report = self.audit(window)
        self.assertFalse(report["joint_limits_passed"])
        self.assertEqual(report["position_excess_joint_samples"], 1)
        self.assertEqual(report["target_position_excess_joint_samples"], 0)
        self.assertEqual(report["tracking_error_excess_joint_samples"], 0)
        self.assertEqual(report["velocity_excess_joint_samples"], 0)

    def test_target_and_tracking_violations_are_not_hidden_by_zero_weights(self):
        window = self.window()
        position = self.pose(window)
        target = position.clone()
        target[0, 2] = -.5  # Outside calf travel; actual state remains legal.
        window.record_command(target, position)
        window.record_state(position, torch.zeros_like(position))
        self.assertGreater(float(window.penalty("target_position_excess")), 0.0)
        self.assertGreater(float(window.penalty("tracking_error")), 0.0)
        self.assertEqual(float(self.weighted_constraint_reward(window)), 0.0)
        report = self.audit(window)
        self.assertEqual(report["target_position_excess_joint_samples"], 1)
        self.assertEqual(report["tracking_error_excess_joint_samples"], 1)
        self.assertEqual(report["position_excess_joint_samples"], 0)
        self.assertEqual(report["velocity_excess_joint_samples"], 0)
        self.assertFalse(report["joint_limits_passed"])

    def test_final_substep_velocity_spike_survives_reset_and_reward_averaging(self):
        window = self.window(steps=4)
        position = self.pose(window)
        for substep in range(4):
            velocity = torch.zeros_like(position)
            if substep == 3:
                velocity[0, 8] = -13.313
            window.record_command(position, position)
            window.record_state(position, velocity)
        reward_before_reset = self.weighted_constraint_reward(window).clone()
        self.assertLess(float(reward_before_reset), 0.0)
        # Environment reset writes must not erase the terminal physics sample.
        window.record_command(position, position)
        window.record_state(position, torch.zeros_like(position))
        self.assertTrue(torch.equal(self.weighted_constraint_reward(window), reward_before_reset))
        report = self.audit(window)
        self.assertFalse(report["joint_limits_passed"])
        self.assertEqual(report["physics_samples_per_joint"], 4)
        self.assertEqual(report["velocity_excess_joint_samples"], 1)
        self.assertEqual(report["velocity_excess_fraction"], 1 / 48)
        self.assertAlmostEqual(report["joints"]["RL_calf_joint"]["velocity_peak_rad_s"], 13.313, places=5)
        self.assertEqual(report["position_excess_joint_samples"], 0)


if __name__ == "__main__":
    unittest.main()
