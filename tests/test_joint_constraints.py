"""Joint constraints: physics-step spikes, command/state distinction and audits."""
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    from go2_him.joint_constraints import (
        JointConstraintAudit, JointConstraintWindow, joint_position_limits,
    )

NAMES = [f"{leg}_{kind}_joint" for leg in ("FL", "FR", "RL", "RR")
         for kind in ("hip", "thigh", "calf")]


@unittest.skipIf(torch is None, "PyTorch required for joint constraint tensor tests")
class JointConstraintTests(unittest.TestCase):
    def window(self, envs=2, steps=4, **kwargs):
        window = JointConstraintWindow(envs, NAMES, **kwargs)
        window.begin(steps, torch.tensor([True] + [False] * (envs - 1)))
        return window

    def pose(self, window):
        return window.center.expand(window.startup.numel(), -1).clone()

    def safe_step(self, window):
        position = self.pose(window)
        window.record_command(position, position)
        window.record_state(position, torch.zeros_like(position))

    def test_asymmetric_limits_follow_names_and_match_actual_urdf(self):
        names = NAMES[::-1]
        limits = joint_position_limits(names)
        source = Path(__file__).resolve().parents[1] / "assets/go2/urdf/go2.urdf"
        reference = {j.attrib["name"]: (float(j.find("limit").attrib["lower"]),
                                        float(j.find("limit").attrib["upper"]))
                     for j in ET.parse(source).getroot().findall("joint")
                     if j.attrib.get("type") == "revolute"}
        for index, name in enumerate(names):
            self.assertTrue(torch.allclose(limits[index], torch.tensor(reference[name]), atol=1e-7))
        self.assertAlmostEqual(float(limits[names.index("RL_thigh_joint"), 0]), -.5236, places=5)
        self.assertAlmostEqual(float(limits[names.index("FL_thigh_joint"), 0]), -1.5708, places=5)
        for bad in (NAMES[:-1], [NAMES[0]] * 12, NAMES[:-1] + ["bad_joint"]):
            with self.assertRaises(ValueError):
                joint_position_limits(bad)

    def test_invalid_numeric_parameters_rejected(self):
        for key, value in (("num_envs", 0), ("num_envs", True), ("velocity_limit", 0),
                           ("velocity_limit", float("nan")), ("velocity_limit", True),
                           ("velocity_soft_ratio", 1), ("position_soft_ratio", 0),
                           ("tracking_error_limit", -1), ("tracking_soft_ratio", float("inf"))):
            kwargs = {"num_envs": 2, key: value}
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                JointConstraintWindow(joint_names=NAMES, **kwargs)

    def test_last_post_integration_spike_survives_terminal_and_reset(self):
        window = self.window()
        for _ in range(3):
            self.safe_step(window)
        position = self.pose(window)
        window.record_command(position, position)
        velocity = torch.zeros_like(position)
        velocity[0, 8] = -13.312929
        position[0, 8] = window.upper[8] + .02
        window.record_state(position, velocity)
        self.assertEqual(window.hard_violation().tolist(), [True, False])
        self.assertGreater(float(window.penalty("velocity_excess")[0]), 0)
        self.assertGreater(float(window.penalty("position_excess")[0]), 0)
        # Reset's extra write and refreshed state must not erase terminal data.
        window.record_command(torch.full_like(position, 999), torch.full_like(position, 999))
        window.record_state(torch.full_like(position, 999), torch.full_like(position, 999))
        audit = JointConstraintAudit(window)
        audit.update(window, torch.tensor([True, False]))
        report = audit.report()
        self.assertFalse(report["joint_limits_passed"])
        self.assertEqual(report["physics_samples_per_joint"], 4)
        self.assertEqual(report["velocity_excess_joint_samples"], 1)
        self.assertEqual(report["position_excess_joint_samples"], 1)
        self.assertEqual(report["startup_velocity_excess_joint_samples"], 1)
        self.assertAlmostEqual(report["joints"]["RL_calf_joint"]["velocity_peak_rad_s"], 13.312929, places=5)
        self.assertAlmostEqual(report["joints"]["RL_calf_joint"]["position_peak_overshoot_rad"], .02, places=5)
        self.assertEqual(report["velocity_excess_fraction"], 1 / 48)

    def test_target_violation_not_hidden_by_legal_measured_state(self):
        window = self.window(envs=1, steps=1)
        position = self.pose(window)
        target = position.clone()
        target[0, 2] = -.5  # Old normalized action clip permits this calf target.
        window.record_command(target, position)
        window.record_state(position, torch.zeros_like(position))
        report_audit = JointConstraintAudit(window)
        report_audit.update(window, torch.tensor([True]))
        report = report_audit.report()
        self.assertFalse(report["joint_limits_passed"])
        self.assertEqual(report["target_position_excess_joint_samples"], 1)
        self.assertEqual(report["position_excess_joint_samples"], 0)
        self.assertEqual(report["tracking_error_excess_joint_samples"], 1)
        self.assertGreater(float(window.penalty("target_position_excess")), 0)

    def test_tracking_error_differs_from_default_pose_deviation(self):
        window = self.window(envs=1, steps=1)
        position = self.pose(window)
        target = position.clone()
        target[0, 1] += .91  # Both values are within thigh joint travel.
        window.record_command(target, position)
        window.record_state(position, torch.zeros_like(position))
        self.assertEqual(int(window.counts["tracking_error"].sum()), 1)
        self.assertEqual(int(window.counts["target_position"].sum()), 0)
        self.assertTrue(bool(window.hard_violation()[0]))

    def test_terminal_tracking_error_crossing_with_legal_velocity_and_position(self):
        window = self.window(envs=1, steps=4)
        for _ in range(3):
            self.safe_step(window)
        position = self.pose(window)
        target = position.clone()
        target[0, 1] += .895
        window.record_command(target, position)
        after = position.clone()
        after[0, 1] -= .01
        velocity = torch.zeros_like(position)
        velocity[0, 1] = -2  # 0.01 rad over 5 ms: well below 13 rad/s.
        window.record_state(after, velocity)
        self.assertTrue(bool(window.hard_violation()[0]))
        audit = JointConstraintAudit(window)
        audit.update(window, torch.tensor([True]))
        report = audit.report()
        self.assertFalse(report["joint_limits_passed"])
        self.assertEqual(report["tracking_error_excess_joint_samples"], 1)
        self.assertEqual(report["velocity_excess_joint_samples"], 0)
        self.assertEqual(report["position_excess_joint_samples"], 0)
        self.assertEqual(report["target_position_excess_joint_samples"], 0)
        self.assertAlmostEqual(report["joints"]["FL_thigh_joint"]["tracking_error_peak_rad"], .905, places=5)
        expected_penalty = ((.905 / .9) - .8) ** 2 / 4
        self.assertAlmostEqual(float(window.penalty("tracking_error")), expected_penalty, places=6)

    def test_tracking_endpoint_max_counted_once_and_target_is_cloned(self):
        window = self.window(envs=1, steps=1)
        position = self.pose(window)
        target = position.clone()
        target[0, 1] += .95
        window.record_command(target, position)
        # Simulate an action buffer reused after actuator compute.
        target.zero_()
        after = position.clone()
        after[0, 1] += .02
        window.record_state(after, torch.zeros_like(position))
        self.assertEqual(int(window.counts["tracking_error"].sum()), 1)
        self.assertAlmostEqual(float(window.peaks["tracking_error"][0, 1]), .95, places=5)
        expected_penalty = ((.95 / .9) - .8) ** 2
        self.assertAlmostEqual(float(window.penalty("tracking_error")), expected_penalty, places=6)

    def test_exact_hard_boundaries_legal_with_nonzero_soft_penalty(self):
        window = self.window(envs=2, steps=1)
        position = torch.stack((window.lower, window.upper))
        window.record_command(position, position)
        window.record_state(position, torch.full_like(position, 13.0))
        self.assertFalse(bool(window.hard_violation().any()))
        self.assertEqual(float(window.penalty("velocity_excess").sum()), 0)
        self.assertEqual(float(window.penalty("position_excess").sum()), 0)
        self.assertEqual(float(window.penalty("target_position_excess").sum()), 0)
        self.assertTrue(bool((window.penalty("velocity_margin") > 0).all()))
        self.assertTrue(bool((window.penalty("position_margin") > 0).all()))
        audit = JointConstraintAudit(window)
        audit.update(window, torch.ones(2, dtype=torch.bool))
        self.assertTrue(audit.report()["joint_limits_passed"])

    def test_soft_penalty_normalization_startup_and_tracking_boundary(self):
        window = self.window(envs=2, steps=1)
        position = self.pose(window)
        position[:, 1] = window.center[1] + .95 * window.half_range[1]
        target = position.clone()
        # Hip center is exactly zero: float32 target equal to the hard limit.
        target[:, 0] = window.tracking_error_limit
        window.record_command(target, position)
        velocity = torch.zeros_like(position)
        velocity[:, 0] = .9 * window.velocity_limit
        window.record_state(position, velocity)
        self.assertFalse(bool(window.hard_violation().any()))
        self.assertTrue(torch.allclose(window.penalty("velocity_margin", 2), torch.tensor([.02, .01]), atol=1e-6))
        self.assertTrue(torch.allclose(window.penalty("position_margin", 2), torch.tensor([.005, .0025]), atol=1e-6))
        self.assertTrue(torch.allclose(window.penalty("tracking_error", 2), torch.tensor([.08, .04]), atol=1e-6))
        for kind, multiplier in (("unknown", 1), ("velocity_margin", .5),
                                 ("velocity_margin", float("nan")), ("velocity_margin", True)):
            with self.assertRaises(ValueError):
                window.penalty(kind, multiplier)

    def test_incomplete_window_empty_audit_and_missing_startup_cannot_pass(self):
        window = self.window(envs=1, steps=2)
        audit = JointConstraintAudit(window)
        self.assertFalse(audit.report()["joint_limits_passed"])
        self.safe_step(window)
        with self.assertRaises(RuntimeError):
            window.require_complete()
        with self.assertRaises(RuntimeError):
            window.hard_violation()
        with self.assertRaises(RuntimeError):
            audit.update(window, torch.tensor([True]))
        self.safe_step(window)
        with self.assertRaises(RuntimeError):
            audit.update(window, torch.tensor([True]))
        self.assertTrue(audit.report()["audit_invalid"])
        window.begin(1, torch.tensor([False]))
        self.safe_step(window)
        no_startup = JointConstraintAudit(window)
        no_startup.update(window, torch.tensor([True]))
        self.assertFalse(no_startup.report()["joint_limits_passed"])

    def test_nonfinite_command_and_post_state_poison_window(self):
        for stage in ("command", "state"):
            for value in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(stage=stage, value=value):
                    window = self.window(envs=1, steps=1)
                    position = self.pose(window)
                    bad = position.clone()
                    bad[0, 0] = value
                    if stage == "command":
                        with self.assertRaises(FloatingPointError):
                            window.record_command(bad, position)
                    else:
                        window.record_command(position, position)
                        with self.assertRaises(FloatingPointError):
                            window.record_state(position, bad)
                    with self.assertRaises(RuntimeError):
                        window.require_complete()
                    with self.assertRaises(RuntimeError):
                        window.begin(1, torch.tensor([True]))
                    audit = JointConstraintAudit(window)
                    with self.assertRaises(RuntimeError):
                        audit.update(window, torch.tensor([True]))
                    self.assertFalse(audit.report()["joint_limits_passed"])

    def test_finite_values_that_overflow_penalty_are_rejected(self):
        window = self.window(envs=1, steps=1)
        position = self.pose(window)
        target = position.clone()
        target[0, 0] = 1e30
        with self.assertRaises(FloatingPointError):
            window.record_command(target, position)
        with self.assertRaises(RuntimeError):
            window.require_complete()

    def test_misordered_sample_and_invalid_shape_rejected(self):
        window = self.window(envs=1, steps=1)
        position = self.pose(window)
        with self.assertRaises(RuntimeError):
            window.record_state(position, torch.zeros_like(position))
        with self.assertRaises(RuntimeError):
            window.record_command(position, position)
        window = self.window(envs=1, steps=2)
        window.record_command(position, position)
        with self.assertRaises(RuntimeError):
            window.record_command(position, position)
        window = self.window(envs=1, steps=1)
        with self.assertRaises(ValueError):
            window.record_command(torch.zeros(1, 11), position)
        with self.assertRaises(ValueError):
            self.window(envs=1, steps=0)

    def test_mask_validation_and_mixed_audit_thresholds(self):
        window = self.window(envs=1, steps=1)
        self.safe_step(window)
        for mask in (torch.ones(1), torch.tensor([True, False]), [True]):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                JointConstraintAudit(window).update(window, mask)
        changed = self.window(envs=1, steps=1, velocity_limit=12)
        self.safe_step(changed)
        audit = JointConstraintAudit(window)
        with self.assertRaises(ValueError):
            audit.update(changed, torch.tensor([True]))
        self.assertFalse(audit.report()["joint_limits_passed"])

    def test_finished_environment_exclusion_and_logger_clones(self):
        window = self.window(envs=2, steps=1)
        position = self.pose(window)
        window.record_command(position, position)
        velocity = torch.zeros_like(position)
        velocity[1, 0] = 99
        window.record_state(position, velocity)
        logs = window.log_metrics()
        audit = JointConstraintAudit(window)
        audit.update(window, torch.tensor([True, False]))
        report = audit.report()
        self.assertTrue(report["joint_limits_passed"])
        self.assertEqual(report["velocity_excess_joint_samples"], 0)
        window.begin(1, torch.tensor([True, False]))
        self.safe_step(window)
        self.assertEqual(float(logs["JointLimits/velocity_peak_rad_s/FL_hip_joint"]), 99)
        self.assertEqual(float(window.log_metrics()["JointLimits/velocity_peak_rad_s/FL_hip_joint"]), 0)
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
