"""Protection cost tests exercise genuine physics windows without a simulator."""
import copy
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(torch is None, "PyTorch required; use .venv-deploy/bin/python")
class GuardBalanceTests(unittest.TestCase):
    def setUp(self):
        from go2_him.settings import load_settings
        self.settings = load_settings(ROOT / "configs/stairs_guard_balanced.yaml")
        self.cfg = self.settings["guard_balance"]
        # Nonstandard order deliberately checks effort limits by joint name.
        self.names = [f"{leg}_{kind}_joint" for kind in ("calf", "hip", "thigh")
                      for leg in ("RR", "FL", "RL", "FR")]

    def windows(self, *, speed=0., effort_ratio=0., tracking=0., target_overshoot=0.,
                actual_overshoot=0., spike_only=True, joint=0):
        from go2_him.joint_constraints import JointConstraintWindow
        from go2_him.torque import TorqueWindow
        j = JointConstraintWindow(2, self.names, velocity_limit=13., tracking_error_limit=.9)
        t = TorqueWindow(2, self.names)
        j.begin(4, torch.tensor([True, False]))
        t.begin(4, torch.tensor([True, False]))
        for step in range(4):
            q = j.center.repeat(2, 1)
            target = q.clone()
            dq = torch.zeros_like(q)
            requested = torch.zeros_like(q)
            if step == 3 or not spike_only:
                dq[0, joint] = speed
                requested[0, joint] = t.limits[joint] * effort_ratio
                target[0, joint] += tracking
                if target_overshoot:
                    target[0, joint] = j.upper[joint] + target_overshoot
                if actual_overshoot:
                    q[0, joint] = j.lower[joint] - actual_overshoot
                    target[0, joint] = q[0, joint]
            j.record_command(target, q)
            t.record(requested, requested.clamp(-t.limits, t.limits))
            j.record_state(q, dq)
        return j, t

    def cost(self, j, t, scale=1.):
        from go2_him.guard_balance import compute_guard_balance
        return compute_guard_balance(j, t, self.cfg, scale=scale)

    def test_safe_midrange_is_zero_and_unaffected_environment_is_zero(self):
        result = self.cost(*self.windows())
        torch.testing.assert_close(result.total, torch.zeros(2))
        result = self.cost(*self.windows(speed=26.))
        self.assertGreater(float(result.total[0]), 0.)
        self.assertEqual(float(result.total[1]), 0.)

    def test_single_final_substep_single_axis_peak_is_not_averaged_away(self):
        j, t = self.windows(speed=13.313)
        result = self.cost(j, t)
        all_steps = self.cost(*self.windows(speed=13.313, spike_only=False))
        torch.testing.assert_close(result.total, all_steps.total)
        self.assertEqual(int(j.counts["velocity"].sum()), 1)
        self.assertGreater(float(result.raw["velocity"][0]), .2)
        self.assertAlmostEqual(float(j.peaks["velocity"][0, 0]), 13.313, places=5)

    def test_reward_cap_does_not_clip_or_certify_audit_evidence(self):
        j, t = self.windows(speed=80., effort_ratio=10., target_overshoot=3.)
        before_speed = j.peaks["velocity"].clone()
        before_effort = t.requested_peak.clone()
        before_counts = j.counts["velocity"].clone()
        result = self.cost(j, t)
        self.assertLess(float(result.total.max()), 3.)
        self.assertTrue(bool(j.hard_violation()[0]))
        self.assertGreater(int(t.excess_count.sum()), 0)
        torch.testing.assert_close(j.peaks["velocity"], before_speed)
        torch.testing.assert_close(t.requested_peak, before_effort)
        torch.testing.assert_close(j.counts["velocity"], before_counts)
        self.assertGreater(float(result.ratios["torque"][0]), 9.99)
        self.assertLessEqual(float((t.applied_peak / t.limits).max()), 1.)

    def test_cost_remains_monotone_above_budget_knee(self):
        costs = [float(self.cost(*self.windows(speed=s)).bounded["velocity"][0])
                 for s in (11., 11.7, 13., 13.313, 20., 26., 50.)]
        self.assertAlmostEqual(costs[0], 0.)
        self.assertAlmostEqual(costs[1], 0., places=6)
        self.assertTrue(all(a < b for a, b in zip(costs[1:], costs[2:])))
        self.assertLess(costs[-1], self.cfg["budget_velocity"])

    def test_actual_and_target_position_inside_inclusive_limits_are_free(self):
        from go2_him.joint_constraints import JointConstraintWindow
        from go2_him.torque import TorqueWindow
        for endpoint in ("lower", "upper"):
            j = JointConstraintWindow(2, self.names)
            t = TorqueWindow(2, self.names)
            mask = torch.zeros(2, dtype=torch.bool)
            j.begin(1, mask)
            t.begin(1, mask)
            q = getattr(j, endpoint).repeat(2, 1)
            j.record_command(q, q)
            j.record_state(q, torch.zeros_like(q))
            t.record(torch.zeros_like(q), torch.zeros_like(q))
            result = self.cost(j, t)
            torch.testing.assert_close(result.total, torch.zeros(2))
        result = self.cost(*self.windows(target_overshoot=.05))
        self.assertGreater(float(result.bounded["target_position"][0]), 0)
        result = self.cost(*self.windows(actual_overshoot=.05))
        self.assertGreater(float(result.bounded["position"][0]), 0)

    def test_effort_normalization_uses_asymmetric_motor_limits_by_name(self):
        calf = self.names.index("RR_calf_joint")
        hip = self.names.index("FR_hip_joint")
        calf_cost = self.cost(*self.windows(effort_ratio=1.2, joint=calf))
        hip_cost = self.cost(*self.windows(effort_ratio=1.2, joint=hip))
        torch.testing.assert_close(calf_cost.bounded["torque"], hip_cost.bounded["torque"])

    def test_exact_hard_boundary_has_margin_but_no_violation(self):
        j, t = self.windows(speed=13., effort_ratio=1., tracking=.9)
        result = self.cost(j, t)
        self.assertEqual(int(j.counts["velocity"].sum()), 0)
        self.assertEqual(int(t.excess_count.sum()), 0)
        self.assertGreater(float(result.bounded["velocity"][0]), 0.)
        self.assertGreater(float(result.bounded["torque"][0]), 0.)

    def test_ramp_budget_is_run_local_and_evaluation_can_disable_ramp(self):
        from go2_him.guard_balance import guard_balance_scale
        for steps, expected in ((0, .2), (150 * 24, .6), (300 * 24, 1.), (1000 * 24, 1.)):
            self.assertAlmostEqual(guard_balance_scale(steps, 24, self.cfg), expected)
        full = self.cost(*self.windows(speed=26., effort_ratio=2.))
        early = self.cost(*self.windows(speed=26., effort_ratio=2.), scale=.2)
        torch.testing.assert_close(early.total, full.total * .2)
        evaluation = {**self.cfg, "ramp_updates": 0}
        self.assertEqual(guard_balance_scale(0, 24, evaluation), 1.)

    def test_incomplete_window_and_mismatched_order_are_rejected(self):
        j, t = self.windows()
        j.state_samples -= 1
        with self.assertRaises(RuntimeError):
            self.cost(j, t)
        j, t = self.windows()
        t.names = t.names[::-1]
        with self.assertRaises(ValueError):
            self.cost(j, t)

    def test_nonfinite_cost_is_rejected_not_bounded_to_false_success(self):
        j, t = self.windows()
        t.requested_peak[0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            self.cost(j, t)
        j, t = self.windows()
        t.requested_peak[0, 0] = torch.finfo(torch.float32).max
        with self.assertRaises(FloatingPointError):
            self.cost(j, t)

    def test_diagnostics_match_cost_and_are_independent_snapshots(self):
        from go2_him.guard_balance import KINDS
        result = self.cost(*self.windows(speed=26., effort_ratio=2.), scale=.2)
        logs = result.log_metrics()
        self.assertAlmostEqual(float(logs["GuardBalance/effective_budget"]), .6, places=6)
        weighted = sum(logs["GuardBalance/weighted_" + kind] for kind in KINDS)
        torch.testing.assert_close(weighted, logs["GuardBalance/total_cost"])
        value = logs["GuardBalance/total_cost"].clone()
        result.total.zero_()
        torch.testing.assert_close(logs["GuardBalance/total_cost"], value)

    def test_config_keeps_old_default_disabled_and_rejects_double_penalties(self):
        from go2_him.settings import load_settings, validate
        defaults = load_settings()
        self.assertFalse(defaults["guard_balance"]["enabled"])
        self.assertEqual(defaults["rewards"]["guard_balance"], 0.)
        self.assertFalse(self.settings["joint_constraints"]["terminate_on_violation"])
        for change in ({"velocity_excess": -2.}, {"torque_excess": -20.}, {"joint_limits": -1.}):
            broken = copy.deepcopy(self.settings)
            broken["rewards"].update(change)
            with self.assertRaises(ValueError):
                validate(broken)
        broken = copy.deepcopy(self.settings)
        broken["guard_balance"]["budget_velocity"] = 4.
        with self.assertRaises(ValueError):
            validate(broken)


if __name__ == "__main__":
    unittest.main()
