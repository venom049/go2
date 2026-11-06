"""Tests of current per-axis guards; no simulator or robot connection required."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from go2_him.official_guard import (DEFAULT_PROFILE_PATH, MOE_NAMES, OfficialGuardAudit,
                                   compute_guard_cost, evaluate_guard, load_official_profile)


class OfficialGuardTests(unittest.TestCase):
    def setUp(self):
        self.p = load_official_profile()
        self.q = (self.p.lower + self.p.upper) / 2

    def sample(self, **kwargs):
        args = dict(q=self.q, dq=np.zeros(12), q_target=self.q, kp=20., kd=.5,
                    applied_torque=np.zeros(12))
        args.update(kwargs)
        return evaluate_guard(self.p, **args)

    def test_profile_name_mapping_and_changed_profile_rejected(self):
        order = [n + "_joint" for n in MOE_NAMES[::-1]]
        p = load_official_profile(joint_names=order)
        self.assertEqual(p.names, MOE_NAMES[::-1])
        for i, name in enumerate(p.names):
            self.assertEqual(p.velocity[i], float(np.float32(15.7 if "calf" in name else 30.1)))
            self.assertEqual(p.effort[i], 45.43 if "calf" in name else 23.7)
        self.assertEqual(p.lower[p.names.index("RL_thigh")], float(np.float32(-.5236)))
        self.assertEqual(p.lower[p.names.index("FR_thigh")], float(np.float32(-1.5708)))
        with self.assertRaises(ValueError):
            load_official_profile(joint_names=["FR_hip"] * 12)
        changed = json.loads(DEFAULT_PROFILE_PATH.read_text())
        changed["limits"][2]["speed_limit_rad_s"] = 30.1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                load_official_profile(path)

    def test_inclusive_float32_boundaries_and_tiny_excess(self):
        for sign in (1, -1):
            s = self.sample(q=self.p.lower, q_target=self.p.lower, dq=sign*self.p.velocity, kp=0., kd=0.)
            self.assertFalse(bool(s.hard_violation))
            s = self.sample(q=self.p.upper, q_target=self.p.upper, dq=sign*self.p.velocity, kp=0., kd=0.)
            self.assertFalse(bool(s.hard_violation))
        dq = self.p.velocity.copy()
        dq[2] = np.nextafter(dq[2], np.inf)
        s = self.sample(dq=dq, kp=0., kd=0.)
        self.assertTrue(s.exceeded["velocity"][2])
        self.assertGreater(s.excess["velocity"][2], 0.)
        q = self.p.upper.copy()
        q[0] = np.nextafter(q[0], np.inf)
        s = self.sample(q=q, q_target=q, kp=0., kd=0.)
        self.assertTrue(s.exceeded["position"][0])
        self.assertTrue(s.exceeded["target_position"][0])

    def test_legal_fast_hip_cannot_hide_slow_overlimit_calf(self):
        dq = np.zeros(12)
        dq[0], dq[2] = 25., 16.
        s = self.sample(dq=dq, kp=0., kd=0.)
        self.assertFalse(s.exceeded["velocity"][0])
        self.assertTrue(s.exceeded["velocity"][2])
        self.assertGreater(s.ratios["velocity"][2], s.ratios["velocity"][0])

    def test_pd_formula_and_output_clamp_do_not_hide_request_violation(self):
        target = self.q.copy()
        target[0] += .4
        dq = np.zeros(12)
        dq[0] = -10.
        ff = np.zeros(12)
        ff[0] = 5.
        applied = np.zeros(12)
        applied[0] = self.p.effort[0]
        s = self.sample(q_target=target, dq=dq, kp=40., kd=1., tau_ff=ff, applied_torque=applied)
        self.assertAlmostEqual(s.values["requested_torque"][0], 31.)
        self.assertTrue(s.exceeded["requested_torque"][0])
        self.assertFalse(s.exceeded["applied_torque"][0])
        self.assertGreater(float(compute_guard_cost(s).bounded["torque"]), 0.)

    def test_tracking_and_target_velocity_are_independent_checks(self):
        target = self.q.copy()
        target[0] += self.p.tracking
        s = self.sample(q_target=target, kp=0., kd=0.)
        self.assertFalse(s.exceeded["tracking"][0])
        target[0] = np.nextafter(target[0], np.inf)
        s = self.sample(q_target=target, kp=0., kd=0.)
        self.assertTrue(s.exceeded["tracking"][0])
        target_dq = np.zeros(12)
        target_dq[2] = self.p.velocity[2] + 1e-8
        s = self.sample(dq_target=target_dq, kp=0., kd=0.)
        self.assertTrue(s.exceeded["target_velocity"][2])
        self.assertFalse(np.any(s.exceeded["velocity"]))

    def test_missing_applied_torque_is_unknown_and_nonfinite_rejected(self):
        audit = OfficialGuardAudit(self.p)
        audit.record(self.sample(applied_torque=None))
        report = audit.summary()
        self.assertIsNone(report["metrics"]["applied_torque"]["excess_fraction"])
        self.assertIsNone(report["joint_and_torque_acceptance_passed"])
        bad = self.q.copy()
        bad[1] = np.nan
        with self.assertRaises(FloatingPointError):
            self.sample(q=bad)

    def test_weighted_counts_and_terminal_substep_not_excluded(self):
        first, second = OfficialGuardAudit(self.p), OfficialGuardAudit(self.p)
        q = np.broadcast_to(self.q, (99, 12))
        first.record(self.sample(q=q, dq=np.zeros_like(q)))
        dq = np.zeros(12)
        dq[2] = self.p.velocity[2] + 1e-8
        second.record(self.sample(dq=dq), terminal=True)
        first.merge(second)
        report = first.summary()
        row = report["metrics"]["velocity"]
        self.assertEqual(report["physics_state_samples"], 100)
        self.assertEqual(report["terminal_physics_state_samples"], 1)
        self.assertEqual(row["excess_count"], 1)
        self.assertEqual(row["sample_count"], 1200)
        self.assertEqual(row["excess_fraction"], 1/1200)
        self.assertEqual(row["per_joint"]["FL_calf"]["excess_fraction"], .01)
        self.assertFalse(report["joint_and_torque_acceptance_passed"])
        self.assertGreater(row["max_excess"], 0.)
        json.dumps(report, allow_nan=False)

    def test_no_startup_grace_and_profile_cannot_be_mixed(self):
        audit = OfficialGuardAudit(self.p)
        dq = np.zeros(12)
        dq[2] = 16.
        audit.record(self.sample(dq=dq))
        self.assertEqual(audit.summary()["metrics"]["velocity"]["excess_count"], 1)
        other = OfficialGuardAudit(load_official_profile(joint_names=MOE_NAMES[::-1]))
        with self.assertRaises(ValueError):
            audit.merge(other)

    def test_cost_bounded_separate_from_evidence_and_uses_worst_substep(self):
        q = np.broadcast_to(self.q, (4, 2, 12)).copy()
        dq = np.zeros_like(q)
        dq[-1, 0, 2] = 100.
        sample = self.sample(q=q, dq=dq)
        before = sample.values["velocity"].copy()
        result = compute_guard_cost(sample, physics_axis=0)
        self.assertEqual(result.total.shape, (2,))
        self.assertGreater(result.total[0], 0.)
        self.assertLess(result.total[0], 3.)
        self.assertEqual(result.total[1], 0.)
        np.testing.assert_array_equal(sample.values["velocity"], before)
        self.assertEqual(np.count_nonzero(sample.exceeded["velocity"]), 1)
        expected = compute_guard_cost(self.sample(dq=dq[-1, 0])).total
        self.assertEqual(result.total[0], expected)

    def test_positions_inside_range_have_zero_position_cost_not_posture_cost(self):
        for endpoint in (self.p.lower, self.p.upper):
            cost = compute_guard_cost(self.sample(q=endpoint, q_target=endpoint))
            self.assertEqual(float(cost.total), 0.)
        q = self.p.upper.copy()
        q[0] += 1e-8
        cost = compute_guard_cost(self.sample(q=q, q_target=q))
        self.assertGreater(float(cost.bounded["position"]), 0.)
        self.assertGreater(float(cost.bounded["target_position"]), 0.)

    def test_margin_cost_and_torque_actual_boundaries(self):
        costs = []
        for ratio in (.5, .9, 1., 1.01, 2.):
            dq = np.zeros(12)
            dq[2] = ratio * self.p.velocity[2]
            s = self.sample(dq=dq, kp=0., kd=0.)
            costs.append(float(compute_guard_cost(s).bounded["velocity"]))
        self.assertEqual(costs[:2], [0., 0.])
        self.assertTrue(all(a < b for a, b in zip(costs[1:], costs[2:])))
        s = self.sample(tau_ff=self.p.effort, kp=0., kd=0., applied_torque=-self.p.effort)
        self.assertFalse(bool(s.hard_violation))
        ff = self.p.effort.copy()
        ff[0] = np.nextafter(ff[0], np.inf)
        self.assertTrue(self.sample(tau_ff=ff, kp=0., kd=0.).exceeded["requested_torque"][0])


if __name__ == "__main__":
    unittest.main()
