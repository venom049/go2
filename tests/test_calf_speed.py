"""Calf-only cost and real four-joint acceptance; no simulator or robot I/O."""
import copy
import json
import unittest

import numpy as np

from go2_him.calf_speed import CALF_NAMES, COST_KEY, calf_speed_summary, compute_calf_speed_cost
from go2_him.official_guard import (MOE_NAMES, OfficialGuardAudit, evaluate_guard,
                                   load_official_profile)


class CalfSpeedTests(unittest.TestCase):
    def setUp(self):
        self.profile = load_official_profile()
        self.q = (self.profile.lower + self.profile.upper) / 2

    def sample(self, profile=None, **kwargs):
        profile = profile or self.profile
        q = (profile.lower + profile.upper) / 2
        args = dict(q=q, dq=np.zeros(12), q_target=q, kp=20., kd=.5,
                    applied_torque=np.zeros(12))
        args.update(kwargs)
        return evaluate_guard(profile, **args)

    def summary(self, sample):
        audit = OfficialGuardAudit(sample.profile)
        audit.record(sample)
        return audit.summary()

    def test_only_calf_actual_speed_contributes_but_other_checks_remain_audited(self):
        dq = np.zeros(12)
        noncalf = [i for i, name in enumerate(self.profile.names) if not name.endswith("_calf")]
        dq[noncalf] = 100.
        target = self.profile.upper + 5.
        sample = self.sample(dq=dq, q_target=target, dq_target=100., tau_ff=100.)
        self.assertTrue(sample.hard_violation)
        self.assertEqual(float(compute_calf_speed_cost(sample).total), 0.)
        guard = self.summary(sample)
        self.assertFalse(guard["joint_and_torque_acceptance_passed"])
        self.assertTrue(calf_speed_summary(guard)["calf_speed_acceptance_passed"])

    def test_name_mapping_survives_joint_order_change(self):
        dq = np.zeros(12)
        dq[self.profile.names.index("RR_calf")] = -26.
        forward = self.sample(dq=dq)
        reverse_profile = load_official_profile(joint_names=MOE_NAMES[::-1])
        reverse = self.sample(profile=reverse_profile, dq=dq[::-1])
        self.assertEqual(float(compute_calf_speed_cost(forward).total), float(compute_calf_speed_cost(reverse).total))
        a, b = calf_speed_summary(self.summary(forward)), calf_speed_summary(self.summary(reverse))
        self.assertEqual(a, b)

    def test_margin_formula_monotonic_budget_and_no_sample_mutation(self):
        costs = []
        for ratio in (0., .89, .90, .95, 1., 1.01, 1.5, 2., 10.):
            dq = np.zeros(12)
            dq[2] = ratio * self.profile.velocity[2]
            sample = self.sample(dq=dq)
            before = sample.values["velocity"].copy()
            result = compute_calf_speed_cost(sample)
            actual_ratio = float(result.ratios[COST_KEY])
            raw = .5 * max((actual_ratio - .9) / .1, 0.)**2 + 4 * max(actual_ratio - 1, 0.) + 4 * max(actual_ratio - 1, 0.)**2
            self.assertAlmostEqual(float(result.total), 4 * raw / (4 + raw))
            self.assertEqual(result.scale, 1.)
            self.assertEqual(result.budget_total, 4.)
            self.assertLess(float(result.total), 4.)
            np.testing.assert_array_equal(sample.values["velocity"], before)
            costs.append(float(result.total))
        self.assertEqual(costs[:3], [0., 0., 0.])
        self.assertTrue(all(a < b for a, b in zip(costs[2:], costs[3:])))

    def test_float32_official_boundaries_and_nextafter_violation(self):
        self.assertEqual(float(self.profile.velocity[2]), float(np.float32(15.7)))
        for sign in (-1., 1.):
            dq = np.zeros(12)
            dq[2] = sign * self.profile.velocity[2]
            result = calf_speed_summary(self.summary(self.sample(dq=dq)))
            self.assertTrue(result["calf_speed_acceptance_passed"])
            self.assertEqual(result["excess_count"], 0)
            dq[2] = sign * np.nextafter(self.profile.velocity[2], np.inf)
            result = calf_speed_summary(self.summary(self.sample(dq=dq)))
            self.assertFalse(result["calf_speed_acceptance_passed"])
            self.assertEqual(result["excess_count"], 1)
            self.assertGreater(result["max_excess"], 0.)

    def test_one_joint_one_boundary_peak_not_diluted_across_axes_or_time(self):
        dq = np.zeros((20, 3, 12))
        dq[-1, 1, 11] = 26.
        result = compute_calf_speed_cost(self.sample(dq=dq), physics_axis=0)
        expected = compute_calf_speed_cost(self.sample(dq=dq[-1, 1])).total
        self.assertEqual(result.total.shape, (3,))
        np.testing.assert_array_equal(result.total, [0., expected, 0.])
        negative_axis = compute_calf_speed_cost(self.sample(dq=dq), physics_axis=-2)
        np.testing.assert_array_equal(negative_axis.total, result.total)
        control_only = compute_calf_speed_cost(self.sample(dq=dq[-1]))
        np.testing.assert_array_equal(control_only.total, result.total)

    def test_summary_uses_exact_four_calf_denominator_and_merged_counts(self):
        first, second = OfficialGuardAudit(self.profile), OfficialGuardAudit(self.profile)
        first.record(self.sample(dq=np.zeros((99, 12))))
        dq = np.zeros(12)
        dq[2] = self.profile.velocity[2] + 1e-8
        second.record(self.sample(dq=dq), terminal=True)
        first.merge(second)
        guard = first.summary()
        # Fractions supplied by a caller are not inputs to the aggregation.
        guard["metrics"]["velocity"]["per_joint"]["FL_calf"]["excess_fraction"] = .99
        result = calf_speed_summary(guard)
        self.assertEqual(guard["metrics"]["velocity"]["sample_count"], 1200)
        self.assertEqual(result["sample_count"], 400)
        self.assertEqual(result["excess_count"], 1)
        self.assertEqual(result["excess_fraction"], 1 / 400)
        self.assertEqual(result["per_joint"]["FL_calf"]["excess_fraction"], .01)
        self.assertEqual(set(result["per_joint"]), set(CALF_NAMES))
        self.assertFalse(result["calf_speed_acceptance_passed"])
        json.dumps(result, allow_nan=False)

    def test_missing_or_unobserved_data_is_unknown_not_a_zero_pass(self):
        for guard in ({}, OfficialGuardAudit(self.profile).summary()):
            result = calf_speed_summary(guard)
            self.assertIsNone(result["calf_speed_acceptance_passed"])
            self.assertIsNone(result["excess_fraction"])
        good = self.summary(self.sample())
        missing = copy.deepcopy(good)
        del missing["metrics"]["velocity"]["per_joint"]["RR_calf"]
        result = calf_speed_summary(missing)
        self.assertIsNone(result["calf_speed_acceptance_passed"])
        self.assertIsNone(result["sample_count"])
        self.assertIsNone(result["peak_absolute_value"])
        good["metrics"]["velocity"]["unobserved_state_samples"] = 1
        self.assertIsNone(calf_speed_summary(good)["calf_speed_acceptance_passed"])
        # Missing another axis never erases a known terminal-boundary violation.
        dq = np.zeros(12);dq[2] = 16.
        bad = self.summary(self.sample(dq=dq))
        del bad["metrics"]["velocity"]["per_joint"]["RR_calf"]
        self.assertFalse(calf_speed_summary(bad)["calf_speed_acceptance_passed"])

    def test_invalid_nonfinite_or_inconsistent_evidence_rejected(self):
        for invalid in (np.nan, np.inf, -np.inf):
            dq = np.zeros(12);dq[2] = invalid
            with self.assertRaises(FloatingPointError):
                self.sample(dq=dq)
            sample = self.sample()
            sample.values["velocity"][0] = invalid
            with self.assertRaises(FloatingPointError):
                compute_calf_speed_cost(sample)
            guard = self.summary(self.sample())
            guard["metrics"]["velocity"]["per_joint"]["FL_calf"]["peak_absolute_value"] = invalid
            with self.assertRaises(ValueError):
                calf_speed_summary(guard)
        for field, value in (("sample_count", 1.5), ("excess_count", True), ("excess_count", 2)):
            guard = self.summary(self.sample())
            guard["metrics"]["velocity"]["per_joint"]["FL_calf"][field] = value
            with self.assertRaises(ValueError):
                calf_speed_summary(guard)
        guard = self.summary(self.sample())
        guard["profile"]["velocity_limits_rad_s"][2] = 13.
        with self.assertRaises(ValueError):
            calf_speed_summary(guard)

    def test_numeric_failure_missing_final_boundary_cannot_certify_a_pass(self):
        for marker in ({"incomplete_due_to_numeric_failure": True},
                       {"incomplete_episodes": 1}):
            with self.subTest(marker=marker):
                # The prior, recorded physical boundaries were all legal;
                # the simulator failed before the last boundary was audited.
                guard = self.summary(self.sample(dq=np.zeros((5, 12))))
                guard.update(marker)
                result = calf_speed_summary(guard)
                self.assertEqual(result["sample_count"], 20)
                self.assertEqual(result["excess_count"], 0)
                self.assertEqual(result["excess_fraction"], 0.)
                self.assertFalse(result["all_calf_metrics_observed"])
                self.assertIsNone(result["calf_speed_acceptance_passed"])
                self.assertTrue(all(row["calf_speed_acceptance_passed"] is None
                                    for row in result["per_joint"].values()))
                # An earlier measured violation stays a failure despite the
                # later missing boundary, at both total and individual level.
                dq = np.zeros((5, 12));dq[0, 2] = 16.
                guard = self.summary(self.sample(dq=dq));guard.update(marker)
                result = calf_speed_summary(guard)
                self.assertEqual(result["excess_count"], 1)
                self.assertFalse(result["calf_speed_acceptance_passed"])
                self.assertFalse(result["per_joint"]["FL_calf"]["calf_speed_acceptance_passed"])
                self.assertIsNone(result["per_joint"]["RR_calf"]["calf_speed_acceptance_passed"])
        complete = self.summary(self.sample())
        complete.update(incomplete_due_to_numeric_failure=False, incomplete_episodes=0)
        self.assertTrue(calf_speed_summary(complete)["calf_speed_acceptance_passed"])
        for marker in ({"incomplete_due_to_numeric_failure": "false"},
                       {"incomplete_episodes": -1}, {"incomplete_episodes": 1.5}):
            with self.subTest(invalid_marker=marker), self.assertRaises(ValueError):
                calf_speed_summary({**complete, **marker})

    def test_invalid_cost_configuration_and_joint_axis_rejected(self):
        sample = self.sample()
        for budget in (0., -1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                compute_calf_speed_cost(sample, budget=budget)
        for soft_ratio in (0., 1., -1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                compute_calf_speed_cost(sample, soft_ratio=soft_ratio)
        for physics_axis in (0, -1, True, .0):
            with self.assertRaises(ValueError):
                compute_calf_speed_cost(sample, physics_axis=physics_axis)
        batched = self.sample(dq=np.zeros((2, 12)))
        with self.assertRaises(ValueError):
            compute_calf_speed_cost(batched, physics_axis=1)


if __name__ == "__main__":
    unittest.main()
