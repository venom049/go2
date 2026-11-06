"""Reward-only joint recovery tests; no simulator or robot connection."""
from copy import deepcopy
from dataclasses import replace
import unittest

import numpy as np

from go2_him.joint_recovery import CONFIG, compute_joint_recovery_cost
from go2_him.official_guard import compute_guard_cost, evaluate_guard, load_official_profile


class JointRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.profile = load_official_profile()
        self.center = (self.profile.lower + self.profile.upper) / 2

    def sample(self, **kwargs):
        values = dict(q=self.center, dq=np.zeros(12), q_target=self.center,
                      kp=0., kd=0., applied_torque=np.zeros(12))
        values.update(kwargs)
        return evaluate_guard(self.profile, **values)

    def cost(self, **kwargs):
        return compute_joint_recovery_cost(self.sample(**kwargs))

    def test_safe_interior_zero_and_config_budget(self):
        c = self.cost()
        self.assertEqual(float(c.total), 0.)
        self.assertEqual(c.budget_total, 8.)
        self.assertEqual(sum(CONFIG['component_budgets_reward_units_per_second'].values()), 8.)
        self.assertEqual(CONFIG['target_position_soft_margin_rad'], .03)
        self.assertEqual(CONFIG['component_budgets_reward_units_per_second']['target_position'], 4.)
        self.assertFalse(CONFIG['physical_state_or_target_clipping'])

    def test_target_margin_symmetric_and_hard_boundary_is_not_excess(self):
        for joint in range(12):
            costs = []
            for bound, sign in ((self.profile.lower[joint], -1), (self.profile.upper[joint], 1)):
                target = self.center.copy()
                target[joint] = bound - sign * .03
                self.assertAlmostEqual(float(self.cost(q_target=target).bounded['target_position']), 0., places=12)
                target[joint] = bound - sign * .015
                costs.append(float(self.cost(q_target=target).bounded['target_position']))
                s = self.sample(q_target=np.where(np.arange(12) == joint, bound, self.center))
                self.assertFalse(s.exceeded['target_position'][joint])
                self.assertGreater(float(compute_joint_recovery_cost(s).bounded['target_position']), 0.)
            self.assertAlmostEqual(*costs, places=12)

    def test_both_position_bounds_and_every_axis_contribute(self):
        for joint in range(12):
            costs = []
            for bound, sign in ((self.profile.lower[joint], -1), (self.profile.upper[joint], 1)):
                q = self.center.copy()
                q[joint] = bound + sign * .05
                c = self.cost(q=q, q_target=q)
                self.assertGreater(float(c.bounded['position']), 0.)
                self.assertGreater(float(c.bounded['target_position']), 0.)
                self.assertEqual(float(c.bounded['tracking']), 0.)
                costs.append(float(c.total))
            self.assertAlmostEqual(*costs, places=12)

    def test_actual_position_has_no_in_range_posture_or_margin_cost(self):
        for q in (self.profile.lower, self.profile.upper,
                  self.profile.lower + .0001, self.profile.upper - .0001):
            self.assertEqual(float(self.cost(q=q).bounded['position']), 0.)

    def test_target_overshoot_monotone_bounded_and_stronger_than_old(self):
        joint = self.profile.names.index('RL_calf')
        results = []
        for excess in (0., 1e-6, .001, .01, .045, .1, .3, .7, 10.):
            target = self.center.copy()
            target[joint] = self.profile.upper[joint] + excess
            s = self.sample(q_target=target)
            new = float(compute_joint_recovery_cost(s).bounded['target_position'])
            old = float(compute_guard_cost(s).bounded['target_position'])
            self.assertGreater(new, old)
            self.assertLess(new, 4.)
            results.append(new)
        self.assertTrue(np.all(np.diff(results) > 0.))

    def test_worst_joint_not_twelve_joint_mean(self):
        target = self.center.copy()
        target[2] = self.profile.upper[2] + .04
        one = float(self.cost(q_target=target).bounded['target_position'])
        target[5] = self.profile.upper[5] + .02
        self.assertAlmostEqual(float(self.cost(q_target=target).bounded['target_position']), one)
        target[5] = self.profile.upper[5] + .08
        self.assertGreater(float(self.cost(q_target=target).bounded['target_position']), one)

    def test_physics_window_reduces_each_component_independently(self):
        q = np.broadcast_to(self.center, (3, 2, 12)).copy()
        target = q.copy()
        target[0, 0, 2] = self.profile.upper[2] + .05
        target[1, 1, 11] = self.profile.lower[11] - .08
        dq = np.zeros_like(q)
        dq[2, 0, 5] = self.profile.velocity[5] * 1.2
        tau = np.zeros_like(q)
        tau[1, 0, 3] = self.profile.effort[3] * 1.1
        s = self.sample(q=q, q_target=target, dq=dq, tau_ff=tau)
        full = compute_joint_recovery_cost(s)
        reduced = compute_joint_recovery_cost(s, physics_axis=0)
        negative_axis = compute_joint_recovery_cost(s, physics_axis=-2)
        self.assertEqual(reduced.total.shape, (2,))
        for key in full.bounded:
            np.testing.assert_allclose(reduced.bounded[key], np.max(full.bounded[key], axis=0))
            np.testing.assert_allclose(reduced.bounded[key], negative_axis.bounded[key])
        np.testing.assert_allclose(reduced.total, sum(np.max(v, axis=0) for v in full.bounded.values()))

    def test_velocity_tracking_and_requested_effort_are_separate(self):
        for sign in (-1, 1):
            dq = np.zeros(12)
            dq[2] = sign * self.profile.velocity[2] * 1.1
            velocity = self.cost(dq=dq)
            self.assertGreater(float(velocity.bounded['velocity']), 0.)
            self.assertEqual(float(velocity.bounded['torque']), 0.)
            target = self.center.copy()
            target[1] += sign * 1.0
            tracking = self.cost(q_target=target)
            self.assertGreater(float(tracking.bounded['tracking']), 0.)
            self.assertEqual(float(tracking.bounded['torque']), 0.)
            ff = np.zeros(12)
            ff[1] = sign * self.profile.effort[1] * 1.1
            torque = self.cost(tau_ff=ff)
            self.assertGreater(float(torque.bounded['torque']), 0.)
            self.assertEqual(float(torque.bounded['tracking']), 0.)

    def test_applied_effort_and_target_velocity_not_double_penalized(self):
        s = self.sample(applied_torque=self.profile.effort * 2., dq_target=self.profile.velocity * 2.)
        self.assertTrue(np.any(s.exceeded['applied_torque']))
        self.assertTrue(np.any(s.exceeded['target_velocity']))
        self.assertEqual(float(compute_joint_recovery_cost(s).total), 0.)
        self.assertEqual(float(self.cost(applied_torque=None).total), 0.)

    def test_combined_budget_and_scale(self):
        q = self.profile.lower - .1
        target = self.profile.upper + .1
        s = self.sample(q=q, q_target=target, dq=self.profile.velocity * 2.,
                        tau_ff=self.profile.effort * 2.)
        full = compute_joint_recovery_cost(s)
        scaled = compute_joint_recovery_cost(s, scale=.25)
        self.assertGreater(float(full.total), 0.)
        self.assertLessEqual(float(full.total), 8.)
        self.assertAlmostEqual(float(scaled.total), float(full.total) * .25)
        for key, budget in CONFIG['component_budgets_reward_units_per_second'].items():
            self.assertGreater(float(full.bounded[key]), 0.)
            self.assertLess(float(full.bounded[key]), budget)
            self.assertEqual(float(scaled.bounded[key]), float(full.bounded[key]))

    def test_inputs_and_audit_evidence_not_modified(self):
        q = self.center.copy()
        target = self.center.copy()
        target[8] = self.profile.upper[8] + .06
        s = self.sample(q=q, q_target=target, dq=self.profile.velocity * 1.1)
        before = deepcopy(s)
        compute_joint_recovery_cost(s)
        for key in ('values', 'ratios', 'excess', 'exceeded'):
            for metric, value in getattr(s, key).items():
                np.testing.assert_array_equal(value, getattr(before, key)[metric])
        np.testing.assert_array_equal(s.profile.lower, before.profile.lower)
        np.testing.assert_array_equal(s.profile.upper, before.profile.upper)
        np.testing.assert_array_equal(q, self.center)
        self.assertEqual(target[8], self.profile.upper[8] + .06)

    def test_nonfinite_and_missing_evidence_rejected(self):
        for collection, key in (('values', 'velocity'), ('values', 'target_position'),
                                ('values', 'applied_torque'), ('ratios', 'tracking'),
                                ('excess', 'position')):
            for value in (np.nan, np.inf, -np.inf):
                s = self.sample()
                changed = dict(getattr(s, collection))
                changed[key] = changed[key].copy()
                changed[key][0] = value
                with self.assertRaises(FloatingPointError):
                    compute_joint_recovery_cost(replace(s, **{collection: changed}))
        s = self.sample()
        with self.assertRaises(ValueError):
            compute_joint_recovery_cost(replace(s, values={**s.values, 'velocity': None}))

    def test_malformed_axis_scale_shape_and_profile_rejected(self):
        s = self.sample()
        for axis in (0, -1, True, 1.0, '0'):
            with self.assertRaises(ValueError):
                compute_joint_recovery_cost(s, physics_axis=axis)
        for scale in (0., -1., 1.1, np.nan, np.inf, True):
            with self.assertRaises(ValueError):
                compute_joint_recovery_cost(s, scale=scale)
        with self.assertRaises(ValueError):
            compute_joint_recovery_cost(replace(s, ratios={**s.ratios, 'velocity': np.zeros(11)}))
        with self.assertRaises(ValueError):
            compute_joint_recovery_cost(replace(s, excess={**s.excess, 'position': -np.ones(12)}))
        with self.assertRaises(ValueError):
            compute_joint_recovery_cost(replace(s, profile=replace(self.profile, upper=self.profile.lower)))

    def test_overflow_is_failure_instead_of_zero_or_nan_cost(self):
        s = self.sample()
        huge = np.full(12, np.finfo(np.float64).max)
        with self.assertRaises(FloatingPointError):
            compute_joint_recovery_cost(replace(s, ratios={**s.ratios, 'velocity': huge}))


if __name__ == '__main__':
    unittest.main()
