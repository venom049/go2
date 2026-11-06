"""Target projection, raw-cost evidence and weighted control-step telemetry."""
import json
import unittest

import numpy as np

from go2_him.official_guard import load_official_profile
from go2_him.target_clip import (
    TARGET_CLIP_CONFIG, TargetClipAccumulator, aggregate_target_clip_summaries,
    transform_target_action,
)


class TargetClipTests(unittest.TestCase):
    def setUp(self):
        self.profile = load_official_profile()
        self.scale = np.full(12, .25, dtype=np.float32)
        self.offset = np.asarray([.1, .8, -1.5, -.1, .8, -1.5,
                                  .1, 1., -1.5, -.1, 1., -1.5], dtype=np.float32)

    def convert(self, action, **changes):
        kwargs = dict(scale=self.scale, offset=self.offset,
                      lower=self.profile.lower, upper=self.profile.upper)
        kwargs.update(changes)
        return transform_target_action(action, **kwargs)

    def test_zero_action_preserves_default_and_types(self):
        result = self.convert(np.zeros(12))
        np.testing.assert_array_equal(result.raw_target, self.offset)
        np.testing.assert_array_equal(result.sent_target, self.offset.astype(np.float64))
        np.testing.assert_array_equal(result.sent_action, np.zeros(12, dtype=np.float32))
        self.assertEqual(result.raw_target.dtype, np.float32)
        self.assertEqual(result.sent_target.dtype, np.float64)
        self.assertEqual(result.sent_action.dtype, np.float32)
        self.assertEqual(float(result.cost_rate), 0.)
        self.assertFalse(TARGET_CLIP_CONFIG["saturating_cost"])

    def test_all_joints_both_sides_use_asymmetric_bounds(self):
        targets = np.stack((self.profile.lower - .2, self.profile.upper + .3))
        actions = ((targets - self.offset) / self.scale).astype(np.float32)
        result = self.convert(actions)
        np.testing.assert_array_equal(result.sent_target, np.stack((self.profile.lower, self.profile.upper)))
        self.assertTrue(np.all(result.raw_excess > 0))
        np.testing.assert_array_equal(result.sent_excess, np.zeros((2, 12)))
        np.testing.assert_allclose(result.cost_rate, 20 * result.raw_excess.sum(-1), rtol=0, atol=0)
        # Check history encoding directly; reapplying a rounded inverse must
        # not replace q_sent as the physically submitted target.
        expected = ((result.sent_target - self.offset.astype(np.float64)) / self.scale.astype(np.float64)).astype(np.float32)
        np.testing.assert_array_equal(result.sent_action, expected)

    def test_exact_float32_boundaries_have_zero_excess(self):
        # Identity scale and zero default need bounds containing zero; this
        # isolates exact-boundary handling from inverse affine rounding.
        lower = np.arange(-12, 0, dtype=np.float32).astype(np.float64)
        upper = np.arange(1, 13, dtype=np.float32).astype(np.float64)
        result = self.convert(np.stack((lower, upper)), scale=np.ones(12),
                              offset=np.zeros(12), lower=lower, upper=upper)
        np.testing.assert_array_equal(result.raw_excess, np.zeros((2, 12)))
        np.testing.assert_array_equal(result.sent_excess, np.zeros((2, 12)))
        np.testing.assert_array_equal(result.sent_target, np.stack((lower, upper)))

    def test_reported_calf_overshoot_is_costly_although_submitted_target_is_legal(self):
        action = np.zeros(12, dtype=np.float32)
        action[11] = 5.4441
        result = self.convert(action)
        self.assertAlmostEqual(float(result.raw_excess[11]), .6988, places=4)
        self.assertGreater(float(result.cost_rate), 13.9)
        self.assertEqual(result.sent_target[11], self.profile.upper[11])
        self.assertFalse(result.sent_excess.any())
        more = self.convert(action * 2)
        self.assertGreater(float(more.cost_rate), float(result.cost_rate) * 2)
        np.testing.assert_array_equal(result.raw_action, action)

    def test_batched_shape_and_raw_float32_affine(self):
        action = np.linspace(-3, 5, 72, dtype=np.float32).reshape(2, 3, 12)
        result = self.convert(action)
        np.testing.assert_array_equal(result.raw_target, action * self.scale + self.offset)
        self.assertEqual(result.cost_rate.shape, (2, 3))
        self.assertEqual(result.sent_action.shape, action.shape)

    def test_affine_preserves_supplied_double_configuration_arithmetic(self):
        action = np.linspace(-2, 3, 12, dtype=np.float32)
        scale = self.scale.astype(np.float64) + 1e-9
        offset = self.offset.astype(np.float64) + 1e-9
        result = self.convert(action, scale=scale, offset=offset)
        expected = action * scale + offset
        self.assertEqual(result.raw_target.dtype, expected.dtype)
        np.testing.assert_array_equal(result.raw_target, expected)
        self.assertFalse(np.array_equal(result.raw_target, action * scale.astype(np.float32) + offset.astype(np.float32)))

    def test_invalid_inputs_fail_without_silent_repair(self):
        for value in (np.nan, np.inf, -np.inf):
            bad = np.zeros(12); bad[2] = value
            with self.assertRaises(FloatingPointError): self.convert(bad)
        for action in (np.zeros(11), np.zeros((0, 12)), np.zeros((12, 1)), np.zeros(12, dtype=complex)):
            with self.assertRaises(ValueError): self.convert(action)
        for value in (0., -.25):
            scale = self.scale.copy(); scale[1] = value
            with self.assertRaises(ValueError): self.convert(np.zeros(12), scale=scale)
        with self.assertRaises(ValueError): self.convert(np.zeros(12), scale=.25)
        offset = self.offset.copy(); offset[2] = 0.
        with self.assertRaises(ValueError): self.convert(np.zeros(12), offset=offset)
        lower = self.profile.lower.copy(); lower[0] += 1e-10
        with self.assertRaisesRegex(ValueError, "float32-promoted"):
            self.convert(np.zeros(12), lower=lower)

    def test_baseline_record_keeps_actual_sent_violations(self):
        result = self.convert(np.full(12, 10.))
        accumulator = TargetClipAccumulator(self.profile.names)
        accumulator.record(result.raw_target, result.raw_target, self.profile.lower, self.profile.upper)
        report = accumulator.summary()
        self.assertGreater(report["raw"]["excess_count"], 0)
        self.assertEqual(report["raw"], report["sent"])
        self.assertEqual(report["clipping"]["control_steps"], 0)
        self.assertEqual(report["clipping"]["sum_l1_rad"], 0.)

    def test_accumulator_and_aggregation_are_weighted_by_control_steps(self):
        first = TargetClipAccumulator(self.profile.names)
        second = TargetClipAccumulator(self.profile.names)
        together = TargetClipAccumulator(self.profile.names)
        a = np.zeros((1, 12), dtype=np.float32); a[:, 2] = 5.4441
        b = np.zeros((3, 12), dtype=np.float32); b[2, 11] = -10.
        for accumulator, action in ((first, a), (second, b)):
            result = self.convert(action)
            accumulator.update(result)
            together.update(result)
        empty = TargetClipAccumulator(self.profile.names).summary()
        report = aggregate_target_clip_summaries([first.summary(), empty, second.summary()])
        self.assertEqual(report, together.summary())
        self.assertEqual(report["control_steps"], 4)
        self.assertEqual(report["raw"]["sample_count"], 48)
        self.assertEqual(report["raw"]["excess_count"], 2)
        self.assertEqual(report["raw"]["excess_fraction"], 2 / 48)
        self.assertEqual(report["raw"]["any_step_excess_fraction"], .5)
        self.assertEqual(report["sent"]["excess_count"], 0)
        self.assertEqual(report["clipping"]["control_steps"], 2)
        self.assertEqual(report["clipping"]["mean_l1_rad_per_step"], report["clipping"]["sum_l1_rad"] / 4)
        self.assertEqual(report["raw_cost_rate"]["mean"], report["raw_cost_rate"]["sum"] / 4)
        json.dumps(report, allow_nan=False)
        json.dumps(empty, allow_nan=False)

    def test_failed_record_does_not_mutate_accumulator(self):
        accumulator = TargetClipAccumulator(self.profile.names)
        before = accumulator.summary()
        with self.assertRaises(FloatingPointError):
            accumulator.record(np.full(12, np.nan), self.offset, self.profile.lower, self.profile.upper)
        self.assertEqual(accumulator.summary(), before)
        with self.assertRaises(ValueError): aggregate_target_clip_summaries([])
        other = TargetClipAccumulator(tuple(reversed(self.profile.names))).summary()
        with self.assertRaises(ValueError): aggregate_target_clip_summaries([before, other])


if __name__ == "__main__":
    unittest.main()
