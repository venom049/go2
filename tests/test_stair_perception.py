"""Synthetic geometry and timing contracts; no transport or robot dependency."""
from dataclasses import replace
import json
import unittest

import numpy as np

from deploy.go2_stairs.perception import (
    PerceptionConfig, TemporalConfirmation, analyze_stairs,
)


def cloud(kind="up", steps=4, noise=0.002, height=0.14, first=0.60,
          depth=0.32, base=-0.30):
    cfg = PerceptionConfig()
    x = cfg.x_min + np.arange(118)*0.02 + 0.005
    x = x[x < cfg.x_max]
    ys = np.concatenate([c+np.array([-.08, -.04, 0., .04, .08]) for c in cfg.lane_centres])
    xx, yy = np.meshgrid(x, ys, indexing="ij")
    count = np.clip(np.floor((xx-first)/depth)+1, 0, steps)
    if kind == "flat": z = np.full_like(xx, base)
    elif kind == "ramp": z = base+0.15*xx
    elif kind == "gentle_ramp": z = base+0.04*xx
    elif kind == "box": z = base+0.18*((xx >= 0.70) & (xx < 1.35))
    else: z = base + (-1 if kind == "down" else 1)*height*count
    z += np.random.default_rng(7301).normal(0, noise, z.shape)
    return np.stack((xx, yy, z), axis=-1).reshape(-1, 3)


class GeometryTests(unittest.TestCase):
    def test_noisy_stairs_have_metric_geometry_but_never_control_permission(self):
        result = analyze_stairs(cloud())
        self.assertEqual(result["state"], "stairs_up")
        geometry = result["geometry"]
        self.assertEqual(geometry["visible_steps"], 4)
        self.assertAlmostEqual(geometry["distance_m"], .60, delta=.035)
        self.assertAlmostEqual(geometry["step_height_m"], .14, delta=.006)
        self.assertAlmostEqual(geometry["tread_depth_m"], .32, delta=.045)
        self.assertGreaterEqual(geometry["supported_lanes"], 3)
        self.assertFalse(result["control_authorized"])
        self.assertFalse(result["coverage"]["rear_footprint_observed"])
        json.dumps(result, allow_nan=False)

    def test_three_steps_are_sufficient_but_two_or_one_are_not(self):
        self.assertEqual(analyze_stairs(cloud(steps=3))["state"], "stairs_up")
        for steps in (1, 2):
            with self.subTest(steps=steps):
                self.assertNotIn(analyze_stairs(cloud(steps=steps))["state"], ("stairs_up", "stairs_down"))

    def test_downstairs_is_geometry_only(self):
        result = analyze_stairs(cloud("down", steps=3, height=.12))
        self.assertEqual(result["state"], "stairs_down")
        self.assertLess(result["geometry"]["signed_step_height_m"], 0)
        self.assertFalse(result["control_authorized"])

    def test_fully_observed_flat_but_not_ramps_boxes_or_wall(self):
        self.assertEqual(analyze_stairs(cloud("flat"))["state"], "flat")
        for kind in ("ramp", "gentle_ramp", "box"):
            with self.subTest(kind=kind):
                self.assertNotIn(analyze_stairs(cloud(kind))["state"], ("stairs_up", "stairs_down", "flat"))
        floor = cloud("flat")
        y, z = np.meshgrid(np.linspace(-.40, .40, 45), np.linspace(-.30, 1., 30))
        wall = np.column_stack((np.full(y.size, 1.15), y.ravel(), z.ravel()))
        result = analyze_stairs(np.vstack((floor, wall)))
        self.assertNotIn(result["state"], ("stairs_up", "stairs_down", "flat"))

    def test_missing_footprint_cannot_be_filled(self):
        points = cloud()
        cases = (points[np.abs(points[:, 1]) < .11],
                 points[(points[:, 0] < 1.0) | (points[:, 0] > 1.25)],
                 points[points[:, 0] > .70], points[points[:, 0] < 1.8])
        for subset in cases:
            with self.subTest(count=len(subset)):
                self.assertNotIn(analyze_stairs(subset)["state"], ("stairs_up", "stairs_down", "flat"))
        flat = cloud("flat")
        flat = flat[(flat[:, 0] < 1.1) | (flat[:, 0] > 1.16)]
        self.assertNotEqual(analyze_stairs(flat)["state"], "flat")

    def test_same_increment_on_disconnected_height_lanes_is_not_stairs(self):
        points = cloud()
        points[points[:, 1] < -.15, 2] += .30
        points[points[:, 1] > .15, 2] -= .30
        self.assertNotIn(analyze_stairs(points)["state"], ("stairs_up", "stairs_down"))

    def test_side_lanes_with_shifted_stair_edges_are_not_accepted(self):
        points = cloud()
        points[points[:, 1] > .15, 0] += .18
        self.assertNotIn(analyze_stairs(points)["state"], ("stairs_up", "stairs_down"))

    def test_width_must_be_observed_at_each_bin_not_once(self):
        for kind in ("flat", "up"):
            points = cloud(kind, noise=0)
            centres = np.array([-.3, 0., .3])
            nearest = centres[np.argmin(np.abs(points[:, 1, None]-centres), axis=1)]
            mask = points[:, 0] > .20
            points[mask, 1] = nearest[mask]+(points[mask, 1]-nearest[mask])*.01
            self.assertNotIn(analyze_stairs(points)["state"], ("flat", "stairs_up", "stairs_down"))

    def test_one_riser_bin_may_be_occluded_but_not_an_entire_tread(self):
        points = cloud()
        # Remove an edge bin: adjacent measured levels still establish a jump.
        index = np.floor((points[:, 0]-.15)/.04).astype(int)
        result = analyze_stairs(points[index != 11])
        self.assertEqual(result["state"], "stairs_up")
        # Removing a full tread leaves no direct evidence of a regular flight.
        points = points[(points[:, 0] < .92) | (points[:, 0] >= 1.24)]
        self.assertNotIn(analyze_stairs(points)["state"], ("stairs_up", "stairs_down"))

    def test_observed_wall_at_step_edge_is_not_treated_as_missing_riser(self):
        points = cloud()
        ys = np.concatenate([c+np.array([-.08, -.04, 0., .04, .08]) for c in (-.3, 0., .3)])
        for top in (1.5, -.08):
            # The first edge joins -.30 and -.16 m. Even a low protrusion
            # exceeding that measured envelope must not be ignored as a hole.
            y, z = np.meshgrid(ys, np.linspace(-.30, top, 20))
            wall = np.column_stack((np.full(y.size, .61), y.ravel(), z.ravel()))
            result = analyze_stairs(np.vstack((points, wall)))
            self.assertNotIn(result["state"], ("stairs_up", "stairs_down", "flat"))

    def test_mixed_up_and_down_edges_and_nonuniform_steps_are_rejected(self):
        points = cloud()
        points[points[:, 0] > 1.56, 2] -= .28
        self.assertNotIn(analyze_stairs(points)["state"], ("stairs_up", "stairs_down"))
        points = cloud()
        points[points[:, 0] > 1.24, 2] += .08
        self.assertNotIn(analyze_stairs(points)["state"], ("stairs_up", "stairs_down"))

    def test_order_duplicates_and_height_translation_do_not_change_evidence(self):
        points = cloud()
        result = analyze_stairs(points)
        shuffled = points[np.random.default_rng(12).permutation(len(points))]
        shifted = points.copy(); shifted[:, 2] += .20
        for modified in (shuffled, np.repeat(points, 3, axis=0), shifted):
            actual = analyze_stairs(modified)
            self.assertEqual(actual["state"], result["state"])
            self.assertAlmostEqual(actual["geometry"]["step_height_m"], result["geometry"]["step_height_m"], places=10)
        sparse = np.array([[.2, -.3, 0.], [.2, 0., 0.], [.2, .3, 0.]])
        self.assertEqual(analyze_stairs(np.repeat(sparse, 100, axis=0))["state"], "unknown")

    def test_empty_nonfinite_shape_and_point_limit(self):
        for points in (np.empty((0, 3)), np.full((10, 3), np.nan)):
            result = analyze_stairs(points)
            self.assertEqual(result["state"], "unknown")
            json.dumps(result, allow_nan=False)
        points = np.vstack((cloud(), [[np.nan, 0, 0], [0, np.inf, 0]]))
        self.assertEqual(analyze_stairs(points)["coverage"]["nonfinite_points_removed"], 2)
        self.assertEqual(analyze_stairs(points)["state"], "stairs_up")
        with self.assertRaises(ValueError): analyze_stairs(np.zeros((10, 4)))
        with self.assertRaises(ValueError): analyze_stairs(np.zeros((5, 3)), PerceptionConfig(max_input_points=4))

    def test_config_rejects_impossible_geometry_and_resolution(self):
        for changes in ({"x_max": .1}, {"min_steps": 2}, {"lane_centres": (0.,)},
                        {"bin_width": .2}, {"min_step_height": float("nan")},
                        {"min_points_per_bin": 1}, {"max_edge_missing_bins": 2}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(PerceptionConfig(), **changes)


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.result = {"state": "stairs_up", "control_authorized": False}

    def test_confirm_only_independent_frames_and_source_time_duration(self):
        gate = TemporalConfirmation()
        for stamp in (1., 1.10):
            self.assertEqual(gate.update(self.result, stamp, stamp+.01)["state"], "unknown")
        result = gate.update(self.result, 1.20, 1.21)
        self.assertEqual(result["state"], "stairs_up")
        self.assertTrue(result["confirmation"]["confirmed"])
        self.assertFalse(result["control_authorized"])

    def test_duplicate_backward_future_stale_and_unknown_clear_confirmation(self):
        for bad_stamp, now in ((1.2, 1.21), (1.1, 1.22), (2., 1.21), (1.3, 2.)):
            gate = TemporalConfirmation()
            for stamp in (1., 1.1, 1.2): gate.update(self.result, stamp, stamp)
            result = gate.update(self.result, bad_stamp, now)
            self.assertEqual(result["state"], "unknown")
            self.assertEqual(result["confirmation"]["frames"], 0)
        gate = TemporalConfirmation()
        for stamp in (1., 1.1, 1.2): gate.update(self.result, stamp, stamp)
        result = gate.update({"state": "unknown"}, 1.3, 1.3)
        self.assertFalse(result["confirmation"]["confirmed"])
        self.assertEqual(gate.update(self.result, 1.4, 1.4)["confirmation"]["frames"], 1)

    def test_gap_geometry_change_and_nan_time_do_not_continue_old_evidence(self):
        gate = TemporalConfirmation()
        gate.update(self.result, 1., 1.)
        self.assertEqual(gate.update(self.result, 1.5, 1.5)["confirmation"]["frames"], 1)
        self.assertEqual(gate.update({"state": "flat"}, 1.6, 1.6)["confirmation"]["frames"], 1)
        self.assertEqual(gate.update(self.result, float("nan"), 1.7)["state"], "unknown")
        with self.assertRaises(ValueError): TemporalConfirmation(required_frames=1)


if __name__ == "__main__":
    unittest.main()
