"""Geometry invariants that prevent spawning on top or rewarding a descent."""

from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from go2_him.terrain import StairCourse, StairsTerrain, StairsTerrainConfig
from go2_him.lab_terrain import ascending_stairs, course_geometry


class StairsTerrainTests(unittest.TestCase):
    def setUp(self):
        self.cfg = StairsTerrainConfig(num_rows=4, num_cols=3)
        self.terrain = StairsTerrain(self.cfg)

    def tile(self, row, col):
        terrain = self.terrain
        x = terrain.border + row * terrain.length_per_env_pixels
        y = terrain.border + col * terrain.width_per_env_pixels
        return terrain.height_field_raw[x:x + terrain.length_per_env_pixels, y:y + terrain.width_per_env_pixels]

    def test_spawn_is_on_lower_approach_for_every_level(self):
        for row in range(self.cfg.num_rows):
            for col in range(self.cfg.num_cols):
                origin = self.terrain.env_origins[row, col]
                np.testing.assert_allclose(origin, (row * 8 + 0.8, col * 3 + 1.5, 0.0))
                spawn_index = int(round(0.8 / self.cfg.horizontal_scale))
                self.assertEqual(int(self.tile(row, col)[spawn_index, 30]), 0)

    def test_all_courses_ascend_without_descent_and_have_eight_risers(self):
        for row in range(1, self.cfg.num_rows):
            for col in range(self.cfg.num_cols):
                tile = self.tile(row, col)
                profile = tile[:, tile.shape[1] // 2].astype(np.int32)
                changes = np.diff(profile)
                self.assertTrue(np.all(changes >= 0))
                self.assertEqual(np.count_nonzero(changes), self.cfg.stair_num_steps)
                self.assertAlmostEqual(profile[-1] * self.cfg.vertical_scale, self.terrain.upper_heights[row, col])
                self.assertTrue(np.all(tile == tile[:, :1]))

    def test_curriculum_includes_flat_and_eighteen_centimeter_endpoints(self):
        np.testing.assert_array_equal(self.tile(0, 0), 0)
        np.testing.assert_allclose(self.terrain.step_heights[0], 0.0)
        np.testing.assert_allclose(self.terrain.step_heights[-1], 0.18)
        self.assertTrue(np.all(np.diff(self.terrain.step_heights, axis=0) > 0))
        self.assertTrue(np.all(np.diff(self.terrain.stair_treads, axis=1) >= 0))

    def test_goal_is_on_landing_and_inside_termination_bounds(self):
        for row in range(self.cfg.num_rows):
            for col in range(self.cfg.num_cols):
                goal_x = self.terrain.goal_x[row, col]
                self.assertGreater(goal_x, self.terrain.stair_end_x[row, col])
                xmin, xmax, ymin, ymax = self.terrain.course_bounds[row, col]
                self.assertLess(xmin, 0.0)
                self.assertLess(0.0, goal_x)
                self.assertLess(goal_x, xmax)
                self.assertLess(ymin, 0.0)
                self.assertGreater(ymax, 0.0)
                goal_pixel = int((goal_x + self.cfg.stair_spawn_x) / self.cfg.horizontal_scale)
                actual_z = self.tile(row, col)[goal_pixel, 30] * self.cfg.vertical_scale
                self.assertAlmostEqual(actual_z, self.terrain.upper_heights[row, col])

    def test_raster_border_and_alias(self):
        self.assertIs(self.terrain.heightsamples, self.terrain.height_field_raw)
        self.assertEqual(self.terrain.height_field_raw.dtype, np.int16)
        self.assertEqual(self.terrain.height_field_raw.shape, (720, 260))
        border = self.terrain.border
        self.assertFalse(self.terrain.height_field_raw[:border].any())
        self.assertFalse(self.terrain.height_field_raw[-border:].any())
        self.assertFalse(self.terrain.height_field_raw[:, :border].any())
        self.assertFalse(self.terrain.height_field_raw[:, -border:].any())

    def test_invalid_geometry_fails_before_silent_truncation(self):
        invalid = (
            replace(self.cfg, terrain_length=8.03),
            replace(self.cfg, stair_num_steps=40),
            replace(self.cfg, stair_spawn_x=3.0),
            replace(self.cfg, stair_goal_offset=8.0),
            replace(self.cfg, vertical_scale=0.0),
            replace(self.cfg, stair_height_max=-0.1),
            replace(self.cfg, stair_tread_min=0.01, stair_tread_max=0.01),
        )
        for cfg in invalid:
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                StairsTerrain(cfg)


class LabTerrainTests(unittest.TestCase):
    def setUp(self):
        self.cfg = SimpleNamespace(
            size=(8.0, 3.0), step_height_range=(0.0, 0.18), step_width=0.39,
            num_steps=8, approach_length=2.0, landing_length=2.0, spawn_x=0.8,
            goal_offset=0.7, boundary_margin=0.35,
        )

    def test_lab_geometry_retains_exact_dimensions(self):
        course = course_geometry(1.0, self.cfg)
        self.assertEqual(course.step_width, 0.39)
        self.assertEqual(course.step_height, 0.18)
        self.assertAlmostEqual(course.upper_height, 1.44)
        self.assertAlmostEqual(course.goal_x, 5.02)
        np.testing.assert_allclose(course.origin, [0.8, 1.5, 0.0])
        np.testing.assert_allclose(course.height_at([0.8, 1.99, 2.01, 2.4, 5.82]), [0, 0, .18, .36, 1.44])

    def test_segments_cover_entire_tile_without_gaps(self):
        course = course_geometry(1.0, self.cfg)
        segments = course.segments()
        self.assertEqual(segments[0][0], 0)
        self.assertEqual(segments[-1][1], 8)
        self.assertEqual(len(segments), 9)
        for left, right in zip(segments, segments[1:]):
            self.assertAlmostEqual(left[1], right[0])
            self.assertLess(left[2], right[2])

    def test_flat_course_remains_solid(self):
        self.assertEqual(course_geometry(0.0, self.cfg).segments(), [(0.0, 8.0, 0.0)])

    def test_invalid_difficulty_is_rejected(self):
        for difficulty in (-.01, 1.01, float("nan")):
            with self.assertRaises(ValueError):
                course_geometry(difficulty, self.cfg)

    def test_mesh_has_correct_extents_and_spawn_when_trimesh_installed(self):
        try:
            import trimesh
        except ImportError:
            self.skipTest("trimesh is optional for NumPy geometry tests")
        meshes, origin = ascending_stairs(1.0, self.cfg)
        self.assertTrue(all(mesh.is_watertight for mesh in meshes))
        combined = trimesh.util.concatenate(meshes)
        np.testing.assert_allclose(combined.bounds, [[0, 0, -.1], [8, 3, 1.44]])
        np.testing.assert_allclose(origin, [.8, 1.5, 0.0])


if __name__ == "__main__":
    unittest.main()
