"""Finite-width evaluation geometry and false-success regressions; CPU only."""
import copy
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from go2_him.evaluation_narrow_stairs import (
    FOOT_NAMES, ascending_narrow_stairs, ascending_narrow_stairs_09,
    configure, narrow_stairs_success,
    upper_landing_occupancy,
)


def original_success(env, allowed=True):
    return env.original_result & allowed


def configuration():
    stairs = SimpleNamespace(size=(8., 3.), step_height_range=(.15, .15),
        step_width=.30, num_steps=2, approach_length=2., landing_length=2.,
        spawn_x=.8, goal_offset=.7, boundary_margin=.35, function=None)
    generator = SimpleNamespace(size=(8., 3.), sub_terrains={"stairs": stairs}, use_cache=True)
    return SimpleNamespace(go2_settings={"terrain": {"curriculum": False}},
        scene=SimpleNamespace(terrain=SimpleNamespace(terrain_generator=generator)),
        terminations=SimpleNamespace(success=SimpleNamespace(func=original_success, params={"allowed": True})))


class NarrowStairsTests(unittest.TestCase):
    def setUp(self):
        self.cfg = configuration()
        self.geometry = configure(self.cfg)

    def test_both_widths_with_full_floor_and_neighbor_gap(self):
        # Inspect the exact boxes passed to trimesh, without requiring its extra
        # dependencies. Independent interval probes check floor/side-edge height.
        module = ModuleType("trimesh")
        def box(extents, transform):
            center = transform[:3, 3].copy()
            half = np.asarray(extents) / 2
            return np.stack((center - half, center + half))
        module.creation = SimpleNamespace(box=box)
        for width, expected_y in ((.6, (1.2, 1.8)), (.9, (1.05, 1.95))):
            with self.subTest(width=width):
                cfg = configuration(); geometry = configure(cfg, width)
                stairs = cfg.scene.terrain.terrain_generator.sub_terrains["stairs"]
                with patch.dict("sys.modules", {"trimesh": module}):
                    meshes, origin = stairs.function(0., stairs)
                np.testing.assert_allclose(origin, [.8, 1.5, 0])
                np.testing.assert_allclose(meshes[0], [[0, 0, -.1], [8, 3, 0]])
                self.assertEqual(len(meshes), 3)
                lo, hi = expected_y
                np.testing.assert_allclose(meshes[1], [[2., lo, 0], [2.3, hi, .15]])
                np.testing.assert_allclose(meshes[2], [[2.3, lo, 0], [8., hi, .30]])
                def height(x, y):
                    return max(bounds[1, 2] for bounds in meshes
                               if bounds[0, 0] < x < bounds[1, 0] and bounds[0, 1] < y < bounds[1, 1])
                self.assertEqual(height(2.15, 1.5), .15)
                self.assertEqual(height(2.45, 1.5), .30)
                self.assertEqual(height(2.45, lo - .01), 0.)
                self.assertEqual(height(2.45, hi + .01), 0.)
                self.assertEqual(geometry["neighboring_flight_ground_gap_m"], 3. - width)
                self.assertFalse(geometry["assumed_upper_landing"]["real_dimensions_known"])

    def samples(self, count=1):
        root = np.tile([2.55, 0., .62], (count, 1))
        feet = np.tile([[2.80, .15, .325], [2.80, -.15, .325],
                        [2.30, .15, .325], [2.30, -.15, .325]], (count, 1, 1))
        forces = np.tile([20., 0., 0., 20.], (count, 1))
        return root, feet, forces

    def test_lower_floor_detour_airborne_edge_and_one_rear_foot_left_fail(self):
        root, feet, force = self.samples(7)
        root[1, 2] -= .30; feet[1, :, 2] -= .30  # bypass on lower floor
        feet[2, 3, 1] = .29                      # foot over margin/edge
        feet[3, 2, 0] = 1.79                     # rear foot not on landing
        force[4] = 0.                           # airborne
        feet[5, 1, 2] = .50                     # foot well above platform
        root[6, 1] = .4                         # outside narrow landing
        result = upper_landing_occupancy(root, feet, force, self.geometry)
        np.testing.assert_array_equal(result, [True, False, False, False, False, False, False])

    def test_nonfinite_state_or_contacts_cannot_pass(self):
        root, feet, force = self.samples(4)
        root[0, 0] = np.nan; feet[1, 0, 2] = np.inf
        force[2, 1] = np.nan; force[3, 2] = np.inf
        with np.errstate(invalid="ignore"):
            self.assertFalse(upper_landing_occupancy(root, feet, force, self.geometry).any())

    def test_nine_tenths_width_keeps_two_point_five_cm_foot_margin(self):
        geometry = configure(configuration(), .9)
        self.assertAlmostEqual(geometry["flight_width_m"] / 2 - geometry["foot_center_edge_margin_m"], .425)
        root, feet, force = self.samples(4)
        feet[0, 0, 1] = .425
        feet[1, 0, 1] = -.425
        feet[2, 0, 1] = .425001
        feet[3, 0, 1] = -.425001
        np.testing.assert_array_equal(upper_landing_occupancy(root, feet, force, geometry),
                                      [True, True, False, False])
        root, feet, force = self.samples(4)
        root[:, 1] = [.425, -.425, .425001, -.425001]
        np.testing.assert_array_equal(upper_landing_occupancy(root, feet, force, geometry),
                                      [True, True, False, False])

    def test_success_uses_named_body_mapping_world_offsets_and_original_predicate(self):
        root, feet, force = self.samples(2)
        origin = np.array([[100., 200., 3.], [-10., -20., 1.]])
        names = ["base_link", "RR_foot", "FL_foot", "FR_foot", "RL_foot"]
        sensor_names = ["RL_foot", "FR_foot", "RR_foot", "FL_foot", "base_link"]
        positions = np.zeros((2, 5, 3)); contact = np.zeros((2, 5, 3))
        for i, name in enumerate(FOOT_NAMES):
            positions[:, names.index(name)] = feet[:, i] + origin
            contact[:, sensor_names.index(name), 2] = force[:, i]
        class Scene(dict):
            pass
        scene = Scene(robot=SimpleNamespace(body_names=names,
            data=SimpleNamespace(root_pos_w=root + origin, body_pos_w=positions)),
            contact_forces=SimpleNamespace(body_names=sensor_names,
                data=SimpleNamespace(net_forces_w=contact)))
        scene.env_origins = origin
        env = SimpleNamespace(scene=scene, original_result=np.array([True, False]))
        term = self.cfg.terminations.success
        np.testing.assert_array_equal(term.func(env, **term.params), [True, False])

    def test_configuration_does_not_narrow_tiles_or_change_original_settings(self):
        for width in (.6, .9):
            with self.subTest(width=width):
                cfg = configuration(); before = copy.deepcopy(cfg.go2_settings)
                geometry = configure(cfg, width)
                self.assertEqual(cfg.go2_settings, before)
                self.assertEqual(cfg.scene.terrain.terrain_generator.size, (8., 3.))
                self.assertIs(cfg.terminations.success.func, narrow_stairs_success)
                self.assertIs(cfg.terminations.success.params["original_success"], original_success)
                self.assertEqual(cfg.terminations.success.params["original_params"], {"allowed": True})
                geometry["flight_width_m"] = 99
                self.assertEqual(cfg.terminations.success.params["geometry"]["flight_width_m"], width)

    def test_terrain_copy_without_dynamic_fields_retains_explicit_site_width(self):
        # Isaac Lab's cfg.copy() reconstructs from declared fields, dropping
        # dynamically attached flight_width_m. The terrain function must work
        # without that attribute and must never substitute another requested size.
        declared_fields = ("size", "step_height_range", "step_width", "num_steps",
                           "approach_length", "landing_length", "spawn_x",
                           "goal_offset", "boundary_margin", "function")
        module = ModuleType("trimesh")
        def box(extents, transform):
            boxes.append(tuple(extents))
            return boxes[-1]
        module.creation = SimpleNamespace(box=box)
        for width, expected_func in ((.6, ascending_narrow_stairs), (.9, ascending_narrow_stairs_09)):
            with self.subTest(width=width):
                cfg = configuration(); configure(cfg, width)
                stairs = cfg.scene.terrain.terrain_generator.sub_terrains["stairs"]
                copied = SimpleNamespace(**{name: copy.deepcopy(getattr(stairs, name))
                                            for name in declared_fields})
                self.assertFalse(hasattr(copied, "flight_width_m"))
                self.assertIs(copied.function, expected_func)
                boxes = []
                with patch.dict("sys.modules", {"trimesh": module}):
                    copied.function(.5, copied)
                self.assertEqual([extent[1] for extent in boxes], [3., width, width])
        for width in (.59, .61, .89, .91, 1.0, True):
            with self.subTest(width=width), self.assertRaisesRegex(ValueError, "only 0.60 m or 0.90 m"):
                configure(configuration(), width)

    def test_training_repeated_or_wrong_geometry_rejected_before_mutation(self):
        for kind in ("training", "tile", "steps", "height", "spawn", "width", "repeat"):
            with self.subTest(kind=kind):
                cfg = configuration(); gen = cfg.scene.terrain.terrain_generator
                stairs = gen.sub_terrains["stairs"]
                if kind == "training": cfg.go2_settings["terrain"]["curriculum"] = True
                if kind == "tile": gen.size = (8., .6)
                if kind == "steps": stairs.num_steps = 3
                if kind == "height": stairs.step_height_range = (.1, .15)
                if kind == "spawn": stairs.spawn_x = .9
                if kind == "repeat": configure(cfg)
                with self.assertRaises(ValueError):
                    configure(cfg, float("nan") if kind == "width" else .6)


if __name__ == "__main__":
    unittest.main()
