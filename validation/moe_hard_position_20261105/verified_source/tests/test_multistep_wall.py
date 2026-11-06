"""Geometry regression for the two- and six-riser training course."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import test_robogauge_two_steps_wall as wall


def scene_args(steps=2, **changes):
    values = dict(steps=steps, height=.15, tread=.30, width=.90, platform=.90,
                  physics_dt=.002, video_width=640, video_height=480)
    values.update(changes)
    return SimpleNamespace(**values)


def scene_geoms(args):
    with TemporaryDirectory() as directory:
        robot = Path(directory) / "robot.xml"
        robot.write_text('<mujoco><compiler/><worldbody/></mujoco>')
        root = ET.fromstring(wall.make_scene(robot, args))
    return {geom.get("name"): geom for geom in root.find("worldbody").findall("geom")}


def vector(geom, name):
    return np.fromstring(geom.get(name), sep=" ")


class MultiStepWallTest(unittest.TestCase):
    def test_two_risers_preserve_original_geometry_exactly(self):
        geoms = scene_geoms(scene_args())
        self.assertEqual(set(geoms), {"floor", "stair_1", "top_platform", "terminal_wall"})
        expected = {
            "stair_1": ([.15, .45, .075], [.15, 0., .075]),
            "top_platform": ([.45, .45, .15], [.75, 0., .15]),
            "terminal_wall": ([.05, 1.5, 1.], [1.25, 0., 1.]),
        }
        for name, (size, position) in expected.items():
            np.testing.assert_array_equal(vector(geoms[name], "size"), size)
            np.testing.assert_array_equal(vector(geoms[name], "pos"), position)
        self.assertEqual(wall.platform_geometry(scene_args()), (.3, 1.2, .3))

    def test_six_risers_have_five_lower_treads_and_one_upper_platform(self):
        geoms = scene_geoms(scene_args(6))
        self.assertEqual(set(geoms), {"floor", "top_platform", "terminal_wall"}
                         | {f"stair_{i}" for i in range(1, 6)})
        for level in range(1, 6):
            size = vector(geoms[f"stair_{level}"], "size")
            pos = vector(geoms[f"stair_{level}"], "pos")
            np.testing.assert_allclose(size, [.15, .45, level*.15/2], atol=1e-15, rtol=0)
            self.assertAlmostEqual(pos[0]-size[0], (level-1)*.3)
            self.assertAlmostEqual(pos[0]+size[0], level*.3)
            self.assertAlmostEqual(pos[2]+size[2], level*.15)
        platform_size, platform_pos = (vector(geoms["top_platform"], field) for field in ("size", "pos"))
        wall_size, wall_pos = (vector(geoms["terminal_wall"], field) for field in ("size", "pos"))
        self.assertAlmostEqual(platform_pos[0]-platform_size[0], 1.5)
        self.assertAlmostEqual(platform_pos[0]+platform_size[0], 2.4)
        self.assertAlmostEqual(platform_pos[2]+platform_size[2], .9)
        self.assertAlmostEqual(wall_pos[0]-wall_size[0], 2.4)
        np.testing.assert_array_equal(wall_size, [.05, 1.5, 1.])

    def test_dimensions_are_taken_from_args(self):
        args = scene_args(4, height=.10, tread=.25, width=1.1, platform=1.2)
        geoms = scene_geoms(args)
        start, end, top = wall.platform_geometry(args)
        self.assertEqual((start, end, top), (.75, 1.95, .4))
        np.testing.assert_array_equal(vector(geoms["top_platform"], "size"), [.6, .55, .2])
        np.testing.assert_array_equal(vector(geoms["terminal_wall"], "pos"), [2., 0., 1.])

    def test_wall_clearance_tracks_each_course_wall_position(self):
        exp = wall.Experiment.__new__(wall.Experiment)
        exp.np = np
        exp.mj = SimpleNamespace(mjtGeom=SimpleNamespace(mjGEOM_SPHERE=1))
        exp.model = SimpleNamespace(geom_type=np.array([1]), geom_size=np.array([[.1, 0., 0.]]))
        exp.data = SimpleNamespace(geom_xmat=np.eye(3).reshape(1, 9), geom_xpos=np.array([[.5, 0., .3]]))
        exp.robot_geoms = [0]
        exp.geom_description = lambda _: {"geom_id": 0}
        for steps, expected in ((2, .6), (6, 1.8)):
            exp.platform_start, exp.platform_end, exp.top_z = wall.platform_geometry(scene_args(steps))
            self.assertAlmostEqual(exp.wall_clearance()["signed_clearance_m"], expected)

    def test_invalid_courses_fail_before_building_scene(self):
        for args in (scene_args(0), scene_args(1), scene_args(-1), scene_args(2.5),
                     scene_args(6, height=0), scene_args(6, platform=float("nan"))):
            with self.assertRaises(ValueError):
                wall.platform_geometry(args)


if __name__ == "__main__":
    unittest.main()
