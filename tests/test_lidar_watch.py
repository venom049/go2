"""Offline binary-layout, transform and time-gating tests; no ROS graph or robot."""
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "deploy.go2_stairs.lidar_watch", Path(__file__).resolve().parents[1] / "deploy/go2_stairs/lidar_watch.py")
watch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watch)


def header(stamp=100., frame="odom"):
    sec = int(stamp)
    return NS(frame_id=frame, stamp=NS(sec=sec, nanosec=int(round((stamp-sec)*1e9))))


def cloud(points=None, *, stamp=100., big=False, width=2, row_padding=0):
    if points is None:
        points = [(1., 2., 3.), (4., 5., 6.)]
    height = len(points) // width
    point_step, row_step = 32, width * 32 + row_padding
    data = bytearray(height * row_step)
    for i, p in enumerate(points):
        struct.pack_into((">" if big else "<") + "fff", data,
                         (i // width) * row_step + (i % width) * point_step, *p)
    return NS(header=header(stamp), width=width, height=height, point_step=point_step,
              row_step=row_step, is_bigendian=big, data=data,
              fields=[NS(name=n, offset=i*4, datatype=7, count=1) for i, n in enumerate("xyz")])


def odom(stamp=100., position=(0., 0., 0.), quaternion=(0., 0., 0., 1.)):
    return NS(header=header(stamp), child_frame_id="base_link",
              pose=NS(pose=NS(position=NS(**dict(zip("xyz", position))),
                              orientation=NS(**dict(zip("xyzw", quaternion))))))


class BinaryCloudTests(unittest.TestCase):
    def test_both_endiannesses_with_row_padding(self):
        expected = np.array([[1., 2., 3.], [4., -5., 6.], [7., 8., 9.], [-1., -2., -3.]])
        for big in (False, True):
            with self.subTest(big=big):
                msg = cloud(expected, big=big, row_padding=17)
                np.testing.assert_array_equal(watch.parse_cloud(msg), expected)

    def test_field_offsets_not_assumed(self):
        msg = cloud()
        msg.fields = [NS(name=n, offset=o, datatype=7, count=1)
                      for n, o in zip("xyz", (20, 4, 12))]
        for i, vals in enumerate(((1., 2., 3.), (4., 5., 6.))):
            for offset, value in zip((20, 4, 12), vals):
                struct.pack_into("<f", msg.data, i*32+offset, value)
        np.testing.assert_array_equal(watch.parse_cloud(msg), [[1, 2, 3], [4, 5, 6]])

    def test_malformed_layouts_rejected(self):
        def field_wrong_type(m): m.fields[0].datatype = 8
        def field_count(m): m.fields[0].count = 2
        def overlap(m): m.fields[1].offset = 0
        def duplicate(m): m.fields[1].name = "x"
        def field_overflow(m): m.fields[2].offset = 31
        def bad_length(m): m.data.pop()
        def bad_stride(m): m.row_step = 63
        def wrong_frame(m): m.header.frame_id = "base_link"
        def enormous(m): m.width = 9999999
        for mutate in (field_wrong_type, field_count, overlap, duplicate, field_overflow,
                       bad_length, bad_stride, wrong_frame, enormous):
            with self.subTest(case=mutate.__name__):
                msg = cloud()
                mutate(msg)
                with self.assertRaises(ValueError):
                    watch.parse_cloud(msg)

    def test_nonfinite_points_preserved_for_counting(self):
        actual = watch.parse_cloud(cloud([(float("nan"), 0, 0), (0, float("inf"), 0)]))
        self.assertTrue(np.isnan(actual[0, 0]))
        self.assertTrue(np.isinf(actual[1, 1]))


class TransformTests(unittest.TestCase):
    def test_yaw_only_preserves_world_vertical_difference(self):
        # A base with yaw=90 degrees, pitch=30 degrees, roll=20 degrees.
        roll, pitch, yaw = map(math.radians, (20, 30, 90))
        cr, sr = math.cos(roll/2), math.sin(roll/2)
        cp, sp = math.cos(pitch/2), math.sin(pitch/2)
        cy, sy = math.cos(yaw/2), math.sin(yaw/2)
        q = (sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy,
             cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy)
        pose = watch.parse_odometry(odom(position=(10, 20, 2), quaternion=q))
        points = watch.yaw_local_points([[10, 21, 2.18], [9, 20, 1.7]],
                                       pose["position"], pose["yaw_rad"])
        np.testing.assert_allclose(points, [[1, 0, .18], [0, 1, -.3]], atol=1e-12)

    def test_small_quaternion_error_normalized_large_or_invalid_rejected(self):
        pose = watch.parse_odometry(odom(quaternion=(0, 0, 0, 1.005)))
        self.assertEqual(pose["yaw_rad"], 0)
        for q in ((0, 0, 0, 0), (0, 0, 0, 1.1), (0, 0, 0, float("nan")), (1e308,)*4):
            with self.subTest(q=q), self.assertRaises(ValueError):
                watch.parse_odometry(odom(quaternion=q))
        for attr in ("header", "child_frame_id"):
            msg = odom()
            if attr == "header": msg.header.frame_id = "map"
            else: msg.child_frame_id = "camera"
            with self.assertRaises(ValueError): watch.parse_odometry(msg)

    def test_stamp_validation(self):
        for s, ns in ((0, 0), (-1, 0), (1, -1), (1, 1000000000), (1.5, 1)):
            with self.assertRaises(ValueError):
                watch.stamp_seconds(NS(stamp=NS(sec=s, nanosec=ns)))


class WatcherTests(unittest.TestCase):
    def make_summary(self, preview=False):
        self.calls = []
        def analyzer(points):
            self.calls.append(points.copy())
            return {"state": "stairs_up", "geometry": {"step_height_m": .18},
                    "control_authorized": True}  # Watcher must enforce false itself.
        return watch.LidarSummary(analyzer, allow_unsynced_preview=preview)

    def feed(self, summary, stamp=100., mono=10., source=None, odom_stamp=None):
        summary.receive_odom(odom(stamp if odom_stamp is None else odom_stamp), mono)
        summary.receive_cloud(cloud(stamp=stamp), mono, stamp if source is None else source)

    def test_nearest_odom_and_nonfinite_filter(self):
        summary = self.make_summary()
        summary.receive_odom(odom(99.98, position=(1, 2, 3)), 10.)
        summary.receive_odom(odom(100.001, position=(4, 5, 6)), 10.)
        summary.receive_cloud(cloud([(4, 6, 6.18), (float("nan"), 0, 0)]), 10., 100.01)
        summary.process(10.01, 100.02)
        result = summary.report(10.02, 100.03)
        self.assertEqual(result["state"], "stairs_up")
        self.assertFalse(result["latest"]["control_authorized"])
        self.assertEqual(result["latest_input"]["nonfinite_points"], 1)
        self.assertAlmostEqual(result["latest_odom_match_difference_s"], .001)
        np.testing.assert_allclose(self.calls[0], [[0, 1, .18]], atol=1e-6)

    def test_unsynchronized_clock_default_rejects_but_explicit_preview_does_not_authorize(self):
        for preview in (False, True):
            with self.subTest(preview=preview):
                summary = self.make_summary(preview)
                self.feed(summary)
                summary.process(10.01, 599.1)
                report = summary.report(10.02, 599.11)
                self.assertEqual(report["state"], "unknown")
                self.assertFalse(report["control_authorized"])
                self.assertFalse(report["source_time"]["source_clock_synchronized"])
                self.assertIsNone(report["source_time"]["source_age_s"])
                self.assertAlmostEqual(report["source_time"]["observed_clock_offset_s"], 499.11)
                self.assertEqual(len(self.calls), int(preview))
                self.assertEqual(report["geometry_preview"] is not None, preview)
                json.dumps(report, allow_nan=False)

    def test_stale_receipts_and_mismatched_odometry_do_not_reach_geometry(self):
        for now, odom_stamp in ((10.31, 100.), (10.01, 99.9)):
            summary = self.make_summary(True)
            self.feed(summary, odom_stamp=odom_stamp)
            summary.process(now, 599.)
            self.assertFalse(self.calls)
            self.assertEqual(summary.latest["state"], "unknown")

    def test_repeated_stamp_does_not_refresh_data_or_reanalyze(self):
        summary = self.make_summary(True)
        self.feed(summary)
        summary.process(10., 599.)
        summary.receive_cloud(cloud(), 10.2, 599.2)
        summary.process(10.21, 599.21)
        report = summary.report(10.21, 599.21)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(report["state"], "unknown")
        self.assertIsNone(report["geometry_preview"])
        self.assertEqual(report["counters"]["new_cloud_frames"], 1)

    def test_rate_limit_latest_frame_and_expiration(self):
        summary = self.make_summary(True)
        self.feed(summary)
        summary.process(10., 599.)
        self.feed(summary, 100.05, 10.05)
        summary.process(10.05, 599.05)
        self.assertEqual(len(self.calls), 1)
        self.assertIsNotNone(summary.report(10.1, 599.1)["geometry_preview"])
        self.feed(summary, 100.15, 10.15)
        summary.process(10.21, 599.21)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(summary.latest_input["source_stamp_s"], 100.15)
        self.assertIsNone(summary.report(10.51, 599.51)["geometry_preview"])

    def test_invalid_odometry_revokes_preview_and_cache(self):
        summary = self.make_summary(True)
        self.feed(summary)
        summary.process(10., 599.)
        summary.receive_odom(odom(100.01, quaternion=(0, 0, 0, 2)), 10.01)
        self.assertFalse(summary.odometry)
        self.assertIsNone(summary.geometry_preview)
        self.assertEqual(summary.latest["state"], "unknown")

    def test_cli_refuses_overwrite_before_ros_import_or_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing.json"
            path.write_text("keep")
            with patch.object(watch, "collect") as collect, self.assertRaises(SystemExit):
                watch.main(["--output", str(path)])
            collect.assert_not_called()
            self.assertEqual(path.read_text(), "keep")

    def test_cli_writes_bounded_npz_and_no_nan_json(self):
        report = {"state": "unknown", "control_authorized": False}
        points = np.zeros((2, 3), np.float32)
        with tempfile.TemporaryDirectory() as directory:
            output, capture = Path(directory)/"report.json", Path(directory)/"capture.npz"
            with patch.object(watch, "collect", return_value=(report, points, {"preview_only": True})):
                watch.main(["--output", str(output), "--capture", str(capture)])
            with np.load(capture, allow_pickle=False) as data:
                np.testing.assert_array_equal(data["points_xyz"], points)
                self.assertTrue(json.loads(str(data["metadata_json"]))["preview_only"])
            self.assertTrue(json.loads(output.read_text())["capture"]["written"])

    def test_collection_only_creates_two_best_effort_depth_one_subscriptions_and_tears_down(self):
        simulated = [0.]
        nodes = []
        class FakeNode:
            def __init__(self, name, **kwargs):
                self.kwargs, self.subs, self.destroyed = kwargs, [], False
                nodes.append(self)
            def get_clock(self):
                return NS(now=lambda: NS(nanoseconds=int((100. + simulated[0])*1e9)))
            def create_subscription(self, kind, topic, callback, qos):
                self.subs.append((topic, callback, qos))
            def destroy_node(self): self.destroyed = True
        def spin(node, timeout_sec):
            simulated[0] += max(timeout_sec, .001)
            if simulated[0] < .021:
                node.subs[0][1](odom())
                node.subs[1][1](cloud())
        fake_rclpy = NS(init=lambda args: None, shutdown=lambda: None,
                        ok=lambda: True, spin_once=spin)
        modules = {
            "rclpy": fake_rclpy, "rclpy.node": NS(Node=FakeNode),
            "rclpy.qos": NS(QoSProfile=lambda **kw: kw,
                QoSHistoryPolicy=NS(KEEP_LAST="keep_last"),
                QoSReliabilityPolicy=NS(BEST_EFFORT="best_effort"),
                QoSDurabilityPolicy=NS(VOLATILE="volatile")),
            "sensor_msgs.msg": NS(PointCloud2=object), "nav_msgs.msg": NS(Odometry=object),
            "deploy.go2_stairs.perception": NS(analyze_stairs=lambda pts: {"state": "flat"}),
        }
        with patch.dict(sys.modules, modules), patch.object(watch, "__package__", "deploy.go2_stairs"), \
                patch.object(watch.time, "monotonic", side_effect=lambda: simulated[0]):
            report, points, metadata = watch.collect(.25)
        node = nodes[0]
        self.assertFalse(node.kwargs["enable_rosout"])
        self.assertFalse(node.kwargs["start_parameter_services"])
        self.assertFalse(node.kwargs["use_global_arguments"])
        self.assertTrue(node.destroyed)
        self.assertEqual([s[0] for s in node.subs], [watch.ODOM_TOPIC, watch.CLOUD_TOPIC])
        for _, _, qos in node.subs:
            self.assertEqual(qos["depth"], 1)
            self.assertEqual(qos["reliability"], "best_effort")
        self.assertEqual(report["state"], "flat")
        self.assertFalse(report["temporal_confirmation_applied"])
        self.assertFalse(report["control_authorized"])


if __name__ == "__main__":
    unittest.main()
