"""Bounded, receive-only ROS 2 lidar geometry diagnostic (no motor SDK).

The only application ROS interfaces are two sensor subscriptions. ROS discovery
and the framework's parameter events may still produce network traffic. No
motion commands, policy-switch messages, clients, or services are created.

Example: python -m go2_stairs.lidar_watch --duration 10 --output report.json
Use --allow-unsynced-preview only to inspect geometry when the sensor clock is
not aligned with the host: overall state remains unknown and source age unknown.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import contextlib
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


CLOUD_TOPIC = "/utlidar/cloud_deskewed"
ODOM_TOPIC = "/utlidar/robot_odom"
MAX_AGE = .3
MAX_SYNC_ERROR = .05
MAX_CLOUD_POINTS = 250_000
MAX_CLOUD_BYTES = 32_000_000
MAX_CAPTURE_POINTS = 50_000
POINT_FIELD_SIZES = {1: 1, 2: 1, 3: 2, 4: 2, 5: 4, 6: 4, 7: 4, 8: 8}
ASSUMPTIONS = [
    "odom z is gravity-aligned; this is an assumption pending calibration",
    "local x/y follow base yaw; local z is relative to the base origin",
    "cloud and odometry source stamps share a clock; host synchronization is checked separately",
    "geometry preview is not a controller readiness or traversability assessment",
]


def stamp_seconds(header):
    stamp = header.stamp
    sec, nsec = stamp.sec, stamp.nanosec
    if (isinstance(sec, bool) or not isinstance(sec, int) or sec < 0
            or isinstance(nsec, bool) or not isinstance(nsec, int)
            or not 0 <= nsec < 1_000_000_000 or sec + nsec == 0):
        raise ValueError("invalid_source_stamp")
    return sec + nsec * 1e-9


def _integer(value, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError("invalid_" + name)
    return value


def parse_cloud(message):
    """Decode XYZ FLOAT32 with explicit byte strides, including organized padding.

    Nonfinite point coordinates are retained here, counted and discarded by the
    watcher before geometry processing. Invalid binary layouts raise ValueError.
    """
    if message.header.frame_id != "odom":
        raise ValueError("cloud_frame_must_be_odom")
    stamp_seconds(message.header)
    width = _integer(message.width, "width", 1, MAX_CLOUD_POINTS)
    height = _integer(message.height, "height", 1, MAX_CLOUD_POINTS)
    if width * height > MAX_CLOUD_POINTS:
        raise ValueError("point_count_limit")
    point_step = _integer(message.point_step, "point_step", 12, 4096)
    row_step = _integer(message.row_step, "row_step", width * point_step, MAX_CLOUD_BYTES)
    if height * row_step > MAX_CLOUD_BYTES:
        raise ValueError("cloud_byte_limit")
    if not isinstance(message.is_bigendian, bool):
        raise ValueError("invalid_endianness")
    fields, occupied = {}, set()
    for field in message.fields:
        if not isinstance(field.name, str) or not field.name or field.name in fields:
            raise ValueError("invalid_or_duplicate_field_name")
        count = _integer(field.count, "field_count", 1, point_step)
        datatype = _integer(field.datatype, "field_datatype", 1, 8)
        offset = _integer(field.offset, "field_offset", 0, point_step - 1)
        end = offset + count * POINT_FIELD_SIZES[datatype]
        if end > point_step:
            raise ValueError("field_exceeds_point_step")
        # Aliased fields are unnecessary for this sensor and are rejected.
        span = set(range(offset, end))
        if occupied.intersection(span):
            raise ValueError("overlapping_fields")
        occupied.update(span)
        fields[field.name] = field
    if any(name not in fields or fields[name].datatype != 7 or fields[name].count != 1
           for name in ("x", "y", "z")):
        raise ValueError("xyz_must_be_unique_scalar_float32")
    try:
        data = memoryview(message.data)
    except TypeError as error:
        raise ValueError("cloud_data_not_buffer") from error
    if not data.contiguous or data.nbytes != height * row_step:
        raise ValueError("cloud_data_length_or_contiguity")
    endian = ">" if message.is_bigendian else "<"
    dtype = np.dtype({"names": ["x", "y", "z"], "formats": [endian + "f4"] * 3,
                      "offsets": [fields[n].offset for n in ("x", "y", "z")],
                      "itemsize": point_step})
    array = np.ndarray((height, width), dtype=dtype, buffer=data,
                       strides=(row_step, point_step))
    return np.stack([array[n].reshape(-1) for n in ("x", "y", "z")], axis=1).astype(np.float32)


def parse_odometry(message):
    if message.header.frame_id != "odom" or message.child_frame_id != "base_link":
        raise ValueError("odom_frames_must_be_odom_and_base_link")
    stamp = stamp_seconds(message.header)
    pose = message.pose.pose
    position = np.asarray([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64)
    q = np.asarray([pose.orientation.x, pose.orientation.y, pose.orientation.z,
                    pose.orientation.w], dtype=np.float64)
    if not np.isfinite(position).all() or not np.isfinite(q).all():
        raise ValueError("nonfinite_odometry")
    norm = math.hypot(*q)
    if not math.isfinite(norm) or abs(norm - 1.) > .01:
        raise ValueError("odometry_quaternion_not_near_unit")
    q /= norm
    x, y, z, w = q
    yaw = math.atan2(2. * (w*z + x*y), 1. - 2. * (y*y + z*z))
    return {"stamp_s": stamp, "position": position, "yaw_rad": yaw,
            "quaternion_original_norm": norm}


def yaw_local_points(points, position, yaw):
    """Undo yaw only: do not rotate gravity by the robot's roll or pitch."""
    points = np.asarray(points, dtype=np.float64)
    position = np.asarray(position, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 3 or position.shape != (3,)
            or not np.isfinite(points).all() or not np.isfinite(position).all()
            or not math.isfinite(yaw)):
        raise ValueError("invalid_transform_input")
    c, s = math.cos(yaw), math.sin(yaw)
    relative = points - position
    local = relative @ np.asarray([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    if not np.isfinite(local).all() or (np.abs(local) > np.finfo(np.float32).max).any():
        raise ValueError("nonfinite_or_out_of_range_local_points")
    return local


def receipt_fresh(received_mono, now_mono):
    return math.isfinite(now_mono) and 0. <= now_mono - received_mono <= MAX_AGE


def source_clock_check(stamp, now_source):
    """Do not estimate a clock correction or misreport an offset as source age."""
    offset = now_source - stamp
    compatible = math.isfinite(offset) and 0. <= offset <= MAX_AGE
    return {"source_clock_synchronized": compatible,
            "source_age_s": offset if compatible else None,
            "observed_clock_offset_s": offset if math.isfinite(offset) else None,
            "clock_note": "Timestamp proximity alone does not establish clock synchronization; "
                          "a mismatch may be clock offset or delayed data. No offset correction applied."}


def unknown(reason):
    return {"state": "unknown", "geometry": None, "reason": reason,
            "control_authorized": False}


class LidarSummary:
    """Bounded state; sensor callbacks only retain the latest cloud and short odom cache."""
    def __init__(self, analyzer, *, allow_unsynced_preview=False):
        self.analyzer = analyzer
        self.allow_unsynced_preview = bool(allow_unsynced_preview)
        self.odometry = deque(maxlen=256)
        self.cloud = None
        self.last_cloud_stamp = None
        self.last_odom_stamp = None
        self.processed_generation = 0
        self.generation = 0
        self.last_process_mono = None
        self.counts = Counter({key: 0 for key in (
            "odom_messages", "invalid_odom_messages", "cloud_messages", "new_cloud_frames",
            "duplicate_or_reversed_cloud_stamps", "processed_frames", "valid_input_frames",
            "unsynchronized_preview_frames")})
        self.states, self.reasons = Counter(), Counter()
        self.timings_ms = deque(maxlen=301)
        self.latest = unknown("no_cloud_received")
        self.geometry_preview = None
        self.latest_clock = source_clock_check(0., float("nan"))
        self.latest_sync_error = None
        self.latest_input = None
        self.result_context = None
        self.last_points = None
        self.capture_metadata = None

    def receive_odom(self, message, received_mono):
        self.counts["odom_messages"] += 1
        try:
            pose = parse_odometry(message)
            if self.last_odom_stamp is not None and pose["stamp_s"] <= self.last_odom_stamp:
                raise ValueError("odom_stamp_not_increasing")
        except (AttributeError, TypeError, ValueError, OverflowError) as error:
            self.counts["invalid_odom_messages"] += 1
            self.reasons[str(error)] += 1
            self.odometry.clear()
            self.latest = unknown("invalid_latest_odometry")
            self.geometry_preview = None
            return
        pose["received_mono"] = received_mono
        self.last_odom_stamp = pose["stamp_s"]
        self.odometry.append(pose)

    def receive_cloud(self, message, received_mono, received_source):
        self.counts["cloud_messages"] += 1
        try:
            if message.header.frame_id != "odom":
                raise ValueError("cloud_frame_must_be_odom")
            stamp = stamp_seconds(message.header)
            if self.last_cloud_stamp is not None and stamp <= self.last_cloud_stamp:
                self.counts["duplicate_or_reversed_cloud_stamps"] += 1
                raise ValueError("cloud_stamp_not_increasing")
        except (AttributeError, TypeError, ValueError, OverflowError) as error:
            self.reasons[str(error)] += 1
            self.latest = unknown(str(error))
            self.geometry_preview = None
            self.cloud = None
            return
        self.last_cloud_stamp = stamp
        self.generation += 1
        self.counts["new_cloud_frames"] += 1
        self.cloud = (message, received_mono, received_source, stamp, self.generation)

    def process(self, now_mono, now_source):
        if self.last_process_mono is not None and now_mono - self.last_process_mono < .2:
            return
        if self.cloud is None or self.cloud[4] == self.processed_generation:
            return
        self.last_process_mono = now_mono
        message, received, _, stamp, generation = self.cloud
        self.processed_generation = generation
        self.counts["processed_frames"] += 1
        started = time.perf_counter()
        self.geometry_preview = None
        self.latest_sync_error = None
        self.latest_clock = source_clock_check(stamp, now_source)
        self.result_context = {"stamp_s": stamp, "received_mono": received}
        try:
            if not receipt_fresh(received, now_mono):
                raise ValueError("cloud_receipt_stale")
            if not self.odometry:
                raise ValueError("no_valid_odometry")
            pose = min(self.odometry, key=lambda p: abs(p["stamp_s"] - stamp))
            self.latest_sync_error = abs(pose["stamp_s"] - stamp)
            if self.latest_sync_error > MAX_SYNC_ERROR:
                raise ValueError("odom_cloud_stamp_mismatch")
            if not receipt_fresh(pose["received_mono"], now_mono):
                raise ValueError("odom_receipt_stale")
            self.result_context["odom_received_mono"] = pose["received_mono"]
            self.result_context["odom_stamp_s"] = pose["stamp_s"]
            odom_clock = source_clock_check(pose["stamp_s"], now_source)
            clocks_ok = (self.latest_clock["source_clock_synchronized"]
                         and odom_clock["source_clock_synchronized"])
            if not clocks_ok and not self.allow_unsynced_preview:
                raise ValueError("source_clock_unsynchronized_or_data_delayed")
            points = parse_cloud(message)
            finite = np.isfinite(points).all(axis=1)
            self.latest_input = {"point_count": len(points), "nonfinite_points": int((~finite).sum()),
                                 "finite_points": int(finite.sum()), "source_stamp_s": stamp,
                                 "odom_stamp_s": pose["stamp_s"],
                                 "quaternion_original_norm": pose["quaternion_original_norm"]}
            local = yaw_local_points(points[finite], pose["position"], pose["yaw_rad"])
            if not len(local):
                raise ValueError("no_finite_points")
            result = self.analyzer(local)
            if not isinstance(result, dict) or not isinstance(result.get("state"), str):
                raise ValueError("invalid_geometry_result")
            result = dict(result, control_authorized=False)
            # Catch accidental NaNs or non-JSON output before recording a result.
            json.dumps(result, allow_nan=False)
            self.counts["valid_input_frames"] += 1
            self.states[result["state"]] += 1
            if clocks_ok:
                self.latest = result
            else:
                self.latest = unknown("source_clock_unsynchronized_or_data_delayed")
                self.geometry_preview = result
                self.counts["unsynchronized_preview_frames"] += 1
            stride = max(1, math.ceil(len(local) / MAX_CAPTURE_POINTS))
            self.last_points = local[::stride].astype(np.float32)
            self.capture_metadata = dict(self.latest_input, sampled_stride=stride,
                                         yaw_rad=pose["yaw_rad"], preview_only=not clocks_ok)
        except (AttributeError, TypeError, ValueError, OverflowError) as error:
            self.latest = unknown(str(error))
            self.reasons[str(error)] += 1
        finally:
            self.timings_ms.append((time.perf_counter() - started) * 1000.)

    def report(self, now_mono, now_source):
        latest = self.latest
        preview = self.geometry_preview
        age = now_mono - self.cloud[1] if self.cloud is not None else None
        context = self.result_context
        clock = source_clock_check(context["stamp_s"], now_source) if context is not None else self.latest_clock
        odom_clock = (source_clock_check(context["odom_stamp_s"], now_source)
                      if context is not None and "odom_stamp_s" in context else None)
        if age is None or not receipt_fresh(self.cloud[1], now_mono):
            latest, preview = unknown("no_fresh_cloud_receipt"), None
        elif context is None:
            latest, preview = unknown("no_processed_cloud"), None
        elif not receipt_fresh(context["received_mono"], now_mono):
            latest, preview = unknown("last_processed_cloud_receipt_stale"), None
        elif ("odom_received_mono" not in context
              or not receipt_fresh(context["odom_received_mono"], now_mono)):
            latest, preview = unknown("last_matched_odometry_receipt_stale"), None
        elif (not clock["source_clock_synchronized"]
              or odom_clock is None or not odom_clock["source_clock_synchronized"]):
            latest = unknown("source_clock_unsynchronized_or_data_delayed")
        return {
            "schema_version": 1, "read_only": True, "control_authorized": False,
            "control_or_motion_performed": False, "state": latest["state"],
            "latest": latest, "geometry_preview": preview,
            "cloud_topic": CLOUD_TOPIC, "odom_topic": ODOM_TOPIC,
            "coordinate_frame": "base_yaw_aligned_z_relative_to_base",
            "gravity_alignment_verified": False, "assumptions": ASSUMPTIONS,
            "temporal_confirmation_applied": False,
            "recognition_scope": "single-frame geometric candidates only; no multi-frame confirmation",
            "allow_unsynced_preview": self.allow_unsynced_preview,
            "transport_note": "Only sensor subscriptions; DDS discovery and ROS framework "
                              "parameter events may transmit. No application publishers or services.",
            "source_time": clock, "odometry_source_time": odom_clock,
            "cloud_receipt_age_s": age,
            "processed_cloud_receipt_age_s": now_mono - context["received_mono"] if context else None,
            "freshness_threshold_s": MAX_AGE,
            "odom_match_method": "nearest source timestamp; no interpolation",
            "odom_match_max_difference_s": MAX_SYNC_ERROR,
            "latest_odom_match_difference_s": self.latest_sync_error,
            "counters": dict(self.counts), "geometry_state_counts": dict(self.states),
            "diagnostic_reason_counts": dict(self.reasons), "latest_input": self.latest_input,
            "processing_rate_limit_hz": 5.,
            "processing_ms_p95": float(np.percentile(self.timings_ms, 95)) if self.timings_ms else None,
            "processing_ms_max": max(self.timings_ms) if self.timings_ms else None,
            "valid_input_definition": "parsed finite cloud plus matching fresh odometry; "
                                      "may be unsynchronized preview, not recognition accuracy",
        }


def collect(duration, *, allow_unsynced_preview=False):
    if not math.isfinite(duration) or not 0 < duration <= 60:
        raise ValueError("duration must be > 0 and <= 60 seconds")
    # Lazy ROS imports keep parser, transforms and offline tests independent of ROS.
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
    from sensor_msgs.msg import PointCloud2
    from nav_msgs.msg import Odometry
    from .perception import analyze_stairs

    summary = LidarSummary(analyze_stairs, allow_unsynced_preview=allow_unsynced_preview)
    rclpy.init(args=[])
    node = None
    started = time.monotonic()
    try:
        node = Node("go2_stairs_lidar_watch", enable_rosout=False,
                    start_parameter_services=False, use_global_arguments=False)
        source_now = lambda: node.get_clock().now().nanoseconds * 1e-9
        qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1,
                         reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         durability=QoSDurabilityPolicy.VOLATILE)
        node.create_subscription(Odometry, ODOM_TOPIC,
                                 lambda msg: summary.receive_odom(msg, time.monotonic()), qos)
        node.create_subscription(PointCloud2, CLOUD_TOPIC,
                                 lambda msg: summary.receive_cloud(msg, time.monotonic(), source_now()), qos)
        while rclpy.ok() and time.monotonic() - started < duration:
            rclpy.spin_once(node, timeout_sec=min(.02, max(0., duration - (time.monotonic() - started))))
            summary.process(time.monotonic(), source_now())
        report = summary.report(time.monotonic(), source_now())
        report["elapsed_s"] = time.monotonic() - started
        report["requested_duration_s"] = duration
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        return report, summary.last_points, summary.capture_metadata
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=10.)
    parser.add_argument("--output", type=Path, help="New JSON report path; otherwise stdout only")
    parser.add_argument("--capture", type=Path, help="New .npz path for last finite local frame (max 50000 points)")
    parser.add_argument("--allow-unsynced-preview", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration) or not 0 < args.duration <= 60:
        parser.error("--duration must be > 0 and <= 60 seconds")
    for path in (args.output, args.capture):
        if path is not None and (path.exists() or path.is_symlink() or not path.parent.is_dir()):
            parser.error("output paths must be new files in existing directories")
    if args.output is not None and args.capture is not None and args.output.absolute() == args.capture.absolute():
        parser.error("--capture and --output must be different files")
    try:
        with contextlib.redirect_stdout(sys.stderr):
            report, points, metadata = collect(args.duration, allow_unsynced_preview=args.allow_unsynced_preview)
    except (Exception, KeyboardInterrupt) as error:
        report = {"schema_version": 1, "read_only": True, "control_authorized": False,
                  "control_or_motion_performed": False, "state": "unknown",
                  "error_type": type(error).__name__, "reason": str(error)}
        points = metadata = None
    if args.capture is not None:
        if points is None:
            report["capture"] = {"written": False, "reason": "no_valid_local_frame"}
        else:
            with args.capture.open("xb") as target:
                np.savez_compressed(target, points_xyz=points,
                                    metadata_json=np.array(json.dumps(metadata, allow_nan=False)))
            report["capture"] = {"written": True, "path": str(args.capture),
                                 "point_count": len(points), "metadata": metadata,
                                 "note": "Last analyzed finite frame; may be older than final report. No control readiness implied."}
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as target:
            target.write(encoded)
    print(encoded, end="")
    return 1 if "error_type" in report else 0


if __name__ == "__main__":
    raise SystemExit(main())
