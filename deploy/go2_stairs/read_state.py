"""Bounded Go2 telemetry subscription; never requests or writes motor control.

Only the SDK's receive channel is created, on rt/lowstate. DDS discovery traffic
is necessary for subscribing, but no command topic or motion-service RPC is
used. Dependencies are imported only by open_subscription(), so the conversion
and tests also work on machines without the SDK. Reports deliberately omit the
robot serial number, raw remote bytes, and arbitrary message attributes.

Example (after explicitly verifying the wired interface):
  python3 deploy/go2_stairs/read_state.py --interface enx... --duration 5

Duration is the sampling window; use an outer process timeout when supervising
SDK initialization/teardown as well. Receiving plausible telemetry is not a
mechanical safety assessment, controller handoff, or permission to move.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import socket
import sys
import threading
import time


TOPIC = "rt/lowstate"
SDK_JOINT_NAMES = tuple(f"{leg}_{joint}_joint"
                        for leg in ("FR", "FL", "RR", "RL")
                        for joint in ("hip", "thigh", "calf"))
ASSESSMENT_SCOPE = (
    "Telemetry format and local receipt freshness only. This does not verify "
    "mechanical safety, sensor calibration, motor-control access, or policy readiness."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_sample(message) -> dict:
    """Copy an allowlist from unitree_go LowState_; unknown values stay null.

    The SDK IDL contains motor mode/lost, bit_flag and BMS status but no general
    motor error-code field. Those values are reported raw, without interpreting
    them as fault-free. No missing field is substituted with a zero.
    """
    missing, nonfinite, invalid = [], [], []

    def number(obj, name, path, integer=False):
        value = getattr(obj, name, None)
        if value is None:
            missing.append(path)
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            invalid.append(path)
            return None
        if not math.isfinite(value):
            nonfinite.append(path)
            return None
        if integer and not isinstance(value, int):
            invalid.append(path)
            return None
        return int(value) if integer else float(value)

    def vector(obj, name, count, path):
        values = getattr(obj, name, None)
        if values is None:
            missing.append(path)
            return None
        if not isinstance(values, (list, tuple)) or len(values) != count:
            invalid.append(path)
            return None
        result = []
        for i, value in enumerate(values):
            key = f"{path}[{i}]"
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                invalid.append(key)
                result.append(None)
            elif not math.isfinite(value):
                nonfinite.append(key)
                result.append(None)
            else:
                result.append(float(value))
        return result

    imu = getattr(message, "imu_state", None)
    imu_summary = {
        "quaternion_wxyz": vector(imu, "quaternion", 4, "imu.quaternion"),
        "gyroscope_rad_s": vector(imu, "gyroscope", 3, "imu.gyroscope"),
        "accelerometer_m_s2": vector(imu, "accelerometer", 3, "imu.accelerometer"),
        "rpy_rad": vector(imu, "rpy", 3, "imu.rpy"),
        "temperature_raw": number(imu, "temperature", "imu.temperature", True),
    }
    quaternion = imu_summary["quaternion_wxyz"]
    quaternion_norm = (math.hypot(*quaternion)
                       if quaternion is not None and all(v is not None for v in quaternion)
                       else None)
    if quaternion_norm is not None and not math.isfinite(quaternion_norm):
        nonfinite.append("imu.quaternion_norm")
        quaternion_norm = None
    imu_summary["quaternion_norm"] = quaternion_norm
    imu_summary["quaternion_unit_within_0p01"] = (
        abs(quaternion_norm - 1.) <= .01 if quaternion_norm is not None else None)

    motors = getattr(message, "motor_state", None)
    if not isinstance(motors, (list, tuple)):
        missing.append("motor_state") if motors is None else invalid.append("motor_state")
        motors = ()
    motor_summary = []
    for i, joint in enumerate(SDK_JOINT_NAMES):
        motor = motors[i] if i < len(motors) else None
        prefix = f"motor_state[{i}]"
        motor_summary.append({
            "sdk_index": i, "joint_name": joint,
            "q_rad": number(motor, "q", f"{prefix}.q"),
            "dq_rad_s": number(motor, "dq", f"{prefix}.dq"),
            "tau_est_Nm": number(motor, "tau_est", f"{prefix}.tau_est"),
            "temperature_raw": number(motor, "temperature", f"{prefix}.temperature", True),
            "mode_raw": number(motor, "mode", f"{prefix}.mode", True),
            "lost_raw": number(motor, "lost", f"{prefix}.lost", True),
        })
    bms = getattr(message, "bms_state", None)
    gyro = imu_summary["gyroscope_rad_s"]
    core_fields_finite = (
        quaternion is not None and all(v is not None for v in quaternion)
        and gyro is not None and all(v is not None for v in gyro)
        and all(m["q_rad"] is not None and m["dq_rad_s"] is not None for m in motor_summary)
    )
    return {
        "tick_raw": number(message, "tick", "tick", True),
        "imu": imu_summary, "motors": motor_summary,
        "power_v": number(message, "power_v", "power_v"),
        "power_a": number(message, "power_a", "power_a"),
        "status_raw": {
            "level_flag": number(message, "level_flag", "level_flag", True),
            "bit_flag": number(message, "bit_flag", "bit_flag", True),
            "bms_status": number(bms, "status", "bms_state.status", True),
            "bms_soc": number(bms, "soc", "bms_state.soc", True),
            "interpretation": "Raw status fields; no fault-clear interpretation is made.",
        },
        "checks": {
            "policy_input_fields_finite": bool(core_fields_finite),
            "quaternion_unit_within_0p01": imu_summary["quaternion_unit_within_0p01"],
            "missing_fields": missing, "nonfinite_fields": nonfinite, "invalid_fields": invalid,
        },
    }


class StateSummary:
    """Keeps one sanitized sample and counters, never an unbounded capture."""
    def __init__(self, clock=time.monotonic):
        self.clock, self.lock = clock, threading.Lock()
        self.started = clock()
        self.started_utc = utc_now()
        self.first_received = self.last_received = None
        self.last_received_utc = None
        self.latest = None
        self.count = self.changed_ticks = self.bad_core_samples = self.callback_errors = 0
        self.last_tick = None

    def receive(self, message):
        now = self.clock()
        try:
            sample = sanitize_sample(message)
        except (TypeError, ValueError, OverflowError, AttributeError):
            with self.lock:
                self.callback_errors += 1
            return
        with self.lock:
            tick = sample["tick_raw"]
            if self.last_tick is not None and tick is not None and tick != self.last_tick:
                self.changed_ticks += 1
            self.last_tick = tick
            self.count += 1
            if not sample["checks"]["policy_input_fields_finite"] or not sample["checks"]["quaternion_unit_within_0p01"]:
                self.bad_core_samples += 1
            if self.first_received is None:
                self.first_received = now
            self.last_received, self.last_received_utc = now, utc_now()
            self.latest = sample

    def report(self, *, interface: str, max_age: float) -> dict:
        now = self.clock()
        with self.lock:
            age = max(0., now - self.last_received) if self.last_received is not None else None
            span = self.last_received - self.first_received if self.count > 1 else 0.
            latest = copy.deepcopy(self.latest)
            return {
                "schema_version": 1, "read_only": True, "topic": TOPIC,
                "interface": interface, "started_utc": self.started_utc,
                "finished_utc": utc_now(), "elapsed_s": max(0., now-self.started),
                "assessment_scope": ASSESSMENT_SCOPE,
                "control_or_motion_performed": False,
                "sample_count": self.count, "callback_error_count": self.callback_errors,
                "core_input_problem_sample_count": self.bad_core_samples,
                "last_received_utc": self.last_received_utc,
                "last_receipt_age_s": age, "fresh_receipt": age <= max_age if age is not None else None,
                "fresh_receipt_threshold_s": max_age,
                "observed_receive_rate_hz": (self.count-1)/span if span > 0 else None,
                "tick_change_count": self.changed_ticks,
                "tick_progress_observed": self.changed_ticks > 0,
                "source_sample_age_s": None,
                "source_age_note": "Local arrival freshness cannot establish source capture age.",
                "latest": latest,
            }


def open_subscription(interface: str, callback):
    """Only construction site for the SDK subscriber; no control APIs exist here."""
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,15}", interface):
        raise ValueError("Invalid explicit network interface name")
    socket.if_nametoindex(interface)
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

    ChannelFactoryInitialize(0, interface)
    subscriber = ChannelSubscriber(TOPIC, LowState_)
    subscriber.Init(callback, 1)
    return subscriber


def collect(interface: str, duration: float, max_age: float) -> dict:
    if not math.isfinite(duration) or not 0. < duration <= 60.:
        raise ValueError("Sampling duration must be > 0 and <= 60 seconds")
    if not math.isfinite(max_age) or not 0. < max_age <= 5.:
        raise ValueError("Freshness threshold must be > 0 and <= 5 seconds")
    summary = StateSummary()
    subscriber = open_subscription(interface, summary.receive)
    try:
        end = time.monotonic() + duration
        while time.monotonic() < end:
            time.sleep(min(.05, max(0., end-time.monotonic())))
        report = summary.report(interface=interface, max_age=max_age)
    finally:
        subscriber.Close()
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", required=True, help="Verified wired interface; never auto-selected")
    parser.add_argument("--duration", type=float, default=5., help="Sampling seconds, max 60")
    parser.add_argument("--max-age", type=float, default=.2, help="Local receipt freshness threshold seconds")
    parser.add_argument("--output", type=Path, help="Optional new local JSON file (will not overwrite)")
    args = parser.parse_args(argv)
    if args.output is not None and (args.output.exists() or not args.output.parent.is_dir()):
        parser.error("--output must be a new file in an existing directory")
    try:
        # SDK diagnostic prints belong on stderr, leaving stdout valid JSON.
        with contextlib.redirect_stdout(sys.stderr):
            report = collect(args.interface, args.duration, args.max_age)
    except (Exception, KeyboardInterrupt) as error:
        report = {
            "schema_version": 1, "read_only": True, "topic": TOPIC,
            "interface": args.interface, "finished_utc": utc_now(),
            "control_or_motion_performed": False, "assessment_scope": ASSESSMENT_SCOPE,
            "status": "probe_error", "error_type": type(error).__name__,
            "detail": ("Install Unitree SDK2 Python/CycloneDDS in the deployment environment."
                       if isinstance(error, ImportError) else
                       "No telemetry result; verify the interface, dependencies, and DDS transport."),
        }
    else:
        report["status"] = "received" if report["sample_count"] else "no_data"
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as output:
            output.write(encoded)
    print(encoded, end="")
    return 0 if report["status"] == "received" else 1


if __name__ == "__main__":
    raise SystemExit(main())
