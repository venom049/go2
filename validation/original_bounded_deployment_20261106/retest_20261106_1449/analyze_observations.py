"""Offline analysis of the receive-only recorder, without any robot connection."""
import bisect
import datetime as dt
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / "onboard_audit"
log = (AUDIT / "terminal.log").read_text()
tz = dt.timezone(dt.timedelta(hours=8))


def event_time(text):
    match = re.search(r"\[([\d-]+ [\d:.]+)\].*" + re.escape(text), log)
    return dt.datetime.strptime(match[1], "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=tz).timestamp()


start = event_time("FIRST_TEST state FixStand -> Stairs")
stop = event_time("FIRST_TEST_LATCHED")
limits = json.loads((ROOT / "prepared/model/limit_profile.json").read_text())["limits"]
states, commands = [], []
with (AUDIT / "test_observations.jsonl").open() as stream:
    for line in stream:
        row = json.loads(line)
        (states if row["event"] == "state" else commands).append(row)
states.sort(key=lambda row: row["wall_s"])
commands.sort(key=lambda row: row["wall_s"])
state_times = [row["wall_s"] for row in states]
active_states = [row for row in states if start <= row["wall_s"] <= stop]
policy_commands = [row for row in commands if start <= row["wall_s"] <= stop + 0.1
                   and all(abs(k - 20) < 1e-6 for k in row["kp"])
                   and all(abs(k - 0.5) < 1e-6 for k in row["kd"])]
assert active_states and policy_commands
guard = []
for line in log.splitlines():
    if "GUARD_JOINT joint=" in line:
        fields = dict(re.findall(r"(\w+)=([^ ]+)", line.split("GUARD_JOINT ")[1]))
        guard.append({k: v if k == "joint" else float(v) for k, v in fields.items()})

bound_violations = []
max_pair_age = 0.0
tracking = [0.0] * 12
for row in policy_commands:
    # A receive-time alignment, not the exact controller state at publication.
    index = bisect.bisect_right(state_times, row["wall_s"]) - 1
    state = states[index] if index >= 0 else None
    age = row["wall_s"] - state["wall_s"] if state else math.inf
    if age <= 0.01:
        max_pair_age = max(max_pair_age, age)
        for i in range(12):
            tracking[i] = max(tracking[i], abs(row["q"][i] - state["q"][i]))
    for i, limit in enumerate(limits):
        q = row["q"][i]
        if not math.isfinite(q) or not limit["position_lower_rad"] - 1e-6 <= q <= limit["position_upper_rad"] + 1e-6:
            bound_violations.append({"wall_s": row["wall_s"], "joint": limit["joint"], "q": q})

joint_stats = []
for i, limit in enumerate(limits):
    qs = [row["q"][i] for row in policy_commands]
    speeds = [abs(row["dq"][i]) for row in active_states]
    joint_stats.append({
        "joint": limit["joint"],
        "observed_policy_target_min_rad": min(qs),
        "observed_policy_target_max_rad": max(qs),
        "observed_policy_bound_hits": sum(abs(q - limit["position_lower_rad"]) <= 1e-6 or abs(q - limit["position_upper_rad"]) <= 1e-6 for q in qs),
        "active_interval_observed_max_abs_dq_rad_s": max(speeds),
        "receive_time_paired_max_abs_tracking_rad": tracking[i],
    })

def tilt(row):
    w, x, y, z = row["quat"]
    norm = math.sqrt(w*w+x*x+y*y+z*z)
    return math.acos(max(-1., min(1., 1 - 2 * (x*x+y*y) / norm**2)))

result = {
    "date": "2026-11-06", "timezone": "Asia/Shanghai",
    "policy": "Original RoboGauge / MoE-CTS + hard target joint clamp; no lidar",
    "policy_onnx_sha256": json.loads((ROOT / "pretest.json").read_text())["onnx_sha256"],
    "remote_run": "/home/unitree/go2-original-bounded-20261106/logs/20261106T145525-5642",
    "start": dt.datetime.fromtimestamp(start, tz).isoformat(),
    "latched": dt.datetime.fromtimestamp(stop, tz).isoformat(),
    "active_elapsed_log_s": round(stop-start, 3),
    "command_vx_m_s": 0.6, "software_budget_s": 2.0,
    "tracking_guard_rad": 2.0,
    "stop_reason": "measured_speed_outside_urdf_range",
    "stop_joint": "FL_calf", "stop_dq_rad_s": guard[5]["dq"],
    "stop_abs_speed_limit_rad_s": guard[5]["dq_limit_rad_s"],
    "guard_rejection_snapshot": guard,
    "guard_rejection_max_tracking_rad": max(abs(row["q"]-row["target_q"]) for row in guard),
    "accepted_stairs_publication_requests": int(re.search(r"accepted_stairs_publications=(\d+)", log)[1]),
    "observed_counts": {"all_state": len(states), "all_command": len(commands), "active_state": len(active_states), "policy_command": len(policy_commands)},
    "policy_first_command_time": dt.datetime.fromtimestamp(policy_commands[0]["wall_s"], tz).isoformat(),
    "policy_last_command_time": dt.datetime.fromtimestamp(policy_commands[-1]["wall_s"], tz).isoformat(),
    "policy_target_bound_violations": bound_violations,
    "pairing_max_age_s": max_pair_age,
    "joint_stats": joint_stats,
    "active_max_observed_body_tilt_rad": max(map(tilt, active_states)),
    "exit": {"site_support_confirmed": True, "controller_code": 0, "terminal_script_code": 0,
             "damping_request_count": 121, "damping_request_period_ms": 250,
             "hardware_delivery_acknowledged": False, "automatic_factory_restore": False},
    "stair_completion": "not verified; no stair-relative pose or video recorded",
    "post_exit_checks": json.loads((ROOT / "post_exit_checks.json").read_text()),
    "scope": ["ROS subscriptions may drop data; observed commands are not motor acknowledgments",
              "Command/state pairing uses observer receive times, not synchronized controller snapshots",
              "Hard position clipping does not bound measured joint velocity",
              "Guard source is authoritative for the rejected snapshot; limit_profile.json retained_application_guards tracking_error_rad=0.9 is stale documentation",
              "No radar/LIO process was started; the separate lidar reliability task remains paused"],
}
(ROOT / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k: result[k] for k in ["active_elapsed_log_s", "observed_counts", "policy_first_command_time", "policy_last_command_time", "guard_rejection_max_tracking_rad", "active_max_observed_body_tilt_rad"]}, ensure_ascii=False, indent=2))
print("Target bound violations:", len(bound_violations))
for row in joint_stats:
    print(row)
