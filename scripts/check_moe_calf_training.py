#!/usr/bin/env python3
"""Read-only monitoring of the 500-update MoE1000 calf-speed continuation.

Acknowledgements update only local monitor state. No training, evaluation or
robot process is started or modified by inspection.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
CALVES = ("FL_calf", "FR_calf", "RL_calf", "RR_calf")
KINDS = ("velocity", "position", "target_position", "tracking", "requested_torque", "applied_torque")
EVENTS = ("evaluating", "completed", "failed", "stopped", "stalled", "unreachable")


def read_jsonl(path):
    if not path.exists():
        return []
    raw = path.read_text()
    lines = raw.splitlines()
    # A writer may currently be appending the final record.
    if raw and not raw.endswith("\n"):
        lines = lines[:-1]
    return [json.loads(line) for line in lines if line.strip()]


def aggregate_metric(rows):
    """A missing member is unknown; observed integer counts are never rounded."""
    required = ("excess_count", "sample_count", "peak_absolute_value", "max_ratio", "max_excess")
    complete = bool(rows) and all(
        isinstance(row, dict) and all(row.get(key) is not None for key in required)
        and type(row["excess_count"]) is int and type(row["sample_count"]) is int
        and 0 <= row["excess_count"] <= row["sample_count"]
        and row["sample_count"] > 0
        and all(math.isfinite(row[key]) for key in required)
        for row in rows)
    if complete:
        count = sum(row["excess_count"] for row in rows)
        samples = sum(row["sample_count"] for row in rows)
        result = {"excess_count": count, "sample_count": samples, "excess_fraction": count / samples,
                  "all_evidence_observed": True,
                  **{key: max(row[key] for row in rows) for key in required[2:]}}
    else:
        result = {key: None for key in (*required, "excess_fraction")} | {"all_evidence_observed": False}
    present = [row for row in rows if isinstance(row, dict)]
    result["observed_violation"] = any(
        row.get("observed_violation", False)
        or (row.get("excess_count") is not None and row["excess_count"] > 0)
        or (row.get("max_excess") is not None and row["max_excess"] > 0)
        or (row.get("max_ratio") is not None and row["max_ratio"] > 1.) for row in present)
    names = set().union(*(row.get("per_joint", {}).keys() for row in present))
    if names:
        result["per_joint"] = {
            name: aggregate_metric([row.get("per_joint", {}).get(name) if isinstance(row, dict) else None for row in rows])
            for name in sorted(names)}
    return result


def calf_window_summary(velocity, profiles, incomplete=False):
    """Select only four calf axes; never divide their violations by 12 axes."""
    normalized = {name.removesuffix("_joint"): row
                  for name, row in velocity.get("per_joint", {}).items()}
    rows = [normalized.get(name) for name in CALVES]
    result = aggregate_metric(rows)
    result["per_joint"] = {name: normalized.get(name) for name in CALVES}
    result["joint_names"] = list(CALVES)
    result["limit_rad_s"] = 15.7
    result["scope"] = "Four calf axes only; pre/post physics boundaries including failed terminal states"
    # SDK float32-promoted boundary, exactly as in the installed profile.
    expected_limit = 15.699999809265137
    profile_ok = bool(profiles)
    for profile in profiles:
        names = profile.get("joint_names", [])
        limits = profile.get("velocity_limits_rad_s", [])
        by_name = {name.removesuffix("_joint"): value for name, value in zip(names, limits)}
        profile_ok = profile_ok and len(names) == 12 and len(limits) == 12 and all(
            by_name.get(name) == expected_limit for name in CALVES)
    result["all_evidence_observed"] = result["all_evidence_observed"] and profile_ok and not incomplete
    violated = any(isinstance(row, dict) and (
        row.get("observed_violation", False)
        or (row.get("excess_count") is not None and row["excess_count"] > 0)
        or (row.get("max_ratio") is not None and row["max_ratio"] > 1.)
        or (row.get("peak_absolute_value") is not None and row["peak_absolute_value"] > expected_limit))
        for row in rows)
    result["calf_speed_acceptance_passed"] = (
        False if violated else True if result["all_evidence_observed"] else None)
    return result


def window_guard(episodes, node):
    subset = [episode for episode in episodes if episode["update"] <= node][-256:]
    guards = [episode.get("guard", {}) for episode in subset]
    metrics = {kind: aggregate_metric([guard.get("metrics", {}).get(kind) for guard in guards])
               for kind in KINDS}
    incomplete = any(guard.get("incomplete_due_to_numeric_failure", False) for guard in guards)
    calf = calf_window_summary(metrics["velocity"], [guard.get("profile", {}) for guard in guards], incomplete)
    return {"scope": "Latest up to 256 completed training episodes through this local update; "
                     "pre/post boundaries include failed terminal states; not all partial episodes of this update",
            "episodes": len(subset), "metrics": metrics, "calf_speed": calf,
            "calf_speed_acceptance_passed": calf["calf_speed_acceptance_passed"],
            "incomplete_episodes": sum(bool(g.get("incomplete_due_to_numeric_failure", False)) for g in guards)}


def inspect_run(root):
    root = Path(root)
    job = json.loads((root / "job.json").read_text())
    training_dir = Path(job["training_dir"])
    training_job = training_dir / "job.json"
    training = json.loads(training_job.read_text()) if training_job.exists() else {}
    records = read_jsonl(training_dir / "progress.jsonl")
    episodes = read_jsonl(training_dir / "episodes.jsonl")
    milestones = []
    record = next((row for row in records if row["update"] == 500), None)
    if record is not None:
        milestones.append({"node": 500, "cumulative_update": 1500, "record": record,
                           "guard_window": window_guard(episodes, 500),
                           "checkpoint_exists": (training_dir / "model_500.pt").is_file()})
    pid = job.get("pid")
    command_file = Path(f"/proc/{pid}/cmdline")
    try:
        alive = bool(pid and b"run_moe_calf_training.py" in command_file.read_bytes())
    except (FileNotFoundError, ProcessLookupError):
        alive = False
    status = job["status"]
    if status in ("training", "preparing", "evaluating") and not alive:
        status = "stopped"
    if status == "training" and time.time() - training.get("last_update_unix", job["started_unix"]) > 900:
        status = "stalled"
    summary = Path(job["eval_summary_path"])
    final = Path(job["final_checkpoint"])
    verified = False
    if status == "completed" and summary.is_file() and final.is_file():
        rows = json.loads(summary.read_text()).get("rows", [])
        verified = (job.get("completion_verified") is True and len(rows) == 2
                    and [row.get("label") for row in rows] == ["reference_moe1000", "calf1500"]
                    and all(row.get("episodes") == 129 and isinstance(row.get("calf_speed"), dict)
                            and "calf_speed_acceptance_passed" in row
                            and isinstance(row.get("guard"), dict)
                            and "joint_and_torque_acceptance_passed" in row for row in rows))
    return {"status": status, "pipeline_alive": alive, "pipeline": job, "training": training,
            "milestones": milestones, "completion_verified": verified,
            "final_checkpoint_exists": final.is_file(), "eval_summary_exists": summary.is_file(),
            "eval_summary_path": str(summary)}


def remote_code(run_path):
    functions = (read_jsonl, aggregate_metric, calf_window_summary, window_guard, inspect_run)
    return ("import json, math, time\nfrom pathlib import Path\n"
            + f"CALVES = {CALVES!r}\nKINDS = {KINDS!r}\n"
            + "\n\n".join(inspect.getsource(function) for function in functions)
            + f"\nprint(json.dumps(inspect_run({str(run_path)!r}), allow_nan=False))\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ack", type=int, choices=[500])
    parser.add_argument("--ack-event", choices=EVENTS)
    args = parser.parse_args()
    state_path = ROOT / ".monitor/moe_calf_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"ack": 0, "events": []}
    if args.ack is not None or args.ack_event is not None:
        if args.ack is not None:
            state["ack"] = max(state["ack"], args.ack)
        if args.ack_event is not None:
            state["events"] = sorted(set(state["events"]) | {args.ack_event})
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(state_path)
        print(json.dumps({"acknowledged": state}))
        return
    cfg = json.loads((ROOT / ".monitor/moe_calf_job.json").read_text())
    command = ["ssh", "-S", cfg["ssh_control"], "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
               "-o", "UserKnownHostsFile=" + cfg["known_hosts"], "-p", str(cfg["port"]),
               cfg["host"], "python3 -"]
    try:
        result = subprocess.run(command, input=remote_code(cfg["remote_run"]), text=True,
                                capture_output=True, timeout=60, check=True)
        data = json.loads(result.stdout)
        data["pending_milestones"] = [m for m in data.pop("milestones") if m["node"] > state["ack"]]
        event = data["status"] if data["status"] in EVENTS else None
    except (subprocess.SubprocessError, OSError, ValueError) as exc:
        data = {"status": "unreachable", "error": str(exc), "details": getattr(exc, "stderr", None),
                "pending_milestones": []}
        event = "unreachable"
    data["event_to_report"] = event if event and event not in state["events"] else None
    data["checked_unix"] = time.time()
    print(json.dumps(data, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
