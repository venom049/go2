#!/usr/bin/env python3
"""Read-only SSH monitoring of the additional 1000-update MoE joint recovery.

Acknowledgements only update local monitor state. Inspection never changes a
training, evaluation or physical-robot process.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
KINDS = ("velocity", "target_velocity", "position", "target_position", "tracking",
         "requested_torque", "applied_torque")
LABELS = ("reference_moe1000", "joint1500", "joint2000")
PROFILE_SHA256 = "bd3ea1830ca2ff7d8eb6eda1fb0b5077e33192a1ee663f2c16f96195d948c8af"
JOINT_NAMES = tuple(f"{leg}_{kind}" for leg in ("FL", "FR", "RL", "RR") for kind in ("hip", "thigh", "calf"))
EVENTS = ("evaluating", "completed", "failed", "stopped", "stalled", "unreachable")


def read_jsonl(path):
    if not path.exists():
        return []
    raw = path.read_text()
    lines = raw.splitlines()
    if raw and not raw.endswith("\n"):
        lines = lines[:-1]
    return [json.loads(line) for line in lines if line.strip()]


def aggregate_metric(rows):
    required = ("excess_count", "sample_count", "peak_absolute_value", "max_ratio", "max_excess")
    complete = bool(rows) and all(
        isinstance(row, dict) and all(row.get(key) is not None for key in required)
        and type(row["excess_count"]) is int and type(row["sample_count"]) is int
        and 0 <= row["excess_count"] <= row["sample_count"] and row["sample_count"] > 0
        and all(type(row[key]) in (int, float) and math.isfinite(row[key]) for key in required)
        and row.get("unobserved_state_samples", 0) == 0 for row in rows)
    if complete:
        count, samples = sum(row["excess_count"] for row in rows), sum(row["sample_count"] for row in rows)
        result = {"excess_count": count, "sample_count": samples, "excess_fraction": count / samples,
                  **{key: max(row[key] for row in rows) for key in required[2:]}}
    else:
        result = {key: None for key in (*required, "excess_fraction")}
    result["all_evidence_observed"] = complete
    present = [row for row in rows if isinstance(row, dict)]
    result["observed_violation"] = any(row.get("observed_violation", False) or any(
        type(row.get(key)) in (int, float) and row[key] > boundary
        for key, boundary in (("excess_count", 0), ("max_excess", 0), ("max_ratio", 1))) for row in present)
    for key, reduction in (("min_value", min), ("max_value", max)):
        result[key] = (reduction(row[key] for row in rows) if complete and all(
            type(row.get(key)) in (int, float) and math.isfinite(row[key]) for row in rows) else None)
    if any("per_joint" in row for row in present):
        result["per_joint"] = {name: aggregate_metric([
            row.get("per_joint", {}).get(name) if isinstance(row, dict) else None for row in rows])
            for name in JOINT_NAMES}
        result["all_evidence_observed"] = complete and all(
            row["all_evidence_observed"] for row in result["per_joint"].values())
        result["observed_violation"] |= any(row["observed_violation"] for row in result["per_joint"].values())
    return result


def window_guard(episodes, node):
    subset = [episode for episode in episodes if episode["update"] <= node][-256:]
    guards = [episode.get("guard", {}) for episode in subset]
    metrics = {kind: aggregate_metric([guard.get("metrics", {}).get(kind) for guard in guards]) for kind in KINDS}
    incomplete = any(guard.get("incomplete_due_to_numeric_failure", False) for guard in guards)
    complete = bool(guards) and not incomplete and all(
        guard.get("all_metrics_observed") is True
        and guard.get("profile", {}).get("source_sha256") == PROFILE_SHA256
        and guard.get("profile", {}).get("joint_names") == list(JOINT_NAMES)
        and all(isinstance(guard.get("metrics", {}).get(kind, {}).get("per_joint"), dict)
                and set(JOINT_NAMES) <= set(guard["metrics"][kind]["per_joint"])
                for kind in KINDS) for guard in guards)
    complete = complete and all(metric["all_evidence_observed"] for metric in metrics.values())
    violated = any(metric["observed_violation"] for metric in metrics.values())
    return {"scope": "Latest up to 256 completed training episodes through this local update; pre/post physics boundaries include failed terminal states; not all partial episodes of this update",
            "episodes": len(subset), "metrics": metrics, "all_metrics_observed": complete,
            "joint_and_torque_acceptance_passed": False if violated else True if complete else None,
            "incomplete_episodes": sum(bool(g.get("incomplete_due_to_numeric_failure", False)) for g in guards)}


def comparison_complete(path, mode):
    if not path.is_file():
        return False
    summary = json.loads(path.read_text())
    rows = summary.get("rows", [])
    if summary.get("guard_mode") != mode or len(rows) != 3 or [row.get("label") for row in rows] != list(LABELS):
        return False
    for row in rows:
        if (row.get("episodes") != 129 or type(row.get("successes")) is not int
                or not 0 <= row["successes"] <= 129 or not isinstance(row.get("gait_success_only"), dict)
                or "joint_and_torque_acceptance_passed" not in row or not isinstance(row.get("guard"), dict)):
            return False
        metrics = row["guard"].get("metrics", {})
        if not isinstance(metrics, dict):
            return False
        for kind in KINDS:
            metric = metrics.get(kind, {})
            if not isinstance(metric, dict):
                return False
            count, samples = metric.get("excess_count"), metric.get("sample_count")
            if type(count) is not int or type(samples) is not int or not 0 <= count <= samples:
                return False
            for key in ("peak_absolute_value", "max_ratio", "max_excess"):
                value = metric.get(key)
                if key not in metric or (samples > 0 and (type(value) not in (int, float) or not math.isfinite(value))):
                    return False
                if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                    return False
    return True


def inspect_run(root):
    root = Path(root)
    job = json.loads((root / "job.json").read_text())
    training_dir = Path(job["training_dir"])
    training_job = training_dir / "job.json"
    training = json.loads(training_job.read_text()) if training_job.exists() else {}
    records = read_jsonl(training_dir / "progress.jsonl")
    episodes = read_jsonl(training_dir / "episodes.jsonl")
    milestones = []
    for node in (500, 1000):
        record = next((row for row in records if row["update"] == node), None)
        if record is not None:
            milestones.append({"node": node, "cumulative_update": job["source_update"] + node,
                               "record": record, "guard_window": window_guard(episodes, node),
                               "checkpoint_exists": (training_dir / f"model_{node}.pt").is_file()})
    pid = job.get("pid")
    try:
        alive = bool(pid and b"run_moe_joint_recovery.py" in Path(f"/proc/{pid}/cmdline").read_bytes())
    except OSError:
        alive = False
    status = job["status"]
    if status in ("training", "preparing", "evaluating") and not alive:
        status = "stopped"
    if status == "training" and time.time() - training.get("last_update_unix", job["started_unix"]) > 900:
        status = "stalled"
    summary, strict = Path(job["eval_summary_path"]), Path(job["strict_eval_summary_path"])
    final = Path(job["final_checkpoint"])
    verified = False
    if status == "completed" and final.is_file():
        verified = (job.get("completion_verified") is True and comparison_complete(summary, "audit")
                    and comparison_complete(strict, "strict")
                    and hashlib.sha256(final.read_bytes()).hexdigest() == job.get("final_checkpoint_sha256"))
    return {"status": status, "pipeline_alive": alive, "pipeline": job, "training": training,
            "milestones": milestones, "completion_verified": verified,
            "final_checkpoint_exists": final.is_file(), "eval_summary_exists": summary.is_file(),
            "strict_eval_summary_exists": strict.is_file(), "eval_summary_path": str(summary),
            "strict_eval_summary_path": str(strict)}


def remote_code(run_path):
    functions = (read_jsonl, aggregate_metric, window_guard, comparison_complete, inspect_run)
    return ("import hashlib, json, math, time\nfrom pathlib import Path\n"
            + f"KINDS={KINDS!r}\nLABELS={LABELS!r}\nJOINT_NAMES={JOINT_NAMES!r}\nPROFILE_SHA256={PROFILE_SHA256!r}\n"
            + "\n\n".join(inspect.getsource(function) for function in functions)
            + f"\nprint(json.dumps(inspect_run({str(run_path)!r}), allow_nan=False))\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ack", type=int, choices=[500, 1000])
    parser.add_argument("--ack-event", choices=EVENTS)
    args = parser.parse_args()
    state_path = ROOT / ".monitor/moe_joint_recovery_state.json"
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
    cfg = json.loads((ROOT / ".monitor/moe_joint_recovery_job.json").read_text())
    command = ["ssh", "-S", cfg["ssh_control"], "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
               "-o", "UserKnownHostsFile=" + cfg["known_hosts"], "-p", str(cfg["port"]), cfg["host"], "python3 -"]
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
