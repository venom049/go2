#!/usr/bin/env python3
"""Read-only SSH monitor for the stable-fast continuation and its evaluation.

Connection details live in .monitor/continuation_job.json, without passwords.
Acknowledgements and snapshots are local and isolated by remote job path.
"""
import argparse
import hashlib
import json
import math
import re
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ("completed", "stopped", "stalled", "unreachable", "evaluating")
ACK_MILESTONES = (1100, 1600)
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$|^[+-]?(?:nan|inf(?:inity)?)$", re.I)
MAIN_LOSSES = ("Mean value_function loss", "Mean surrogate loss", "Mean estimation loss", "Mean swap loss")
JOINT_METRICS = tuple(
    f"JointLimits/{kind}_{suffix}"
    for kind in ("velocity", "position", "target_position", "tracking_error")
    for suffix in ("excess_fraction", "peak_ratio")
) + ("JointLimits/violation_environment_fraction",)


def parse_iterations(text, start_iteration=600):
    """Discard partial log blocks and translate the runner's zero-based index."""
    text = ANSI.sub("", text)
    headings = list(re.finditer(r"Learning iteration (\d+)/(\d+)", text))
    records = []
    for index, match in enumerate(headings):
        block = text[match.end():headings[index + 1].start() if index + 1 < len(headings) else len(text)]
        values = {key.strip(): value.strip() for key, value in
                  re.findall(r"^\s*([^:\n]+):\s*([^\n]+)", block, re.M)}
        if "Total timesteps" not in values or "ETA" not in values:
            continue
        numeric = {key: float(value) for key, value in values.items() if NUMBER.fullmatch(value)}
        nonfinite = [key for key, value in numeric.items() if not math.isfinite(value)]
        loss_names = [key for key in values if key.lower().endswith(" loss") or key.startswith("Loss/")]
        complete_losses = all(key in numeric for key in MAIN_LOSSES)
        records.append({
            "completed_updates": int(match[1]) + 1,
            "new_updates": max(0, int(match[1]) + 1 - start_iteration),
            "target_updates": int(match[2]), "metrics": values,
            "numeric_metrics": {key: value if math.isfinite(value) else None for key, value in numeric.items()},
            "nonfinite_metrics": nonfinite,
            "metrics_finite": not nonfinite,
            "main_losses_present": complete_losses,
            "losses_finite": (all(key in numeric and math.isfinite(numeric[key]) for key in loss_names)
                               if complete_losses else None),
        })
    return records


def process_info(pid, expected_command=None):
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return {"pid": pid, "alive": False, "detail": "PID not recorded"}
    try:
        proc = Path("/proc") / str(pid)
        state = proc.joinpath("stat").read_text().rsplit(") ", 1)[1].split()[0]
        command = proc.joinpath("cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        alive = state not in ("Z", "X") and bool(command)
        if expected_command and expected_command not in command:
            alive = False
        return {"pid": pid, "alive": alive, "state": state}
    except (OSError, IndexError) as exc:
        return {"pid": pid, "alive": False, "detail": str(exc)}


def verified_completion(completion, checkpoint_exists, target):
    return bool(isinstance(completion, dict)
                and isinstance(completion.get("iterations"), int)
                and not isinstance(completion["iterations"], bool)
                and completion["iterations"] >= target
                and completion.get("all_model_tensors_finite") is True
                and checkpoint_exists)


def classify_status(pipeline_status, pipeline_alive, train_alive, artifacts_verified,
                    training_age, evaluation_age, startup_age, stall_seconds=300):
    """Never call a run completed from a log line or a live PID alone."""
    if pipeline_status == "completed":
        return "completed" if artifacts_verified else "stopped"
    if pipeline_status == "failed":
        return "stopped"
    if pipeline_status == "evaluating" or artifacts_verified:
        if not pipeline_alive:
            return "stopped"
        return "stalled" if evaluation_age > stall_seconds else "evaluating"
    if train_alive:
        return "stalled" if training_age > stall_seconds else "running"
    if pipeline_alive and startup_age <= stall_seconds:
        return "running"
    return "stopped"


def read_optional_json(path, errors):
    if not path or not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        errors.append(f"Cannot read {path}: {exc}")
        return None


def age_seconds(path, now, fallback):
    try:
        return max(0.0, now - path.stat().st_mtime) if path else fallback
    except OSError:
        return fallback


def joint_metrics_at_updates(accumulator, updates):
    """Read exact zero-based TensorBoard steps, never the nearest/newest step.

    The runner already stores a MAX over physics/rollout samples for peak tags.
    If a step has duplicate records (e.g. a writer reopened), retain their MAX
    for peaks and the newest wall-time value for fractions. Preserve nonfinite
    evidence rather than letting a later finite record erase it.
    """
    selected = {int(update) - 1: {} for update in updates}
    for tag in accumulator.Tags().get("scalars", []):
        # Preserve exact torque fractions and bounded-reward diagnostics too;
        # console rounding must not turn a small violation into a reported zero.
        if not tag.startswith(("JointLimits/", "Torque/", "GuardBalance/")):
            continue
        for event in accumulator.Scalars(tag):
            if event.step not in selected:
                continue
            previous = selected[event.step].get(tag)
            candidate = (float(event.wall_time), float(event.value))
            if previous is None:
                selected[event.step][tag] = candidate
            elif not math.isfinite(previous[1]):
                continue
            elif not math.isfinite(candidate[1]):
                selected[event.step][tag] = candidate
            elif "peak" in tag:
                selected[event.step][tag] = max(previous, candidate, key=lambda pair: pair[1])
            elif candidate[0] >= previous[0]:
                selected[event.step][tag] = candidate
    result = {}
    for step, values in selected.items():
        nonfinite = sorted(tag for tag, (_, value) in values.items() if not math.isfinite(value))
        missing = [tag for tag in JOINT_METRICS if tag not in values]
        result[step + 1] = {
            "source": "tensorboard", "step": step,
            "metrics": {tag: value if math.isfinite(value) else None
                        for tag, (_, value) in values.items()},
            "missing_metrics": missing, "nonfinite_metrics": nonfinite,
            "all_metrics_finite": False if nonfinite else (None if missing else True),
        }
    return result


def read_joint_tensorboard(run, updates, errors):
    """Optional precision/peak supplement, enabled only by new job metadata."""
    if run is None:
        return {}
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        accumulator = EventAccumulator(str(run), size_guidance={"scalars": 0})
        accumulator.Reload()
        return joint_metrics_at_updates(accumulator, updates)
    except Exception as exc:
        # An unavailable diagnostic source must not turn a live run into a
        # connection failure, nor silently manufacture zero violations.
        errors.append(f"Cannot read optional constraint TensorBoard metrics: {type(exc).__name__}: {exc}")
        return {}


def remote_snapshot(remote_job, now=None):
    """Runs remotely via stdin; all operations in this function are reads."""
    now = time.time() if now is None else now
    job_path = Path(remote_job)
    job = json.loads(job_path.read_text())
    start = int(job.get("start_iteration", 600))
    additional = int(job.get("additional_updates", 1000))
    target = int(job.get("target_iteration", start + additional))
    if target <= start or additional != target - start:
        raise ValueError("Inconsistent continuation iteration bounds in job metadata")
    errors = []
    console = Path(job["console_log"])
    try:
        text = ANSI.sub("", console.read_text(errors="replace"))
    except OSError as exc:
        text = ""
        errors.append(f"Cannot read training console: {exc}")
    run_value = job.get("run_directory")
    if not run_value:
        matches = re.findall(r"^GO2_RUN_DIR=(.+)$", text, re.M)
        run_value = matches[-1].strip() if matches else None
    run = Path(run_value) if run_value else None
    records = parse_iterations(text, start)
    records = [record for record in records if start < record["completed_updates"] <= target]
    latest = records[-1] if records else {
        "completed_updates": start, "new_updates": 0, "target_updates": target,
        "metrics": {}, "numeric_metrics": {}, "nonfinite_metrics": [],
        "main_losses_present": False, "losses_finite": None, "metrics_finite": None,
    }
    completion_file = run / "completed.json" if run else None
    completion = read_optional_json(completion_file, errors)
    final_checkpoint = run / f"model_{target}.pt" if run else None
    final_exists = bool(final_checkpoint and final_checkpoint.is_file())
    artifacts_verified = verified_completion(completion, final_exists, target)
    # completed.json also allows reporting nodes when the console was rotated.
    completed_count = target if artifacts_verified else latest["completed_updates"]
    if artifacts_verified and latest["completed_updates"] < target:
        latest = {"completed_updates": target, "new_updates": target - start,
                  "target_updates": target, "metrics": {}, "numeric_metrics": {},
                  "metrics_unavailable": True, "main_losses_present": False,
                  "losses_finite": None, "metrics_finite": None,
                  "last_log_completed_updates": latest["completed_updates"]}
    milestones = []
    by_iteration = {record["completed_updates"]: record for record in records}
    for node in range(start + 500, target + 1, 500):
        if node > completed_count:
            continue
        record = dict(by_iteration.get(node, {
            "completed_updates": node, "new_updates": node - start,
            "target_updates": target, "metrics": {}, "metrics_unavailable": True,
            "main_losses_present": False, "losses_finite": None,
        }))
        checkpoint = run / f"model_{node}.pt" if run else None
        record.update(checkpoint=str(checkpoint) if checkpoint else None,
                      checkpoint_exists=bool(checkpoint and checkpoint.is_file()))
        milestones.append(record)
    latest = dict(latest)
    latest_checkpoint = run / f"model_{latest['completed_updates']}.pt" if run else None
    latest.update(checkpoint=str(latest_checkpoint) if latest_checkpoint else None,
                  checkpoint_exists=bool(latest_checkpoint and latest_checkpoint.is_file()))
    if job.get("tensorboard_joint_metrics") is True:
        joint_metrics = read_joint_tensorboard(
            run, {latest["completed_updates"], *(record["completed_updates"] for record in milestones)}, errors)
        for record in [latest, *milestones]:
            record["joint_limits"] = joint_metrics.get(record["completed_updates"])
    train = process_info(job.get("train_pid"), "scripts/train_lab.py")
    pipeline = process_info(job.get("pid"))
    job_age = age_seconds(job_path, now, 0.0)
    training_age = age_seconds(console, now, job_age)
    evaluation_console = Path(job["evaluation_console"]) if job.get("evaluation_console") else None
    evaluation_age = age_seconds(evaluation_console, now, job_age)
    phase = job.get("pipeline_status", "training")
    status = classify_status(phase, pipeline["alive"], train["alive"], artifacts_verified,
                             training_age, evaluation_age, job_age)
    detail = None
    if phase == "completed" and not artifacts_verified:
        detail = "Pipeline claims completion, but completed.json / finite tensors / final checkpoint could not be verified"
    elif phase == "failed":
        detail = "Pipeline recorded failure: " + str(job.get("error") or job.get("detail") or "see console logs")
    elif status == "stopped":
        detail = "Expected training or evaluation pipeline process is no longer running"
    elif status == "stalled":
        detail = "Live process has not updated its phase console for more than 300 seconds"
    errors.extend(line for line in text.splitlines() if re.search(
        r"Traceback|NaN/Inf detected|CUDA out of memory|FloatingPointError", line))
    if evaluation_console and evaluation_console.is_file():
        # Keep a bounded diagnostic tail while avoiding large evaluation output.
        with evaluation_console.open("rb") as stream:
            stream.seek(max(0, evaluation_console.stat().st_size - 16384))
            evaluation_tail = stream.read().decode(errors="replace")
        errors.extend(line for line in evaluation_tail.splitlines() if re.search(
            r"Traceback|NaN/Inf detected|CUDA out of memory|FloatingPointError|FAILED|ERROR", line))
    summary_value = job.get("eval_summary_path") or job.get("evaluation_summary")
    if not summary_value and job.get("output_dir"):
        summary_value = str(Path(job["output_dir"]) / "comparison.json")
    return {
        "status": status, "detail": detail, "pipeline_status": phase,
        "remote_job": str(job_path), "run_directory": str(run) if run else None,
        "console_log": str(console), "evaluation_console": str(evaluation_console) if evaluation_console else None,
        "start_iteration": start, "additional_updates": additional, "target_iteration": target,
        "latest": latest, "completed_updates": completed_count, "new_updates": max(0, completed_count - start),
        "milestones": milestones, "completion": completion,
        "completion_path": str(completion_file) if completion_file else None,
        "training_artifacts_verified": artifacts_verified,
        "completion_verified": phase == "completed" and artifacts_verified,
        "final_checkpoint": str(final_checkpoint) if final_checkpoint else None,
        "final_checkpoint_exists": final_exists,
        "eval_summary_path": summary_value,
        "eval_summary_exists": bool(summary_value and Path(summary_value).is_file()),
        "train_process": train, "pipeline_process": pipeline,
        "seconds_since_log_update": round(training_age, 1),
        "seconds_since_evaluation_log_update": round(evaluation_age, 1), "errors": errors[-10:],
    }


def state_paths(remote_job, directory=None):
    directory = Path(directory) if directory else ROOT / ".monitor"
    identity = hashlib.sha256(remote_job.encode("utf-8")).hexdigest()[:16]
    return directory / f"continuation_{identity}_state.json", directory / f"continuation_{identity}_snapshot.json"


def acknowledge(state, milestone=None, event=None):
    result = dict(state)
    if milestone is not None:
        if milestone not in ACK_MILESTONES:
            raise ValueError("Milestone must be 1100 or 1600 (500 or 1000 additional updates)")
        result["last_reported_milestone"] = max(int(result.get("last_reported_milestone", 0)), milestone)
    if event is not None:
        if event not in EVENTS:
            raise ValueError("Unsupported event")
        result["last_reported_event"] = event
        seen = set(result.get("reported_phase_events", []))
        if event in ("evaluating", "completed"):
            seen.add(event)
        result["reported_phase_events"] = sorted(seen)
    return result


def report_filter(data, state):
    data, state = dict(data), dict(state)
    data["pending_milestones"] = [record for record in data.pop("milestones", [])
                                  if record["completed_updates"] > int(state.get("last_reported_milestone", 0))]
    status = data["status"]
    seen = state.get("reported_phase_events", [])
    data["event_to_report"] = (status if status in EVENTS
                               and status != state.get("last_reported_event")
                               and status not in seen else None)
    if status in ("running", "evaluating") and state.get("last_reported_event") in ("unreachable", "stalled", "stopped"):
        data["connection_recovered"] = state.get("last_reported_event") == "unreachable"
        state["last_reported_event"] = None
    return data, state


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def fetch_snapshot(config):
    remote_command = shlex.join([str(config["python"]), "-", "--remote", str(config["remote_job"])])
    command = ["ssh", "-S", str(config["ssh_control"]), "-o", "BatchMode=yes",
               "-o", "PasswordAuthentication=no", "-o", "StrictHostKeyChecking=yes",
               "-o", "UserKnownHostsFile=" + str(config["known_hosts"]), "-o", "ConnectTimeout=10",
               "-p", str(config["port"]), str(config["host"]), remote_command]
    result = subprocess.run(command, input=Path(__file__).read_text(), capture_output=True, text=True, timeout=40)
    if result.returncode:
        raise RuntimeError(f"SSH/read command exited {result.returncode}: {result.stderr.strip()[-4000:]}")
    data = json.loads(result.stdout)
    if not isinstance(data, dict) or "status" not in data:
        raise ValueError("SSH response is not a monitor snapshot")
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / ".monitor/continuation_job.json")
    parser.add_argument("--ack", type=int, choices=ACK_MILESTONES,
                        help="Record a reported cumulative milestone locally; no SSH")
    parser.add_argument("--ack-event", choices=EVENTS, help="Record a reported event locally; no SSH")
    parser.add_argument("--remote", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.remote:
        try:
            data = remote_snapshot(args.remote)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            data = {"status": "unreachable", "monitor_error": True,
                    "detail": f"SSH connected but remote monitor could not read job: {exc}",
                    "latest": None, "completion": None, "eval_summary_path": None, "milestones": [], "errors": [str(exc)]}
        print(json.dumps(data, ensure_ascii=False, allow_nan=False))
        return
    try:
        config = json.loads(args.config.read_text())
        remote_job = config["remote_job"]
        if not isinstance(remote_job, str) or not remote_job:
            raise ValueError("remote_job must be a nonempty path")
        state_path, snapshot_path = state_paths(remote_job)
        state = json.loads(state_path.read_text()) if state_path.exists() else {
            "last_reported_milestone": 0, "last_reported_event": None, "reported_phase_events": []}
        if args.ack is not None or args.ack_event:
            state = acknowledge(state, args.ack, args.ack_event)
            write_json(state_path, state)
            print(json.dumps(state, ensure_ascii=False))
            return
        try:
            data = fetch_snapshot(config)
        except (subprocess.TimeoutExpired, RuntimeError, OSError, ValueError, KeyError) as exc:
            data = {"status": "unreachable", "detail": str(exc), "connection_error": True,
                    "latest": None, "completion": None, "eval_summary_path": None, "milestones": [], "errors": [str(exc)]}
        data, state = report_filter(data, state)
        data["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
        data["state_path"] = str(state_path)
        state["last_check"] = data["checked_at_utc"]
        write_json(state_path, state)
        write_json(snapshot_path, data)
        print(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "unreachable", "monitor_error": True, "detail": str(exc),
                          "latest": None, "completion": None, "eval_summary_path": None,
                          "pending_milestones": [], "event_to_report": "unreachable", "errors": [str(exc)]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
