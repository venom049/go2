#!/usr/bin/env python3
"""Bounded simulation prerequisites for the requested two-step physical test.

This runner has no robot connection, training, policy-selection or motor API.
Run only from an isolated copy of the evaluated source tree.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from run_guard_balanced_training import aggregate_scene


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    profile = root / "validation/onboard_joint_limits_20261104/joint_limits.json"
    checkpoints = {
        "guard1000": args.checkpoint_root / "logs/20261104_111052_stairs_guard_balanced/model_1000.pt",
        "guard500": args.checkpoint_root / "logs/20261104_111052_stairs_guard_balanced/model_500.pt",
        "reference_torque1000": args.checkpoint_root / "validation/joint_constraints/source_torque1000/model_1000.pt",
    }
    rows = []
    job = {"status": "preparing", "pid": os.getpid(), "started_unix": time.time(),
           "expected_conditions": 12, "completed_conditions": 0,
           "geometry": {"height_m": .15, "tread_m": .30, "steps": 2, "width_m": .60},
           "real_front_foot_distance_m": .50, "real_upper_landing_known": False,
           "evaluation_start_scope": "Original randomized spawn; not an exact measured real-robot start pose.",
           "motor_commands_sent": False, "training_started": False,
           "profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest()}
    write_json(out / "job.json", job)
    try:
        hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in checkpoints.items()}
        if hashes["guard1000"] != "b8e2e3c12474e3166c5ebd84187ea7364d3a3ba28ccba0131854f8de8964c31f":
            raise ValueError("Unexpected guard1000 checkpoint")
        if hashes["reference_torque1000"] != "508f5435b1a9de3ffe3bbf24ea3bd700c241e97dcb8671d5dba2b6c7ed3f325f":
            raise ValueError("Unexpected torque1000 reference")
        job["checkpoint_sha256"] = hashes
        for label, checkpoint in checkpoints.items():
            for speed in (.45, .6):
                for seed in (42, 43):
                    stem = f"{label}_speed{speed}_seed{seed}"
                    report_path = out / (stem + ".json")
                    cmd = ["bash", "scripts/run_lab.sh", "scripts/play_lab.py", "--headless",
                           "--checkpoint", str(checkpoint), "--controller", "stairs",
                           "--num-envs", "64", "--steps", "1500", "--episodes-per-env", "1",
                           "--gait-metrics", "--torque-metrics", "--joint-metrics",
                           "--onboard-velocity-limits", str(profile), "--stairs-flight-width", "0.6",
                           "--output", str(report_path)]
                    overrides = {"seed": seed, "terrain.rows": 1, "terrain.cols": 4,
                                 "terrain.initial_level": 0, "terrain.num_steps": 2,
                                 "terrain.height_min": .15, "terrain.height_max": .15, "terrain.tread": .30,
                                 "commands.forward_min": speed, "commands.forward_max": speed,
                                 "joint_constraints.velocity_limit": 13.0,
                                 "joint_constraints.tracking_error_limit": .9,
                                 "joint_constraints.startup_seconds": 2.0}
                    for key, value in overrides.items():
                        cmd.extend(("--set", f"{key}={value}"))
                    job.update(status="evaluating", condition=stem, command=cmd, updated_unix=time.time())
                    write_json(out / "job.json", job)
                    env = dict(os.environ, GO2_PYTHON="/opt/conda/envs/isaaclab/bin/python")
                    with (out / (stem + ".log")).open("x") as log:
                        process = subprocess.Popen(cmd, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                                   stdout=log, stderr=subprocess.STDOUT)
                        job["eval_pid"] = process.pid
                        write_json(out / "job.json", job)
                        code = process.wait()
                    if code:
                        raise RuntimeError(f"Evaluation process failed: {stem}, return code {code}")
                    report = json.loads(report_path.read_text())
                    if report.get("episodes") != 64 or report.get("envs_incomplete") != 0:
                        raise RuntimeError(f"Incomplete episode quota: {stem}")
                    calibration = report["joint_velocity_calibration"]
                    if calibration["profile_sha256"] != job["profile_sha256"] or not calibration.get("final_readback"):
                        raise RuntimeError("Calibration evidence missing or mismatched")
                    row = {"label": label, "command_m_s": speed, "seed": seed, "condition": stem,
                           "report_path": str(report_path),
                           **{key: report[key] for key in ("successes", "episodes", "envs_incomplete",
                               "joint_acceptance_passed", "torque_acceptance_passed", "joint_audit", "torque_audit")},
                           "successful_episode_gait": report.get("gait_metrics_successful_episodes", {})}
                    rows.append(row)
                    write_json(out / "partial_comparison.json", {"complete": False, "rows": rows})
                    job.update(completed_conditions=len(rows), updated_unix=time.time())
                    write_json(out / "job.json", job)
                    print(json.dumps({"finished": stem, "successes": row["successes"],
                                      "joint_passed": row["joint_acceptance_passed"],
                                      "torque_passed": row["torque_acceptance_passed"]}), flush=True)
        summaries = []
        for speed in (.45, .6):
            reference = sum(row["successes"] for row in rows if row["label"] == "reference_torque1000" and row["command_m_s"] == speed)
            for label in checkpoints:
                aggregate = aggregate_scene([row for row in rows if row["label"] == label and row["command_m_s"] == speed])
                aggregate.update(label=label, command_m_s=speed, reference_successes=reference,
                                 task_nonregression_passed=aggregate["successes"] >= reference and aggregate["successes"] > 0)
                aggregate["sampled_simulation_eligible"] = (aggregate["task_nonregression_passed"] and
                    aggregate["joint_acceptance_passed"] and aggregate["torque_acceptance_passed"])
                summaries.append(aggregate)
        summary = {"complete": len(rows) == 12, "rows": rows, "summaries": summaries,
                   "physical_motion_authorized_by_report": False,
                   "limitations": ["URDF speed specifications are not drive firmware readback.",
                       "Upper landing dimensions remain unconfirmed; geometry includes an explicitly assumed long narrow landing.",
                       "Original randomized simulation spawn is not the current measured 50 cm front-foot offset.",
                       "Success requires upper-landing foot occupancy; it is not standing-stop or balance certification.",
                       "No model is automatically activated, selected or deployed."]}
        write_json(out / "comparison.json", summary)
        job.update(status="completed", finished_unix=time.time(), eval_pid=None)
        write_json(out / "job.json", job)
    except BaseException as exc:
        job.update(status="failed", error=str(exc), traceback=traceback.format_exc(), finished_unix=time.time())
        write_json(out / "job.json", job)
        raise


if __name__ == "__main__":
    main()
