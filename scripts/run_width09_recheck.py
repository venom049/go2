#!/usr/bin/env python3
"""Evaluate the requested 0.9 m two-step flight; no training or robot APIs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from run_guard_balanced_training import aggregate_scene


CHECKPOINT_SHA = "b8e2e3c12474e3166c5ebd84187ea7364d3a3ba28ccba0131854f8de8964c31f"
PROFILE_SHA = "185d1082f317e9d260fa7fd13c1d2e557159cfa3c67369a7f1be2aeba09b405f"


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    profile = root / "validation/onboard_joint_limits_20261104/joint_limits.json"
    checkpoint = args.checkpoint_root / "logs/20261104_111052_stairs_guard_balanced/model_1000.pt"
    rows = []
    job = {"status": "preparing", "pid": os.getpid(), "started_unix": time.time(),
           "expected_conditions": 4, "completed_conditions": 0, "model": "guard1000",
           "geometry": {"height_m": .15, "tread_m": .30, "steps": 2, "width_m": .90},
           "command_speeds_m_s": [.45, .6], "seeds": [42, 43], "episodes_per_condition": 64,
           "start_scope": "Body origin 1.2 +/- 0.1 m before first riser; lateral +/- 0.1 m; yaw +/- 0.05 rad; initial body z 0.4 m. Not a measured front-foot offset.",
           "upper_landing_scope": "Assumed long upper landing, width 0.9 m; actual site dimensions unknown.",
           "motor_commands_sent": False, "training_started": False}
    write_json(out / "job.json", job)
    try:
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == CHECKPOINT_SHA, "Checkpoint identity mismatch"
        assert hashlib.sha256(profile.read_bytes()).hexdigest() == PROFILE_SHA, "Velocity profile identity mismatch"
        baseline = json.loads(args.baseline.read_text())
        assert baseline.get("complete") is True, "Incomplete baseline comparison"
        previous = [r for r in baseline["rows"] if r["label"] == "guard1000"]
        assert len(previous) == 4
        assert {(r["command_m_s"], r["seed"]) for r in previous} == {(v, s) for v in (.45, .6) for s in (42, 43)}
        # Verify the saved 0.6 m conditions before describing this as a width comparison.
        for row in previous:
            report = json.loads(Path(row["report_path"]).read_text())
            geometry = report["narrow_stairs_geometry"]
            assert (geometry["flight_width_m"], geometry["num_steps"], geometry["step_height_m"], geometry["tread_m"]) == (.6, 2, .15, .3)
            assert report["controller"] == "stairs" and report["spawn_x_offset_m"] == 0
            assert report["episodes"] == 64 and report["envs_incomplete"] == 0
            assert report["joint_velocity_calibration"]["profile_sha256"] == PROFILE_SHA
            assert report["checkpoint"] == str(checkpoint)
            assert report["settings"]["seed"] == row["seed"]
            for bound in ("forward_min", "forward_max"):
                assert report["settings"]["commands"][bound] == row["command_m_s"]
        write_json(out / "baseline_060.json", baseline)
        job.update(checkpoint_sha256=CHECKPOINT_SHA, profile_sha256=PROFILE_SHA,
                   baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest())
        for speed in (.45, .6):
            for seed in (42, 43):
                stem = f"guard1000_width0.9_speed{speed}_seed{seed}"
                report_path = out / (stem + ".json")
                cmd = ["bash", "scripts/run_lab.sh", "scripts/play_lab.py", "--headless",
                       "--checkpoint", str(checkpoint), "--controller", "stairs",
                       "--num-envs", "64", "--steps", "1500", "--episodes-per-env", "1",
                       "--gait-metrics", "--torque-metrics", "--joint-metrics",
                       "--onboard-velocity-limits", str(profile), "--stairs-flight-width", "0.9",
                       "--output", str(report_path)]
                overrides = {"seed": seed, "terrain.rows": 1, "terrain.cols": 4,
                             "terrain.initial_level": 0, "terrain.num_steps": 2,
                             "terrain.height_min": .15, "terrain.height_max": .15, "terrain.tread": .30,
                             "commands.forward_min": speed, "commands.forward_max": speed,
                             "joint_constraints.velocity_limit": 13.,
                             "joint_constraints.tracking_error_limit": .9,
                             "joint_constraints.startup_seconds": 2.}
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
                    raise RuntimeError(f"Evaluation failed: {stem}, return code {code}")
                report = json.loads(report_path.read_text())
                assert report["episodes"] == 64 and report["envs_incomplete"] == 0, "Incomplete episode quota"
                calibration = report["joint_velocity_calibration"]
                assert calibration["profile_sha256"] == PROFILE_SHA and calibration.get("final_readback")
                assert calibration["initial_readback"] == calibration["final_readback"]
                assert report["narrow_stairs_geometry"]["flight_width_m"] == .9
                row = {"label": "guard1000", "width_m": .9, "command_m_s": speed, "seed": seed,
                       "condition": stem, "report_path": str(report_path),
                       **{key: report[key] for key in ("successes", "episodes", "failures", "timeouts",
                           "envs_incomplete", "joint_acceptance_passed", "torque_acceptance_passed",
                           "joint_audit", "torque_audit")},
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
            current = aggregate_scene([r for r in rows if r["command_m_s"] == speed])
            old = aggregate_scene([r for r in previous if r["command_m_s"] == speed])
            current.update(model="guard1000", width_m=.9, command_m_s=speed,
                           baseline_width_m=.6, baseline_successes=old["successes"],
                           baseline_episodes=old["episodes"],
                           baseline_joint_velocity_peak_rad_s=old["joint_audit"]["velocity_peak_rad_s"],
                           success_rate_delta_from_baseline=current["success_rate"] - old["success_rate"],
                           sampled_simulation_eligible=(current["successes"] > 0 and
                               current["joint_acceptance_passed"] and current["torque_acceptance_passed"]))
            summaries.append(current)
        write_json(out / "comparison.json", {"complete": len(rows) == 4, "rows": rows, "summaries": summaries,
            "limitations": ["Two-step assumption follows the described physical stairs; this is not the old eight-step case.",
                "Spawn is the same randomized body-origin distribution as the 0.6 m comparison, not a measured 50 cm front-foot offset.",
                "Upper landing is assumed; successful completion does not certify stable stopping.",
                "Velocity caps come from onboard URDF specifications, not proprietary drive firmware.",
                "Command speeds are not measured climbing speeds. Gait statistics include successful first episodes only.",
                "Joint/torque audits include failures and terminal physics substeps; no model is automatically activated or deployed."]})
        job.update(status="completed", finished_unix=time.time(), eval_pid=None)
        write_json(out / "job.json", job)
    except BaseException as exc:
        job.update(status="failed", error=str(exc), traceback=traceback.format_exc(), finished_unix=time.time())
        write_json(out / "job.json", job)
        raise


if __name__ == "__main__":
    main()
