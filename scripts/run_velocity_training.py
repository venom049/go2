#!/usr/bin/env python3
"""Run the authorized velocity-focused warm start and 12 evaluations once.

Run on the training server. This script never retries training, selects a
replacement policy, or deploys to a robot. Constraint violations are evaluation
results, not a reason to discard an otherwise complete evaluation report.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation/velocity_training"
JOB = OUT / "job.json"
CONSOLE = OUT / "training.console.log"
EVAL_CONSOLE = OUT / "evaluation.console.log"
SUMMARY = OUT / "comparison.json"
SNAPSHOT = OUT / "training_source_snapshot.tar.gz"
PYTHON = "/opt/conda/envs/isaaclab/bin/python"
SOURCE = ROOT / "validation/joint_constraints/source_torque1000/model_1000.pt"
SOURCE_SHA256 = "508f5435b1a9de3ffe3bbf24ea3bd700c241e97dcb8671d5dba2b6c7ed3f325f"
WIRING_REPORT = ROOT / "validation/velocity_training_setup/isaac_wiring_check.json"
WALK_CHECKPOINT = ROOT / "assets/policies/upstream_walk/walk.pt"
LABELS = ("reference_torque1000", "velocity500", "velocity1000")
CONDITIONS = ((0.15, 0.30, "stairs"), (0.18, 0.32, "hybrid"))
SEEDS = (42, 43)
VELOCITY_LIMIT_RAD_S = 13.0
JOINT_NAMES = {f"{leg}_{kind}_joint" for leg in ("FL", "FR", "RL", "RR")
               for kind in ("hip", "thigh", "calf")}


def condition_name(label, height, controller, seed):
    return f"{label}_{controller}_{height}m_seed{seed}"


def expected_conditions():
    return {
        condition_name(label, height, controller, seed)
        for label in LABELS for height, _, controller in CONDITIONS for seed in SEEDS
    }


def encode(value):
    return json.dumps(value, indent=2, allow_nan=False) + "\n"


def write_job(job, **updates):
    """Update only this invocation's reserved job, atomically for the monitor."""
    job.update(updates, updated_at_unix=time.time())
    with tempfile.NamedTemporaryFile("w", dir=OUT, prefix=".job-", suffix=".tmp",
                                     delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(encode(job))
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(JOB)
    finally:
        temporary.unlink(missing_ok=True)


def reserve_job(job):
    OUT.mkdir(parents=True, exist_ok=True)
    expected = [JOB, CONSOLE, EVAL_CONSOLE, SUMMARY, SNAPSHOT]
    for stem in expected_conditions():
        expected.extend((OUT / f"{stem}.json", OUT / f"{stem}.console.log"))
    for path in expected:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite previous job/output: {path}")
    # Exclusive creation prevents two concurrent launchers from starting twice.
    with JOB.open("x") as stream:
        stream.write(encode(job))


def snapshot_source():
    names = ("go2_him", "configs", "scripts", "tests",
             "upstream/himloco_lab/source", "upstream/SOURCES.json")
    with tarfile.open(SNAPSHOT, "x:gz") as archive:
        for name in names:
            path = ROOT / name
            if not path.exists():
                raise FileNotFoundError(f"Missing source snapshot input: {path}")
            candidates = sorted(path.rglob("*")) if path.is_dir() else [path]
            for candidate in candidates:
                if candidate.is_file() and not any(
                        part in ("__pycache__", ".git") for part in candidate.parts):
                    archive.add(candidate, arcname=str(candidate.relative_to(ROOT)))


def _finite_nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _complete_episode_quota(row):
    return (type(row.get("episodes")) is int and row["episodes"] == 64
            and type(row.get("envs_incomplete")) is int and row["envs_incomplete"] == 0)


def _complete_audit_samples(audit):
    count = audit.get("physics_samples_per_joint")
    startup = audit.get("startup_physics_samples_per_joint")
    return (type(count) is int and type(startup) is int and 0 < startup <= count)


def speed_acceptance(row):
    """Check speed evidence directly; other joint violations remain observations.

    Neither a true summary flag nor an absent/empty audit can establish a pass.
    Measured speed includes every active physics substep, including failed and
    terminal episodes, and must stay within the existing 13 rad/s guard.
    """
    audit = row.get("joint_audit")
    if not isinstance(audit, dict) or not _complete_episode_quota(row):
        return False
    if (audit.get("audit_invalid") is not False or not _complete_audit_samples(audit)
            or audit.get("velocity_limit_rad_s") != VELOCITY_LIMIT_RAD_S):
        return False
    for field in ("velocity_excess_joint_samples", "startup_velocity_excess_joint_samples"):
        if type(audit.get(field)) is not int or audit[field] != 0:
            return False
    for field in ("velocity_excess_fraction", "startup_velocity_excess_fraction"):
        if not _finite_nonnegative(audit.get(field)) or audit[field] != 0:
            return False
    peak_ratio = audit.get("velocity_peak_ratio")
    if not _finite_nonnegative(peak_ratio) or peak_ratio > 1:
        return False
    joints = audit.get("joints")
    if not isinstance(joints, dict) or set(joints) != JOINT_NAMES:
        return False
    for joint in joints.values():
        if not isinstance(joint, dict):
            return False
        peak = joint.get("velocity_peak_rad_s")
        count = joint.get("velocity_excess_samples")
        if (not _finite_nonnegative(peak) or peak > VELOCITY_LIMIT_RAD_S
                or type(count) is not int or count != 0):
            return False
    return True


def torque_acceptance(row):
    """Keep the previous requested/applied torque acceptance, fail closed."""
    audit = row.get("torque_audit")
    if not isinstance(audit, dict) or not _complete_episode_quota(row):
        return False
    if not _complete_audit_samples(audit) or audit.get("torque_limits_passed") is not True:
        return False
    for field in ("requested_excess_joint_samples", "applied_excess_joint_samples",
                  "startup_excess_joint_samples"):
        if type(audit.get(field)) is not int or audit[field] != 0:
            return False
    fraction = audit.get("requested_excess_fraction")
    if not _finite_nonnegative(fraction) or fraction != 0:
        return False
    joints = audit.get("joints")
    if not isinstance(joints, dict) or set(joints) != JOINT_NAMES:
        return False
    for name, joint in joints.items():
        if not isinstance(joint, dict):
            return False
        limit = joint.get("limit_nm")
        expected_limit = 45.43 if "_calf_" in name else 23.7
        if (not _finite_nonnegative(limit)
                or not math.isclose(limit, expected_limit, rel_tol=0, abs_tol=1e-5)):
            return False
        for field in ("requested_peak_nm", "applied_peak_nm"):
            if not _finite_nonnegative(joint.get(field)) or joint[field] > limit:
                return False
        if type(joint.get("requested_excess_samples")) is not int or joint["requested_excess_samples"] != 0:
            return False
    # Both audits must describe the same active-episode substeps.
    joint_audit = row.get("joint_audit", {})
    return all(audit[field] == joint_audit.get(field) for field in (
        "physics_samples_per_joint", "startup_physics_samples_per_joint"))


def build_summary(rows):
    names = [row["condition"] for row in rows]
    complete = (len(rows) == 12 and len(set(names)) == 12
                and set(names) == expected_conditions()
                and all(_complete_episode_quota(row) for row in rows))
    if not complete:
        raise RuntimeError("Refusing incomplete or duplicate 12-condition comparison")
    # Preserve the full joint audit while explicitly separating this experiment's
    # speed objective from position/target/tracking observations.
    rows = [dict(row, speed_acceptance_passed=speed_acceptance(row),
                 speed_and_torque_acceptance_passed=(speed_acceptance(row) and torque_acceptance(row)))
            for row in rows]
    eligible = []
    for label in LABELS[1:]:
        candidate = [row for row in rows if row["label"] == label]
        if len(candidate) == 4 and all(
                row["speed_and_torque_acceptance_passed"] is True
                and row["successes"] == row["episodes"] for row in candidate):
            eligible.append(label)
    return {
        "source_checkpoint": str(SOURCE), "source_sha256": SOURCE_SHA256,
        "expected_conditions": sorted(expected_conditions()),
        "all_expected_conditions_complete": complete, "rows": rows,
        "constraint_eligible_candidates": eligible,
        "eligibility_requires": "All 12 expected reports complete; a candidate must have 64/64 successes in all four conditions, finite complete measured-speed evidence with zero samples above 13 rad/s, and zero requested/applied torque violations. Position, target position, and tracking-error violations are reported separately and do not disqualify this velocity-focused experiment.",
        "note": "0.6 m/s is commanded speed. Gait statistics select successful first episodes; full joint and torque audits include failures and terminal substeps. All position and tracking-error observations remain in each row's joint_audit, regardless of velocity eligibility. Existing real-robot position, tracking, speed and torque guards remain unchanged. Eligibility is a sampled simulation result, not real-robot certification or deployment approval. No automatic selection, policy replacement, or deployment.",
    }


def main():
    command = ["bash", "scripts/run_lab.sh", "scripts/train_lab.py", "--headless",
               "--config", "configs/stairs_velocity_limited.yaml", "--warm-start", str(SOURCE),
               "--num-envs", "1024", "--max-iterations", "1000",
               "--run-name", "stairs_velocity_limited"]
    job = dict(
        pid=os.getpid(), train_pid=None, start_iteration=0, additional_updates=1000,
        target_iteration=1000, num_envs=1024, initialization="warm_start",
        fresh_optimizers=True, tensorboard_joint_metrics=True,
        source_checkpoint=str(SOURCE), source_sha256=None,
        expected_source_sha256=SOURCE_SHA256, command=command,
        output_dir=str(OUT), console_log=str(CONSOLE), evaluation_console=str(EVAL_CONSOLE),
        run_directory=None, pipeline_status="starting", started_at_unix=time.time(),
        eval_summary_path=str(SUMMARY), source_snapshot=str(SNAPSHOT),
        evaluation_plan={
            "iterations": [500, 1000], "reference_source_iteration": 1000,
            "labels": list(LABELS), "expected_report_count": 12,
            "seeds": list(SEEDS), "num_envs": 64, "episodes_per_env": 1,
            "conditions": [{"height": h, "tread": t, "controller": c}
                           for h, t, c in CONDITIONS],
            "command_m_s": 0.6, "gate_lookahead_m": 1.1,
            "joint_metrics": True, "torque_metrics": True,
            "primary_joint_objective": "measured_velocity_at_most_13_rad_s",
            "secondary_observations": ["position", "target_position", "tracking_error"],
            "constraint_early_termination": False,
        },
    )
    reserve_job(job)
    try:
        source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
        write_job(job, source_sha256=source_hash)
        if source_hash != SOURCE_SHA256:
            raise ValueError("Warm-start checkpoint SHA-256 differs from the authorized torque1000 model")
        if not WALK_CHECKPOINT.is_file():
            raise FileNotFoundError(f"Hybrid evaluation requires {WALK_CHECKPOINT}")
        wiring = json.loads(WIRING_REPORT.read_text())
        if wiring.get("status") != "passed":
            raise RuntimeError("Isaac velocity-focused wiring check has not passed")
        env = dict(os.environ, GO2_PYTHON=PYTHON)
        snapshot_source()
        with CONSOLE.open("x") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT)
            write_job(job, train_pid=process.pid, pipeline_status="training")
            code = process.wait()
        write_job(job, train_returncode=code)
        output = CONSOLE.read_text(errors="replace")
        matches = re.findall(r"^GO2_RUN_DIR=(.+)$", output, re.M)
        if code or len(matches) != 1:
            raise RuntimeError(f"Training exited with code {code}, run directories={len(matches)}; see {CONSOLE}")
        run = Path(matches[0].strip()).resolve()
        if run.parent != (ROOT / "logs").resolve() or not run.is_dir():
            raise RuntimeError(f"Unexpected training output directory: {run}")
        write_job(job, run_directory=str(run))
        snapshot_copy = run / SNAPSHOT.name
        with SNAPSHOT.open("rb") as source, snapshot_copy.open("xb") as target:
            shutil.copyfileobj(source, target)
        completed = json.loads((run / "completed.json").read_text())
        if completed.get("iterations") != 1000 or completed.get("all_model_tensors_finite") is not True:
            raise RuntimeError("Training completion count/model finiteness check failed")
        for iteration in (500, 1000):
            checkpoint = run / f"model_{iteration}.pt"
            if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
                raise FileNotFoundError(f"Expected nonempty evaluation checkpoint missing: {checkpoint}")
        write_job(job, pipeline_status="evaluating", training_completed_at_unix=time.time())
        rows = []
        checkpoints = (SOURCE, run / "model_500.pt", run / "model_1000.pt")
        with EVAL_CONSOLE.open("x") as log:
            for label, checkpoint in zip(LABELS, checkpoints):
                for height, tread, controller in CONDITIONS:
                    for seed in SEEDS:
                        stem = condition_name(label, height, controller, seed)
                        result_path = OUT / f"{stem}.json"
                        condition_log_path = OUT / f"{stem}.console.log"
                        if result_path.exists() or result_path.is_symlink():
                            raise FileExistsError(result_path)
                        cmd = ["bash", "scripts/run_lab.sh", "scripts/play_lab.py", "--headless",
                               "--checkpoint", str(checkpoint), "--controller", controller,
                               "--num-envs", "64", "--steps", "1500", "--episodes-per-env", "1",
                               "--gait-metrics", "--torque-metrics", "--joint-metrics",
                               "--gate-lookahead", "1.1", "--output", str(result_path)]
                        for key, value in {
                            "seed": seed, "terrain.rows": 1, "terrain.cols": 4,
                            "terrain.initial_level": 0, "terrain.height_min": height,
                            "terrain.height_max": height, "terrain.tread": tread,
                            "commands.forward_min": 0.6, "commands.forward_max": 0.6,
                            "targets.hip_abduction_max": 0.22, "targets.feet_lateral_min": 0.12,
                            "targets.feet_lateral_max": 0.205,
                            "joint_constraints.velocity_limit": 13.0,
                            "joint_constraints.tracking_error_limit": 0.9,
                            "joint_constraints.startup_seconds": 2.0,
                        }.items():
                            cmd += ["--set", f"{key}={value}"]
                        print("START " + stem, file=log, flush=True)
                        with condition_log_path.open("x") as condition_log:
                            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                                       stdout=condition_log, stderr=subprocess.STDOUT)
                            write_job(job, eval_pid=process.pid, evaluating_condition=stem,
                                      evaluation_console=str(condition_log_path), eval_command=cmd)
                            code = process.wait()
                        if code:
                            raise RuntimeError(f"Evaluation process failed {stem}: {code}")
                        result = json.loads(result_path.read_text())
                        if result["envs_incomplete"] or result["episodes"] != 64:
                            raise RuntimeError(f"Incomplete first-episode evaluation: {stem}")
                        rows.append(dict(
                            label=label, condition=stem, height_m=height, tread_m=tread,
                            controller=controller, seed=seed, command_m_s=0.6,
                            successes=result["successes"], episodes=result["episodes"],
                            envs_incomplete=result["envs_incomplete"],
                            joint_acceptance_passed=result["joint_acceptance_passed"],
                            torque_acceptance_passed=result["torque_acceptance_passed"],
                            joint_audit=result["joint_audit"], torque_audit=result["torque_audit"],
                            successful_episode_gait=result.get("gait_metrics_successful_episodes", {}),
                            report_path=str(result_path),
                        ))
                        print(json.dumps({"finished": stem, "successes": result["successes"],
                                          "joint_acceptance_passed": result["joint_acceptance_passed"],
                                          "torque_acceptance_passed": result["torque_acceptance_passed"]}),
                              file=log, flush=True)
                        write_job(job, evaluation_reports_completed=len(rows))
        summary = build_summary(rows)
        with SUMMARY.open("x") as stream:
            stream.write(encode(summary))
        write_job(job, pipeline_status="completed", finished_at_unix=time.time(),
                  evaluation_reports_completed=len(rows), evaluation_all_conditions_complete=True)
    except BaseException as exc:
        write_job(job, pipeline_status="failed", error=f"{type(exc).__name__}: {exc}",
                  error_traceback=traceback.format_exc(), finished_at_unix=time.time())
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
