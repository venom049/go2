#!/usr/bin/env python3
"""Run the authorized guard-balanced warm start and 12 evaluations once.

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
OUT = ROOT / "validation/guard_balanced_training"
JOB = OUT / "job.json"
CONSOLE = OUT / "training.console.log"
EVAL_CONSOLE = OUT / "evaluation.console.log"
SUMMARY = OUT / "comparison.json"
SNAPSHOT = OUT / "training_source_snapshot.tar.gz"
MANIFEST = OUT / "source_manifest.json"
CONFIG = ROOT / "configs/stairs_guard_balanced.yaml"
PYTHON = "/opt/conda/envs/isaaclab/bin/python"
SOURCE = ROOT / "validation/joint_constraints/source_torque1000/model_1000.pt"
SOURCE_SHA256 = "508f5435b1a9de3ffe3bbf24ea3bd700c241e97dcb8671d5dba2b6c7ed3f325f"
WIRING_REPORT = ROOT / "validation/guard_balanced_training_setup/isaac_wiring_check.json"
PROTECTION_EVIDENCE = tuple(ROOT / "validation/onboard_joint_limits_20261104" / name
                            for name in ("protection_audit.json", "protection_incidents.json", "joint_limits.json"))
WALK_CHECKPOINT = ROOT / "assets/policies/upstream_walk/walk.pt"
LABELS = ("reference_torque1000", "guard500", "guard1000")
CONDITIONS = ((0.15, 0.30, "stairs"), (0.18, 0.32, "hybrid"))
SEEDS = (42, 43)
VELOCITY_LIMIT_RAD_S = 13.0
TRACKING_LIMIT_RAD = 0.9
JOINT_KINDS = ("velocity", "position", "target_position", "tracking_error")
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
    expected = [JOB, CONSOLE, EVAL_CONSOLE, SUMMARY, SNAPSHOT, MANIFEST]
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
    hashes = {}
    evidence = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in PROTECTION_EVIDENCE}
    with tarfile.open(SNAPSHOT, "x:gz") as archive:
        for name in names:
            path = ROOT / name
            if not path.exists():
                raise FileNotFoundError(f"Missing source snapshot input: {path}")
            candidates = sorted(path.rglob("*")) if path.is_dir() else [path]
            for candidate in candidates:
                if candidate.is_file() and not any(
                        part in ("__pycache__", ".git") for part in candidate.parts):
                    relative = str(candidate.relative_to(ROOT))
                    hashes[relative] = hashlib.sha256(candidate.read_bytes()).hexdigest()
                    archive.add(candidate, arcname=relative)
        for path in PROTECTION_EVIDENCE:
            archive.add(path, arcname=str(path.relative_to(ROOT)))
    manifest = {
        "files": hashes,
        "config_path": str(CONFIG),
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "snapshot_sha256": hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest(),
        "wiring_report_path": str(WIRING_REPORT),
        "wiring_report_sha256": hashlib.sha256(WIRING_REPORT.read_bytes()).hexdigest(),
        "warm_start_sha256": SOURCE_SHA256,
        "protection_evidence": evidence,
        "protection_evidence_scope": "Read-only deployed application/controller code and incident logs; not a claim to have read proprietary motor firmware limits.",
    }
    if hashes.get(str(CONFIG.relative_to(ROOT))) != manifest["config_sha256"]:
        raise RuntimeError("Training configuration changed while snapshotting")
    with MANIFEST.open("x") as stream:
        stream.write(encode(manifest))
    return manifest


def _finite_nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _complete_episode_quota(row):
    return (type(row.get("episodes")) is int and row["episodes"] == 64
            and type(row.get("envs_incomplete")) is int and row["envs_incomplete"] == 0)


def _complete_audit_samples(audit):
    count = audit.get("physics_samples_per_joint")
    startup = audit.get("startup_physics_samples_per_joint")
    return (type(count) is int and type(startup) is int and 0 < startup <= count)


def position_bounds(name):
    """The bounds read from the deployed FirstTestGuard, keyed by joint name."""
    leg, kind, _ = name.split("_")
    if kind == "hip":
        return -1.0472, 1.0472
    if kind == "calf":
        return -2.7227, -0.83776
    return (-1.5708, 3.4907) if leg in ("FL", "FR") else (-0.5236, 4.5379)


def joint_acceptance(row):
    """Fail closed on missing, rounded, mismatched, or nonfinite evidence.

    Audits include every active physics substep and failed terminal steps.
    No measured state or requested target is clipped to establish acceptance.
    """
    audit = row.get("joint_audit")
    if not isinstance(audit, dict) or not _complete_episode_quota(row):
        return False
    if (audit.get("audit_invalid") is not False or not _complete_audit_samples(audit)
            or audit.get("velocity_limit_rad_s") != VELOCITY_LIMIT_RAD_S
            or audit.get("tracking_error_limit_rad") != TRACKING_LIMIT_RAD):
        return False
    for kind in JOINT_KINDS:
        for field in (f"{kind}_excess_joint_samples", f"startup_{kind}_excess_joint_samples"):
            if type(audit.get(field)) is not int or audit[field] != 0:
                return False
        for field in (f"{kind}_excess_fraction", f"startup_{kind}_excess_fraction"):
            if not _finite_nonnegative(audit.get(field)) or audit[field] != 0:
                return False
        peak_ratio = audit.get(f"{kind}_peak_ratio")
        if not _finite_nonnegative(peak_ratio) or peak_ratio > 1:
            return False
    joints = audit.get("joints")
    if not isinstance(joints, dict) or set(joints) != JOINT_NAMES:
        return False
    for name, joint in joints.items():
        if not isinstance(joint, dict):
            return False
        peak = joint.get("velocity_peak_rad_s")
        if not _finite_nonnegative(peak) or peak > VELOCITY_LIMIT_RAD_S:
            return False
        tracking = joint.get("tracking_error_peak_rad")
        if not _finite_nonnegative(tracking) or tracking > TRACKING_LIMIT_RAD:
            return False
        for kind in JOINT_KINDS:
            count = joint.get(f"{kind}_excess_samples")
            if type(count) is not int or count != 0:
                return False
        for kind in ("position", "target_position"):
            peak = joint.get(f"{kind}_peak_overshoot_rad")
            if not _finite_nonnegative(peak) or peak != 0:
                return False
        for field, expected in zip(("position_lower_rad", "position_upper_rad"), position_bounds(name)):
            value = joint.get(field)
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or not math.isclose(value, expected, rel_tol=0, abs_tol=1e-5)):
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


def aggregate_scene(rows):
    """Pool two seeds by actual audit sample counts, never by rounded rates."""
    episodes = sum(row["episodes"] for row in rows)
    successes = sum(row["successes"] for row in rows)
    result = {
        "successes": successes, "episodes": episodes,
        "success_rate": successes / episodes,
        "joint_acceptance_passed": all(joint_acceptance(row) for row in rows),
        "torque_acceptance_passed": all(torque_acceptance(row) for row in rows),
        "joint_audit": {}, "torque_audit": {}, "successful_episode_gait": {},
    }
    for section, kinds in (("joint_audit", JOINT_KINDS), ("torque_audit", ("requested", "applied"))):
        audits = [row.get(section, {}) for row in rows]
        counts = [audit.get("physics_samples_per_joint") for audit in audits]
        count_valid = all(type(n) is int and n > 0 for n in counts)
        total = sum(counts) if count_valid else None
        result[section]["physics_samples_per_joint"] = total
        for kind in kinds:
            values = [audit.get(f"{kind}_excess_joint_samples") for audit in audits]
            valid = count_valid and all(type(n) is int and n >= 0 for n in values)
            result[section][f"{kind}_excess_joint_samples"] = sum(values) if valid else None
            result[section][f"{kind}_excess_fraction"] = sum(values) / (12 * total) if valid else None
        field_names = (("velocity_peak_rad_s", "position_peak_overshoot_rad",
                        "target_position_peak_overshoot_rad", "tracking_error_peak_rad")
                       if section == "joint_audit" else ("requested_peak_nm", "applied_peak_nm"))
        peaks = {}
        for field in field_names:
            values = [audit.get("joints", {}).get(name, {}).get(field)
                      for audit in audits for name in JOINT_NAMES]
            peaks[field] = max(values) if all(_finite_nonnegative(v) for v in values) else None
        result[section].update(peaks)
        if section == "torque_audit":
            for kind in ("requested", "applied"):
                ratios = []
                for audit in audits:
                    for name in JOINT_NAMES:
                        joint = audit.get("joints", {}).get(name, {})
                        peak, limit = joint.get(f"{kind}_peak_nm"), joint.get("limit_nm")
                        ratios.append(peak / limit if _finite_nonnegative(peak)
                                      and _finite_nonnegative(limit) and limit > 0 else None)
                result[section][f"{kind}_peak_ratio"] = max(ratios) if all(
                    _finite_nonnegative(value) for value in ratios) else None
    metrics = ("stairs_traversal_speed_mean_m_s", "stairs_body_roll_rms_rad",
               "stairs_hip_outward_component_mean_rad", "stairs_stance_foot_slip_rms_m_s")
    for key in metrics:
        weighted_sum = count = 0
        valid = True
        for row in rows:
            gait = row.get("successful_episode_gait", {})
            n = gait.get("metric_element_counts", {}).get(key)
            value = gait.get(key)
            if row["successes"] == 0:
                continue
            if type(n) is not int or n <= 0 or not _finite_nonnegative(value):
                valid = False
                break
            weighted_sum += (value ** 2 if "_rms_" in key else value) * n
            count += n
        value = weighted_sum / count if valid and count else None
        if value is not None and "_rms_" in key:
            value = math.sqrt(value)
        result["successful_episode_gait"][key] = value
    return result


def build_summary(rows):
    expected = {
        condition_name(label, height, controller, seed): (label, height, tread, controller, seed)
        for label in LABELS for height, tread, controller in CONDITIONS for seed in SEEDS
    }
    names = [row.get("condition") for row in rows]
    complete = (len(rows) == 12 and len(set(names)) == 12
                and set(names) == expected_conditions()
                and all(_complete_episode_quota(row) for row in rows))
    if complete:
        for row in rows:
            fields = (row.get("label"), row.get("height_m"), row.get("tread_m"),
                      row.get("controller"), row.get("seed"))
            complete &= (fields == expected[row["condition"]]
                         and type(row.get("successes")) is int
                         and 0 <= row["successes"] <= row["episodes"])
    if not complete:
        raise RuntimeError("Refusing incomplete, mismatched or duplicate 12-condition comparison")
    rows = [dict(row, joint_acceptance_passed=joint_acceptance(row),
                 torque_acceptance_passed=torque_acceptance(row),
                 joint_and_torque_acceptance_passed=(joint_acceptance(row) and torque_acceptance(row)))
            for row in rows]
    scenes = []
    for height, tread, controller in CONDITIONS:
        reference = [row for row in rows if row["label"] == LABELS[0]
                     and row["height_m"] == height and row["controller"] == controller]
        reference_rate = sum(row["successes"] for row in reference) / sum(row["episodes"] for row in reference)
        for label in LABELS:
            selected = [row for row in rows if row["label"] == label
                        and row["height_m"] == height and row["controller"] == controller]
            item = aggregate_scene(selected)
            item.update(label=label, height_m=height, tread_m=tread, controller=controller,
                        seeds=list(SEEDS), reference_success_rate=reference_rate,
                        success_rate_delta_from_reference=item["success_rate"] - reference_rate,
                        task_nonregression_passed=item["success_rate"] >= reference_rate)
            scenes.append(item)
    eligible = []
    candidate_results = {}
    for label in LABELS[1:]:
        selected = [scene for scene in scenes if scene["label"] == label]
        guards = all(scene["joint_acceptance_passed"] and scene["torque_acceptance_passed"] for scene in selected)
        task = all(scene["task_nonregression_passed"] and scene["successes"] > 0 for scene in selected)
        candidate_results[label] = {"joint_and_torque_acceptance_passed": guards,
                                    "task_nonregression_passed": task,
                                    "simulation_eligible": guards and task}
        if guards and task:
            eligible.append(label)
    return {
        "source_checkpoint": str(SOURCE), "source_sha256": SOURCE_SHA256,
        "expected_conditions": sorted(expected_conditions()),
        "all_expected_conditions_complete": complete, "rows": rows,
        "scene_comparisons": scenes, "candidate_results": candidate_results,
        "constraint_eligible_candidates": eligible,
        "eligibility_requires": "All 12 expected reports complete. Every candidate condition must have finite complete audits with zero violations of 13 rad/s measured velocity, deployed actual/target angle bounds, 0.9 rad tracking error, and requested/applied per-joint effort bounds. In each stair scenario its combined two-seed success rate must be no lower than reference_torque1000 and include successful climbs. Task outcomes and guard acceptance are reported separately; 100% success is not required when the reference succeeds less often.",
        "note": "0.6 m/s is commanded speed. Gait statistics select successful first episodes and use their metric-element counts; zero successful samples produce null gait metrics. Joint and torque audits include failures and terminal substeps; pooled excess fractions use sample counts and peaks use maxima. Existing real-robot guards remain unchanged. This acceptance covers audited joint and torque quantities only, not communication watchdogs, the 45-degree body-tilt stop, or the protected two-second real-robot test duration. Eligibility is sampled simulation evidence, not real-robot certification or deployment approval. No automatic selection, policy replacement, or deployment.",
    }


def main():
    command = ["bash", "scripts/run_lab.sh", "scripts/train_lab.py", "--headless",
               "--config", "configs/stairs_guard_balanced.yaml", "--warm-start", str(SOURCE),
               "--num-envs", "1024", "--max-iterations", "1000",
               "--run-name", "stairs_guard_balanced"]
    job = dict(
        pid=os.getpid(), train_pid=None, start_iteration=0, additional_updates=1000,
        target_iteration=1000, num_envs=1024, initialization="warm_start",
        fresh_optimizers=True, tensorboard_joint_metrics=True,
        source_checkpoint=str(SOURCE), source_sha256=None,
        expected_source_sha256=SOURCE_SHA256, command=command,
        output_dir=str(OUT), console_log=str(CONSOLE), evaluation_console=str(EVAL_CONSOLE),
        run_directory=None, pipeline_status="starting", started_at_unix=time.time(),
        eval_summary_path=str(SUMMARY), source_snapshot=str(SNAPSHOT),
        source_manifest=str(MANIFEST), training_config=str(CONFIG),
        evaluation_plan={
            "iterations": [500, 1000], "reference_source_iteration": 1000,
            "labels": list(LABELS), "expected_report_count": 12,
            "seeds": list(SEEDS), "num_envs": 64, "episodes_per_env": 1,
            "conditions": [{"height": h, "tread": t, "controller": c}
                           for h, t, c in CONDITIONS],
            "command_m_s": 0.6, "gate_lookahead_m": 1.1,
            "joint_metrics": True, "torque_metrics": True,
            "guard_objectives": ["velocity_13_rad_s", "actual_angle_bounds", "target_angle_bounds",
                                 "tracking_error_0.9_rad", "requested_and_applied_torque"],
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
            raise RuntimeError("Isaac guard-balanced wiring check has not passed")
        config_hash = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
        if wiring.get("config_sha256", config_hash) != config_hash:
            raise RuntimeError("Isaac wiring report belongs to a different training configuration")
        env = dict(os.environ, GO2_PYTHON=PYTHON)
        manifest = snapshot_source()
        write_job(job, config_sha256=manifest["config_sha256"],
                  source_snapshot_sha256=manifest["snapshot_sha256"],
                  source_manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
                  wiring_report_sha256=manifest["wiring_report_sha256"],
                  protection_evidence=manifest["protection_evidence"],
                  protection_evidence_scope=manifest["protection_evidence_scope"])
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
