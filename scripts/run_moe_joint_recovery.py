#!/usr/bin/env python3
"""Resume MoE1000 for 1000 joint-recovery updates, then audit and strict tests.

All artifacts are isolated from the source model. This offline pipeline performs
one declared training run and one three-model comparison in each guard mode.
No robot connection, model replacement or deployment is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SOURCE_CHECKPOINT = ROOT / "validation/moe_finetune_20261104/formal_run/training/model_1000.pt"
SOURCE_SHA256 = "3bad0efcc2e751e834bea1e85a5914c410a2a1ff1a1b80193426805f69e4881f"
PROFILE_SHA256 = "bd3ea1830ca2ff7d8eb6eda1fb0b5077e33192a1ee663f2c16f96195d948c8af"
LABELS = ("reference_moe1000", "joint1500", "joint2000")
MILESTONES = (500, 1000)
UPDATES = 1000
EPISODES_PER_MODEL = 129
KINDS = ("velocity", "target_velocity", "position", "target_position", "tracking",
         "requested_torque", "applied_torque")
SOURCE_FILES = (
    "scripts/train_moe_stairs.py", "scripts/evaluate_moe_finetune.py",
    "scripts/run_moe_joint_recovery.py", "scripts/check_moe_joint_recovery.py",
    "scripts/test_robogauge_stairs.py", "scripts/test_robogauge_two_steps_wall.py",
    "go2_him/moe_stair_env.py", "go2_him/moe_stair_rewards.py",
    "go2_him/official_guard.py", "go2_him/moe_cts_actor.py", "go2_him/calf_speed.py",
    "go2_him/joint_recovery.py", "validation/official_joint_limits_20261104/limit_profile.json",
)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_source(path):
    actual = digest(path)
    if actual != SOURCE_SHA256:
        raise ValueError(f"Refusing unexpected MoE1000 checkpoint: {actual}")
    return actual


def training_command(source, training):
    return [sys.executable, str(ROOT / "scripts/train_moe_stairs.py"),
            "--actor", str(source), "--out", str(training), "--resume-training",
            "--guard-objective", "joint_recovery", "--updates", str(UPDATES),
            "--num-envs", "128", "--workers", "32", "--critic-warmup", "20",
            "--guard-ramp-start", ".25", "--guard-ramp-updates", "200"]


def evaluation_command(source, training, evaluation, guard_mode="audit"):
    if guard_mode not in ("audit", "strict"):
        raise ValueError("Unknown comparison guard mode")
    return [sys.executable, str(ROOT / "scripts/evaluate_moe_finetune.py"),
            "--actor", f"{LABELS[0]}={source}",
            f"{LABELS[1]}={training / 'model_500.pt'}",
            f"{LABELS[2]}={training / 'model_1000.pt'}",
            "--out", str(evaluation), "--episodes-per-seed", "64",
            "--seeds", "42", "43", "--device", "cuda:0", "--guard-mode", guard_mode]


def strict_evaluation_command(source, training, strict_evaluation):
    return evaluation_command(source, training, strict_evaluation, guard_mode="strict")


def validate_comparison(summary, guard_mode):
    """Require complete declared records, not successful safety/task acceptance.

    A strict pre-send rejection can legitimately leave applied-torque evidence
    unobserved (zero samples and null peaks). Such a result is not a safety pass.
    """
    if summary.get("guard_mode") != guard_mode:
        raise ValueError(f"Expected {guard_mode} comparison")
    rows = summary.get("rows", [])
    if len(rows) != len(LABELS) or [row.get("label") for row in rows] != list(LABELS):
        raise ValueError("Expected reference_moe1000, joint1500 and joint2000 rows")
    for row in rows:
        if row.get("episodes") != EPISODES_PER_MODEL:
            raise ValueError("Comparison is missing declared first episodes")
        successes = row.get("successes")
        if type(successes) is not int or not 0 <= successes <= EPISODES_PER_MODEL:
            raise ValueError("Comparison is missing task success counts")
        if not isinstance(row.get("gait_success_only"), dict):
            raise ValueError("Comparison is missing successful-episode gait evidence")
        guard = row.get("guard", {})
        if not isinstance(guard, dict) or "joint_and_torque_acceptance_passed" not in row:
            raise ValueError("Comparison is missing full joint/torque acceptance evidence")
        if not isinstance(guard.get("metrics"), dict) or any(kind not in guard["metrics"] for kind in KINDS):
            raise ValueError("Comparison is missing joint/torque metric records")
        for kind in KINDS:
            metric = guard["metrics"][kind]
            if not isinstance(metric, dict):
                raise ValueError(f"Invalid {kind} evidence")
            count, samples = metric.get("excess_count"), metric.get("sample_count")
            if type(count) is not int or type(samples) is not int or not 0 <= count <= samples:
                raise ValueError(f"Invalid {kind} integer counts")
            for key in ("peak_absolute_value", "max_ratio", "max_excess"):
                value = metric.get(key)
                if key not in metric or (samples > 0 and (type(value) not in (int, float) or not math.isfinite(value))):
                    raise ValueError(f"Missing or invalid {kind}/{key}")
                if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                    raise ValueError(f"Nonfinite {kind}/{key}")
    return True


def verify_inputs(source, source_hashes):
    verify_source(source)
    verify_source(SOURCE_CHECKPOINT)
    for filename, expected in source_hashes.items():
        if digest(ROOT / filename) != expected:
            raise RuntimeError(f"Run source changed after snapshot: {filename}")


def run(out):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    training, evaluation, strict = out / "training", out / "evaluation", out / "strict_evaluation"
    source = out / "source_checkpoint/model_1000.pt"
    final = training / "model_1000.pt"
    job = {
        "status": "preparing", "pid": os.getpid(), "started_unix": time.time(),
        "training_dir": str(training), "training_job": str(training / "job.json"),
        "eval_summary_path": str(evaluation / "comparison.json"),
        "strict_eval_summary_path": str(strict / "comparison.json"),
        "source_checkpoint": str(SOURCE_CHECKPOINT), "source_actor": str(source),
        "source_actor_sha256": SOURCE_SHA256, "final_checkpoint": str(final),
        "source_update": 1000, "max_updates": UPDATES, "final_cumulative_update": 2000,
        "milestones": list(MILESTONES), "num_envs": 128, "workers": 32,
        "guard_objective": "joint_recovery", "guard_ramp_start": .25, "guard_ramp_updates": 200,
        "resume_actor_critic_optimizer": True, "critic_warmup": 20,
        "environment_and_rng_restored": False, "reference_anchor": "source MoE1000 actor",
        "profile_source_sha256": PROFILE_SHA256, "training_calf_velocity_limit_rad_s": 15.7,
        "profile_note": "Source training profile retained. Its calf 15.7 rad/s threshold is stricter than the current application calf 30 rad/s setting; neither is firmware readback.",
        "all_guard_audits_and_physical_torque_clipping_retained": True,
        "comparison_labels": list(LABELS), "episodes_per_model_per_guard_mode": EPISODES_PER_MODEL,
        "evaluation_modes": ["audit", "strict"],
        "strict_guard_scope": "Joint/torque violation rejects at first pre/post physics boundary; excludes communications, two-second runtime limit and factory handoff. Not equivalent to complete real-robot protection.",
        "external_platform_stop": True, "simulation_pass_is_not_real_robot_permission": True,
        "real_robot_driven": False,
    }
    write(out / "job.json", job)
    try:
        verify_source(SOURCE_CHECKPOINT)
        source.parent.mkdir()
        shutil.copy2(SOURCE_CHECKPOINT, source)
        verify_source(source)
        hashes = {}
        for filename in SOURCE_FILES:
            destination = out / "source" / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / filename, destination)
            hashes[filename] = digest(destination)
        write(out / "source_manifest.json", {"files": hashes, "source_checkpoint_sha256": SOURCE_SHA256})
        train_cmd = training_command(source, training)
        eval_cmd = evaluation_command(source, training, evaluation)
        strict_cmd = strict_evaluation_command(source, training, strict)
        job.update(status="training", training_command=train_cmd, evaluation_command=eval_cmd,
                   strict_evaluation_command=strict_cmd, source_hashes=hashes)
        verify_inputs(source, hashes)
        write(out / "job.json", job)
        with (out / "training.log").open("x") as stream:
            subprocess.run(train_cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        checkpoints = [training / f"model_{node}.pt" for node in MILESTONES]
        if not all(path.is_file() for path in checkpoints):
            raise RuntimeError("Trainer exited without local-500/1000 (cumulative-1500/2000) checkpoints")
        verify_inputs(source, hashes)
        checkpoint_hashes = {str(path): digest(path) for path in checkpoints}
        job.update(status="evaluating", evaluation_phase="audit", evaluation_started_unix=time.time(),
                   final_checkpoint_sha256=checkpoint_hashes[str(final)], checkpoint_hashes=checkpoint_hashes)
        write(out / "job.json", job)
        for mode, command, directory in (("audit", eval_cmd, evaluation), ("strict", strict_cmd, strict)):
            verify_inputs(source, hashes)
            job.update(evaluation_phase=mode)
            write(out / "job.json", job)
            with (out / ("evaluation.log" if mode == "audit" else "strict_evaluation.log")).open("x") as stream:
                subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
            validate_comparison(json.loads((directory / "comparison.json").read_text()), mode)
        verify_inputs(source, hashes)
        if any(not Path(path).is_file() or digest(path) != expected for path, expected in checkpoint_hashes.items()):
            raise RuntimeError("Comparison checkpoint changed or disappeared during evaluation")
        job.update(status="completed", completed_unix=time.time(), completion_verified=True,
                   final_checkpoint_exists=True, eval_summary_exists=True, strict_eval_summary_exists=True)
        write(out / "job.json", job)
    except BaseException as exc:
        job.update(status="failed", error=repr(exc), traceback=traceback.format_exc(), failed_unix=time.time())
        write(out / "job.json", job)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)


if __name__ == "__main__":
    main()
