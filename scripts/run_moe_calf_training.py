#!/usr/bin/env python3
"""Continue the approved MoE1000 checkpoint for 500 calf-speed-only updates.

This offline pipeline reserves a new directory, snapshots the exact source and
checkpoint, trains once, then runs the declared two-model comparison once.
It neither deploys a model nor communicates with the physical robot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
LABELS = ("reference_moe1000", "calf1500")
UPDATES = 500
EPISODES_PER_MODEL = 129
SOURCE_FILES = (
    "scripts/train_moe_stairs.py", "scripts/evaluate_moe_finetune.py",
    "scripts/run_moe_calf_training.py", "scripts/check_moe_calf_training.py",
    "scripts/test_robogauge_stairs.py", "scripts/test_robogauge_two_steps_wall.py",
    "go2_him/moe_stair_env.py", "go2_him/moe_stair_rewards.py",
    "go2_him/official_guard.py", "go2_him/moe_cts_actor.py", "go2_him/calf_speed.py",
    "validation/official_joint_limits_20261104/limit_profile.json",
)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


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
            "--actor", str(source), "--out", str(training),
            "--resume-training", "--guard-objective", "calf_velocity",
            "--updates", str(UPDATES), "--num-envs", "128", "--workers", "32",
            "--critic-warmup", "0", "--guard-ramp-start", ".5",
            "--guard-ramp-updates", "100"]


def evaluation_command(source, final, evaluation):
    return [sys.executable, str(ROOT / "scripts/evaluate_moe_finetune.py"),
            "--actor", f"{LABELS[0]}={source}", f"{LABELS[1]}={final}",
            "--out", str(evaluation), "--episodes-per-seed", "64",
            "--seeds", "42", "43", "--device", "cuda:0"]


def validate_comparison(summary):
    """Completion means the declared evidence exists, not that either model passed."""
    rows = summary.get("rows", [])
    if len(rows) != len(LABELS) or [row.get("label") for row in rows] != list(LABELS):
        raise ValueError("Expected exactly reference_moe1000 and calf1500 comparison rows")
    for row in rows:
        if row.get("episodes") != EPISODES_PER_MODEL:
            raise ValueError("Comparison is missing declared first episodes")
        if not isinstance(row.get("calf_speed"), dict) or "calf_speed_acceptance_passed" not in row:
            raise ValueError("Comparison is missing calf-only speed evidence")
        if not isinstance(row.get("guard"), dict) or "joint_and_torque_acceptance_passed" not in row:
            raise ValueError("Comparison is missing diagnostic full-guard evidence")
    return True


def run(out):
    out = Path(out).resolve()
    # Reserving a new root, not just a job file, also protects earlier artifacts.
    out.mkdir(parents=True, exist_ok=False)
    training, evaluation = out / "training", out / "evaluation"
    final = training / "model_500.pt"
    source = out / "source_checkpoint/model_1000.pt"
    job = {
        "status": "preparing", "pid": os.getpid(), "started_unix": time.time(),
        "training_dir": str(training), "training_job": str(training / "job.json"),
        "eval_summary_path": str(evaluation / "comparison.json"),
        "source_checkpoint": str(SOURCE_CHECKPOINT), "source_actor": str(source),
        "source_actor_sha256": SOURCE_SHA256, "final_checkpoint": str(final),
        "source_update": 1000, "max_updates": UPDATES, "final_cumulative_update": 1500,
        "num_envs": 128, "workers": 32, "guard_objective": "calf_velocity",
        "guard_ramp_start": .5, "guard_ramp_updates": 100,
        "resume_actor_critic_optimizer": True, "critic_warmup": 0,
        "reference_anchor": "source MoE1000 actor",
        "calf_velocity_limit_rad_s": 15.7,
        "calf_velocity_limit_is_audit_threshold_not_state_clipping": True,
        "calf_speed_reward_budget_per_second": 4., "calf_speed_soft_ratio": .90,
        "other_joint_protection_rewards_enabled": False,
        "all_guard_audits_and_physical_torque_clipping_retained": True,
        "comparison_labels": list(LABELS), "episodes_per_model": EPISODES_PER_MODEL,
        "external_platform_stop": True, "real_robot_driven": False,
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
        eval_cmd = evaluation_command(source, final, evaluation)
        job.update(status="training", training_command=train_cmd, evaluation_command=eval_cmd,
                   source_hashes=hashes)
        write(out / "job.json", job)
        with (out / "training.log").open("x") as stream:
            subprocess.run(train_cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        if not final.is_file():
            raise RuntimeError("Trainer exited without the local-500 / cumulative-1500 checkpoint")
        verify_source(source)
        verify_source(SOURCE_CHECKPOINT)
        job.update(status="evaluating", evaluation_started_unix=time.time(),
                   final_checkpoint_sha256=digest(final))
        write(out / "job.json", job)
        with (out / "evaluation.log").open("x") as stream:
            subprocess.run(eval_cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        summary = json.loads((evaluation / "comparison.json").read_text())
        validate_comparison(summary)
        if not final.is_file() or digest(final) != job["final_checkpoint_sha256"]:
            raise RuntimeError("Final checkpoint changed or disappeared during evaluation")
        job.update(status="completed", completed_unix=time.time(), completion_verified=True,
                   final_checkpoint_exists=True, eval_summary_exists=True)
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
