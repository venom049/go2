#!/usr/bin/env python3
"""One isolated CPU MoE1000 continuation with bounded target commands.

Preserve the original checkpoint and A/B/C preflight evidence, train exactly
1000 additional updates, then compare the clipped source and both milestones
under audit and strict guards. Completion records an experiment, not a safety
or stair-success claim. No robot connection or deployment is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
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
sys.path.insert(0, str(ROOT))
SOURCE_CHECKPOINT = ROOT / "validation/moe_finetune_20261104/formal_run/training/model_1000.pt"
SOURCE_SHA256 = "3bad0efcc2e751e834bea1e85a5914c410a2a1ff1a1b80193426805f69e4881f"
PROFILE_PATH = ROOT / "validation/official_joint_limits_20261104/limit_profile.json"
PROFILE_SHA256 = "bd3ea1830ca2ff7d8eb6eda1fb0b5077e33192a1ee663f2c16f96195d948c8af"
DEFAULT_PREFLIGHT = ROOT / "validation/moe_target_clip_20261104/preflight_20261104_202254/comparison.json"
LABELS = ("reference_moe1000_clipped", "target_clip1500", "target_clip2000")
MILESTONES = (500, 1000)
UPDATES = 1000
EPISODES_PER_MODEL = 129
KINDS = ("velocity", "target_velocity", "position", "target_position", "tracking",
         "requested_torque", "applied_torque")
SOURCE_FILES = (
    "scripts/train_moe_stairs.py", "scripts/evaluate_moe_finetune.py",
    "scripts/run_moe_target_clip_training.py", "scripts/experiment_moe_target_clip.py",
    "scripts/test_robogauge_stairs.py", "scripts/test_robogauge_two_steps_wall.py",
    "go2_him/moe_stair_env.py", "go2_him/moe_stair_rewards.py", "go2_him/target_clip.py",
    "go2_him/official_guard.py", "go2_him/moe_cts_actor.py", "go2_him/calf_speed.py",
    "go2_him/joint_recovery.py", "tests/test_moe_target_clip_pipeline.py",
    "go2_him/hard_joint_limits.py", "tests/test_hard_joint_limits.py",
    "tests/test_moe_six_step_env.py", "tests/test_multistep_wall.py", "tests/test_moe_stair_telemetry.py",
    "validation/official_joint_limits_20261104/limit_profile.json",
    "upstream/robogauge/RoboGauge/resources/robots/go2/go2.xml",
    "upstream/robogauge/unitree_cpp_deploy/logs/rsl_rl/go2_moe_cts_expert_goal_137000_0.6745/params/deploy.yaml",
)
ACTION_CONTRACT = {
    "target_mode": "bounded", "guard_objective": "target_clipped",
    "raw_target": "q_default + 0.25 * raw_action",
    "sent_target": "clip(raw_target, profile.lower, profile.upper)",
    "last_action": "(sent_target - q_default) / 0.25",
    "raw_target_penalty_rate": "20 * sum(c); c is the raw-target violation defined in target_clip.py",
    "penalty_weight": 20., "penalty_evidence": "raw target before the physical target clamp",
    "reference_anchor": "source actor projected into the same feasible target interval",
    "target_slew_limiter_present": False,
    "actual_joint_state_clipping": False,
    "bounds_scope": "sent targets only; not a guarantee on actual angle, velocity, tracking or torque",
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_source(path):
    actual = digest(path)
    if actual != SOURCE_SHA256:
        raise ValueError(f"Refusing unexpected MoE1000 checkpoint: {actual}")
    return actual


def validate_resources(num_envs, workers):
    if (type(num_envs) is not int or type(workers) is not int
            or num_envs <= 0 or workers <= 0 or num_envs % workers):
        raise ValueError("Positive num-envs must be divisible by positive workers")


def environment_contract(stair_steps=2):
    from go2_him.moe_stair_env import stair_environment_contract
    return stair_environment_contract(stair_steps)


def validate_environment_metadata(data, stair_steps, context, *, allow_legacy=False):
    expected = environment_contract(stair_steps)
    if data.get("stair_steps", 2) != stair_steps:
        raise ValueError(f"{context} stair count differs from the requested environment")
    actual = data.get("environment_contract")
    if actual is None and allow_legacy and stair_steps == 2 and expected['actual_position_constraint']['mode']=='native':
        return
    if actual != expected:
        raise ValueError(f"{context} environment contract differs from the requested geometry")


def training_command(source, training, *, num_envs=128, workers=32, stair_steps=2):
    validate_resources(num_envs, workers)
    environment_contract(stair_steps)
    return [sys.executable, str(ROOT / "scripts/train_moe_stairs.py"),
            "--actor", str(source), "--out", str(training), "--resume-training",
            "--guard-objective", "target_clipped", "--updates", str(UPDATES),
            "--num-envs", str(num_envs), "--workers", str(workers), "--steps", "24",
            "--stair-steps", str(stair_steps), "--position-constraint", "hard",
            "--critic-warmup", "20", "--guard-ramp-start", ".25",
            "--guard-ramp-updates", "200", "--anchor-coef", ".2", "--device", "cpu"]


def evaluation_command(source, training, evaluation, guard_mode="audit", *, stair_steps=2):
    if guard_mode not in ("audit", "strict"):
        raise ValueError("Unknown comparison guard mode")
    environment_contract(stair_steps)
    return [sys.executable, str(ROOT / "scripts/evaluate_moe_finetune.py"),
            "--actor", f"{LABELS[0]}={source}",
            f"{LABELS[1]}={training / 'model_500.pt'}", f"{LABELS[2]}={training / 'model_1000.pt'}",
            "--out", str(evaluation), "--episodes-per-seed", "64", "--seeds", "42", "43",
            "--device", "cpu", "--guard-mode", guard_mode, "--target-mode", "bounded",
            "--stair-steps", str(stair_steps), "--position-constraint", "hard"]


def validate_preflight(path, stair_steps=2):
    result = json.loads(Path(path).read_text())
    validate_environment_metadata(result, stair_steps, "Preflight", allow_legacy=True)
    if (result.get("format") != "moe1000_target_clip_abc_v1"
            or result.get("checkpoint_unchanged") is not True
            or result.get("source_files_unchanged") is not True
            or result.get("training_performed") is not False
            or result.get("real_robot_driven") is not False):
        raise ValueError("Expected completed, unchanged offline A/B/C preflight evidence")
    expected = {(mode, arm) for mode in ("audit", "strict")
                for arm in ("A_original", "B_clip_raw_history", "C_clip_sent_history")}
    rows = result.get("rows", [])
    if len(rows) != 6 or {(r.get("guard_mode"), r.get("arm")) for r in rows} != expected:
        raise ValueError("Preflight must retain original, raw-history and sent-history arms in both modes")
    for row in rows:
        n, successes = row.get("episodes"), row.get("successes")
        if type(n) is not int or n <= 0 or type(successes) is not int or not 0 <= successes <= n:
            raise ValueError("Preflight is missing complete episode counts")
        position=row.get('guard',{}).get('metrics',{}).get('position',{})
        if (position.get('excess_count') != 0 or position.get('max_excess') != 0
                or position.get('sample_count',0) <= 0):
            raise ValueError('Preflight is missing passing actual-position hard-bound evidence')
    return result


def validate_comparison(summary, guard_mode, stair_steps=2):
    """Validate complete evidence and the sent-target invariant, not task success."""
    validate_environment_metadata(summary, stair_steps, "Comparison")
    if summary.get("guard_mode") != guard_mode or summary.get("target_mode_requested") != "bounded":
        raise ValueError("Comparison guard/target modes differ from the declared experiment")
    rows = summary.get("rows", [])
    if len(rows) != len(LABELS) or [row.get("label") for row in rows] != list(LABELS):
        raise ValueError("Comparison must include the clipped source and both trained milestones")
    for row in rows:
        if row.get("environment_contract") != environment_contract(stair_steps):
            raise ValueError("Actor comparison used a different stair environment")
        if row.get("target_mode") != "bounded":
            raise ValueError("Actor comparison did not use bounded target commands")
        if row.get("episodes") != EPISODES_PER_MODEL:
            raise ValueError("Comparison is missing declared first episodes")
        if type(row.get("successes")) is not int or not 0 <= row["successes"] <= EPISODES_PER_MODEL:
            raise ValueError("Comparison is missing task success counts")
        if not isinstance(row.get("gait_success_only"), dict):
            raise ValueError("Comparison is missing successful-episode gait evidence")
        if row.get("joint_and_torque_acceptance_passed") not in (True, False, None):
            raise ValueError("Invalid full protection acceptance result")
        if "joint_and_torque_acceptance_passed" not in row:
            raise ValueError("Missing full protection acceptance result")
        commands = row.get("target_commands", {})
        if stair_steps >= 2 and environment_contract(stair_steps)['actual_position_constraint']['mode']=='hard':
            position=row.get('guard',{}).get('metrics',{}).get('position',{})
            if position.get('excess_count') != 0 or position.get('max_excess') != 0:
                raise ValueError('Actual joint position violated hard bounds')
            hard=row.get('position_constraints')
            if not hard or hard.get('accepted_excess_count') != 0:
                raise ValueError('Missing hard-position constraint evidence')
        for kind in ("raw", "sent"):
            metric = commands.get(kind, {})
            samples, count = metric.get("sample_count"), metric.get("excess_count")
            if type(samples) is not int or samples <= 0 or type(count) is not int or not 0 <= count <= samples:
                raise ValueError(f"Missing or invalid {kind} target counts")
            fraction, peak = metric.get("excess_fraction"), metric.get("max_excess_rad")
            if (type(fraction) not in (float, int) or not math.isfinite(fraction)
                    or not math.isclose(fraction, count / samples, rel_tol=1e-9, abs_tol=1e-12)
                    or type(peak) not in (float, int) or not math.isfinite(peak) or peak < 0):
                raise ValueError(f"Invalid {kind} target fraction or peak")
        if commands["raw"]["sample_count"] != commands["sent"]["sample_count"]:
            raise ValueError("Raw and sent target samples must share a denominator")
        if commands["sent"]["excess_count"] != 0 or commands["sent"]["max_excess_rad"] != 0:
            raise ValueError("Bounded target implementation violated its sent-target invariant")
        guard = row.get("guard", {}).get("metrics", {})
        for kind in KINDS:
            metric = guard.get(kind, {})
            count, samples = metric.get("excess_count"), metric.get("sample_count")
            if type(count) is not int or type(samples) is not int or not 0 <= count <= samples:
                raise ValueError(f"Missing or invalid {kind} protection counts")
            for field in ("peak_absolute_value", "max_ratio", "max_excess"):
                value = metric.get(field)
                if field not in metric or (samples > 0 and value is None):
                    raise ValueError(f"Missing {kind}/{field} evidence")
                if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                    raise ValueError(f"Nonfinite {kind}/{field} evidence")
    return True


def validate_checkpoint(path, source, update, stair_steps=2):
    """Check the saved continuation state and actual frozen encoder tensors."""
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    original = torch.load(source, map_location="cpu", weights_only=True)
    validate_environment_metadata(checkpoint, stair_steps, "Checkpoint")
    if (checkpoint.get("format") != "moe_cts_ppo_finetune_v1"
            or checkpoint.get("update") != update
            or checkpoint.get("cumulative_update") != 1000 + update
            or checkpoint.get("guard_objective") != "target_clipped"
            or checkpoint.get("frozen_moe_encoder") is not True
            or checkpoint.get("std") != original.get("std")):
        raise ValueError("Checkpoint is not the declared target-clipped continuation milestone")
    resumed = checkpoint.get("initialization", {})
    if resumed.get("source_cumulative_update") != 1000 or any(resumed.get(key) is not True for key in (
            "full_training_state_resumed", "actor_restored", "critic_restored", "optimizer_restored")):
        raise ValueError("Checkpoint is missing full MoE1000 resume evidence")
    def inspect(value):
        if torch.is_tensor(value):
            if not torch.isfinite(value).all():
                raise ValueError("Nonfinite model or optimizer tensor in checkpoint")
        elif isinstance(value, dict):
            for child in value.values():
                inspect(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                inspect(child)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Nonfinite scalar in checkpoint")
    for name in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict"):
        if not isinstance(checkpoint.get(name), dict) or not checkpoint[name]:
            raise ValueError(f"Checkpoint missing {name}")
        inspect(checkpoint[name])
    encoder = [name for name in original["actor_state_dict"] if name.startswith("student_moe_encoder.")]
    if not encoder or any(name not in checkpoint["actor_state_dict"] or not torch.equal(
            original["actor_state_dict"][name], checkpoint["actor_state_dict"][name]) for name in encoder):
        raise ValueError("Frozen MoE encoder changed")
    return {"update": update, "cumulative_update": 1000 + update, "stair_steps": stair_steps,
            "environment_contract": environment_contract(stair_steps), "all_tensors_finite": True,
            "encoder_unchanged": True, "full_training_state_resumed": True, "sha256": digest(path)}


def verify_inputs(source, hashes, preflight, preflight_sha256):
    verify_source(source)
    verify_source(SOURCE_CHECKPOINT)
    if digest(PROFILE_PATH) != PROFILE_SHA256:
        raise RuntimeError("Audited protection profile changed")
    if digest(preflight) != preflight_sha256:
        raise RuntimeError("Saved A/B/C preflight evidence changed")
    for filename, expected in hashes.items():
        if digest(ROOT / filename) != expected:
            raise RuntimeError(f"Run source changed after snapshot: {filename}")


def acceptance_rows(summary):
    return [{"label": row["label"], "episodes": row["episodes"], "successes": row["successes"],
             "all_task_episodes_succeeded": row["successes"] == row["episodes"],
             "sent_target_bounds_passed": True,
             "actual_position_bounds_passed": row['guard']['metrics']['position']['excess_count']==0,
             "position_constraints": row.get('position_constraints'),
             "raw_target_excess_count": row["target_commands"]["raw"]["excess_count"],
             "sent_target_excess_count": row["target_commands"]["sent"]["excess_count"],
             "target_sample_count": row["target_commands"]["sent"]["sample_count"],
             "raw_target_max_excess_rad": row["target_commands"]["raw"]["max_excess_rad"],
             "joint_and_torque_acceptance_passed": row["joint_and_torque_acceptance_passed"],
             "task_and_joint_acceptance_passed": row["successes"] == row["episodes"]
             and row["joint_and_torque_acceptance_passed"] is True} for row in summary["rows"]]


def run(out, *, preflight_comparison=DEFAULT_PREFLIGHT, num_envs=128, workers=32, stair_steps=2):
    validate_resources(num_envs, workers)
    geometry = environment_contract(stair_steps)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    training, audit, strict = out / "training", out / "evaluation", out / "strict_evaluation"
    source, preflight = out / "source_checkpoint/model_1000.pt", out / "evidence/preflight_comparison.json"
    final = training / "model_1000.pt"
    job = {"status": "preparing", "pid": os.getpid(), "started_unix": time.time(),
           "training_dir": str(training), "training_job": str(training / "job.json"),
           "eval_summary_path": str(audit / "comparison.json"),
           "strict_eval_summary_path": str(strict / "comparison.json"),
           "source_checkpoint": str(SOURCE_CHECKPOINT), "source_actor": str(source),
           "source_actor_sha256": SOURCE_SHA256, "final_checkpoint": str(final),
           "source_update": 1000, "max_updates": UPDATES, "final_cumulative_update": 2000,
           "milestones": list(MILESTONES), "num_envs": num_envs, "workers": workers, "device": "cpu",
           "stair_steps": stair_steps, "environment_contract": geometry,
           "resume_actor_critic_optimizer": True, "environment_and_rng_restored": False,
           "critic_warmup": 20, "guard_ramp_start": .25, "guard_ramp_updates": 200,
           "anchor_coef": .2, "action_contract": ACTION_CONTRACT,
           "profile_source_sha256": PROFILE_SHA256, "training_calf_velocity_limit_rad_s": 15.7,
           "profile_scope": "unchanged application training profile; not firmware readback",
           "comparison_labels": list(LABELS), "episodes_per_model_per_guard_mode": EPISODES_PER_MODEL,
           "evaluation_modes": ["audit", "strict"], "preflight_source": str(Path(preflight_comparison).resolve()),
           "preflight_evidence": str(preflight), "external_platform_stop": True,
           "strict_guard_scope": "sampled joint/torque stop; excludes communication watchdogs and factory handoff",
           "target_slew_limiter_present": False, "real_robot_driven": False,
           "simulation_pass_is_not_real_robot_permission": True, "completion_verified": False}
    write(out / "job.json", job)
    try:
        verify_source(SOURCE_CHECKPOINT)
        validate_preflight(preflight_comparison, stair_steps)
        source.parent.mkdir(); preflight.parent.mkdir()
        shutil.copy2(SOURCE_CHECKPOINT, source)
        shutil.copy2(preflight_comparison, preflight)
        preflight_hash = digest(preflight)
        hashes = {}
        for filename in SOURCE_FILES:
            destination = out / "source" / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / filename, destination)
            hashes[filename] = digest(destination)
        write(out / "source_manifest.json", {"files": hashes, "source_checkpoint_sha256": SOURCE_SHA256,
                                            "preflight_comparison_sha256": preflight_hash})
        versions = {}
        for name in ("torch", "numpy", "mujoco", "onnxruntime", "PyYAML"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        write(out / "runtime.json", {"python": sys.version, "executable": sys.executable, "packages": versions})
        train_cmd = training_command(source, training, num_envs=num_envs, workers=workers,
                                     stair_steps=stair_steps)
        eval_commands = {mode: evaluation_command(source, training, path, mode, stair_steps=stair_steps)
                         for mode, path in (("audit", audit), ("strict", strict))}
        job.update(status="training", training_command=train_cmd, evaluation_command=eval_commands["audit"],
                   strict_evaluation_command=eval_commands["strict"], source_hashes=hashes,
                   preflight_comparison_sha256=preflight_hash)
        verify_inputs(source, hashes, preflight, preflight_hash)
        write(out / "job.json", job)
        with (out / "training.log").open("x") as stream:
            subprocess.run(train_cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        checkpoints = {str(training / f"model_{node}.pt"): validate_checkpoint(
            training / f"model_{node}.pt", source, node, stair_steps) for node in MILESTONES}
        verify_inputs(source, hashes, preflight, preflight_hash)
        job.update(status="evaluating", evaluation_started_unix=time.time(),
                   checkpoint_validation=checkpoints, final_checkpoint_sha256=checkpoints[str(final)]["sha256"])
        summaries = {}
        for mode, directory in (("audit", audit), ("strict", strict)):
            verify_inputs(source, hashes, preflight, preflight_hash)
            job.update(evaluation_phase=mode)
            write(out / "job.json", job)
            with (out / ("evaluation.log" if mode == "audit" else "strict_evaluation.log")).open("x") as stream:
                subprocess.run(eval_commands[mode], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
            summary = json.loads((directory / "comparison.json").read_text())
            validate_comparison(summary, mode, stair_steps)
            summaries[mode] = acceptance_rows(summary)
        verify_inputs(source, hashes, preflight, preflight_hash)
        if any(digest(path) != result["sha256"] for path, result in checkpoints.items()):
            raise RuntimeError("Checkpoint changed during evaluation")
        report = {"format": "moe_target_clip_training_validation_v1", "experiment_complete": True,
                  "source_unchanged": True, "preflight_evidence_sha256": preflight_hash,
                  "checkpoint_validation": checkpoints, "action_contract": ACTION_CONTRACT,
                  "stair_steps": stair_steps, "environment_contract": geometry,
                  "sent_target_bounds_passed": True, "actual_position_bounds_passed": True, "evaluations": summaries,
                  "final_policy_task_and_joint_acceptance_passed": all(
                      rows[-1]["task_and_joint_acceptance_passed"] for rows in summaries.values()),
                  "scope": "Finite offline experiments; command bounds do not guarantee actual-state safety",
                  "real_robot_driven": False}
        write(out / "validation_report.json", report)
        job.update(status="completed", completed_unix=time.time(), completion_verified=True,
                   final_checkpoint_exists=True, eval_summary_exists=True, strict_eval_summary_exists=True,
                   validation_report=str(out / "validation_report.json"),
                   sent_target_bounds_passed=True, actual_position_bounds_passed=True, acceptance=summaries,
                   final_policy_task_and_joint_acceptance_passed=report["final_policy_task_and_joint_acceptance_passed"])
        write(out / "job.json", job)
        return job
    except BaseException as exc:
        job.update(status="failed", completion_verified=False, error=repr(exc),
                   traceback=traceback.format_exc(), failed_unix=time.time())
        write(out / "job.json", job)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight-comparison", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--stair-steps", type=int, default=2)
    args = parser.parse_args()
    run(args.out, preflight_comparison=args.preflight_comparison, num_envs=args.num_envs,
        workers=args.workers, stair_steps=args.stair_steps)


if __name__ == "__main__":
    main()
