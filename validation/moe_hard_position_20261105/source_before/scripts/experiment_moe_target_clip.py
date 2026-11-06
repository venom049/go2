#!/usr/bin/env python3
"""Offline paired MoE1000 target-clipping experiment; no robot transport.

A: original targets and raw-action history.
B: bounded targets and raw-action history (isolates clipping).
C: bounded targets and inverse-mapped sent-action history.

Uses the existing first-episode evaluator and keeps policy weights, simulator,
PD gains, motor effort clipping, protection thresholds and success unchanged.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
SOURCE = ROOT / "validation/moe_finetune_20261104/formal_run/training/model_1000.pt"
SOURCE_SHA256 = "3bad0efcc2e751e834bea1e85a5914c410a2a1ff1a1b80193426805f69e4881f"
ARMS = ("A_original", "B_clip_raw_history", "C_clip_sent_history")
SOURCE_FILES = (
    "scripts/experiment_moe_target_clip.py", "scripts/evaluate_moe_finetune.py",
    "scripts/train_moe_stairs.py", "scripts/test_robogauge_stairs.py",
    "scripts/test_robogauge_two_steps_wall.py", "go2_him/moe_stair_env.py",
    "go2_him/moe_stair_rewards.py", "go2_him/official_guard.py",
    "go2_him/moe_cts_actor.py", "go2_him/calf_speed.py", "go2_him/joint_recovery.py",
    "validation/official_joint_limits_20261104/limit_profile.json",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def decode(raw, scale, offset, clip):
    return np.clip(np.asarray(raw, dtype=np.float32) * scale + offset, clip[:, 0], clip[:, 1])


def sent_action(raw, scale, offset, legacy_clip, lower, upper):
    target = np.clip(decode(raw, scale, offset, legacy_clip), lower, upper)
    return ((target - offset) / scale).astype(np.float32)


def self_check():
    from go2_him.official_guard import load_official_profile
    profile = load_official_profile()
    offset = np.array([.1, .8, -1.5, -.1, .8, -1.5, .1, 1., -1.5, -.1, 1., -1.5])
    scale = np.full(12, .25)
    legacy = np.tile([-100., 100.], (12, 1))
    bounded = np.column_stack((profile.lower, profile.upper))
    np.testing.assert_array_equal(decode(np.zeros(12), scale, offset, bounded), offset)
    for i in range(12):
        for sign in (-1, 1):
            raw = np.zeros(12, dtype=np.float32)
            raw[i] = sign * .1
            q = decode(raw, scale, offset, bounded)
            expected = offset.copy()
            expected[i] += sign * .02500000037252903
            np.testing.assert_allclose(q, expected, atol=1e-14, rtol=0)
    rng = np.random.default_rng(42)
    for raw in np.r_[rng.normal(0, 20, (100, 12)), np.full((1, 12), 1e4), np.full((1, 12), -1e4)]:
        normalized = sent_action(raw, scale, offset, legacy, profile.lower, profile.upper)
        q = decode(normalized, scale, offset, bounded)
        assert np.all(q >= profile.lower) and np.all(q <= profile.upper)
        np.testing.assert_allclose(q, np.clip(decode(raw, scale, offset, legacy), profile.lower, profile.upper),
                                   atol=2e-7, rtol=0)
    assert profile.lower[1] != profile.lower[7] and profile.upper[1] != profile.upper[7]
    assert offset[0] == -offset[3] and offset[6] == -offset[9]
    assert offset[1] != offset[7]
    return {"passed": True, "checks": ["zero action preserves default pose", "12 positive/negative axes",
            "front/rear thigh limit asymmetry", "left/right hip default signs", "front/rear thigh defaults",
            "102 random/extreme actions remain inside exact profile bounds after float32 inverse mapping"]}


class TraceActor:
    def __init__(self, actor, env, arm, legacy_clip):
        self.actor, self.env, self.arm = actor, env, arm
        self.legacy_clip = legacy_clip.copy()
        self.rows = []
        self.previous_raw_target = env.exp.policy.offset.copy()
        self.previous_sent_target = env.exp.policy.offset.copy()

    def __call__(self, observation):
        import torch
        output = self.actor(observation)
        if output.shape != (1, 12) or not torch.isfinite(output).all():
            raise FloatingPointError("Actor emitted invalid actions")
        raw = output[0].cpu().numpy().copy()
        e, profile = self.env.exp, self.env.profile
        legacy_target = decode(raw, e.policy.scale, e.policy.offset, self.legacy_clip)
        forwarded = (sent_action(raw, e.policy.scale, e.policy.offset, self.legacy_clip,
                                 profile.lower, profile.upper)
                     if self.arm == "C_clip_sent_history" else raw.copy())
        sent_target = decode(forwarded, e.policy.scale, e.policy.offset, e.policy.clip)
        q, dq = e.data.qpos[e.qadr].copy(), e.data.qvel[e.vadr].copy()
        # Verify the observation's newest last-action term before this command.
        observed_previous = observation[0, -12:].cpu().numpy()
        np.testing.assert_allclose(observed_previous, np.clip(self.env.last_action, -100., 100.),
                                   atol=1e-6, rtol=0)
        self.rows.append({
            "time_s": float(self.env.elapsed), "observation": observation[0].cpu().numpy().copy(),
            "raw_action": raw, "forwarded_action": forwarded,
            "policy_affine_target_rad": raw * e.policy.scale + e.policy.offset,
            "raw_target_rad": legacy_target, "sent_target_rad": sent_target,
            "filter_delta_rad": sent_target - legacy_target,
            "q_rad": q, "dq_rad_s": dq,
            "raw_pd_requested_nm": e.kp * (legacy_target - q) - e.kd * dq,
            "sent_pd_requested_nm": e.kp * (sent_target - q) - e.kd * dq,
            "raw_target_slew_rad_s": (legacy_target - self.previous_raw_target) / e.control_dt,
            "sent_target_slew_rad_s": (sent_target - self.previous_sent_target) / e.control_dt,
        })
        self.previous_raw_target, self.previous_sent_target = legacy_target.copy(), sent_target.copy()
        return torch.as_tensor(forwarded[None], dtype=output.dtype, device=output.device)

    def arrays(self):
        if not self.rows:
            raise RuntimeError("No actor-call evidence")
        return {key: np.asarray([row[key] for row in self.rows]) for key in self.rows[0]}


def target_metrics(arrays, profile):
    result = {"policy_command_attempts": len(arrays["time_s"]), "per_joint": {}}
    for label in ("raw", "sent"):
        q = arrays[f"{label}_target_rad"]
        excess = np.maximum(np.maximum(profile.lower - q, q - profile.upper), 0.)
        bad = excess > 0.
        result[f"{label}_target"] = {
            "joint_excess_count": int(bad.sum()), "joint_sample_count": int(bad.size),
            "joint_excess_fraction": float(bad.mean()),
            "steps_with_any_excess": int(bad.any(axis=1).sum()),
            "step_excess_fraction": float(bad.any(axis=1).mean()),
            "max_excess_rad": float(excess.max()),
            "peak_slew_rad_s": float(np.abs(arrays[f"{label}_target_slew_rad_s"]).max()),
        }
        for i, name in enumerate(profile.names):
            result["per_joint"].setdefault(name, {})[label] = {
                "excess_count": int(bad[:, i].sum()), "sample_count": len(q),
                "max_excess_rad": float(excess[:, i].max()),
                "min_target_rad": float(q[:, i].min()), "max_target_rad": float(q[:, i].max()),
                "peak_slew_rad_s": float(np.abs(arrays[f"{label}_target_slew_rad_s"][:, i]).max()),
            }
    delta = np.abs(arrays["filter_delta_rad"])
    altered = delta > 1e-7  # excludes harmless inverse-map roundoff; raw/sent violations use strict > 0.
    result["filter"] = {"changed_joint_count": int(altered.sum()), "joint_sample_count": int(delta.size),
        "changed_step_count": int(altered.any(axis=1).sum()), "change_count_tolerance_rad": 1e-7,
        "max_delta_rad": float(delta.max()), "sum_abs_delta_rad": float(delta.sum()),
        "sum_squared_delta_rad2": float(np.square(delta).sum()),
        "max_pd_change_nm": float(np.abs(arrays["sent_pd_requested_nm"] - arrays["raw_pd_requested_nm"]).max())}
    return result


def aggregate_targets(records):
    metrics = [record["target_filter"] for record in records]
    n = sum(m["policy_command_attempts"] for m in metrics)
    result = {"policy_command_attempts": n}
    for label in ("raw", "sent"):
        rows = [m[f"{label}_target"] for m in metrics]
        count, denom = sum(r["joint_excess_count"] for r in rows), sum(r["joint_sample_count"] for r in rows)
        steps = sum(r["steps_with_any_excess"] for r in rows)
        result[f"{label}_target"] = {"joint_excess_count": count, "joint_sample_count": denom,
            "joint_excess_fraction": count / denom, "steps_with_any_excess": steps,
            "step_excess_fraction": steps / n, "max_excess_rad": max(r["max_excess_rad"] for r in rows),
            "peak_slew_rad_s": max(r["peak_slew_rad_s"] for r in rows)}
    rows = [m["filter"] for m in metrics]
    denom = sum(r["joint_sample_count"] for r in rows)
    total = sum(r["sum_abs_delta_rad"] for r in rows)
    squared = sum(r["sum_squared_delta_rad2"] for r in rows)
    result["filter"] = {"joint_sample_count": denom,
        "changed_joint_count": sum(r["changed_joint_count"] for r in rows),
        "changed_step_count": sum(r["changed_step_count"] for r in rows),
        "change_count_tolerance_rad": 1e-7, "max_delta_rad": max(r["max_delta_rad"] for r in rows),
        "mean_abs_delta_rad": total / denom, "rms_delta_rad": (squared / denom) ** .5,
        "max_pd_change_nm": max(r["max_pd_change_nm"] for r in rows)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--episodes-per-seed", type=int, default=8)
    parser.add_argument("--stair-steps", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--guard-modes", nargs="+", choices=("audit", "strict"), default=["audit", "strict"])
    parser.add_argument("--self-check-only", action="store_true")
    args = parser.parse_args()
    if args.stair_steps < 2:
        parser.error("stair-steps must be at least 2")
    checks = self_check()
    if args.self_check_only:
        print(json.dumps(checks))
        return
    if args.episodes_per_seed < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Nonnegative episode count and unique seeds required")
    import torch
    import evaluate_moe_finetune as evaluation
    from train_moe_stairs import load_policy
    from go2_him.moe_stair_env import MoEStairEnv, stair_environment_contract
    from go2_him.official_guard import load_official_profile
    if digest(SOURCE) != SOURCE_SHA256:
        raise ValueError("Source checkpoint hash differs from verified MoE1000")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    out = args.out or ROOT / "validation/moe_target_clip_20261104" / (
        "preflight_" + datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=False)
    hashes = {name: digest(ROOT / name) for name in SOURCE_FILES}
    starts = evaluation.samples(args.episodes_per_seed, args.seeds)
    manifest = {"format": "moe1000_target_clip_abc_v1", "started_unix": time.time(),
        "stair_steps": args.stair_steps, "environment_contract": stair_environment_contract(args.stair_steps),
        "source_checkpoint": str(SOURCE), "source_checkpoint_sha256": SOURCE_SHA256,
        "source_sha256": hashes, "profile": load_official_profile().metadata(),
        "self_checks": checks, "device": "cpu", "dtype": "float32", "samples": starts,
        "arms": {ARMS[0]: "Original targets; raw network-action history",
                 ARMS[1]: "Profile target clipping; raw network-action history",
                 ARMS[2]: "Profile target clipping; float32 inverse-mapped sent-action history"},
        "guard_modes": args.guard_modes, "real_robot_driven": False, "model_weights_modified": False,
        "scenario": {"steps": args.stair_steps, "height_m": .15, "tread_m": .30, "width_m": .90,
            "platform_m": .90, "front_foot_edge_distance_m": .50, "vx_command_m_s": .6,
            "physics_dt_s": .002, "control_dt_s": .02, "timeout_s": 15.},
        "history_reset": "All arms keep existing zero last-action reset; measured real-handoff history is not simulated",
        "reward_semantics": "A/B smoothness uses raw actions; C uses forwarded executed actions; return is not an across-arm performance metric",
        "target_metric_scope": "One sample per policy command attempt, including rejected strict proposals. Sent means target forwarded to environment, not proof it was applied; guard reports acceptance and physical evidence separately.",
        "guard_scope": "Unchanged official joint/target/tracking/torque audit at every pre/post physics boundary; strict rejects first violation without damping continuation. No communication, handoff, two-second watchdog or full real-controller equivalence.",
        "success_scope": "Existing full-platform ascent plus two continuous seconds stable stop; external simulation geometry supplies stop command",
        "slew_scope": "Diagnostic finite difference only; no target rate limit, first command compared with default target",
    }
    write_json(out / "manifest.json", manifest)
    actor = load_policy(SOURCE, device="cpu").float().eval()
    rows = []
    for mode in args.guard_modes:
        for arm in ARMS:
            env = MoEStairEnv(seed=0, evaluation=True, guard_mode=mode, stair_steps=args.stair_steps)
            env.ramp = 1.
            legacy_clip = env.exp.policy.clip.copy()
            if arm != "A_original":
                env.exp.policy.clip = np.column_stack((env.profile.lower, env.profile.upper))
            directory = out / mode / arm
            directory.mkdir(parents=True)
            records = []
            try:
                for index, sample in enumerate(starts):
                    traced = TraceActor(actor, env, arm, legacy_clip)
                    record = evaluation.run_episode(env, traced, sample, torch.device("cpu"))
                    arrays = traced.arrays()
                    record["target_filter"] = target_metrics(arrays, env.profile)
                    record.update(arm=arm, guard_mode=mode)
                    if index == 0:
                        np.savez_compressed(directory / "nominal_trace.npz", **arrays)
                    write_json(directory / f"episode_{index:03d}.json", record)
                    records.append(record)
                    print(json.dumps({"mode": mode, "arm": arm, "episode": index,
                        "status": record["status"], "seconds": record["seconds"],
                        "raw_target_excess_count": record["target_filter"]["raw_target"]["joint_excess_count"],
                        "sent_target_excess_count": record["target_filter"]["sent_target"]["joint_excess_count"]}), flush=True)
                guard = evaluation.aggregate_guards([r["guard"] for r in records])
                row = {"arm": arm, "guard_mode": mode, "episodes": len(records),
                    "successes": sum(r["success"] for r in records),
                    "statuses": dict(Counter(r["status"] for r in records)),
                    "target_filter": aggregate_targets(records), "guard": guard,
                    "gait_success_only": evaluation.gait_summary([r["gait_integrals"] for r in records if r["success"]]),
                    "task_and_joint_acceptance_passed": all(r["success"] for r in records)
                        and guard["joint_and_torque_acceptance_passed"] is True,
                    "nominal": {key: records[0][key] for key in ("status", "seconds", "target_filter")},
                    "episode_statuses": [r["status"] for r in records],
                    "episode_seconds": [r["seconds"] for r in records],
                }
                write_json(directory / "summary.json", row)
                rows.append(row)
            finally:
                env.exp.close()
    end_hashes = {name: digest(ROOT / name) for name in SOURCE_FILES}
    result = {"format": manifest["format"], "completed_unix": time.time(), "rows": rows,
        "stair_steps": args.stair_steps, "environment_contract": manifest["environment_contract"],
        "scenario": manifest["scenario"],
        "source_files_unchanged": hashes == end_hashes, "source_sha256_after": end_hashes,
        "checkpoint_unchanged": digest(SOURCE) == SOURCE_SHA256,
        "real_robot_driven": False, "training_performed": False,
        "finite_test_success_is_not_safety_proof": True}
    for mode in args.guard_modes:
        reference = next(row for row in rows if row["guard_mode"] == mode and row["arm"] == ARMS[0])
        for row in [r for r in rows if r["guard_mode"] == mode]:
            row["success_count_delta_from_original"] = row["successes"] - reference["successes"]
            row["paired_success_to_failure_count"] = sum(a == "success" and b != "success"
                for a, b in zip(reference["episode_statuses"], row["episode_statuses"]))
    write_json(out / "comparison.json", result)
    if not result["source_files_unchanged"] or not result["checkpoint_unchanged"]:
        raise RuntimeError("Input files changed during the run; see comparison.json")
    print(json.dumps({"output": str(out), "rows": [{key: row[key] for key in (
        "arm", "guard_mode", "successes", "episodes", "statuses", "target_filter")}
        for row in rows]}), flush=True)


if __name__ == "__main__":
    main()
