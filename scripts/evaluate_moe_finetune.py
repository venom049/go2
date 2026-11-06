#!/usr/bin/env python3
"""Deterministic first-episode evaluation of MoE actors on stairs and a wall.

Example:
  python scripts/evaluate_moe_finetune.py --actor original=actor.pt \
    --actor model500=run/model_500.pt model1000=run/model_1000.pt --out eval

This program has no robot transport. The platform stop command uses simulation
ground truth, not a learned wall detector. Default: nominal + 8 starts per seed
42 and 43. --episodes-per-seed 0 runs only the nominal regression episode.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def write_json(path, data):
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def samples(episodes_per_seed, seeds):
    result = [{"kind": "nominal", "seed": None, "episode": 0, "y_m": 0., "yaw_rad": 0.}]
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for index in range(episodes_per_seed):
            # Same draw order and bounds as the original two-step evaluator.
            result.append({"kind": "perturbed", "seed": seed, "episode": index,
                           "y_m": float(rng.uniform(-.02, .02)),
                           "yaw_rad": float(rng.uniform(-math.pi / 90, math.pi / 90))})
    return result


class Telemetry:
    """Physics-endpoint metrics, including the last substep before auto-reset."""
    def __init__(self, env):
        e = env.exp
        self.last_time = 0.
        self.start_x = self.last_x = float(e.data.qpos[0])
        self.last_feet = e.data.geom_xpos[e.feet].copy()
        self.entry_time = self.exit_time = None
        self.climb_distance = float(e.platform_start)
        self.values = dict(seconds=0., physics_samples=0, roll_sq_seconds=0.,
                           hip_positive_rad_joint_seconds=0., hip_joint_seconds=0.,
                           stance_slip_sq_contact_seconds=0., stance_contact_seconds=0.)
        self.final_state = None

    def __call__(self, env):
        e, d = env.exp, env.exp.data
        t = float(env.elapsed)
        dt = t - self.last_time
        if not dt > 0:
            raise ValueError("Telemetry callback must run exactly once per advancing physics step")
        x = float(d.qpos[0])
        if self.entry_time is None and self.last_x < 0. <= x:
            self.entry_time = self.last_time + dt * (-self.last_x) / (x - self.last_x)
        if self.entry_time is not None and self.exit_time is None and self.last_x < self.climb_distance <= x:
            self.exit_time = self.last_time + dt * (self.climb_distance - self.last_x) / (x - self.last_x)
        rot = d.xmat[e.base].reshape(3, 3)
        roll = math.atan2(rot[2, 1], rot[2, 2])
        foot = d.geom_xpos[e.feet].copy()
        # Position differences over one physical substep, not a control interval.
        foot_vxy = (foot[:, :2] - self.last_feet[:, :2]) / dt
        touching, _, _, _ = e.contacts()
        mask = np.array([j in touching for j in range(4)])
        hip = d.qpos[e.qadr][::3] * np.array([1., -1., 1., -1.])
        self.values["seconds"] += dt
        self.values["physics_samples"] += 1
        self.values["roll_sq_seconds"] += roll * roll * dt
        self.values["hip_positive_rad_joint_seconds"] += float(np.maximum(hip, 0.).sum()) * dt
        self.values["hip_joint_seconds"] += 4 * dt
        self.values["stance_slip_sq_contact_seconds"] += float(np.square(foot_vxy[mask]).sum()) * dt
        self.values["stance_contact_seconds"] += int(mask.sum()) * dt
        self.final_state = {"time_s": t, "base_position_m": d.qpos[:3].tolist(),
                            "foot_centers_m": foot.tolist(), "joint_position_rad": d.qpos[e.qadr].tolist(),
                            "joint_velocity_rad_s": d.qvel[e.vadr].tolist(),
                            "horizontal_speed_m_s": float(np.linalg.norm(d.qvel[:2]))}
        self.last_time, self.last_x, self.last_feet = t, x, foot

    def result(self):
        result = dict(self.values)
        result["forward_displacement_m"] = self.last_x - self.start_x
        result["climb_entry_time_s"] = self.entry_time
        result["climb_exit_time_s"] = self.exit_time
        result["climb_distance_m"] = self.climb_distance
        result["climb_seconds"] = (self.exit_time - self.entry_time
                                    if self.entry_time is not None and self.exit_time is not None
                                    and self.exit_time > self.entry_time else None)
        return result


def gait_summary(integrals):
    """Only pass successful first-episode integrals to this function."""
    if not integrals:
        return {"successful_episodes": 0, "physics_samples": 0, "seconds": 0.,
                "climb_horizontal_speed_m_s": None, "roll_rms_rad": None,
                "hip_abduction_positive_mean_rad": None, "stance_foot_slip_rms_m_s": None,
                "mean_world_forward_speed_m_s": None}
    summed = {key: sum(row[key] for row in integrals) for key in (
        "seconds", "physics_samples", "roll_sq_seconds", "hip_positive_rad_joint_seconds",
        "hip_joint_seconds", "stance_slip_sq_contact_seconds", "stance_contact_seconds",
        "forward_displacement_m")}
    climb_durations = [row["climb_seconds"] for row in integrals if row["climb_seconds"] is not None]
    climb_distances = [row.get("climb_distance_m", .30) for row in integrals if row["climb_seconds"] is not None]
    return {"successful_episodes": len(integrals), "physics_samples": summed["physics_samples"],
            "seconds": summed["seconds"], "climb_speed_observed_episodes": len(climb_durations),
            "climb_horizontal_speed_m_s": (sum(climb_distances) / sum(climb_durations)
                                             if climb_durations else None),
            "roll_rms_rad": math.sqrt(summed["roll_sq_seconds"] / summed["seconds"]),
            "hip_abduction_positive_mean_rad": (summed["hip_positive_rad_joint_seconds"]
                                                 / summed["hip_joint_seconds"]),
            "stance_foot_slip_rms_m_s": (math.sqrt(summed["stance_slip_sq_contact_seconds"]
                                                   / summed["stance_contact_seconds"])
                                          if summed["stance_contact_seconds"] else None),
            "mean_world_forward_speed_m_s": summed["forward_displacement_m"] / summed["seconds"]}


def aggregate_guards(guards):
    """Sum counts/denominators; take maxima, never average episode percentages."""
    if not guards:
        raise ValueError("No guard evidence")
    profile = guards[0]["profile"]
    names = profile["joint_names"]
    for guard in guards:
        other = guard["profile"]
        if other["source_sha256"] != profile["source_sha256"] or other["joint_names"] != names:
            raise ValueError("Cannot compare or aggregate mismatched guard profiles/joint orders")
    keys = guards[0]["metrics"].keys()
    if any(set(g["metrics"]) != set(keys) for g in guards):
        raise ValueError("Guard metrics missing in an episode; do not interpret missing as zero")
    maximum = lambda rows, key: max((r[key] for r in rows if r.get(key) is not None), default=None)
    minimum = lambda rows, key: min((r[key] for r in rows if r.get(key) is not None), default=None)
    metrics = {}
    for key in keys:
        rows = [guard["metrics"][key] for guard in guards]
        count, denominator = sum(r["excess_count"] for r in rows), sum(r["sample_count"] for r in rows)
        per_joint = {}
        for name in names:
            jr = [row["per_joint"][name] for row in rows]
            jc, jn = sum(r["excess_count"] for r in jr), sum(r["sample_count"] for r in jr)
            per_joint[name] = {"excess_count": jc, "sample_count": jn,
                               "excess_fraction": jc / jn if jn else None,
                               "min_value": minimum(jr, "min_value"), "max_value": maximum(jr, "max_value"),
                               **{field: maximum(jr, field) for field in (
                                   "peak_absolute_value", "max_ratio", "max_excess")}}
        metrics[key] = {"value_and_excess_unit": rows[0].get("value_and_excess_unit"),
                        "excess_count": count, "sample_count": denominator,
                        "excess_fraction": count / denominator if denominator else None,
                        "observed_state_samples": sum(r["observed_state_samples"] for r in rows),
                        "unobserved_state_samples": sum(r["unobserved_state_samples"] for r in rows),
                        **{field: maximum(rows, field) for field in (
                            "peak_absolute_value", "max_ratio", "max_excess")}, "per_joint": per_joint}
    complete = all(g["all_metrics_observed"] and not g.get("incomplete_due_to_numeric_failure", False)
                   for g in guards)
    any_violation = any(row["excess_count"] for row in metrics.values())
    return {"profile": profile, "physics_state_samples": sum(g["physics_state_samples"] for g in guards),
            "terminal_physics_state_samples": sum(g["terminal_physics_state_samples"] for g in guards),
            "all_metrics_observed": complete, "metrics": metrics,
            "joint_and_torque_acceptance_passed": False if any_violation else True if complete else None,
            "episodes_passing_joint_and_torque": sum(g["joint_and_torque_acceptance_passed"] is True for g in guards),
            "incomplete_episodes": sum(g.get("incomplete_due_to_numeric_failure", False) for g in guards),
            "scope": "All first episodes, including failures; pre-send/post-integration physics boundaries; no startup grace"}


def aggregate_position_constraints(rows):
    if all(row is None for row in rows):
        return None
    if any(row is None for row in rows):
        raise ValueError('Missing hard-position diagnostics in an episode')
    result={'mode':'hard','episodes':len(rows)}
    for field in ('physics_steps','joint_endpoint_samples','intervention_steps','predictor_excess_count',
                  'accepted_excess_count','stop_generalized_impulse_sum_abs','kinetic_energy_removed_j'):
        result[field]=sum(row[field] for row in rows)
    for field in ('predictor_max_excess_rad','accepted_max_excess_rad','stop_generalized_impulse_max_abs'):
        result[field]=max(row[field] for row in rows)
    result['impulse_units']=rows[0]['impulse_units']
    return result


def run_episode(env, actor, sample, device):
    import torch
    from go2_him.moe_stair_env import JointGuardStop
    obs = env.reset(sample=sample, height=.15, vx=.6)
    e = env.exp
    initial = {"base_position_m": e.data.qpos[:3].tolist(),
               "front_foot_edge_distances_m": (-(e.data.geom_xpos[e.feet[:2], 0] + e.radii[:2])).tolist()}
    if not math.isclose(min(initial["front_foot_edge_distances_m"]), .50, abs_tol=1e-8):
        raise AssertionError("Initial front-foot edge distance differs from 50 cm")
    telemetry = Telemetry(env)
    env.physics_callback = telemetry
    try:
        with torch.inference_mode():
            for _ in range(math.ceil(15. / .02) + 2):
                action = actor(torch.as_tensor(obs[None], dtype=torch.float32, device=device))
                if action.shape != (1, 12) or not torch.isfinite(action).all():
                    raise FloatingPointError("Actor emitted invalid actions")
                obs, _, done, info = env.step(action[0].cpu().numpy())
                if done:
                    record = info["episode"]
                    break
            else:
                raise RuntimeError("Environment failed to terminate at its 15-second horizon")
    except JointGuardStop as exc:
        record = {'status':'joint_guard_stop','success':False,'seconds':float(env.elapsed),
                  'forward_m':float(e.data.qpos[0])-telemetry.start_x,
                  'return':float(env.episode_return),'guard':env.audit.summary(),
                  'guard_stop':exc.details,'external_stop_command':True,
                  'min_wall_clearance_m':float(e.wall_clearance()['signed_clearance_m'])}
    except FloatingPointError as exc:
        guard = env.audit.summary()
        guard.update(incomplete_due_to_numeric_failure=True, joint_and_torque_acceptance_passed=False)
        record = {"status": "numeric_failure", "success": False, "seconds": telemetry.last_time,
                  "forward_m": telemetry.last_x - telemetry.start_x,
                  "return": float(env.episode_return), "guard": guard, "error": str(exc),
                  "min_wall_clearance_m": (float(env.min_wall_clearance)
                                             if math.isfinite(env.min_wall_clearance) else None),
                  "invalid_physics_samples_excluded": True, "external_stop_command": True}
    finally:
        env.physics_callback = None
    if 'target_commands' not in record:
        record['target_commands']=env.target_commands.summary()
    if 'position_constraints' not in record:
        limits=getattr(e,'hard_joint_limits',None)
        record['position_constraints']=limits.summary() if limits else None
    record['startup_position_constraints']=getattr(env,'startup_position_constraints',None)
    integrals = telemetry.result()
    record.update(initial_state=initial, sample=dict(sample), final_state=telemetry.final_state,
                  gait_metrics=gait_summary([integrals]) if record["success"] else None,
                  gait_integrals=integrals if record["success"] else None)
    return record


def evaluate(args):
    import torch
    from train_moe_stairs import load_policy
    from go2_him.moe_stair_env import MoEStairEnv, stair_environment_contract
    from go2_him.official_guard import load_official_profile
    from go2_him.calf_speed import calf_speed_summary
    from go2_him.target_clip import aggregate_target_clip_summaries
    from train_moe_stairs import target_action_contract

    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    stair_steps = getattr(args, 'stair_steps', 2)
    position_constraint=getattr(args,'position_constraint','hard')
    environment_contract = stair_environment_contract(stair_steps,position_constraint)
    platform_start = (stair_steps - 1) * .30
    platform_end = platform_start + .90
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; select --device cpu explicitly")
    actors = []
    for specification in sum(args.actor, []):
        label, separator, filename = specification.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", label):
            raise ValueError("Actors require label=path; labels must be simple unique names")
        path = Path(filename).expanduser().resolve(strict=True)
        if label in [item[0] for item in actors]:
            raise ValueError(f"Duplicate actor label: {label}")
        actors.append((label, path))
    args.out.mkdir(parents=True, exist_ok=False)
    starts = samples(args.episodes_per_seed, args.seeds)
    manifest = {"format": "moe_stairs_wall_evaluation_v2", "started_unix": time.time(),
                "stair_steps": stair_steps, "environment_contract": environment_contract,
                "guard_mode":getattr(args,'guard_mode','audit'),
                "target_mode_requested":getattr(args,'target_mode','auto'),
                "target_command_scope":"Raw affine request and post-filter command proposal at every control call. Strict rejection may prevent its physical application; physical guard evidence is separate.",
                "strict_guard_scope":"Joint/target/tracking/requested and applied torque stop at first observed violating pre/post boundary. No damping continuation, communication, factory handoff, 45-degree attitude or two-second runtime gate is simulated; not full real-controller equivalence.",
                "device": str(device), "dtype": "float32", "tf32_enabled": False,
                "scenario": {"width_m": .90, "step_height_m": .15, "first_tread_depth_m": .30,
                             "steps": stair_steps, "top_platform_depth_m": .90, "wall_front_x_m": platform_end,
                             "platform_start_x_m": platform_start, "top_height_m": stair_steps * .15,
                             "front_foot_edge_initial_distance_m": .50, "timeout_s": 15.,
                             "command_forward_m_s": .6, "command_value_is_not_measured_speed": True,
                             "physics_dt_s": .002, "control_dt_s": .02,
                             "motor_torque_mode": "physical clipping", "external_platform_stop": True},
                "samples": starts, "profile": load_official_profile().metadata(),
                "actors": [{"label": label, "path": str(path), "sha256": digest(path)} for label, path in actors],
                "gait_scope": "Successful first episodes only, including approach and 2-second platform stop; no resets counted",
                "climb_speed_definition": f"{platform_start:g} m / interpolated base crossing time from first riser x=0 to platform x={platform_start:g}",
                "hip_definition": "Mean positive outward component: FL/RL +q_hip, FR/RR -q_hip, average across four joints",
                "slip_definition": "Contact-foot horizontal-speed RMS; 2 ms endpoint finite differences, contact-time weighted",
                "guard_aggregation": "Integer count sums / sample-count sums; peaks are maxima across every seed and failure",
                "real_robot_driven": False,
                "source_sha256": {path: digest(ROOT / path) for path in (
                    "scripts/evaluate_moe_finetune.py", "scripts/train_moe_stairs.py", "go2_him/moe_stair_env.py",
                    "go2_him/moe_stair_rewards.py", "go2_him/official_guard.py", "go2_him/moe_cts_actor.py",
                    "go2_him/calf_speed.py", "go2_him/joint_recovery.py", "go2_him/target_clip.py", "go2_him/hard_joint_limits.py")}}
    write_json(args.out / "manifest.json", manifest)
    env = MoEStairEnv(seed=0, evaluation=True, guard_mode=getattr(args,'guard_mode','audit'), stair_steps=stair_steps,
                     position_constraint=position_constraint)
    env.ramp = 1.  # Reward accounting only; audits and dynamics are unaffected.
    rows = []
    for label, path in actors:
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        target_mode=getattr(args,'target_mode','auto')
        if target_mode=='auto':
            target_mode='bounded' if checkpoint.get('guard_objective')=='target_clipped' else 'legacy'
        if checkpoint.get('guard_objective')=='target_clipped' and checkpoint.get('action_contract')!=target_action_contract():
            raise ValueError('Checkpoint action contract differs from the current decoder/limits/history')
        if 'environment_contract' in checkpoint and checkpoint['environment_contract'] != environment_contract:
            raise ValueError('Checkpoint training environment differs from the requested stair geometry')
        env.guard_objective='target_clipped' if target_mode=='bounded' else 'balanced'
        actor = load_policy(path, device=device).float().eval()
        directory = args.out / label
        directory.mkdir()
        reports = []
        for index, sample in enumerate(starts):
            report = run_episode(env, actor, sample, device)
            report["calf_speed"] = calf_speed_summary(report["guard"])
            report["calf_speed_acceptance_passed"] = report["calf_speed"]["calf_speed_acceptance_passed"]
            filename = f"episode_{index:03d}.json"
            report.update(actor_label=label, actor_sha256=digest(path), episode_file=f"{label}/{filename}")
            write_json(directory / filename, report)
            reports.append(report)
            print(json.dumps({"actor": label, "episode": index, "sample": sample,
                              "status": report["status"], "seconds": report["seconds"],
                              "guard_pass": report["guard"]["joint_and_torque_acceptance_passed"],
                              "calf_speed_pass": report["calf_speed_acceptance_passed"]}), flush=True)
        passed = sum(report["success"] for report in reports)
        guard = aggregate_guards([report["guard"] for report in reports])
        target_commands=aggregate_target_clip_summaries([report['target_commands'] for report in reports])
        calf_speed = calf_speed_summary(guard)
        row = {"label": label, "actor_path": str(path), "actor_sha256": digest(path),
               "successes": passed, "episodes": len(reports), "success_rate": passed / len(reports),
               "target_mode":target_mode,
               "stair_steps":stair_steps,"environment_contract":environment_contract,
               "action_contract":target_action_contract() if target_mode=='bounded' else {'kind':'legacy_raw_action_history'},
               "target_commands":target_commands,
               "position_constraints":aggregate_position_constraints([report['position_constraints'] for report in reports]),
               "startup_position_constraints":env.startup_position_constraints,
               "target_command_acceptance_passed":target_commands['sent']['sample_count']>0 and target_commands['sent']['excess_count']==0,
               "statuses": dict(Counter(report["status"] for report in reports)),
               "gait_success_only": gait_summary([report["gait_integrals"] for report in reports if report["success"]]),
               "guard": guard, "joint_and_torque_acceptance_passed": guard["joint_and_torque_acceptance_passed"],
               "calf_speed": calf_speed,
               "calf_speed_acceptance_passed": calf_speed["calf_speed_acceptance_passed"],
               "task_and_calf_speed_acceptance_passed": passed == len(reports) and calf_speed["calf_speed_acceptance_passed"] is True,
               "task_and_joint_acceptance_passed": passed == len(reports) and guard["joint_and_torque_acceptance_passed"] is True,
               "episode_files": [report["episode_file"] for report in reports]}
        rows.append(row)
        write_json(directory / "summary.json", row)
        del actor
    env.exp.close()
    comparison = {"format": manifest["format"], "scene": manifest["scenario"],
                  "stair_steps":stair_steps,"environment_contract":environment_contract,
                  "guard_mode":manifest['guard_mode'],
                  "target_mode_requested":manifest['target_mode_requested'],
                  "target_command_scope":manifest['target_command_scope'],
                  "strict_guard_scope":manifest['strict_guard_scope'],
                  "manifest": "manifest.json", "completed_unix": time.time(), "rows": rows,
                  "gait_success_only": True, "guard_covers_failed_episodes": True,
                  "no_success_means_gait_metrics_unknown": True, "external_platform_stop": True,
                  "simulation_pass_is_not_real_robot_permission": True}
    if rows:
        reference = rows[0]
        for row in rows:
            row["reference_label"] = reference["label"]
            row["success_count_delta_from_reference"] = row["successes"] - reference["successes"]
            row["task_regressed_from_reference"] = row["successes"] < reference["successes"]
    write_json(args.out / "comparison.json", comparison)
    print(json.dumps({"comparison": str(args.out / "comparison.json"),
                      "results": [{k: row[k] for k in ("label", "successes", "episodes", "calf_speed_acceptance_passed", "joint_and_torque_acceptance_passed")}
                                  for row in rows]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", action="append", nargs="+", required=True, metavar="LABEL=PATH")
    parser.add_argument("--out", "--outdir", dest="out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stair-steps", type=int, default=2)
    parser.add_argument("--position-constraint",choices=['hard','native'],default='hard')
    parser.add_argument("--guard-mode",choices=['audit','strict'],default='audit')
    parser.add_argument("--target-mode",choices=['auto','legacy','bounded'],default='auto',
                        help='auto honors target_clipped checkpoint metadata; bounded also evaluates a source actor through the new command contract')
    parser.add_argument("--episodes-per-seed", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    args = parser.parse_args()
    if args.stair_steps < 2:
        parser.error("stair-steps must be at least 2")
    if args.episodes_per_seed < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("episodes-per-seed must be nonnegative and seeds must be distinct")
    evaluate(args)


if __name__ == "__main__":
    main()
