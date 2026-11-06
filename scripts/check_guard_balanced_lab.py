#!/usr/bin/env python3
"""Small Isaac Lab wiring check for the bounded, ramped guard reward.

Four simulated robots receive eight zero-action startup steps and one finite
out-of-range calf target. This checks reward wiring and unfiltered diagnostic
coverage; it does not train a policy or assess climbing success. No real robot
is connected or driven.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import _bootstrap
from go2_him.settings import ROOT, load_settings


CONFIG = "configs/stairs_guard_balanced.yaml"
OLD_CONSTRAINT_REWARDS = {
    "torque_margin", "torque_excess", "velocity_margin", "velocity_excess",
    "position_margin", "position_excess", "target_position_excess", "tracking_error",
    "joint_limits", "joint_vel", "joint_deviation", "hip_deviation", "hip_abduction",
}
RETAINED_TASK_REWARDS = {
    "tracking_lin_vel", "tracking_ang_vel", "forward_progress", "success", "failure",
    "roll_orientation", "roll_rate", "lateral_velocity", "feet_lateral", "foot_slip",
    "foot_clearance", "feet_air_time", "feet_stumble",
}


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite existing validation output")
    settings = load_settings(CONFIG, [
        "num_envs=4", "terrain.rows=1", "terrain.cols=1", "terrain.initial_level=0",
        "terrain.height_min=0", "terrain.height_max=0", "randomization.enabled=false",
    ])
    app = AppLauncher(args).app
    env = None
    try:
        import torch
        from go2_him.lab_config import build_env_cfg, JOINT_NAMES
        from go2_him.lab_torque import TorqueHimlocoEnv
        from go2_him.joint_constraints import JointConstraintAudit
        from go2_him.torque import TorqueAudit
        from go2_him.guard_balance import compute_guard_balance, guard_balance_scale

        cfg = build_env_cfg(settings, args.device or "cuda:0", evaluation=False)
        eval_cfg = build_env_cfg(settings, args.device or "cuda:0", evaluation=True)
        guard = cfg.go2_settings["guard_balance"]
        steps_per_update = cfg.go2_settings["training"]["num_steps_per_env"]
        duration = guard["ramp_updates"] * steps_per_update
        initial_scale = guard_balance_scale(0, steps_per_update, guard)
        middle_scale = guard_balance_scale(duration // 2, steps_per_update, guard)
        final_scale = guard_balance_scale(duration, steps_per_update, guard)
        later_scale = guard_balance_scale(duration * 2, steps_per_update, guard)
        eval_scale = guard_balance_scale(0, steps_per_update, eval_cfg.go2_settings["guard_balance"])
        assert guard["enabled"] and guard["ramp_updates"] > 0
        assert math.isclose(initial_scale, guard["initial_scale"])
        assert math.isclose(middle_scale, (guard["initial_scale"] + 1) / 2)
        assert final_scale == later_scale == eval_scale == 1.0
        assert cfg.go2_settings["joint_constraints"]["enabled"]
        assert cfg.go2_settings["torque"]["enabled"]
        assert not cfg.go2_settings["joint_constraints"]["terminate_on_violation"]
        assert cfg.terminations.joint_constraint is None

        env = TorqueHimlocoEnv(cfg)
        env.reset()
        actuator = env.scene["robot"].actuators["legs"]
        joint_window, torque_window = actuator.joint_window, actuator.torque_window
        assert joint_window is not None and torque_window is not None
        assert joint_window.velocity_limit == 13.0
        assert joint_window.tracking_error_limit == 0.9
        assert cfg.scene.robot.actuators["legs"].velocity_limit_sim > joint_window.velocity_limit
        active_rewards = list(env.reward_manager.active_terms)
        assert "guard_balance" in active_rewards
        assert env.reward_manager.get_term_cfg("guard_balance").weight == -1.0
        effective_weights = {}
        for name in sorted(OLD_CONSTRAINT_REWARDS):
            assert getattr(cfg.rewards, name).weight == 0
            effective_weights[name] = env.reward_manager.get_term_cfg(name).weight if name in active_rewards else 0.0
            assert effective_weights[name] == 0
        retained_weights = {}
        for name in sorted(RETAINED_TASK_REWARDS):
            assert name in active_rewards
            retained_weights[name] = env.reward_manager.get_term_cfg(name).weight
            assert retained_weights[name] == settings["rewards"][name]
            assert retained_weights[name] != 0
        zero_columns = [active_rewards.index(name) for name in OLD_CONSTRAINT_REWARDS if name in active_rewards]
        guard_column = active_rewards.index("guard_balance")
        assert "joint_constraint" not in env.termination_manager.active_terms
        action = torch.zeros((env.num_envs, 12), device=env.device)
        active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        safe_joint_audit, safe_torque_audit = JointConstraintAudit(joint_window), TorqueAudit(torque_window)
        observed_costs = []

        def check_step(result):
            joint_window.require_complete()
            torque_window.require_complete()
            assert joint_window.samples == torque_window.samples == cfg.decimation == 4
            assert bool(torch.isfinite(result[2]).all())
            assert bool((env.reward_manager._step_reward[:, zero_columns] == 0).all())
            scale = guard_balance_scale(env.common_step_counter, steps_per_update, guard)
            cost = compute_guard_balance(joint_window, torque_window, guard, scale=scale)
            assert bool(torch.isfinite(cost.total).all())
            assert bool((cost.total >= 0).all())
            assert bool((cost.total <= scale * cost.budget_total + 1e-6).all())
            step_reward = env.reward_manager._step_reward[:, guard_column]
            assert bool(torch.isfinite(step_reward).all() & (step_reward <= 0).all())
            # Isaac Lab 2.3 stores reward rates in _step_reward; the total step
            # reward is separately multiplied by step_dt by RewardManager.
            assert torch.allclose(step_reward, -cost.total, atol=1e-6, rtol=1e-5)
            logs = result[-1].get("log", {})
            assert "GuardBalance/total_cost" in logs
            assert torch.allclose(logs["GuardBalance/total_cost"], cost.total.mean(), atol=1e-6, rtol=1e-5)
            assert "JointLimits/velocity_excess_fraction" in logs
            assert "Torque/requested_excess_fraction" in logs
            observed_costs.append({"control_step": int(env.common_step_counter), "scale": float(scale),
                                   "mean_cost": float(cost.total.mean()), "max_cost": float(cost.total.max())})
            return cost

        with torch.inference_mode():
            for _ in range(8):
                result = env.step(action)
                check_step(result)
                safe_joint_audit.update(joint_window, active)
                safe_torque_audit.update(torque_window, active)
            safe_joint_report = safe_joint_audit.report()
            safe_torque_report = safe_torque_audit.report()
            if not safe_joint_report["joint_limits_passed"]:
                raise AssertionError("Zero-action startup unexpectedly violated joint constraints: " + json.dumps(safe_joint_report))

            # This finite target moves FL calf beyond its upper angle bound.
            # Keep it observable rather than silently projecting it to safety.
            action[:, JOINT_NAMES.index("FL_calf_joint")] = 4.0
            result = env.step(action)
            cost = check_step(result)
            assert bool((joint_window.penalty("target_position_excess") > 0).all())
            assert bool((cost.bounded["target_position"] > 0).all())
            assert bool((cost.total > 0).all())
            assert bool((env.reward_manager._step_reward[:, guard_column] < 0).all())
            assert "joint_constraint" not in env.termination_manager.active_terms
            assert not bool(result[3].any() | result[4].any()), "Deliberate finite target unexpectedly terminated an environment"
            bad_joint_audit, bad_torque_audit = JointConstraintAudit(joint_window), TorqueAudit(torque_window)
            bad_joint_audit.update(joint_window, active)
            bad_torque_audit.update(torque_window, active)
            bad_joint_report, bad_torque_report = bad_joint_audit.report(), bad_torque_audit.report()
            assert not bad_joint_report["joint_limits_passed"]
            assert bad_joint_report["target_position_excess_joint_samples"] >= 16
            assert bad_joint_report["physics_samples_per_joint"] == env.num_envs * cfg.decimation
            assert bad_torque_report["physics_samples_per_joint"] == env.num_envs * cfg.decimation
            full_cost = compute_guard_balance(joint_window, torque_window, guard, scale=final_scale)
            initial_cost = compute_guard_balance(joint_window, torque_window, guard, scale=initial_scale)
            assert torch.allclose(initial_cost.total, full_cost.total * initial_scale, atol=1e-6, rtol=1e-5)

        report = {
            "status": "passed", "scope": "Isaac Lab reward wiring and audit validation only; no climbing quality assessment",
            "config": CONFIG,
            "config_sha256": hashlib.sha256((ROOT / CONFIG).read_bytes()).hexdigest(),
            "num_envs": env.num_envs, "control_steps": 9,
            "physics_substeps_per_control": cfg.decimation,
            "guard_reward_weight": env.reward_manager.get_term_cfg("guard_balance").weight,
            "guard_budget_per_second": float(cost.budget_total),
            "measured_speed_acceptance_rad_s": joint_window.velocity_limit,
            "tracking_error_acceptance_rad": joint_window.tracking_error_limit,
            "simulation_velocity_limit_rad_s": cfg.scene.robot.actuators["legs"].velocity_limit_sim,
            "guard_ramp": {"initial_scale": initial_scale, "midpoint_scale": middle_scale,
                           "final_scale": final_scale, "evaluation_scale": eval_scale,
                           "ramp_updates": guard["ramp_updates"], "control_steps": duration},
            "effective_disabled_constraint_weights": effective_weights,
            "retained_task_reward_weights": retained_weights,
            "guard_cost_observations": observed_costs,
            "active_termination_terms": list(env.termination_manager.active_terms),
            "joint_constraint_termination_enabled": False,
            "zero_action_startup": {"joint_audit": safe_joint_report, "torque_audit": safe_torque_report},
            "deliberate_target_violation": {"joint_audit": bad_joint_report, "torque_audit": bad_torque_report,
                "bounded_cost_mean": {k: float(v.mean()) for k, v in cost.bounded.items()},
                "raw_cost_mean": {k: float(v.mean()) for k, v in cost.raw.items()},
                "terminated_env_count": int(result[3].sum()), "timeout_env_count": int(result[4].sum())},
            "zero_weight_contributions_zero_all_test_steps": True,
            "finite_target_violation_penalized_without_joint_termination": True,
            "precise_audit_retained": True, "real_robot_driven": False, "training_launched": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
        print("GO2_GUARD_BALANCED_WIRING_CHECK_PASSED " + str(args.output), flush=True)
    except BaseException as exc:
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        import omni.kit.app
        code = exc.code if isinstance(exc, SystemExit) and isinstance(exc.code, int) else 1
        omni.kit.app.get_app().post_quit(code)
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
