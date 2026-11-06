#!/usr/bin/env python3
"""Small Isaac Lab wiring test for the velocity-only joint reward configuration.

The deliberate out-of-range calf target is applied only to four simulated
environments. It verifies that position evidence remains available to the audit
without activating position rewards or the joint-constraint termination rule.
No policy is trained, evaluated for locomotion quality, or sent to a real robot.
"""
import argparse
import json
import sys
from pathlib import Path

import _bootstrap
from go2_him.settings import load_settings


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite existing validation output")
    settings = load_settings("configs/stairs_velocity_limited.yaml", [
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

        cfg = build_env_cfg(settings, args.device or "cuda:0", evaluation=True)
        assert cfg.go2_settings["joint_constraints"]["enabled"]
        assert not cfg.go2_settings["joint_constraints"]["terminate_on_violation"]
        assert cfg.terminations.joint_constraint is None
        env = TorqueHimlocoEnv(cfg)
        env.reset()
        window = env.scene["robot"].actuators["legs"].joint_window
        if window is None:
            raise AssertionError("Joint constraint instrumentation absent")
        required_rewards = {"velocity_margin", "velocity_excess", "torque_margin", "torque_excess"}
        disabled_position_rewards = {
            "joint_limits", "joint_deviation", "hip_deviation", "hip_abduction",
            "position_margin", "position_excess", "target_position_excess", "tracking_error",
        }
        active_rewards = set(env.reward_manager.active_terms)
        assert required_rewards.issubset(active_rewards)
        for name in required_rewards:
            assert getattr(cfg.rewards, name).weight < 0
            assert env.reward_manager.get_term_cfg(name).weight < 0
        for name in disabled_position_rewards:
            assert getattr(cfg.rewards, name).weight == 0
            # Isaac Lab versions may retain zero-weight names in active_terms.
            # Their effective manager configuration, not name membership,
            # determines whether a reward contribution is enabled.
            if name in active_rewards:
                assert env.reward_manager.get_term_cfg(name).weight == 0
        assert cfg.rewards.joint_vel.weight == 0
        if "joint_vel" in active_rewards:
            assert env.reward_manager.get_term_cfg("joint_vel").weight == 0
        # Verified against Isaac Lab 2.3 RewardManager.compute: zero-weight
        # retained terms have their _step_reward column cleared and are skipped.
        zero_reward_names = disabled_position_rewards | {"joint_vel"}
        zero_reward_columns = [
            list(env.reward_manager.active_terms).index(name)
            for name in sorted(zero_reward_names.intersection(active_rewards))
        ]
        assert "joint_constraint" not in env.termination_manager.active_terms
        action = torch.zeros((env.num_envs, 12), device=env.device)
        active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        safe_audit = JointConstraintAudit(window)
        with torch.inference_mode():
            for _ in range(8):
                result = env.step(action)
                window.require_complete()
                assert window.samples == cfg.decimation == 4
                assert bool(torch.isfinite(result[2]).all())
                assert bool((env.reward_manager._step_reward[:, zero_reward_columns] == 0).all())
                safe_audit.update(window, active)
            safe_report = safe_audit.report()
            if not safe_report["joint_limits_passed"]:
                raise AssertionError("Zero-action startup unexpectedly violated constraints: " + json.dumps(safe_report))

            # The action wrapper permits [-4, 4]. Keep its unprojected target so
            # this step verifies that position violations are observable even
            # though position penalties and joint-based termination are off.
            action[:, JOINT_NAMES.index("FL_calf_joint")] = 4.0
            result = env.step(action)
            window.require_complete()
            assert window.samples == cfg.decimation == 4
            assert bool(torch.isfinite(result[2]).all())
            assert bool((window.penalty("target_position_excess") > 0).all())
            assert bool((env.reward_manager._step_reward[:, zero_reward_columns] == 0).all())
            assert "joint_constraint" not in env.termination_manager.active_terms
            bad_audit = JointConstraintAudit(window)
            bad_audit.update(window, active)
            bad_report = bad_audit.report()
            assert not bad_report["joint_limits_passed"]
            assert bad_report["target_position_excess_joint_samples"] >= 16
            assert bad_report["physics_samples_per_joint"] == env.num_envs * cfg.decimation

        report = {
            "status": "passed",
            "scope": "Isaac Lab wiring validation only; no learning or policy-performance evaluation",
            "config": "configs/stairs_velocity_limited.yaml",
            "num_envs": env.num_envs,
            "control_steps": 9,
            "physics_substeps_per_control": cfg.decimation,
            "active_constraint_rewards": sorted(required_rewards),
            "disabled_position_rewards": sorted(disabled_position_rewards),
            "effective_position_reward_weights": {
                name: env.reward_manager.get_term_cfg(name).weight if name in active_rewards else 0.0
                for name in sorted(disabled_position_rewards)
            },
            "active_termination_terms": list(env.termination_manager.active_terms),
            "joint_constraint_termination_enabled": False,
            "zero_action_startup": safe_report,
            "deliberate_target_violation": bad_report,
            "other_termination_env_count": int(result[3].sum().item()),
            "timeout_env_count": int(result[4].sum().item()),
            "position_audit_retained_without_position_penalty": True,
            "position_reward_contributions_zero_all_test_steps": True,
            "joint_vel_regularizer_contribution_zero_all_test_steps": True,
            "real_robot_driven": False,
            "training_launched": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
        print("GO2_VELOCITY_WIRING_CHECK_PASSED " + str(args.output), flush=True)
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
