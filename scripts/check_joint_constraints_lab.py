#!/usr/bin/env python3
"""Small Isaac Lab wiring test; no policy training or robot connection.

The deliberate unsafe target is applied only to a four-environment simulation
to check penalty, failure classification and preservation of terminal evidence.
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
    settings = load_settings("configs/stairs_joint_limited.yaml", [
        "num_envs=4", "terrain.rows=1", "terrain.cols=1", "terrain.initial_level=0",
        "terrain.height_min=0", "terrain.height_max=0", "randomization.enabled=false"])
    app = AppLauncher(args).app
    env = None
    try:
        import torch
        from go2_him.lab_config import build_env_cfg, JOINT_NAMES
        from go2_him.lab_torque import TorqueHimlocoEnv
        from go2_him.joint_constraints import JointConstraintAudit
        cfg = build_env_cfg(settings, args.device or "cuda:0", evaluation=True)
        # Explicitly retain the training failure rule in this wiring test.
        assert cfg.go2_settings["joint_constraints"]["terminate_on_violation"]
        env = TorqueHimlocoEnv(cfg)
        env.reset()
        window = env.scene["robot"].actuators["legs"].joint_window
        if window is None:
            raise AssertionError("Joint constraint instrumentation absent")
        action = torch.zeros((env.num_envs, 12), device=env.device)
        required_rewards = set(window.PENALTIES)
        assert required_rewards.issubset(set(env.reward_manager.active_terms))
        safe_audit = JointConstraintAudit(window)
        with torch.inference_mode():
            for _ in range(8):
                result = env.step(action)
                window.require_complete()
                assert window.samples == cfg.decimation == 4
                assert torch.isfinite(result[2]).all()
                safe_audit.update(window, torch.ones(env.num_envs, dtype=torch.bool, device=env.device))
            safe_report = safe_audit.report()
            if not safe_report["joint_limits_passed"]:
                raise AssertionError("Zero-action startup unexpectedly violated constraints: " + json.dumps(safe_report))

            # Existing wrapper allows normalized actions in [-4,4]. This calf
            # target demonstrates why that alone is not a joint-position bound.
            action[:, JOINT_NAMES.index("FL_calf_joint")] = 4.0
            result = env.step(action)
            window.require_complete()
            violation = env.termination_manager.get_term("joint_constraint").clone()
            success = env.termination_manager.get_term("success").clone()
            assert bool(violation.all()) and not bool(success.any())
            assert bool(result[3].all())
            assert bool((window.penalty("target_position_excess") > 0).all())
            assert bool(torch.isfinite(result[2]).all())
            assert bool((env.episode_length_buf == 0).all())
            assert window.samples == 4  # reset's extra write did not erase it
            bad_audit = JointConstraintAudit(window)
            bad_audit.update(window, torch.ones(env.num_envs, dtype=torch.bool, device=env.device))
            bad_report = bad_audit.report()
            assert not bad_report["joint_limits_passed"]
            assert bad_report["target_position_excess_joint_samples"] >= 16
        report = {
            "status": "passed", "scope": "Isaac Lab wiring validation only; no learning or policy-performance evaluation",
            "num_envs": env.num_envs, "control_steps": 9, "physics_substeps_per_control": cfg.decimation,
            "active_joint_rewards": sorted(required_rewards),
            "zero_action_startup": safe_report,
            "deliberate_target_violation": bad_report,
            "violation_termination_all_envs": bool(violation.all()),
            "simultaneous_success_any_env": bool(success.any()),
            "terminal_evidence_survives_reset": True,
            "real_robot_driven": False, "training_launched": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
        print("GO2_JOINT_WIRING_CHECK_PASSED " + str(args.output), flush=True)
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
