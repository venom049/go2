#!/usr/bin/env python3
"""Evaluate or interactively view a checkpoint produced by this project."""
import argparse
import json
import sys
import time
from pathlib import Path
import _bootstrap
from go2_him.settings import load_settings
from go2_him.evaluation_velocity_limits import (
    load_evaluation_velocity_profile, verify_evaluation_velocity_readback)
from go2_him.evaluation_narrow_stairs import configure as configure_narrow_stairs_evaluation


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--controller", choices=("stairs", "walk", "hybrid"), default="stairs",
        help="stairs: HIM checkpoint; walk: upstream flat policy; hybrid: terrain-ray policy switching")
    parser.add_argument("--walk-checkpoint", type=Path,
        default=Path(__file__).resolve().parents[1] / "assets/policies/upstream_walk/walk.pt")
    parser.add_argument("--gate-lookahead", type=float, default=1.10,
        help="Hybrid terrain lookahead in metres; uses a separate scanner to preserve HIM observations")
    parser.add_argument("--spawn-x-offset", type=float, default=0.0,
        help="Evaluation reset x offset, within the lower approach; useful for varying handover gait phase")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--torque-metrics", action="store_true", help="Audit raw/applied torque at every physics substep")
    parser.add_argument("--require-torque-pass", action="store_true", help="Exit nonzero on torque violations or incomplete first episodes")
    parser.add_argument("--joint-metrics", action="store_true",
        help="Audit joint position, angular velocity and target tracking at every physics substep")
    parser.add_argument("--onboard-velocity-limits", type=Path,
        help="Evaluation-only joint_limits.json URDF speed profile; enables raw joint audit at 13 rad/s")
    parser.add_argument("--stairs-flight-width", type=float,
        help="Evaluation-only width for two 15 cm / 30 cm steps with an explicitly assumed upper landing")
    parser.add_argument("--require-joint-pass", action="store_true",
        help="Exit nonzero on joint constraint violations or incomplete first episodes")
    parser.add_argument("--gait-metrics", action="store_true", help="Collect independent pre-step gait diagnostics")
    parser.add_argument("--episodes-per-env", type=int, default=0,
        help="Count at most this many episodes per environment and stop once all finish; 0 preserves unlimited counting")
    parser.add_argument("--output", type=Path, help="Evaluation JSON output path")
    parser.add_argument("--interactive", action="store_true",
        help="Keep the GUI open until closed, pace playback in real time, and skip evaluation output")
    parser.add_argument("--view", choices=("oblique", "front"), default="oblique",
        help="Interactive camera preset; front makes lateral foot spreading easier to inspect")
    parser.add_argument("--set", action="append", default=[])
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.interactive and args.headless:
        parser.error("--interactive requires a graphical session; omit --headless")
    if args.require_torque_pass and (args.interactive or args.episodes_per_env != 1):
        parser.error("--require-torque-pass requires noninteractive --episodes-per-env 1")
    if args.require_joint_pass and (args.interactive or args.episodes_per_env != 1):
        parser.error("--require-joint-pass requires noninteractive --episodes-per-env 1")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.episodes_per_env < 0:
        parser.error("--episodes-per-env must be non-negative")
    if not .4 <= args.gate_lookahead <= 2.0:
        parser.error("--gate-lookahead must be between 0.4 and 2.0 metres")
    if not -.3 <= args.spawn_x_offset <= .3:
        parser.error("--spawn-x-offset must be between -0.3 and 0.3 metres")
    if not args.checkpoint.is_file(): parser.error("Checkpoint not found")
    if args.controller != "stairs" and not args.walk_checkpoint.is_file():
        parser.error("Walking checkpoint not found; obtain the pinned upstream walk.pt first")
    run_dir = args.checkpoint.parent
    if not (run_dir / "manifest.json").is_file(): parser.error("Expected a checkpoint from this project")
    s = load_settings(run_dir / "settings.yaml", args.set + ["num_envs=" + str(args.num_envs)])
    velocity_profile = None
    if args.stairs_flight_width is not None and (args.interactive or args.controller != "stairs"):
        parser.error("--stairs-flight-width requires noninteractive stairs-only evaluation")
    if args.onboard_velocity_limits is not None:
        if args.interactive:
            parser.error("--onboard-velocity-limits requires noninteractive evaluation with a saved report")
        try:
            velocity_profile = load_evaluation_velocity_profile(args.onboard_velocity_limits)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        if s["joint_constraints"]["velocity_limit"] != 13.0:
            parser.error("--onboard-velocity-limits retains the existing 13 rad/s audit threshold")
        args.joint_metrics = True
    if args.torque_metrics or args.require_torque_pass:
        s["torque"]["enabled"] = True
    if args.joint_metrics or args.require_joint_pass:
        s["joint_constraints"]["enabled"] = True
    # Every candidate must retain the same first-episode failure/success rules.
    # Observe violations through the complete rollout instead of letting the
    # training-only joint termination shorten an unsafe candidate's audit.
    s["joint_constraints"]["terminate_on_violation"] = False
    app = AppLauncher(args).app
    env = None
    try:
        import torch
        from go2_him.lab_config import build_env_cfg, agent_config, FOOT_NAMES, JOINT_NAMES
        from go2_him.gait_metrics import EpisodeCounter, GaitMetrics
        from go2_him.lab_torque import TorqueHimlocoEnv
        from go2_him.torque import TorqueAudit
        from go2_him.joint_constraints import JointConstraintAudit
        from himloco_lab.rsl_rl.wrappers.himloco_vec_env_wrapper import HimlocoVecEnvWrapper
        from himloco_lab.rsl_rl.runners.him_on_policy_runner import HIMOnPolicyRunner
        device = args.device or "cuda:0"
        torch.manual_seed(s["seed"])
        cfg = build_env_cfg(s, device, evaluation=True)
        narrow_geometry = None
        if args.stairs_flight_width is not None:
            narrow_geometry = configure_narrow_stairs_evaluation(cfg, args.stairs_flight_width)
        if velocity_profile is not None:
            velocity_profile.apply(cfg.scene.robot.actuators["legs"])
        cfg.go2_settings["joint_constraints"]["terminate_on_violation"] = False
        cfg.terminations.joint_constraint = None
        cfg.events.reset_base.params["pose_range"]["x"] = (-.1+args.spawn_x_offset, .1+args.spawn_x_offset)
        if args.controller == "hybrid":
            import copy
            cfg.scene.transition_scanner = copy.deepcopy(cfg.scene.height_scanner)
            cfg.scene.transition_scanner.pattern_cfg.size = (2*max(.8, args.gate_lookahead+.1), 1.0)
        if args.interactive:
            cfg.viewer.origin_type = "asset_root"
            cfg.viewer.asset_name = "robot"
            cfg.viewer.env_index = 0
            # Look down steeply enough that the ascending treads do not occlude
            # the robot when the camera is ahead of it on the staircase.
            cfg.viewer.eye = (2.2, 0.6, 2.2) if args.view == "front" else (2.5, 3.2, 2.0)
            cfg.viewer.lookat = (0.3, 0.0, 0.10) if args.view == "front" else (0.6, 0.0, 0.15)
        env = TorqueHimlocoEnv(cfg)
        velocity_calibration = None
        if velocity_profile is not None:
            velocity_calibration = velocity_profile.metadata()
            velocity_calibration["initial_readback"] = verify_evaluation_velocity_readback(
                velocity_profile, env.scene["robot"],
                env.scene["robot"].actuators["legs"].joint_window.velocity_limit)
            print("GO2_VELOCITY_CALIBRATION " + json.dumps(velocity_calibration, allow_nan=False), flush=True)
        # Mixed actions encode absolute targets using the HIM offset/scale.
        # Each source controller owns its original clipping before conversion.
        wrapped = HimlocoVecEnvWrapper(env, history_length=5,
            clip_actions=s["control"]["action_clip"] if args.controller == "stairs" else None)
        runner = HIMOnPolicyRunner(wrapped, agent_config(s,device), log_dir=None, device=device)
        runner.load(str(args.checkpoint), load_optimizer=False)
        policy = runner.get_inference_policy(device=device)
        obs = wrapped.get_observations()
        controller = None
        if args.controller != "stairs":
            from go2_him.flat_policy import load_walk_actor, UpstreamWalkPolicy
            from go2_him.policy_switch import HybridLocomotionController
            walk = UpstreamWalkPolicy(load_walk_actor(args.walk_checkpoint, device),
                JOINT_NAMES, env.num_envs, env.device, policy_dt=env.step_dt)
            controller = HybridLocomotionController(env, walk, JOINT_NAMES,
                mode=args.controller, action_scale=s["control"]["action_scale"],
                action_clip=s["control"]["action_clip"], stairs_kp=s["control"]["stiffness"],
                stairs_kd=s["control"]["damping"], gate_lookahead_m=args.gate_lookahead,
                height_sensor_name="transition_scanner" if args.controller == "hybrid" else "height_scanner")
        counter = EpisodeCounter(env.num_envs, env.device, args.episodes_per_env)
        metrics = GaitMetrics(env, FOOT_NAMES, s["control"]["action_scale"], s["control"]["action_clip"]) if args.gait_metrics and not args.interactive else None
        torque_window = getattr(env.scene["robot"].actuators["legs"], "torque_window", None)
        torque_audit = TorqueAudit(torque_window) if torque_window is not None else None
        joint_window = getattr(env.scene["robot"].actuators["legs"], "joint_window", None)
        joint_audit = JointConstraintAudit(joint_window) if joint_window is not None else None
        if (args.joint_metrics or args.require_joint_pass) and joint_audit is None:
            raise RuntimeError("Requested joint audit is unavailable; refusing unaudited playback")
        previous_raw_actions = torch.zeros_like(env.action_manager.action)
        first_terminal_states = [None] * env.num_envs
        steps = 0
        print(f"GO2_PLAYBACK_READY checkpoint={args.checkpoint} controller={args.controller} interactive={args.interactive}", flush=True)
        with torch.inference_mode():
            while app.is_running() and (args.interactive or steps < args.steps):
                started = time.perf_counter()
                if metrics is not None:
                    metrics.update_pre_step(previous_raw_actions, counter.active)
                raw_actions = (torch.zeros_like(env.action_manager.action)
                               if args.controller == "walk" else policy(obs))
                if controller is not None:
                    raw_actions = controller.action(raw_actions, counter.active)
                # Preserve actual policy output before the wrapper's clipping.
                previous_raw_actions = raw_actions.clone()
                if not args.interactive:
                    data = env.scene["robot"].data
                    # Lab resets immediately inside step(), so retain the last
                    # pre-terminal state for transition debugging (within 20 ms).
                    pre_terminal = torch.cat((
                        data.root_pos_w - env.scene.env_origins,
                        data.root_lin_vel_w,
                        data.projected_gravity_b,
                        (env.episode_length_buf * env.step_dt)[:, None],
                        (controller.last_weights if controller is not None else torch.ones(env.num_envs, device=env.device))[:, None],
                    ), dim=-1).clone()
                obs, _, _, done, _, _, _ = wrapped.step(raw_actions)
                if torque_audit is not None:
                    torque_audit.update(torque_window, counter.active)
                if joint_audit is not None:
                    # Keep the terminal physics state, including failures, and
                    # stop counting only after this environment's first done.
                    joint_audit.update(joint_window, counter.active)
                if controller is not None:
                    controller.reset(done)
                steps += 1
                if args.interactive:
                    time.sleep(max(0.0, env.step_dt - (time.perf_counter() - started)))
                    continue
                # v2.3.2 get_term returns the current step termination buffer, retained across reset.
                first_finished = done.bool() & counter.active & (counter.completed == 0)
                for env_id in first_finished.nonzero(as_tuple=False).flatten().tolist():
                    state = pre_terminal[env_id].cpu().tolist()
                    first_terminal_states[env_id] = {
                        "relative_position_m": state[:3], "world_velocity_m_s": state[3:6],
                        "projected_gravity": state[6:9], "elapsed_s": state[9], "stairs_weight": state[10],
                        "success": bool(env.termination_manager.get_term("success")[env_id]),
                        "time_out": bool(env.reset_time_outs[env_id]),
                        "terminated": bool(env.reset_terminated[env_id]),
                    }
                counter.update(done, env.termination_manager.get_term("success"), env.reset_time_outs, env.reset_terminated)
                if steps % 250 == 0:
                    print("GO2_EVAL_PROGRESS " + json.dumps({"step": steps,
                        "completed": int(counter.completed.sum().item()),
                        "successes": int(counter.successes.sum().item())}), flush=True)
                if counter.all_finished():
                    break
        if args.interactive:
            return
        report = dict(steps=steps, max_steps=args.steps, **counter.report(), checkpoint=str(args.checkpoint), settings=s)
        if narrow_geometry is not None:
            report["narrow_stairs_geometry"] = narrow_geometry
        if velocity_profile is not None:
            velocity_calibration["final_readback"] = verify_evaluation_velocity_readback(
                velocity_profile, env.scene["robot"],
                env.scene["robot"].actuators["legs"].joint_window.velocity_limit)
            report["joint_velocity_calibration"] = velocity_calibration
        if torque_audit is not None:
            report["torque_audit"] = torque_audit.report()
            report["torque_acceptance_passed"] = counter.all_finished() and report["torque_audit"]["torque_limits_passed"]
        if joint_audit is not None:
            report["joint_audit"] = joint_audit.report()
            report["joint_acceptance_passed"] = (
                counter.all_finished() and report["episodes"] > 0
                and report["joint_audit"]["joint_limits_passed"])
            report["joint_evaluation"] = {
                "settings": dict(cfg.go2_settings["joint_constraints"]),
                "termination_policy": "observe_without_joint_constraint_early_termination",
                "sample_selection": "all_active_episodes_including_failed_terminal_substeps",
                "acceptance_requires": "complete_nonempty_episode_quota_and_zero_joint_limit_violations",
            }
        report["controller"] = args.controller
        report["spawn_x_offset_m"] = args.spawn_x_offset
        report["first_terminal_pre_step_states"] = first_terminal_states
        if controller is not None:
            import hashlib
            report["policy_switch"] = controller.report()
            report["walk_checkpoint"] = str(args.walk_checkpoint)
            report["walk_checkpoint_sha256"] = hashlib.sha256(args.walk_checkpoint.read_bytes()).hexdigest()
        report["stop_reason"] = "episode_target_reached" if counter.all_finished() else ("step_limit" if steps >= args.steps else "application_closed")
        if metrics is not None:
            report["gait_metrics"] = metrics.report()
            if args.episodes_per_env == 1:
                successful_metrics = metrics.report(counter.successes > 0)
                successful_metrics["selection"] = "completed_successful_first_episodes_only"
                report["gait_metrics_successful_episodes"] = successful_metrics
        output = args.output or run_dir / (args.checkpoint.stem + "_evaluation.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(report, indent=2, allow_nan=False)
        output.write_text(encoded)
        print(encoded)
        failed_acceptance = []
        if args.require_torque_pass and not report.get("torque_acceptance_passed", False):
            failed_acceptance.append("torque")
        if args.require_joint_pass and not report.get("joint_acceptance_passed", False):
            failed_acceptance.append("joint")
        if failed_acceptance:
            raise SystemExit("Acceptance FAILED (" + ", ".join(failed_acceptance) + "); see saved evaluation report")
    except BaseException as exc:
        # Kit shutdown can terminate Python before an uncaught traceback prints.
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        # Preserve failure status across Kit's fast shutdown (which otherwise
        # terminates the process before Python propagates the exception).
        import omni.kit.app
        code = exc.code if isinstance(exc, SystemExit) and isinstance(exc.code, int) else 1
        omni.kit.app.get_app().post_quit(code)
        raise
    finally:
        if env is not None: env.close()
        app.close()


if __name__ == "__main__": main()
