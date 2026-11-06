#!/usr/bin/env python3
"""Train HIMLoco from scratch, resume, or refine this project's own checkpoint."""
import argparse
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
import _bootstrap
from go2_him.settings import ROOT, load_settings, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--run-name", default="stairs")
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", type=Path, help="Continue this project's checkpoint and optimizers")
    initialization.add_argument("--warm-start", type=Path,
        help="Refine this project's model with fresh optimizers, update count and configured exploration noise")
    parser.add_argument("--dry-run", action="store_true", help="Validate settings without importing Isaac Sim")
    if "--dry-run" in sys.argv:
        args = parser.parse_args()
        settings = load_settings(args.config, args.set)
        if args.num_envs is not None: settings["num_envs"] = args.num_envs
        if args.max_iterations is not None: settings["max_iterations"] = args.max_iterations
        validate(settings)
        print(json.dumps(settings, indent=2))
        return
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    s = load_settings(args.config, args.set)
    if args.num_envs is not None: s["num_envs"] = args.num_envs
    if args.max_iterations is not None: s["max_iterations"] = args.max_iterations
    validate(s)
    if not (ROOT / "assets/go2/urdf/go2.urdf").is_file():
        parser.error("Run python scripts/prepare_go2_asset.py first")
    checkpoint = args.resume or args.warm_start
    if checkpoint and not checkpoint.is_file(): parser.error("Checkpoint not found")
    app = AppLauncher(args).app
    env = None
    try:
        import torch
        import yaml
        from isaaclab.utils.io import dump_yaml
        from go2_him.lab_config import build_env_cfg, agent_config, JOINT_NAMES
        from go2_him.lab_torque import TorqueHimlocoEnv, TorqueHIMRunner
        from himloco_lab.rsl_rl.wrappers.himloco_vec_env_wrapper import HimlocoVecEnvWrapper
        from go2_him.validation import validate_runtime
        device = args.device or "cuda:0"
        torch.manual_seed(s["seed"])
        stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
        if Path(args.run_name).name != args.run_name: parser.error("run-name must be a directory name")
        log_dir = ROOT / "logs" / (stamp + "_" + args.run_name)
        log_dir.mkdir(parents=True, exist_ok=False)
        (log_dir / "settings.yaml").write_text(yaml.safe_dump(s, sort_keys=False))
        cfg = build_env_cfg(s, device)
        cfg.log_dir = str(log_dir)
        env = TorqueHimlocoEnv(cfg)
        wrapped = HimlocoVecEnvWrapper(env, history_length=5, privileged_history_length=0,
            clip_actions=s["control"]["action_clip"])
        report = validate_runtime(wrapped)
        (log_dir / "runtime_check.json").write_text(json.dumps(report, indent=2))
        train_cfg = agent_config(s, device)
        dump_yaml(str(log_dir / "env.yaml"), cfg)
        (log_dir / "agent.json").write_text(json.dumps(train_cfg, indent=2))
        manifest = dict(backend="Isaac Sim 5.1 / Isaac Lab 2.3", algorithm="HIMLoco",
            initialization="resume" if args.resume else ("warm_start" if args.warm_start else "random"),
            checkpoint=str(checkpoint) if checkpoint else None,
            source_model=json.loads((ROOT / "assets/go2/PROVENANCE.json").read_text()),
            joint_order=JOINT_NAMES, observation_order=["commands3", "angular_velocity3", "gravity3", "joint_pos12", "joint_vel12", "action12"],
            history_frames=6, critic_layout="policy45, base_linear_velocity3, height_scan187",
            torque_monitoring=s["torque"], torque_acceptance="Separate evaluation required; finite tensors do not certify torque",
            joint_constraints=s["joint_constraints"],
            joint_acceptance="Separate complete-episode evaluation required, including failed terminal substeps; no deployment certification",
            torch=torch.__version__, cuda=torch.version.cuda)
        (log_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        runner = TorqueHIMRunner(wrapped, train_cfg, log_dir=str(log_dir), device=device)
        if checkpoint:
            check_manifest = checkpoint.parent / "manifest.json"
            if not check_manifest.is_file() or json.loads(check_manifest.read_text()).get("algorithm") != "HIMLoco":
                raise ValueError("Loading weights requires a checkpoint with this project's HIMLoco manifest")
            source_manifest = json.loads(check_manifest.read_text())
            for field in ("joint_order", "observation_order", "history_frames", "critic_layout"):
                if source_manifest.get(field) != manifest[field]:
                    raise ValueError("Checkpoint interface mismatch: " + field)
            runner.load(str(checkpoint), load_optimizer=bool(args.resume))
            if args.warm_start:
                runner.current_learning_iteration = 0
                with torch.no_grad():
                    runner.alg.actor_critic.std.fill_(s["training"]["init_noise_std"])
                    runner.alg.actor_critic.project_noise_std_()
        print("GO2_RUN_DIR=" + str(log_dir), flush=True)
        print("INITIALIZATION=" + manifest["initialization"], flush=True)
        runner.learn(s["max_iterations"], init_at_random_ep_len=False)
        if runner.writer: runner.writer.close()
        for name, tensor in runner.alg.actor_critic.state_dict().items():
            if not torch.isfinite(tensor).all(): raise RuntimeError("Nonfinite model tensor: " + name)
        (log_dir / "completed.json").write_text(json.dumps({"iterations":runner.current_learning_iteration,
            "all_model_tensors_finite":True}, indent=2))
        print("GO2_TRAINING_COMPLETE", flush=True)
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


if __name__ == "__main__":
    main()
