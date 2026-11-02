#!/usr/bin/env python3
"""Run reproducible first-episode stability/speed comparisons on the server.

Each condition launches a fresh Isaac Lab process. Completed JSON is never
overwritten; failed/incomplete conditions stop the suite and retain their log.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--controller", choices=("stairs", "walk", "hybrid"), default="stairs")
    p.add_argument("--speeds", type=float, nargs="+", default=[.4, .6])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    p.add_argument("--height", type=float, default=.18)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--gate-lookahead", type=float, default=1.10)
    p.add_argument("--spawn-x-offset", type=float, default=0.0)
    p.add_argument("--output-dir", type=Path, default=ROOT / "validation/stable_fast")
    args = p.parse_args()
    if Path(args.label).name != args.label or not args.label or args.num_envs <= 0:
        p.error("Use a simple label and a positive environment count")
    if not args.checkpoint.is_file():
        p.error("checkpoint does not exist")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for speed in args.speeds:
            stem = f"{args.label}_{args.controller}_{args.height:g}m_{speed:g}mps_seed{seed}"
            output = (args.output_dir / f"{stem}.json").resolve()
            if output.exists():
                raise FileExistsError(f"Refusing to overwrite {output}")
            command = ["bash", str(ROOT / "scripts/run_lab.sh"), str(ROOT / "scripts/play_lab.py"),
                "--headless", "--checkpoint", str(args.checkpoint.resolve()), "--controller", args.controller,
                "--num-envs", str(args.num_envs), "--steps", "1500", "--episodes-per-env", "1",
                "--gait-metrics", "--gate-lookahead", str(args.gate_lookahead),
                "--spawn-x-offset", str(args.spawn_x_offset), "--output", str(output)]
            # Same diagnostic boundaries for every candidate, regardless of its
            # training reward dead zones. Evaluation disables domain/noise randomization.
            settings = {"seed": seed, "terrain.rows":1, "terrain.cols":4,
                "terrain.initial_level":0, "terrain.height_min": args.height, "terrain.height_max": args.height,
                "commands.forward_min": speed, "commands.forward_max": speed,
                "targets.hip_abduction_max":.22, "targets.feet_lateral_min":.12,
                "targets.feet_lateral_max":.205}
            for key, value in settings.items():
                command += ["--set", f"{key}={value}"]
            env = dict(os.environ)
            env.setdefault("GO2_PYTHON", "/opt/conda/envs/isaaclab/bin/python")
            print("START " + stem, flush=True)
            with output.with_suffix(".console.log").open("x") as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            report = json.loads(output.read_text())
            if report["envs_incomplete"] or report["episodes"] != args.num_envs:
                raise RuntimeError(f"Incomplete evaluation: {output}")
            metrics = report.get("gait_metrics_successful_episodes", {})
            print(json.dumps({"condition":stem, "successes":report["successes"],
                "episodes":report["episodes"], "stairs_speed_m_s":metrics.get("stairs_traversal_speed_mean_m_s"),
                "stairs_roll_rad":metrics.get("stairs_body_roll_rms_rad"),
                "stairs_hip_outward_rad":metrics.get("stairs_hip_outward_component_mean_rad")}), flush=True)


if __name__ == "__main__":
    main()
