"""Offline CPU check of a copied stair inference bundle, without robot IO.

Run as ``python -m go2_stairs.offline_check --model PATH`` from a bundle's
parent directory, or ``python -m deploy.go2_stairs.offline_check --model PATH``
from this repository. All observations below are synthetic; no SDK, networking,
sensor reading or actuator command is involved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import torch

from .policy import CONTROL_DT, DEFAULT_JOINT_POSITION, load_exported_policy


def percentile(values: list[float], quantile: float) -> float:
    """Linearly interpolated percentile, requiring no NumPy dependency."""
    ordered = sorted(values)
    position = (len(ordered)-1)*quantile
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper]-ordered[lower])*(position-lower)


def check_model(model_directory: str | Path, iterations: int = 100,
                warmup: int = 20, threads: int = 1) -> dict:
    for name, value, minimum in (("iterations", iterations, 1), ("warmup", warmup, 0),
                                 ("threads", threads, 1)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    model_directory = Path(model_directory).resolve()
    torch.set_num_threads(threads)
    # This verifies the exported file's SHA256 and the exact saved observation,
    # history, nominal pose and control contract before loading TorchScript.
    policy = load_exported_policy(model_directory)
    manifest_path = model_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    model_path = model_directory / "stairs_policy.pt"
    actual_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    command = torch.tensor([[.6, 0., 0.]], dtype=torch.float32)
    gyro = torch.zeros(1, 3)
    quaternion = torch.tensor([[1., 0., 0., 0.]])
    position = torch.tensor([DEFAULT_JOINT_POSITION], dtype=torch.float32)
    velocity = torch.zeros(1, 12)

    def infer_synthetic_frame():
        # Fixed synthetic measured state and fixed previous target. In particular,
        # a proposed output is never presented as a command sent to a real robot.
        policy.observe(command, gyro, quaternion, position, velocity,
                       previous_applied_joint_target=position)
        return policy.infer()

    durations_ms = []
    with torch.inference_mode():
        for _ in range(warmup):
            infer_synthetic_frame()
        for _ in range(iterations):
            started = time.perf_counter_ns()
            result = infer_synthetic_frame()
            durations_ms.append((time.perf_counter_ns()-started)/1e6)
    raw = result.raw_action
    target = result.joint_target
    if raw.shape != (1, 12) or target.shape != (1, 12):
        raise ValueError("Export must produce 12 raw actions and 12 proposed joint targets")
    if not bool(torch.isfinite(raw).all() and torch.isfinite(target).all()):
        raise ValueError("Non-finite synthetic policy result")
    p50, p95 = percentile(durations_ms, .50), percentile(durations_ms, .95)
    budget_ms = CONTROL_DT*1000
    return {
        "kind": "go2-stairs-offline-cpu-check-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_directory": str(model_directory),
        "checkpoint_iteration": manifest.get("checkpoint_iteration"),
        "export_sha256": actual_hash,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "export_hash_verified": actual_hash == manifest["export_sha256"],
        "load_and_contract_verified": True,
        "torch_version": str(torch.__version__),
        "platform": platform.platform(), "machine": platform.machine(),
        "torch_cpu_threads": torch.get_num_threads(),
        "input": {"source": "synthetic only", "command_m_s_m_s_rad_s": [.6, 0., 0.],
                  "joint_position": list(DEFAULT_JOINT_POSITION),
                  "previous_target": "same fixed default pose; normalized previous action is zero",
                  "gyro": [0., 0., 0.], "quaternion_wxyz": [1., 0., 0., 0.],
                  "joint_velocity": "all zero", "history_shape": list(policy.history.shape)},
        "raw_output_shape": list(raw.shape), "all_outputs_finite": True,
        "raw_output": raw[0].tolist(),
        "proposed_joint_target_rad": target[0].tolist(),
        "benchmark": {
            "scope": "observe plus CPU inference plus proposal construction; no IO or deliberate sleep",
            "warmup_iterations": warmup, "timed_iterations": iterations,
            "warmup_excluded_from_statistics": True,
            "p50_ms": p50, "p95_ms": p95,
            "minimum_ms": min(durations_ms), "maximum_ms": max(durations_ms),
            "mean_ms": sum(durations_ms)/len(durations_ms),
            "budget_ms": budget_ms, "policy_frequency_hz": 1/CONTROL_DT,
            "p95_within_budget": p95 <= budget_ms,
            "all_samples_within_budget": max(durations_ms) <= budget_ms,
            "samples_over_budget": sum(value > budget_ms for value in durations_ms),
        },
        "robot_validation": False,
        "limitations": "Synthetic CPU numerical and timing check only. Timing is not a real-time scheduling guarantee, and does not measure robot communications, control ownership, actual motor response or stair performance.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Directory containing manifest.json and stairs_policy.pt")
    parser.add_argument("--iterations", type=int, default=100, help="Timed synthetic frames after warmup (default: 100)")
    parser.add_argument("--warmup", type=int, default=20, help="Untimed warmup frames (default: 20)")
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads (default: 1)")
    parser.add_argument("--output", type=Path, help="Optionally save JSON; refuses to overwrite an existing file")
    args = parser.parse_args()
    try:
        if args.output is not None and args.output.exists():
            raise ValueError("Output report already exists; choose a new path")
        report = check_model(args.model, args.iterations, args.warmup, args.threads)
        rendered = json.dumps(report, indent=2, ensure_ascii=False)+"\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(rendered)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.error(str(exc))
    print(rendered, end="")


if __name__ == "__main__":
    main()
