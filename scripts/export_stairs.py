#!/usr/bin/env python3
"""Export a hash-pinned stair checkpoint, offline and on the CPU.

This performs numerical comparisons against the actual vendored HIMActorCritic.
It neither imports Isaac Sim nor connects to or controls a robot. The output is
a raw 12-action TorchScript network, with its observation/control contract.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstream/himloco_lab/source/himloco_lab"))

import torch
import yaml

from deploy.go2_stairs.policy import (
    CONTROL_DT, DEFAULT_JOINT_POSITION, HISTORY_DIM, HISTORY_FRAMES,
    JOINT_NAMES, OBSERVATION_DIM, OBSERVATION_ORDER, StairsPolicy, load_checkpoint,
)
from himloco_lab.rsl_rl.modules import HIMActorCritic


DEFAULT_CHECKPOINT = ROOT / "artifacts/20261102_195635_stairs_stable_fast_continue/model_1100.pt"
CHECKPOINT_SHA256 = "877ff67ce7619516fa0e9350eccdd6537e7170640e5ef0a8b095a3624b2c27f5"
SOURCE_HASHES = {
    "settings.yaml": "3e59e5c617c619353df1015dd416eee650b584937baba1c1f788eb102de4e40b",
    "agent.json": "854134be9e3ed423a972a03fabc44861f75b4cf415dc6803c0c3ef32e24aad98",
    "manifest.json": "c2735d5c570c3b4a5f14d7ae3a657eb63de267aa6ca7bb9d80626613a3bd93a4",
    # Hash the saved Isaac configuration without executing its Python YAML tags.
    "env.yaml": "9c3066c2b10823acde782b1c9319829ab2da72b4ee6cb5dab3432d7f84037a8c",
}
EXPECTED_CONTROL = {"stiffness": 40., "damping": 1., "action_scale": .25, "action_clip": 4.}
SDK_JOINT_NAMES = tuple(f"{leg}_{joint}_joint" for leg in ("FR", "FL", "RR", "RL")
                        for joint in ("hip", "thigh", "calf"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source_record(source_record: str | Path | dict | None = None) -> dict:
    """Freeze an explicit identity, or retain the original strict 1100 selection.

    A source record only selects the bytes to inspect; it never relaxes the
    observation, control, architecture or completed-update checks below.
    """
    if source_record is None:
        record = {"checkpoint_sha256": CHECKPOINT_SHA256, "checkpoint_iteration": 1100,
                  "source_hashes": dict(SOURCE_HASHES)}
    elif isinstance(source_record, dict):
        record = source_record
    else:
        record = json.loads(Path(source_record).read_text())
    required = {"checkpoint_sha256", "checkpoint_iteration", "source_hashes"}
    if not isinstance(record, dict) or set(record) != required:
        raise ValueError("Source record must contain checkpoint_sha256, checkpoint_iteration and source_hashes")
    iteration = record["checkpoint_iteration"]
    if type(iteration) is not int or iteration < 1:
        raise ValueError("Source checkpoint_iteration must be a positive integer")
    hashes = record["source_hashes"]
    if not isinstance(hashes, dict) or set(hashes) != set(SOURCE_HASHES):
        raise ValueError("Source record must pin settings.yaml, agent.json, manifest.json and env.yaml")
    for digest in (record["checkpoint_sha256"], *hashes.values()):
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("Source record SHA256 values must be 64 lowercase hexadecimal characters")
    return {"checkpoint_sha256": record["checkpoint_sha256"],
            "checkpoint_iteration": iteration, "source_hashes": dict(hashes)}


def validate_source(checkpoint: Path, source_record: str | Path | dict | None = None) -> tuple[dict, dict]:
    """Reject a different model or settings before producing deployment files."""
    identity = load_source_record(source_record)
    if sha256(checkpoint) != identity["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA256 differs from the selected source record")
    for filename, expected in identity["source_hashes"].items():
        if sha256(checkpoint.parent / filename) != expected:
            raise ValueError(f"{filename} SHA256 differs from the saved training contract")
    settings = yaml.safe_load((checkpoint.parent / "settings.yaml").read_text())
    agent = json.loads((checkpoint.parent / "agent.json").read_text())
    manifest = json.loads((checkpoint.parent / "manifest.json").read_text())
    if settings.get("control") != EXPECTED_CONTROL:
        raise ValueError("Saved training control settings do not match the deployment interface")
    if (manifest.get("joint_order") != list(JOINT_NAMES)
            or manifest.get("observation_order") != list(OBSERVATION_ORDER)
            or manifest.get("history_frames") != HISTORY_FRAMES
            or manifest.get("critic_layout") != "policy45, base_linear_velocity3, height_scan187"):
        raise ValueError("Saved observation, history, or joint contract does not match deployment")
    policy = agent["policy"]
    if (policy.get("actor_hidden_dims") != [512, 256, 128]
            or policy.get("critic_hidden_dims") != [512, 256, 128]
            or policy.get("activation") != "elu"
            or agent["algorithm"].get("action_clip") != EXPECTED_CONTROL["action_clip"]):
        raise ValueError("Saved HIM network architecture does not match deployment")
    return settings, agent


def load_reference(checkpoint: Path, agent: dict, expected_iteration: int = 1100) -> HIMActorCritic:
    """Use the real training implementation, not a second hand-built actor."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if type(expected_iteration) is not int or expected_iteration < 1:
        raise ValueError("Expected iteration must be a positive integer")
    if (type(saved.get("iter")) is not int or saved.get("iter") != expected_iteration
            or saved.get("iteration_semantics") != "completed_updates"):
        raise ValueError(f"Checkpoint must record {expected_iteration} completed updates")
    with contextlib.redirect_stdout(io.StringIO()):
        reference = HIMActorCritic(
            num_actor_obs=HISTORY_DIM, num_critic_obs=235,
            num_one_step_obs=OBSERVATION_DIM, num_actions=12, **agent["policy"],
        )
    reference.load_state_dict(saved["model_state_dict"], strict=True)
    return reference.cpu().eval().requires_grad_(False)


@torch.inference_mode()
def verify_against_reference(eager, scripted, reference) -> dict:
    """Check raw output parity and a changing, nonzero 50-Hz observation stream."""
    generator = torch.Generator().manual_seed(1100)
    errors = {"eager_vs_reference_max_abs": 0., "jit_vs_reference_max_abs": 0.}
    cases = 0

    def compare(history):
        nonlocal cases
        expected = reference.act_inference(history)
        for name, model in (("eager", eager), ("jit", scripted)):
            actual = model(history)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
            key = f"{name}_vs_reference_max_abs"
            errors[key] = max(errors[key], float((actual-expected).abs().max()))
        cases += len(history)

    compare(torch.randn(128, HISTORY_DIM, generator=generator)*.4)
    compare(torch.linspace(-1.2, 1.3, HISTORY_DIM).repeat(8, 1)
            + torch.arange(8).float()[:, None]*.07)
    adapter = StairsPolicy(scripted)
    default = torch.tensor([DEFAULT_JOINT_POSITION])
    previous_target = None
    last_q = default.clone()
    for step in range(40):
        cmd = torch.tensor([[.45+.004*step, .02*math.sin(step*.2), .1*math.sin(step*.1)]])
        gyro = torch.tensor([[.2*math.sin(step*.1), -.07, .05]])
        quat = torch.tensor([[1., .03*math.sin(step*.2), .06, -.02]])
        quat /= torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
        q = default + .06*torch.sin(torch.arange(12).float()[None, :]*.3+step*.1)
        dq = (q-last_q)/CONTROL_DT
        adapter.observe(cmd, gyro, quat, q, dq, previous_target)
        compare(adapter.history)
        # Exercise history feedback after an external handover/filter changes
        # the proposed target. This is synthetic input, not a robot simulation.
        previous_target = .85*adapter.infer().joint_target + .15*default
        last_q = q
    return {**errors, "input_histories_checked": cases, "multistep_frames": 40,
            "atol": 2e-6, "rtol": 1e-5, "seed": 1100,
            "reference": "vendored HIMActorCritic.act_inference with trained HIMEstimator",
            "scope": "offline numerical parity only; no physics or real-robot validation"}


def export_policy(checkpoint: str | Path, output: str | Path,
                  source_record: str | Path | dict | None = None) -> dict:
    checkpoint, output = Path(checkpoint).resolve(), Path(output).absolute()
    if output.is_symlink() or (output.exists() and
                              (not output.is_dir() or any(output.iterdir()))):
        raise ValueError("Output must be a new or empty directory; refusing to overwrite files")
    identity = load_source_record(source_record)
    settings, agent = validate_source(checkpoint, identity)
    torch.set_num_threads(1)
    eager, checkpoint_metadata = load_checkpoint(checkpoint)
    reference = load_reference(checkpoint, agent, identity["checkpoint_iteration"])
    scripted = torch.jit.script(eager)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as staging:
        stage = Path(staging)
        exported = stage / "stairs_policy.pt"
        scripted.save(str(exported))
        # Compare the deserialized artifact, which is what a consumer loads.
        reloaded = torch.jit.load(str(exported), map_location="cpu").eval()
        verification = verify_against_reference(eager, reloaded, reference)
        metadata = {
            "format": "go2-him-stairs-cpu-v1",
            "checkpoint_filename": checkpoint.name,
            "checkpoint_sha256": identity["checkpoint_sha256"],
            "checkpoint_iteration": checkpoint_metadata["iteration"],
            "iteration_semantics": checkpoint_metadata["iteration_semantics"],
            "source_training_files_sha256": identity["source_hashes"],
            "export_sha256": sha256(exported),
            "torch_version": str(torch.__version__),
            "dtype": "float32", "device": "cpu",
            "input_shape": ["batch", HISTORY_DIM], "output_shape": ["batch", 12],
            "output_semantics": "raw normalized actions; no clipping, pose offset, PD, or robot IO in TorchScript",
            "joint_order": list(JOINT_NAMES), "sdk_motor_order": list(SDK_JOINT_NAMES),
            "sdk_to_training_indices": [SDK_JOINT_NAMES.index(n) for n in JOINT_NAMES],
            "training_to_sdk_indices": [JOINT_NAMES.index(n) for n in SDK_JOINT_NAMES],
            "observation_order": list(OBSERVATION_ORDER),
            "observation_dim": OBSERVATION_DIM, "history_frames": HISTORY_FRAMES,
            "history_order": "newest_first", "history_reset": "repeat_actual_first_frame_six_times",
            "default_joint_position": list(DEFAULT_JOINT_POSITION),
            "control": {**settings["control"], "control_dt": CONTROL_DT},
            "observation_contract": {
                "commands": "vx m/s, vy m/s, yaw rate rad/s; no scale",
                "angular_velocity": "body rad/s multiplied by 0.25",
                "gravity": "world (0,0,-1) rotated into body; unit body-to-world quaternion wxyz",
                "joint_pos": "radians minus default_joint_position, in joint_order",
                "joint_vel": "rad/s multiplied by 0.05, in joint_order",
                "previous_action": "(previous actually applied target-default_joint_position)/0.25; after blending/filtering/clipping",
                "normalization": "no running observation normalization; encoder latent16 L2 normalized, eps=1e-12",
            },
            "simulation_contract": {"physics_dt": .005, "decimation": 4,
                                    "effort_limit_per_leg_nm": [23.7, 23.7, 45.43]},
            "verification": verification,
            "deployment_scope": "additive stair inference artifact; preserve original flat policy and underlying control services",
            "limitations": [
                "No actuator commands, control ownership, watchdog or joint-limit enforcement are supplied by this artifact.",
                "The evaluated hybrid terrain gate used simulation raycasts; real-robot perception and switching are not exported.",
                "Simulation success is not evidence of real-robot validation.",
            ],
        }
        (stage / "manifest.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+"\n")
        validate_source(checkpoint, identity)
        # Directory rename refuses a concurrently populated destination. Existing
        # empty directories may be replaced; no existing files are overwritten.
        if output.is_symlink() or (output.exists() and
                                  (not output.is_dir() or any(output.iterdir()))):
            raise ValueError("Output became nonempty; refusing to overwrite files")
        os.rename(stage, output)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help="Checkpoint beside its pinned training sidecars; defaults to the selected 1100 model")
    parser.add_argument("--source-record", type=Path,
                        help="Explicit JSON identity for another checkpoint; omission keeps strict original 1100 hashes")
    parser.add_argument("--output", type=Path, required=True, help="New or empty export directory")
    args = parser.parse_args()
    try:
        metadata = export_policy(args.checkpoint, args.output, args.source_record)
    except (OSError, ValueError, RuntimeError, AssertionError) as exc:
        parser.error(str(exc))
    print(json.dumps({"output": str(args.output.absolute()),
                      "checkpoint_iteration": metadata["checkpoint_iteration"],
                      "export_sha256": metadata["export_sha256"],
                      "verification": metadata["verification"]}, indent=2))


if __name__ == "__main__":
    main()
