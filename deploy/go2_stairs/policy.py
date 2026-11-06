"""Offline CPU inference for this project's 45 x 6 HIM stair policy.

No Isaac, DDS, Unitree SDK, networking, or actuator-writing dependency exists
in this module. Outputs are proposals for an existing control loop to review
and apply. This interface is NOT the upstream 49-observation walking interface.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


JOINT_NAMES = tuple(f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR")
                    for joint in ("hip", "thigh", "calf"))
DEFAULT_JOINT_POSITION = (.1, .8, -1.5, -.1, .8, -1.5,
                          .1, 1., -1.5, -.1, 1., -1.5)
OBSERVATION_ORDER = ("commands3", "angular_velocity3", "gravity3",
                     "joint_pos12", "joint_vel12", "action12")
OBSERVATION_DIM = 45
HISTORY_FRAMES = 6
HISTORY_DIM = 270
CONTROL_DT = .02


class StairsNetwork(nn.Module):
    """Stateless TorchScript-compatible estimator + actor, returning raw actions.

    Input: float32 (batch, 270), six 45D frames, newest first. Encoder's velocity
    is not rescaled: it is consumed exactly as produced during training. Only
    the 16D latent part receives L2 normalization, with PyTorch eps=1e-12.
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(270, 128), nn.ELU(),
                                     nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 19))
        self.actor = nn.Sequential(nn.Linear(64, 512), nn.ELU(),
                                   nn.Linear(512, 256), nn.ELU(),
                                   nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 12))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.dim() != 2 or history.size(1) != 270:
            raise ValueError("Expected float32 history shape (N, 270), newest frame first")
        if history.dtype != torch.float32:
            raise ValueError("HIM deployment history must be float32")
        if not bool(torch.isfinite(history).all()):
            raise ValueError("Non-finite HIM deployment observation history")
        parts = self.encoder(history)
        latent = F.normalize(parts[:, 3:], p=2.0, dim=-1, eps=1e-12)
        output = self.actor(torch.cat((history[:, :45], parts[:, :3], latent), dim=-1))
        if not bool(torch.isfinite(output).all()):
            raise ValueError("Non-finite HIM deployment output")
        return output


def load_checkpoint(checkpoint: str | Path) -> tuple[StairsNetwork, dict]:
    """Safely load only this architecture's inference tensors on the CPU."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or not isinstance(saved.get("model_state_dict"), dict):
        raise ValueError("Expected model_state_dict in the HIM training checkpoint")
    state = {}
    for name, tensor in saved["model_state_dict"].items():
        if name.startswith("actor."):
            state[name] = tensor
        elif name.startswith("estimator.encoder."):
            state[name.removeprefix("estimator.")] = tensor
    if not state or any(not isinstance(t, torch.Tensor) or not t.is_floating_point()
                        or not bool(torch.isfinite(t).all()) for t in state.values()):
        raise ValueError("Inference weights must be finite floating-point tensors")
    network = StairsNetwork()
    network.load_state_dict(state, strict=True)
    network.eval().requires_grad_(False)
    return network, {"iteration": saved.get("iter"),
                     "iteration_semantics": saved.get("iteration_semantics")}


def _cpu_float(value: torch.Tensor, shape: tuple[int, ...], name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise ValueError(f"{name} must be a tensor of shape {shape}")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be on the CPU")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain finite floating-point values")
    return value.to(dtype=torch.float32)


def projected_gravity(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    """World (0,0,-1) rotated into the body frame using a unit wxyz quaternion."""
    if quaternion_wxyz.ndim != 2 or quaternion_wxyz.shape[1] != 4:
        raise ValueError("Quaternion must have shape (N, 4), wxyz, body-to-world")
    q = _cpu_float(quaternion_wxyz, tuple(quaternion_wxyz.shape), "quaternion_wxyz")
    if not bool(torch.isclose(q.square().sum(-1), torch.ones(q.shape[0]), atol=1e-3, rtol=0.).all()):
        raise ValueError("Quaternion must be unit length; do not substitute xyzw")
    xyz = q[:, 1:]
    down = q.new_tensor((0., 0., -1.)).expand_as(xyz)
    cross = torch.cross(xyz, down, dim=-1)
    return down + 2 * (-q[:, :1] * cross + torch.cross(xyz, cross, dim=-1))


def build_observation(command: torch.Tensor, angular_velocity_body: torch.Tensor,
                      quaternion_wxyz: torch.Tensor, joint_position: torch.Tensor,
                      joint_velocity: torch.Tensor, previous_applied_action: torch.Tensor) -> torch.Tensor:
    """Construct one actor frame, with joints in JOINT_NAMES training order.

    Commands are physical (vx m/s, vy m/s, yaw rate rad/s), with NO scaling.
    Gyro is body-frame rad/s. Positions and velocities are radians and rad/s.
    Previous action is the actual target's HIM encoding, before this step, not
    the previous proposed network output. It is deliberately not clipped here.
    """
    if joint_position.ndim != 2:
        raise ValueError("joint_position must have shape (N, 12)")
    n = joint_position.shape[0]
    cmd = _cpu_float(command, (n, 3), "command")
    gyro = _cpu_float(angular_velocity_body, (n, 3), "angular_velocity_body")
    q = _cpu_float(joint_position, (n, 12), "joint_position")
    dq = _cpu_float(joint_velocity, (n, 12), "joint_velocity")
    previous = _cpu_float(previous_applied_action, (n, 12), "previous_applied_action")
    quat = _cpu_float(quaternion_wxyz, (n, 4), "quaternion_wxyz")
    return torch.cat((cmd, gyro*.25, projected_gravity(quat),
                      q-q.new_tensor(DEFAULT_JOINT_POSITION), dq*.05, previous), dim=-1)


@dataclass(frozen=True)
class StairsCommand:
    """A proposal only: all joint tensors use the caller's joint-name order."""
    joint_target: torch.Tensor
    stiffness: torch.Tensor
    damping: torch.Tensor
    effort_limit: torch.Tensor
    raw_action: torch.Tensor
    clipped_action: torch.Tensor


class StairsPolicy:
    """CPU observation-history adapter; it never sends control commands.

    Call observe() every 20 ms, including while another policy controls walking,
    and infer() only when a stair proposal is needed. Supply the actual previous
    applied joint target after blending/filtering/clipping, in caller order.
    Do not reset history merely because walk/stairs mode changes. reset() is for
    a new episode or restarting control. The first frame is repeated six times,
    exactly as the project's HIM wrapper handles unavailable reset history.
    """

    def __init__(self, network: nn.Module, joint_names: Sequence[str] = JOINT_NAMES,
                 num_envs: int = 1, action_scale: float = .25, action_clip: float = 4.,
                 stiffness: float = 40., damping: float = 1., control_dt: float = CONTROL_DT):
        if len(joint_names) != 12 or set(joint_names) != set(JOINT_NAMES):
            raise ValueError("Specify each of the 12 Go2 joint names exactly once")
        if type(num_envs) is not int or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        for name, value in (("action_scale", action_scale), ("action_clip", action_clip),
                            ("stiffness", stiffness), ("damping", damping)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isclose(control_dt, .02, abs_tol=1e-9, rel_tol=0):
            raise ValueError("This checkpoint expects 50 Hz inference (0.02 s)")
        self.network = network.cpu().eval()
        # ScriptModule does not implement Module.requires_grad_(). Disabling
        # gradients on its tensors works for eager and serialized networks.
        for parameter in self.network.parameters():
            parameter.requires_grad_(False)
        self.num_envs = num_envs
        self.to_policy = [joint_names.index(name) for name in JOINT_NAMES]
        self.to_caller = [JOINT_NAMES.index(name) for name in joint_names]
        self.default_policy = torch.tensor(DEFAULT_JOINT_POSITION)
        self.history = torch.zeros(num_envs, HISTORY_DIM)
        self.initialized = torch.zeros(num_envs, dtype=torch.bool)
        self.action_scale, self.action_clip = action_scale, action_clip
        self.stiffness, self.damping = stiffness, damping
        self.effort = torch.tensor((23.7, 23.7, 45.43)*4)[self.to_caller].expand(num_envs, -1)

    def reset(self, reset_mask: torch.Tensor | None = None) -> None:
        if reset_mask is None:
            self.history.zero_()
            self.initialized.zero_()
            return
        if reset_mask.device.type != "cpu" or reset_mask.dtype != torch.bool or reset_mask.shape != (self.num_envs,):
            raise ValueError("reset_mask must be a CPU boolean vector with one entry per environment")
        self.history[reset_mask] = 0
        self.initialized[reset_mask] = False

    @torch.inference_mode()
    def observe(self, command: torch.Tensor, angular_velocity_body: torch.Tensor,
                quaternion_wxyz: torch.Tensor, joint_position: torch.Tensor,
                joint_velocity: torch.Tensor,
                previous_applied_joint_target: torch.Tensor | None = None) -> torch.Tensor:
        q = _cpu_float(joint_position, (self.num_envs, 12), "joint_position")[:, self.to_policy]
        dq = _cpu_float(joint_velocity, (self.num_envs, 12), "joint_velocity")[:, self.to_policy]
        if previous_applied_joint_target is None:
            if bool(self.initialized.any()):
                raise ValueError("Supply the actually applied previous joint target on every non-reset step")
            previous_action = torch.zeros(self.num_envs, 12)
        else:
            target = _cpu_float(previous_applied_joint_target, (self.num_envs, 12), "previous_applied_joint_target")
            previous_action = (target[:, self.to_policy] - self.default_policy) / self.action_scale
        frame = build_observation(command, angular_velocity_body, quaternion_wxyz, q, dq, previous_action)
        advanced = torch.cat((frame, self.history[:, :-OBSERVATION_DIM]), dim=-1)
        repeated = frame.repeat(1, HISTORY_FRAMES)
        self.history.copy_(torch.where(self.initialized[:, None], advanced, repeated))
        self.initialized[:] = True
        return frame.clone()

    @torch.inference_mode()
    def infer(self) -> StairsCommand:
        if not bool(self.initialized.all()):
            raise RuntimeError("observe() must initialize every environment before infer()")
        raw = self.network(self.history)
        raw = _cpu_float(raw, (self.num_envs, 12), "network output")
        clipped = raw.clamp(-self.action_clip, self.action_clip)
        target = self.default_policy + self.action_scale * clipped
        target_caller = target[:, self.to_caller]
        return StairsCommand(target_caller, torch.full_like(target_caller, self.stiffness),
                             torch.full_like(target_caller, self.damping), self.effort.clone(),
                             raw[:, self.to_caller], clipped[:, self.to_caller])


def load_exported_policy(directory: str | Path, joint_names: Sequence[str] = JOINT_NAMES,
                         num_envs: int = 1) -> StairsPolicy:
    """Load a generated export directory, checking its contract and file hash."""
    directory = Path(directory)
    metadata = json.loads((directory / "manifest.json").read_text())
    if (metadata.get("format") != "go2-him-stairs-cpu-v1"
            or metadata.get("joint_order") != list(JOINT_NAMES)
            or metadata.get("observation_order") != list(OBSERVATION_ORDER)
            or metadata.get("history_frames") != HISTORY_FRAMES
            or metadata.get("history_order") != "newest_first"
            or metadata.get("default_joint_position") != list(DEFAULT_JOINT_POSITION)):
        raise ValueError("Incompatible stair export interface")
    expected_control = {"action_scale": .25, "action_clip": 4., "stiffness": 40.,
                        "damping": 1., "control_dt": CONTROL_DT}
    control = metadata.get("control")
    if (not isinstance(control, dict) or set(control) != set(expected_control)
            or any(type(control[key]) not in (int, float)
                   or control[key] != expected for key, expected in expected_control.items())):
        raise ValueError("Exported control settings differ from the trained stair contract")
    path = directory / "stairs_policy.pt"
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata.get("export_sha256"):
        raise ValueError("Exported policy checksum does not match manifest")
    return StairsPolicy(torch.jit.load(str(path), map_location="cpu"), joint_names, num_envs,
                        action_scale=control["action_scale"], action_clip=control["action_clip"],
                        stiffness=control["stiffness"], damping=control["damping"],
                        control_dt=control["control_dt"])
