"""Pure PyTorch adapter for the upstream Go2 ``final/walk.pt`` policy.

The source interface is pinned in ``assets/policies/upstream_walk/manifest.json``.
The upstream BSD-3-Clause notice is retained beside that manifest in LICENSE.
This is a separate 49-observation / 16-action policy, not a HIM checkpoint. Its
last four actions control per-leg PD gains and must not be silently discarded.
All public joint tensors use the caller's explicitly supplied joint-name order.
No Isaac Sim, MuJoCo, or robot SDK imports or hardware operations are performed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


POLICY_JOINT_NAMES = tuple(
    f"{leg}_{joint}_joint"
    for leg in ("FR", "FL", "RR", "RL")
    for joint in ("hip", "thigh", "calf")
)
DEFAULT_DOF_POS = (0.0, 0.8, -1.5, 0.0, 0.8, -1.5,
                   0.0, 1.0, -1.5, 0.0, 1.0, -1.5)
TORQUE_LIMITS = (23.7, 23.7, 45.0) * 4
POLICY_DT = 0.02


class WalkActor(nn.Module):
    """The upstream ELU actor architecture; critic/optimizer are not needed."""

    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(49, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 16),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor(observation)


def load_walk_actor(checkpoint: str | Path, device: str | torch.device = "cpu") -> WalkActor:
    """Load tensor weights strictly, without executing pickled Python objects."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or not isinstance(saved.get("model_state_dict"), dict):
        raise ValueError("Expected an upstream checkpoint containing model_state_dict")
    state = {name: value for name, value in saved["model_state_dict"].items()
             if name.startswith("actor.")}
    if not state or any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("Checkpoint actor weights must be tensors")
    if any(not torch.isfinite(value).all().item() for value in state.values()):
        raise ValueError("Checkpoint contains non-finite actor weights")
    actor = WalkActor()
    actor.load_state_dict(state, strict=True)
    return actor.eval().requires_grad_(False).to(device)


def build_walk_observation(
    angular_velocity_body: torch.Tensor,
    quaternion_wxyz: torch.Tensor,
    command: torch.Tensor,
    joint_position_policy_order: torch.Tensor,
    joint_velocity_policy_order: torch.Tensor,
    last_action: torch.Tensor,
) -> torch.Tensor:
    """Build upstream observations, with joints ordered FR, FL, RR, RL.

    Angular velocity is in the body frame, commands are (vx, vy, yaw_rate), and
    quaternion is the unit body-to-world wxyz quaternion supplied by Isaac Lab.
    Last action contains all 16 *clipped network outputs*, before target slew.
    """
    if angular_velocity_body.ndim != 2 or angular_velocity_body.shape[-1] != 3:
        raise ValueError("angular_velocity_body must have shape (N, 3)")
    n = angular_velocity_body.shape[0]
    for name, value, width in (
        ("quaternion_wxyz", quaternion_wxyz, 4), ("command", command, 3),
        ("joint_position_policy_order", joint_position_policy_order, 12),
        ("joint_velocity_policy_order", joint_velocity_policy_order, 12),
        ("last_action", last_action, 16),
    ):
        if value.shape != (n, width):
            raise ValueError(f"{name} must have shape (N, {width})")
    xyz = quaternion_wxyz[:, 1:]
    gravity_w = xyz.new_tensor((0.0, 0.0, -1.0)).expand_as(xyz)
    cross = torch.cross(xyz, gravity_w, dim=-1)
    gravity_b = gravity_w + 2.0 * (-quaternion_wxyz[:, :1] * cross
                                  + torch.cross(xyz, cross, dim=-1))
    return torch.cat((
        angular_velocity_body * 0.25, gravity_b,
        command * command.new_tensor((2.0, 2.0, 0.25)),
        joint_position_policy_order - joint_position_policy_order.new_tensor(DEFAULT_DOF_POS),
        joint_velocity_policy_order * 0.05, last_action,
    ), dim=-1)


@dataclass(frozen=True)
class WalkCommand:
    """Position and gains in caller joint order; raw_action is upstream order."""
    joint_target: torch.Tensor
    stiffness: torch.Tensor
    damping: torch.Tensor
    effort_limit: torch.Tensor
    raw_action: torch.Tensor


class UpstreamWalkPolicy:
    """Batched stateful 50 Hz adapter with explicit name-based joint mapping.

    Pass articulation joint names (exactly the 12 actuated joints, any order).
    Apply target positions, returned gains, and effort limits each control step. A
    manager's default pose / action clip must not reinterpret these positions.
    When switching back from another controller call ``reset`` for only the
    affected environments, passing their current target positions in caller
    order; this bounds the first target change to the original 0.1 rad / step.
    """

    def __init__(self, actor: nn.Module, joint_names: Sequence[str], num_envs: int,
                 device: str | torch.device = "cpu", policy_dt: float = POLICY_DT,
                 kp_factor: float = 1.0, kd_factor: float = 1.0):
        if len(joint_names) != 12 or set(joint_names) != set(POLICY_JOINT_NAMES):
            raise ValueError("joint_names must contain each of the 12 Go2 policy joints exactly once")
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if not math.isclose(policy_dt, POLICY_DT, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("The upstream walking policy requires a 50 Hz control period (0.02 s)")
        if any(not math.isfinite(x) or x <= 0.0 for x in (kp_factor, kd_factor)):
            raise ValueError("kp_factor and kd_factor must be finite and positive")
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.actor = actor.eval().requires_grad_(False).to(self.device)
        self.to_policy = [joint_names.index(name) for name in POLICY_JOINT_NAMES]
        self.to_caller = [POLICY_JOINT_NAMES.index(name) for name in joint_names]
        self.default_policy = torch.tensor(DEFAULT_DOF_POS, device=self.device)
        self.effort_limit = torch.tensor(TORQUE_LIMITS, device=self.device)[self.to_caller].expand(num_envs, -1)
        self.previous_target = self.default_policy.expand(num_envs, -1).clone()
        self.last_action = torch.zeros((num_envs, 16), device=self.device)
        self.kp_factor, self.kd_factor = kp_factor, kd_factor

    def reset(self, env_ids=None, joint_position: torch.Tensor | None = None):
        """Clear policy memory, initializing slew from actual supplied targets.

        ``joint_position`` has shape (N,12) for a full reset or (len(ids),12)
        for a partial reset. Boolean masks and integer indices are supported.
        A reset without positions initializes to the upstream default pose.
        """
        ids = slice(None) if env_ids is None else env_ids
        if joint_position is None:
            self.previous_target[ids] = self.default_policy
        else:
            expected = self.previous_target[ids].shape
            if joint_position.shape != expected:
                raise ValueError(f"joint_position must have shape {tuple(expected)} for the selected environments")
            self.previous_target[ids] = joint_position[:, self.to_policy]
        self.last_action[ids] = 0.0

    @torch.inference_mode()
    def step(self, angular_velocity_body: torch.Tensor, quaternion_wxyz: torch.Tensor,
             command: torch.Tensor, joint_position: torch.Tensor,
             joint_velocity: torch.Tensor) -> WalkCommand:
        if joint_position.shape != (self.num_envs, 12) or joint_velocity.shape != (self.num_envs, 12):
            raise ValueError("joint_position and joint_velocity must have shape (num_envs, 12)")
        observation = build_walk_observation(
            angular_velocity_body, quaternion_wxyz, command,
            joint_position[:, self.to_policy], joint_velocity[:, self.to_policy], self.last_action,
        )
        action = self.actor(observation)
        if action.shape != (self.num_envs, 16):
            raise ValueError("Upstream walking actor must return (num_envs, 16) actions")
        clipped = action.clamp(-100.0, 100.0)
        desired = self.default_policy + 0.25 * clipped[:, :12]
        target = self.previous_target + (desired - self.previous_target).clamp(-0.1, 0.1)
        kp_unscaled = (40.0 + 20.0 * clipped[:, 12:]).clamp(20.0, 60.0).repeat_interleave(3, dim=-1)
        self.previous_target.copy_(target)
        self.last_action.copy_(clipped)
        return WalkCommand(
            joint_target=target[:, self.to_caller],
            stiffness=(kp_unscaled * self.kp_factor)[:, self.to_caller],
            damping=(0.2 * kp_unscaled.sqrt() * self.kd_factor)[:, self.to_caller],
            effort_limit=self.effort_limit,
            raw_action=clipped,
        )
