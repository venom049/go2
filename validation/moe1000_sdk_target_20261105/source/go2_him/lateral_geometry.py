"""Tensor-only geometry shared by lateral posture rewards and evaluation."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def side_sign_tensor(reference: torch.Tensor, side_signs: Sequence[float]) -> torch.Tensor:
    """Build static left/right signs without inspecting tensor values on the GPU.

    Callers must provide Python signs matching the explicit FL, FR, RL, RR
    entity ordering. Left is +Y in the robot frame and right is -Y.
    """
    if isinstance(side_signs, torch.Tensor):
        raise TypeError("side_signs must be a Python sequence, not a device tensor")
    if len(side_signs) != 4 or any(sign not in (-1, 1) for sign in side_signs):
        raise ValueError("side_signs must contain four values, each +1 or -1")
    return reference.new_tensor(side_signs)


def signed_foot_lateral_positions(
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    feet_pos_w: torch.Tensor,
    side_signs: Sequence[float] = (1, -1, 1, -1),
) -> torch.Tensor:
    """Outward foot distance from the body centerline, in metres, shape (N, 4).

    Inputs use world-frame positions and Isaac Lab's unit wxyz root quaternion.
    Feet and signs must share an explicit FL, FR, RL, RR order. The full inverse
    rotation removes roll, pitch, and yaw: climbing pitch and heading cannot
    masquerade as foot spreading. Negative output means a foot crossed the
    centerline. Unit quaternions are supplied by the simulator; this helper does
    not normalize or silently replace invalid simulation state.

    All validation uses tensor metadata or static Python signs, so this function
    performs no device-to-host synchronization in the training reward loop.
    """
    if root_pos_w.ndim != 2 or root_pos_w.shape[-1] != 3:
        raise ValueError("root_pos_w must have shape (N, 3)")
    if root_quat_w.shape != (root_pos_w.shape[0], 4):
        raise ValueError("root_quat_w must have shape (N, 4), in wxyz order")
    if feet_pos_w.shape != (root_pos_w.shape[0], 4, 3):
        raise ValueError("feet_pos_w must have shape (N, 4, 3)")
    signs = side_sign_tensor(feet_pos_w, side_signs)
    relative_w = feet_pos_w - root_pos_w[:, None, :]
    # Quaternion inverse has vector part -xyz for a unit quaternion.
    inverse_xyz = -root_quat_w[:, None, 1:].expand_as(relative_w)
    twice_cross = 2.0 * torch.cross(inverse_xyz, relative_w, dim=-1)
    feet_b = relative_w + root_quat_w[:, None, :1] * twice_cross + torch.cross(
        inverse_xyz, twice_cross, dim=-1
    )
    return feet_b[..., 1] * signs
