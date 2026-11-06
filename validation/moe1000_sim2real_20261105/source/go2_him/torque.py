"""Simulator-free, per-physics-step torque accounting (N m).

Requested PD effort is measured BEFORE actuator clipping. A successful climb
or clipped applied effort cannot make a requested-effort violation disappear.
"""
from __future__ import annotations

import math
import torch

EFFORT_LIMITS_NM = {"hip": 23.7, "thigh": 23.7, "calf": 45.43}


def joint_effort_limits(names, device="cpu", dtype=torch.float32):
    values = []
    if len(names) != 12 or len(set(names)) != 12:
        raise ValueError("Expected 12 distinct Go2 joint names")
    expected = {f"{leg}_{kind}_joint" for leg in ("FL", "FR", "RL", "RR") for kind in EFFORT_LIMITS_NM}
    if set(names) != expected:
        raise ValueError("Unexpected Go2 joints; do not assume their ordering")
    for name in names:
        values.append(EFFORT_LIMITS_NM[name.split("_")[1]])
    return torch.tensor(values, device=device, dtype=dtype)


class TorqueWindow:
    """One control interval, including every physics substep and terminal step.

    begin() explicitly opens a window. After exactly `substeps` samples it is
    frozen, so an extra scene.write_data_to_sim() during auto-reset cannot erase
    or contaminate the just-finished step. Actuator reset must NOT clear it.
    """
    def __init__(self, num_envs, names, device="cpu", soft_ratio=.8):
        if not math.isfinite(soft_ratio) or not 0 < soft_ratio < 1:
            raise ValueError("Torque soft ratio must be in (0, 1)")
        self.names, self.soft_ratio = list(names), soft_ratio
        self.limits = joint_effort_limits(names, device)
        shape = (num_envs, len(names))
        self.requested_peak = torch.zeros(shape, device=device)
        self.applied_peak = torch.zeros_like(self.requested_peak)
        self.excess_count = torch.zeros(shape, device=device, dtype=torch.long)
        self.applied_excess_count = torch.zeros_like(self.excess_count)
        self.saturation_count = torch.zeros_like(self.excess_count)
        self.margin_sum = torch.zeros(num_envs, device=device)
        self.excess_sum = torch.zeros_like(self.margin_sum)
        self.startup = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.samples = self.expected = 0

    def begin(self, substeps, startup):
        if type(substeps) is not int or substeps < 1:
            raise ValueError("Positive integer physics substep count required")
        if startup.shape != self.startup.shape or startup.dtype != torch.bool:
            raise ValueError("One boolean startup flag per environment required")
        for value in (self.requested_peak, self.applied_peak, self.excess_count,
                      self.applied_excess_count, self.saturation_count, self.margin_sum, self.excess_sum):
            value.zero_()
        self.startup.copy_(startup)
        self.samples, self.expected = 0, substeps

    def record(self, requested, applied):
        if self.samples >= self.expected:
            return  # initialization/reset writes are outside the physics window
        if requested.shape != self.requested_peak.shape or applied.shape != requested.shape:
            raise ValueError("Torque sample shape mismatch")
        if not bool(torch.isfinite(requested).all() & torch.isfinite(applied).all()):
            raise FloatingPointError("Nonfinite torque; abort instead of reporting a pass")
        requested, applied = requested.detach().abs(), applied.detach().abs()
        ratio = requested / self.limits
        self.requested_peak.copy_(torch.maximum(self.requested_peak, requested))
        self.applied_peak.copy_(torch.maximum(self.applied_peak, applied))
        self.excess_count.add_(requested > self.limits)
        self.applied_excess_count.add_(applied > self.limits)
        self.saturation_count.add_((requested - applied).abs() > 1e-6)
        self.margin_sum.add_((ratio - self.soft_ratio).clamp_min(0).square().sum(-1))
        self.excess_sum.add_((ratio - 1.).clamp_min(0).square().sum(-1))
        self.samples += 1

    def require_complete(self):
        if self.expected < 1 or self.samples != self.expected:
            raise RuntimeError("Torque window incomplete; missing substeps cannot be certified")

    def penalty(self, kind, startup_multiplier=1.):
        self.require_complete()
        if not math.isfinite(startup_multiplier) or startup_multiplier < 1:
            raise ValueError("Startup multiplier must be >= 1")
        if kind not in ("margin", "excess"):
            raise ValueError("Unknown torque penalty")
        value = self.margin_sum if kind == "margin" else self.excess_sum
        weight = torch.where(self.startup, startup_multiplier, 1.)
        return value / self.samples * weight

    def log_metrics(self):
        self.require_complete()
        result = {
            "Torque/requested_excess_fraction": self.excess_count.float().mean() / self.samples,
            "Torque/applied_excess_fraction": self.applied_excess_count.float().mean() / self.samples,
            "Torque/saturation_fraction": self.saturation_count.float().mean() / self.samples,
            "Torque/requested_peak_ratio": (self.requested_peak / self.limits).amax(),
            "Torque/applied_peak_ratio": (self.applied_peak / self.limits).amax(),
        }
        for i, name in enumerate(self.names):
            result[f"Torque/requested_peak_nm/{name}"] = self.requested_peak[:, i].amax()
            result[f"Torque/applied_peak_nm/{name}"] = self.applied_peak[:, i].amax()
        # Independent tensors: the rollout logger retains references for 24 steps.
        return {key: value.detach().clone() for key, value in result.items()}


class TorqueAudit:
    """Accumulate active evaluation episodes, retaining failed/terminal samples."""
    def __init__(self, window):
        self.names, self.limits = window.names, window.limits
        self.requested_peak = torch.zeros_like(window.limits)
        self.applied_peak = torch.zeros_like(window.limits)
        self.requested_excess = torch.zeros_like(window.limits, dtype=torch.long)
        self.applied_excess = torch.zeros_like(self.requested_excess)
        self.saturation = torch.zeros_like(self.requested_excess)
        self.samples_per_joint = torch.zeros((), device=window.limits.device, dtype=torch.long)
        self.startup_samples_per_joint = torch.zeros_like(self.samples_per_joint)
        self.startup_excess = torch.zeros_like(self.requested_excess)

    def update(self, window, active):
        window.require_complete()
        if active.shape != window.startup.shape or active.dtype != torch.bool:
            raise ValueError("One active flag per environment required")
        self.samples_per_joint += active.sum() * window.samples
        self.startup_samples_per_joint += (active & window.startup).sum() * window.samples
        selected = active[:, None]
        self.requested_peak = torch.maximum(self.requested_peak, window.requested_peak.masked_fill(~selected, 0).amax(0))
        self.applied_peak = torch.maximum(self.applied_peak, window.applied_peak.masked_fill(~selected, 0).amax(0))
        self.requested_excess += (window.excess_count * selected).sum(0)
        self.applied_excess += (window.applied_excess_count * selected).sum(0)
        self.saturation += (window.saturation_count * selected).sum(0)
        self.startup_excess += (window.excess_count * (active & window.startup)[:, None]).sum(0)

    def report(self):
        count = int(self.samples_per_joint.item())
        startup_count = int(self.startup_samples_per_joint.item())
        requested = int(self.requested_excess.sum().item())
        applied = int(self.applied_excess.sum().item())
        return {
            "scope": "Every physics substep in active episodes, including failures and terminal steps; not successful episodes only",
            "physics_samples_per_joint": count, "startup_physics_samples_per_joint": startup_count,
            "requested_excess_joint_samples": requested, "applied_excess_joint_samples": applied,
            "startup_excess_joint_samples": int(self.startup_excess.sum().item()),
            "requested_excess_fraction": requested / (count * len(self.names)) if count else None,
            "torque_limits_passed": count > 0 and startup_count > 0 and requested == 0 and applied == 0,
            "note": "Zero sampled violations is an evaluation result, not a guarantee for unseen real-world states.",
            "joints": {name: {"limit_nm": float(self.limits[i]),
                               "requested_peak_nm": float(self.requested_peak[i]),
                               "applied_peak_nm": float(self.applied_peak[i]),
                               "requested_excess_samples": int(self.requested_excess[i]),
                               "saturation_samples": int(self.saturation[i])}
                       for i, name in enumerate(self.names)},
        }
