"""Bounded training costs for the existing Go2 application guards.

This module consumes unmodified, frozen physics-window evidence. Only the
reward is bounded: measured states, requested PD efforts and audit counts are
never clipped. A finite reward or successful episode is not a guard pass.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


KINDS = ("velocity", "torque", "tracking", "target_position", "position")


def guard_balance_scale(control_steps, steps_per_update, settings):
    """Run-local ramp, independent of episode resets and checkpoint numbering."""
    if type(control_steps) is not int or control_steps < 0:
        raise ValueError("control_steps must be a nonnegative integer")
    if type(steps_per_update) is not int or steps_per_update < 1:
        raise ValueError("steps_per_update must be a positive integer")
    updates = settings["ramp_updates"]
    initial = settings["initial_scale"]
    if type(updates) is not int or updates < 0:
        raise ValueError("ramp_updates must be a nonnegative integer")
    if type(initial) not in (int, float) or not math.isfinite(initial) or not 0 < initial <= 1:
        raise ValueError("initial_scale must be in (0, 1]")
    fraction = min(control_steps / (updates * steps_per_update), 1.) if updates else 1.
    return float(initial + (1. - initial) * fraction)


def _guard_cost(ratio, soft_ratio):
    # A visible soft margin, plus a linear term for small hard-limit crossings.
    # Pure squared hard excess almost ignores a 13.313/13 speed violation.
    margin = ((ratio - soft_ratio).clamp_min(0) / (1. - soft_ratio)).square()
    excess = (ratio - 1.).clamp_min(0)
    return .1 * margin + 2. * excess + excess.square()


@dataclass
class GuardBalanceResult:
    total: torch.Tensor
    raw: dict[str, torch.Tensor]
    bounded: dict[str, torch.Tensor]
    ratios: dict[str, torch.Tensor]
    scale: float
    budget_total: float

    def log_metrics(self):
        # Device-side reductions only; the normal rollout logger owns copying
        # to the host. Do not synchronize separately for each reward component.
        result = {
            "GuardBalance/scale": self.total.new_tensor(self.scale),
            "GuardBalance/budget_total": self.total.new_tensor(self.budget_total),
            "GuardBalance/effective_budget": self.total.new_tensor(self.scale * self.budget_total),
            "GuardBalance/total_cost": self.total.mean(),
        }
        for name in KINDS:
            result[f"GuardBalance/raw_{name}"] = self.raw[name].mean()
            result[f"GuardBalance/bounded_{name}"] = self.bounded[name].mean()
            result[f"GuardBalance/weighted_{name}"] = self.bounded[name].mean() * self.scale
        return {name: value.detach().clone() for name, value in result.items()}


def compute_guard_balance(joint_window, torque_window, settings, *, scale=1.):
    """One positive cost per environment, preserving short single-axis spikes.

    Position costs are zero everywhere inside the actual asymmetric URDF
    interval. They do not penalize distance from a standing or central pose.
    """
    joint_window.require_complete()
    torque_window.require_complete()
    if joint_window.names != torque_window.names or joint_window.samples != torque_window.samples:
        raise ValueError("Guard costs require matching joint order and physics windows")
    if not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError("Guard cost scale must be in (0, 1]")
    budgets = {kind: float(settings[f"budget_{kind}"]) for kind in KINDS}
    if any(not math.isfinite(value) or value <= 0 for value in budgets.values()):
        raise ValueError("Every guard budget must be finite and positive")
    if sum(budgets.values()) > 3.0 + 1e-9:
        raise ValueError("Total guard budget cannot exceed 3.0 per second")
    for kind in ("velocity", "torque", "tracking"):
        soft = settings[f"{kind}_soft_ratio"]
        if not math.isfinite(soft) or not 0 < soft < 1:
            raise ValueError("Guard soft ratios must be finite and in (0, 1)")
    for key in ("position_overshoot_scale_rad", "target_position_overshoot_scale_rad"):
        if not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError("Position overshoot scales must be finite and positive")
    # Raw joint_window.peaks contain all substeps, including the terminal one.
    ratios = {
        "velocity": (joint_window.peaks["velocity"] / joint_window.velocity_limit).amax(-1),
        "torque": (torque_window.requested_peak / torque_window.limits).amax(-1),
        "tracking": (joint_window.peaks["tracking_error"] / joint_window.tracking_error_limit).amax(-1),
        "target_position": joint_window.peaks["target_position"].amax(-1) / settings["target_position_overshoot_scale_rad"],
        "position": joint_window.peaks["position"].amax(-1) / settings["position_overshoot_scale_rad"],
    }
    raw = {kind: _guard_cost(ratios[kind], settings[f"{kind}_soft_ratio"])
           for kind in ("velocity", "torque", "tracking")}
    for kind in ("position", "target_position"):
        raw[kind] = ratios[kind] + ratios[kind].square()
    # One grouped check, following the existing window's fail-closed contract.
    # Do not hide NaN/overflow with nan_to_num or accept an invalid cost as safe.
    joint_window._check_finite(*raw.values())
    # Strictly increasing soft upper bound. Unlike min(x,B), it retains a
    # preference for reducing violations beyond the budget knee.
    bounded = {kind: budgets[kind] * raw[kind] / (budgets[kind] + raw[kind]) for kind in KINDS}
    total = torch.stack(tuple(bounded.values())).sum(0) * scale
    return GuardBalanceResult(total, raw, bounded, ratios, float(scale), sum(budgets.values()))
