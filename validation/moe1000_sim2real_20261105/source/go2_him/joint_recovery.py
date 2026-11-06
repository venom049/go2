"""Stronger, bounded joint costs for continuing the audited MoE stair policy.

Costs are reward units per second. The caller applies its curriculum ramp and
the elapsed control-window duration once. No actions, measurements, audit
evidence or protection boundaries are clipped or changed by this module.
"""
from __future__ import annotations

import math
from numbers import Real

import numpy as np

from go2_him.official_guard import COST_KINDS, GuardCost, GuardSample


_BUDGETS = {"velocity": 1., "torque": .5, "tracking": 1.5,
            "target_position": 4., "position": 1.}
_SOFT_RATIOS = {"velocity": .90, "torque": .85, "tracking": .85}
_TARGET_MARGIN = .03
_POSITION_SCALE = .10

CONFIG = {
    "kind": "joint_recovery",
    "budget_reward_units_per_second": 8.,
    "component_budgets_reward_units_per_second": dict(_BUDGETS),
    "soft_ratios": dict(_SOFT_RATIOS),
    "target_position_soft_margin_rad": _TARGET_MARGIN,
    "target_position_overshoot_scale_rad": _POSITION_SCALE,
    "position_overshoot_scale_rad": _POSITION_SCALE,
    "ratio_raw_cost": ".1*soft_margin^2 + 2*hard_ratio_excess + hard_ratio_excess^2",
    "target_position_raw_cost": ".1*soft_margin^2 + 2*(overshoot/.10) + (overshoot/.10)^2",
    "position_raw_cost": "overshoot/.10 + (overshoot/.10)^2; no in-range posture cost",
    "bounded_cost": "component_budget * raw / (1 + raw)",
    "reduction": "worst joint and optional pre/post physics boundary per cost component",
    "physical_state_or_target_clipping": False,
    "limits": "unchanged supplied audited profile; not firmware readback",
}


def compute_joint_recovery_cost(sample: GuardSample, *, physics_axis=None,
                                scale=1.) -> GuardCost:
    """Return five stronger costs without treating a bounded cost as a pass.

    The target-position margin starts 0.03 rad inside either angular bound;
    actual position is penalized only outside its bounds. Velocity, requested
    torque and tracking use their independent profile ratios. Target velocity
    and applied torque remain audit-only, avoiding duplicate effort penalties.
    ``physics_axis`` indexes a batch dimension, never the final joint axis:
    for (T,N,12) evidence, axis 0 produces one cost per environment.
    """
    if (isinstance(scale, bool) or not isinstance(scale, Real)
            or not math.isfinite(scale) or not 0 < scale <= 1):
        raise ValueError("scale must be finite and in (0,1]")
    target = np.asarray(sample.values["target_position"], dtype=np.float64)
    if target.ndim == 0 or target.shape[-1] != 12 or not target.size:
        raise ValueError("Joint evidence needs a nonempty trailing dimension 12")
    shape = target.shape
    rank = target.ndim - 1
    if physics_axis is not None:
        if type(physics_axis) is not int or not -rank <= physics_axis < rank:
            raise ValueError("physics_axis must index a batch axis, not the joint axis")

    def field(mapping, key, *, nonnegative=False):
        value = mapping.get(key)
        if value is None:
            raise ValueError(f"Missing required {key} training evidence")
        result = np.asarray(value, dtype=np.float64)
        if result.shape != shape:
            raise ValueError(f"Inconsistent {key} evidence shape")
        if not np.isfinite(result).all():
            raise FloatingPointError(f"Nonfinite {key} is invalid training evidence")
        if nonnegative and np.any(result < 0):
            raise ValueError(f"{key} ratios and excesses must be nonnegative")
        return result

    for key in ("velocity", "position", "target_position", "tracking", "requested_torque"):
        field(sample.values, key)
    # Validate even the audit-only observations, when present, instead of
    # silently interpreting nonfinite evidence as a finite guard cost.
    for key, value in sample.values.items():
        if value is not None:
            field(sample.values, key)
    lower = np.asarray(sample.profile.lower, dtype=np.float64)
    upper = np.asarray(sample.profile.upper, dtype=np.float64)
    if (lower.shape != (12,) or upper.shape != (12,)
            or not np.isfinite(lower).all() or not np.isfinite(upper).all()
            or np.any(upper - lower <= 2 * _TARGET_MARGIN)):
        raise ValueError("Profile needs 12 finite angular intervals wider than the target margin")

    def peak(value):
        result = np.max(value, axis=-1)
        return np.max(result, axis=physics_axis) if physics_axis is not None else result

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        ratios = {
            "velocity": peak(field(sample.ratios, "velocity", nonnegative=True)),
            "torque": peak(field(sample.ratios, "requested_torque", nonnegative=True)),
            "tracking": peak(field(sample.ratios, "tracking", nonnegative=True)),
            "position": peak(field(sample.excess, "position", nonnegative=True)) / _POSITION_SCALE,
            "target_position": peak(field(sample.excess, "target_position", nonnegative=True)) / _POSITION_SCALE,
        }
        clearance = np.minimum(target - lower, upper - target)
        target_soft = peak(np.maximum((_TARGET_MARGIN - clearance) / _TARGET_MARGIN, 0.))
        raw = {}
        for key in COST_KINDS:
            if key in _SOFT_RATIOS:
                soft = _SOFT_RATIOS[key]
                margin = np.maximum((ratios[key] - soft) / (1. - soft), 0.)
                hard = np.maximum(ratios[key] - 1., 0.)
                raw[key] = .1 * margin**2 + 2. * hard + hard**2
            elif key == "target_position":
                hard = ratios[key]
                raw[key] = .1 * target_soft**2 + 2. * hard + hard**2
            else:
                raw[key] = ratios[key] + ratios[key]**2
        bounded = {key: _BUDGETS[key] * raw[key] / (1. + raw[key]) for key in COST_KINDS}
        total = np.sum(np.stack(list(bounded.values())), axis=0) * scale
    if any(not np.isfinite(value).all() for value in (*raw.values(), *bounded.values(), total)):
        raise FloatingPointError("Nonfinite recovery cost is invalid training evidence")
    return GuardCost(total, raw, bounded, ratios, float(scale), sum(_BUDGETS.values()))
