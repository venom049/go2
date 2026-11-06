"""Calf-only velocity training cost and an independent four-joint audit.

The cost is measured in reward units per second. The caller multiplies it by
the actual control-window duration once. Neither function changes joint state,
actions, official protection limits, or the full joint-and-torque audit.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Integral, Real

import numpy as np

from go2_him.official_guard import GuardCost, GuardSample, load_official_profile


CALF_NAMES = ("FL_calf", "FR_calf", "RL_calf", "RR_calf")
COST_KEY = "calf_velocity"


def _calf_indices(names):
    names = tuple(names)
    if len(names) != 12 or len(set(names)) != 12:
        raise ValueError("Expected 12 unique named joints")
    calves = tuple(name for name in names if name.endswith("_calf"))
    if len(calves) != 4 or set(calves) != set(CALF_NAMES):
        raise ValueError("Expected exactly the four named Go2 calf joints")
    return np.asarray([names.index(name) for name in CALF_NAMES], dtype=np.int64)


def compute_calf_speed_cost(sample: GuardSample, *, physics_axis=None,
                            budget=4., soft_ratio=.90) -> GuardCost:
    """Penalize the worst actual calf velocity ratio in a control window.

    Only actual calf speed contributes. Position, target position, tracking,
    target velocity, torque and other joints remain available in the original
    sample for auditing but do not contribute to this training cost. There is
    no hard-violation termination or sample clipping. ``physics_axis`` indexes
    a batch dimension: e.g. (T,N,12), axis 0 produces N costs. Defaults use a
    90% margin and a strict upper budget of 4 reward units/second. A bounded
    training cost is not proof that the measured speed obeys the hard limit.
    """
    if not isinstance(budget, Real) or not math.isfinite(budget) or budget <= 0:
        raise ValueError("budget must be positive and finite")
    if (not isinstance(soft_ratio, Real) or not math.isfinite(soft_ratio)
            or not 0 < soft_ratio < 1):
        raise ValueError("soft_ratio must be finite and in (0,1)")
    indices = _calf_indices(sample.profile.names)
    velocity = np.asarray(sample.values["velocity"], dtype=np.float64)
    limits = np.asarray(sample.profile.velocity, dtype=np.float64)
    if velocity.ndim == 0 or velocity.shape[-1] != 12 or not velocity.size:
        raise ValueError("Actual joint velocity needs a nonempty trailing dimension 12")
    if limits.shape != (12,) or not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError("Profile speed limits must be 12 positive finite values")
    if not np.isfinite(velocity).all():
        raise FloatingPointError("Nonfinite joint velocity is invalid training evidence")
    batch_rank = velocity.ndim - 1
    if physics_axis is not None:
        if type(physics_axis) is not int or not -batch_rank <= physics_axis < batch_rank:
            raise ValueError("physics_axis must index a batch axis, not the joint axis")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        ratio = np.max(np.abs(velocity[..., indices]) / limits[indices], axis=-1)
        if physics_axis is not None:
            ratio = np.max(ratio, axis=physics_axis)
        margin = np.maximum((ratio - soft_ratio) / (1. - soft_ratio), 0.)
        hard = np.maximum(ratio - 1., 0.)
        raw = .5 * margin**2 + 4. * hard + 4. * hard**2
        bounded = budget * (raw / (budget + raw))
    if not np.isfinite(raw).all() or not np.isfinite(bounded).all():
        raise FloatingPointError("Nonfinite calf speed cost is invalid training evidence")
    return GuardCost(bounded, {COST_KEY: raw}, {COST_KEY: bounded},
                     {COST_KEY: ratio}, scale=1., budget_total=float(budget))


def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer count")
    return int(value)


def _peak(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative measurement")
    return float(value)


def calf_speed_summary(guard_summary: Mapping) -> dict:
    """Aggregate the existing audit's per-joint actual-velocity evidence.

    Fractions use integer counts and the four calf sample counts only, so an
    all-12-joint denominator cannot dilute this result. This accepts either a
    single-episode or an already merged official guard summary. Missing or
    unobserved calf evidence stays unknown, never zero. An observed violation
    remains a failure even if another calf's evidence is missing. This verdict
    deliberately excludes other joint/torque categories and task success.
    """
    if not isinstance(guard_summary, Mapping):
        raise ValueError("Expected an official guard summary mapping")
    metrics = guard_summary.get("metrics")
    velocity = metrics.get("velocity") if isinstance(metrics, Mapping) else None
    source = velocity.get("per_joint") if isinstance(velocity, Mapping) else None
    source = source if isinstance(source, Mapping) else {}
    official = load_official_profile()
    limits = {name: float(official.velocity[official.names.index(name)]) for name in CALF_NAMES}
    metadata = guard_summary.get("profile")
    if isinstance(metadata, Mapping) and "joint_names" in metadata and "velocity_limits_rad_s" in metadata:
        names = tuple(metadata["joint_names"])
        _calf_indices(names)
        supplied_limits = np.asarray(metadata["velocity_limits_rad_s"], dtype=np.float64)
        if supplied_limits.shape != (12,) or not np.isfinite(supplied_limits).all():
            raise ValueError("Malformed guard velocity-limit metadata")
        if any(float(supplied_limits[names.index(name)]) != limits[name] for name in CALF_NAMES):
            raise ValueError("Calf audit does not use the current official speed limit")

    numeric_failure = guard_summary.get("incomplete_due_to_numeric_failure", False)
    if not isinstance(numeric_failure, (bool, np.bool_)):
        raise ValueError("incomplete_due_to_numeric_failure must be boolean")
    incomplete_episodes = _count(guard_summary.get("incomplete_episodes", 0), "incomplete_episodes")
    incomplete_audit = bool(numeric_failure or incomplete_episodes > 0)
    complete = not incomplete_audit
    observed_violation = False
    rows = {}
    for name in CALF_NAMES:
        row = source.get(name)
        result = {"speed_limit_rad_s": limits[name], "sample_count": None,
                  "excess_count": None, "excess_fraction": None,
                  "peak_absolute_value": None, "max_ratio": None, "max_excess": None,
                  "calf_speed_acceptance_passed": None}
        if isinstance(row, Mapping):
            count = row.get("excess_count")
            samples = row.get("sample_count")
            if count is not None:
                result["excess_count"] = _count(count, f"{name}.excess_count")
                observed_violation |= count > 0
            if samples is not None:
                result["sample_count"] = _count(samples, f"{name}.sample_count")
            if count is not None and samples is not None and count > samples:
                raise ValueError(f"{name} excess count exceeds sample count")
            if samples and count is not None:
                result["excess_fraction"] = int(count) / int(samples)
            for key in ("peak_absolute_value", "max_ratio", "max_excess"):
                if row.get(key) is not None:
                    result[key] = _peak(row[key], f"{name}.{key}")
            if (result["max_excess"] is not None and result["max_excess"] > 0
                    or result["max_ratio"] is not None and result["max_ratio"] > 1
                    or result["peak_absolute_value"] is not None and result["peak_absolute_value"] > limits[name]):
                observed_violation = True
                result["calf_speed_acceptance_passed"] = False
            required = ("sample_count", "excess_count", "peak_absolute_value", "max_ratio", "max_excess")
            observed = all(result[key] is not None for key in required) and result["sample_count"] > 0
            if observed:
                result["calf_speed_acceptance_passed"] = (
                    result["excess_count"] == 0 and result["max_excess"] == 0
                    and result["max_ratio"] <= 1 and result["peak_absolute_value"] <= limits[name])
            elif count:
                result["calf_speed_acceptance_passed"] = False
            complete &= observed
        else:
            complete = False
        rows[name] = result
    unobserved = velocity.get("unobserved_state_samples") if isinstance(velocity, Mapping) else None
    if unobserved is not None and _count(unobserved, "velocity.unobserved_state_samples") > 0:
        complete = False
        incomplete_audit = True
    if incomplete_audit:
        # A numeric failure can abort before recording its final boundary even
        # though all previously recorded rows have finite zero-excess values.
        # Those observed counts remain useful, but they cannot certify a pass.
        for row in rows.values():
            if row["calf_speed_acceptance_passed"] is True:
                row["calf_speed_acceptance_passed"] = None
    # Unknown aggregate values stay unknown rather than reporting a partial
    # denominator as if it covered every calf. Per-joint evidence is retained.
    counts_known = all(row["excess_count"] is not None and row["sample_count"] is not None for row in rows.values())
    count = sum(row["excess_count"] for row in rows.values()) if counts_known else None
    samples = sum(row["sample_count"] for row in rows.values()) if counts_known else None
    peaks = {}
    for key in ("peak_absolute_value", "max_ratio", "max_excess"):
        peaks[key] = max(row[key] for row in rows.values()) if all(row[key] is not None for row in rows.values()) else None
    return {"scope": "four named calves, actual velocity only; all supplied audit samples",
            "joint_names": list(CALF_NAMES), "value_and_excess_unit": "rad/s",
            "speed_limits_rad_s": limits, "sample_count": samples, "excess_count": count,
            "excess_fraction": count / samples if samples and count is not None else None,
            **peaks, "per_joint": rows, "all_calf_metrics_observed": bool(complete),
            "incomplete_due_to_numeric_failure": bool(numeric_failure),
            "incomplete_episodes": incomplete_episodes,
            "calf_speed_acceptance_passed": False if observed_violation else True if complete else None}
