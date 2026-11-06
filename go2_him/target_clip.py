"""Explicit target projection and control-step diagnostics for offline MoE RL.

The raw policy action remains the PPO random variable. Only ``sent_action``
belongs in the new mode's observation history. This module does not alter
measured states, effort limits, physical guards, or PPO returns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


TARGET_CLIP_CONFIG = {
    "kind": "target_clipped",
    "method": "PPO with executed target projection and explicit raw-target penalty; not CaT",
    "raw_target": "float32(action) * supplied scale + supplied offset; preserve original NumPy affine arithmetic",
    "raw_excess": "max(lower - q_raw, q_raw - upper, 0) per joint, float64",
    "sent_target": "clip(q_raw, supplied exact float32-promoted lower, upper), float64",
    "observation_action": "float32((q_sent - supplied offset) / supplied scale)",
    "cost_rate_formula": "20 * sum(raw_excess_rad) reward units per second",
    "cost_per_rad_per_second": 20.0,
    "other_costs": "retain the existing balanced guard cost evaluated on the post-clipping command and measured state",
    "reduction": "sum of 12 joint excesses, once per control step",
    "saturating_cost": False,
    "caller_applies_elapsed_dt_once": True,
    "ppo_log_probability_action": "original raw sampled action",
    "target_rate_limiting": False,
    "physical_state_or_effort_changes": False,
}


def _array(value, name, *, vector=False):
    value = np.asarray(value)
    if not np.issubdtype(value.dtype, np.number) or np.iscomplexobj(value):
        raise ValueError(f"{name} must contain real numbers")
    if (value.shape != (12,) if vector else
            value.ndim == 0 or value.shape[-1] != 12 or value.size == 0):
        raise ValueError(f"{name} must have {'shape (12,)' if vector else 'a nonempty trailing dimension 12'}")
    if not np.isfinite(value).all():
        raise FloatingPointError(f"Nonfinite {name}")
    return value


def _bounds(lower, upper):
    lower = _array(lower, "lower", vector=True).astype(np.float64)
    upper = _array(upper, "upper", vector=True).astype(np.float64)
    with np.errstate(over="raise", invalid="raise"):
        if (not np.array_equal(lower, lower.astype(np.float32).astype(np.float64))
                or not np.array_equal(upper, upper.astype(np.float32).astype(np.float64))):
            raise ValueError("Bounds must already be exact float32-promoted protocol boundaries")
    if np.any(lower >= upper):
        raise ValueError("Each lower bound must be strictly below its upper bound")
    return lower, upper


def _excess(target, lower, upper):
    return np.maximum(np.maximum(lower - target, target - upper), 0.0)


@dataclass(frozen=True)
class TargetClipResult:
    raw_action: np.ndarray
    raw_target: np.ndarray
    raw_excess: np.ndarray
    sent_target: np.ndarray
    sent_action: np.ndarray
    sent_excess: np.ndarray
    cost_rate: np.ndarray


def transform_target_action(action, scale, offset, lower, upper) -> TargetClipResult:
    """Project any nonempty batch of 12 actions without changing the action law.

    Configuration arrays have shape ``(12,)``. Bounds must already use the
    SDK float32 boundary values promoted to float64 (as the audited profile
    does). Refusing inexact bounds avoids silently widening a supplied limit.
    Scale and offset retain their supplied dtypes, matching the original
    environment's ``float32(action) * scale + offset`` NumPy expression.
    """
    action = _array(action, "action")
    scale = _array(scale, "scale", vector=True)
    offset = _array(offset, "offset", vector=True)
    lower, upper = _bounds(lower, upper)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        raw_action = action.astype(np.float32, copy=True)
        if np.any(scale <= 0):
            raise ValueError("Every action scale must be positive and nonzero")
        if np.any(offset.astype(np.float64) < lower) or np.any(offset.astype(np.float64) > upper):
            raise ValueError("The default offset must be inside every supplied angular interval")
        raw_target = raw_action * scale + offset
        raw_target64 = raw_target.astype(np.float64)
        raw_excess = _excess(raw_target64, lower, upper)
        sent_target = np.clip(raw_target64, lower, upper)
        sent_action = ((sent_target - offset.astype(np.float64)) / scale.astype(np.float64)).astype(np.float32)
        sent_excess = _excess(sent_target, lower, upper)
        cost_rate = 20.0 * np.sum(raw_excess, axis=-1)
    for name, value in (("raw target", raw_target), ("sent action", sent_action), ("cost", cost_rate)):
        if not np.isfinite(value).all():
            raise FloatingPointError(f"Nonfinite {name} after target conversion")
    return TargetClipResult(raw_action, raw_target, raw_excess, sent_target,
                            sent_action, sent_excess, cost_rate)


class TargetClipAccumulator:
    """Accumulate each control command once, independently of physics substeps.

    A batch with shape ``(..., 12)`` contributes one control sample for each
    leading index. ``record`` also supports an unclipped baseline, where the
    actually submitted target can legitimately remain outside the limits.
    """
    def __init__(self, joint_names: Sequence[str]):
        names = tuple(joint_names)
        if len(names) != 12 or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != 12:
            raise ValueError("Expected 12 distinct nonempty joint names")
        self.names = names
        self.control_steps = 0
        self._stats = {name: {
            "counts": np.zeros(12, dtype=np.int64), "any_count": 0,
            "minimum": np.full(12, np.inf), "maximum": np.full(12, -np.inf),
            "max_excess": np.zeros(12),
        } for name in ("raw", "sent")}
        self.clipped_steps = self.clipped_joints = 0
        self.sum_l1 = self.max_l1 = self.max_abs = 0.0
        self.sum_cost = self.max_cost = 0.0

    def update(self, result: TargetClipResult):
        """Accumulate already validated transform output without recomputation."""
        self._record(result.raw_target, result.sent_target,
                     result.raw_excess, result.sent_excess)

    def record(self, raw_target, sent_target, lower, upper):
        """Record actual raw and submitted targets for any execution mode."""
        raw = _array(raw_target, "raw_target").astype(np.float64)
        sent = _array(sent_target, "sent_target").astype(np.float64)
        if raw.shape != sent.shape:
            raise ValueError("Raw and sent targets must have identical shapes")
        lower, upper = _bounds(lower, upper)
        with np.errstate(over="raise", invalid="raise"):
            self._record(raw, sent, _excess(raw, lower, upper), _excess(sent, lower, upper))

    def _record(self, raw, sent, raw_excess, sent_excess):
        raw = _array(raw, "raw_target").astype(np.float64).reshape(-1, 12)
        sent = _array(sent, "sent_target").astype(np.float64).reshape(-1, 12)
        excesses = [_array(v, "excess").astype(np.float64).reshape(-1, 12)
                    for v in (raw_excess, sent_excess)]
        if any(v.shape != raw.shape for v in (sent, *excesses)) or any(np.any(v < 0) for v in excesses):
            raise ValueError("Target and nonnegative excess arrays must have matching shapes")
        # Compute all potentially failing arithmetic before changing counters.
        with np.errstate(over="raise", invalid="raise"):
            correction = np.abs(raw - sent)
            l1 = correction.sum(axis=-1)
            costs = 20.0 * excesses[0].sum(axis=-1)
            sum_l1 = self.sum_l1 + float(l1.sum())
            sum_cost = self.sum_cost + float(costs.sum())
        if not np.isfinite([sum_l1, sum_cost]).all():
            raise FloatingPointError("Nonfinite accumulated target diagnostics")
        self.control_steps += len(raw)
        for key, values, excess in zip(("raw", "sent"), (raw, sent), excesses):
            stats = self._stats[key]
            exceeded = excess > 0
            stats["counts"] += exceeded.sum(axis=0)
            stats["any_count"] += int(exceeded.any(axis=-1).sum())
            stats["minimum"] = np.minimum(stats["minimum"], values.min(axis=0))
            stats["maximum"] = np.maximum(stats["maximum"], values.max(axis=0))
            stats["max_excess"] = np.maximum(stats["max_excess"], excess.max(axis=0))
        self.clipped_steps += int((correction > 0).any(axis=-1).sum())
        self.clipped_joints += int((correction > 0).sum())
        self.sum_l1, self.sum_cost = sum_l1, sum_cost
        self.max_l1 = max(self.max_l1, float(l1.max()))
        self.max_abs = max(self.max_abs, float(correction.max()))
        self.max_cost = max(self.max_cost, float(costs.max()))

    def summary(self) -> dict:
        n = self.control_steps
        result = {"joint_names": list(self.names), "control_steps": n,
                  "sampling": "one sample per policy command proposal; strict guard may reject before application; no physics-boundary duplication"}
        for key, stats in self._stats.items():
            total = int(stats["counts"].sum())
            result[key] = {
                "sample_count": 12 * n, "excess_count": total,
                "excess_fraction": total / (12 * n) if n else None,
                "any_step_excess_count": stats["any_count"],
                "any_step_excess_fraction": stats["any_count"] / n if n else None,
                "max_excess_rad": float(stats["max_excess"].max()) if n else None,
                "per_joint": {name: {
                    "sample_count": n, "excess_count": int(stats["counts"][i]),
                    "excess_fraction": int(stats["counts"][i]) / n if n else None,
                    "min_target_rad": float(stats["minimum"][i]) if n else None,
                    "max_target_rad": float(stats["maximum"][i]) if n else None,
                    "max_excess_rad": float(stats["max_excess"][i]) if n else None,
                } for i, name in enumerate(self.names)},
            }
        result["clipping"] = {
            "control_steps": self.clipped_steps, "control_step_fraction": self.clipped_steps / n if n else None,
            "joint_count": self.clipped_joints, "joint_fraction": self.clipped_joints / (12 * n) if n else None,
            "sum_l1_rad": self.sum_l1, "mean_l1_rad_per_step": self.sum_l1 / n if n else None,
            "max_l1_rad_per_step": self.max_l1 if n else None,
            "max_abs_rad": self.max_abs if n else None,
        }
        result["raw_cost_rate"] = {"sum": self.sum_cost, "mean": self.sum_cost / n if n else None,
                                   "max": self.max_cost if n else None, "units": "reward units per second"}
        return result


def aggregate_target_clip_summaries(summaries: Iterable[dict]) -> dict:
    """Merge accumulator summaries using counts/sums, never mean-of-means.

    Input summaries must come from this module and have the same joint order.
    Empty accumulators may be included; an empty iterable has no joint contract.
    """
    summaries = list(summaries)
    if not summaries:
        raise ValueError("At least one target summary is required")
    accumulator = TargetClipAccumulator(summaries[0]["joint_names"])
    for summary in summaries:
        if tuple(summary["joint_names"]) != accumulator.names:
            raise ValueError("Cannot aggregate different joint orders")
        n = summary["control_steps"]
        if type(n) is not int or n < 0:
            raise ValueError("Invalid control-step count")
        if n == 0:
            continue
        accumulator.control_steps += n
        for key, stats in accumulator._stats.items():
            metric = summary[key]
            if metric["sample_count"] != n * 12:
                raise ValueError("Target sample count does not match control-step count")
            joints = [metric["per_joint"][name] for name in accumulator.names]
            if any(j["sample_count"] != n for j in joints):
                raise ValueError("Per-joint sample count does not match control-step count")
            stats["counts"] += np.asarray([j["excess_count"] for j in joints], dtype=np.int64)
            stats["any_count"] += metric["any_step_excess_count"]
            stats["minimum"] = np.minimum(stats["minimum"], [j["min_target_rad"] for j in joints])
            stats["maximum"] = np.maximum(stats["maximum"], [j["max_target_rad"] for j in joints])
            stats["max_excess"] = np.maximum(stats["max_excess"], [j["max_excess_rad"] for j in joints])
        clipping = summary["clipping"]
        accumulator.clipped_steps += clipping["control_steps"]
        accumulator.clipped_joints += clipping["joint_count"]
        accumulator.sum_l1 += clipping["sum_l1_rad"]
        accumulator.max_l1 = max(accumulator.max_l1, clipping["max_l1_rad_per_step"])
        accumulator.max_abs = max(accumulator.max_abs, clipping["max_abs_rad"])
        accumulator.sum_cost += summary["raw_cost_rate"]["sum"]
        accumulator.max_cost = max(accumulator.max_cost, summary["raw_cost_rate"]["max"])
    return accumulator.summary()
