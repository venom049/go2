"""NumPy training costs and independent audits of the installed Go2 profile.

Only application joint checks are reproduced here. Communication, orientation,
gain, handoff and the two-second real-test watchdog are separate requirements.
No measurements or requests are clipped, and there is no startup grace period.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


PROFILE_ID = "unitree_go2_2ded279_hip48"
SDK_NAMES = tuple(f"{leg}_{kind}" for leg in ("FR", "FL", "RR", "RL")
                  for kind in ("hip", "thigh", "calf"))
MOE_NAMES = tuple(f"{leg}_{kind}" for leg in ("FL", "FR", "RL", "RR")
                  for kind in ("hip", "thigh", "calf"))
URDF_SHA256 = "7d19fe48e2e689ee1a032ab99f2a4a8b671d87e73de48d3e65811682a5b48b9e"
DEFAULT_PROFILE_PATH = Path(__file__).resolve().parents[1] / "validation/official_joint_limits_20261104/limit_profile.json"
METRICS = ("velocity", "target_velocity", "position", "target_position",
           "tracking", "requested_torque", "applied_torque")
METRIC_UNITS = {"velocity": "rad/s", "target_velocity": "rad/s", "position": "rad",
                "target_position": "rad", "tracking": "rad", "requested_torque": "Nm",
                "applied_torque": "Nm"}
COST_KINDS = ("velocity", "torque", "tracking", "target_position", "position")
DEFAULT_BUDGETS = {"velocity": 1.25, "torque": 1., "tracking": .35,
                   "target_position": .25, "position": .15}


def _names(names: Sequence[str]) -> tuple[str, ...]:
    result = tuple(n.removesuffix("_joint") if isinstance(n, str) else n for n in names)
    if len(result) != 12 or len(set(result)) != 12 or set(result) != set(SDK_NAMES):
        raise ValueError("Expected each of the 12 named Go2 joints exactly once")
    return result


def _readonly(values) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class OfficialGuardProfile:
    names: tuple[str, ...]
    velocity: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    effort: np.ndarray
    tracking: float
    source_path: str
    source_sha256: str
    profile_id: str = PROFILE_ID

    def metadata(self) -> dict:
        return {"profile_id": self.profile_id, "source_path": self.source_path,
                "source_sha256": self.source_sha256, "joint_names": list(self.names),
                "velocity_limits_rad_s": self.velocity.tolist(),
                "position_lower_rad": self.lower.tolist(),
                "position_upper_rad": self.upper.tolist(),
                "effort_limits_nm": self.effort.tolist(),
                "tracking_error_limit_rad": self.tracking,
                "q_dq_boundary_semantics": "SDK float32 boundary, strict outside comparison",
                "scope": "application joint checks; not firmware readback or permission to actuate"}


def load_official_profile(path=None, joint_names: Sequence[str] | None = None) -> OfficialGuardProfile:
    """Load and verify the exact installed profile, then remap by joint names.

    The default order is MoE-CTS FL/FR/RL/RR. `_joint` name suffixes are accepted.
    Unknown/missing/duplicate joints, altered values or source identity fail.
    q/dq limits use promoted float32 boundaries, just as Go2JointLimits.h does;
    effort limits stay double and tracking uses the policy's float32 0.9 value.
    """
    path = Path(path) if path is not None else DEFAULT_PROFILE_PATH
    payload = path.read_bytes()
    data = json.loads(payload)
    if data.get("units") != {"q": "joint output rad", "dq": "joint output rad/s", "effort": "Nm"}:
        raise ValueError("Official profile units do not match SDK joint-output units")
    if data.get("sdk_order") != "FR,FL,RR,RL; each hip,thigh,calf":
        raise ValueError("Official profile SDK order is missing or changed")
    sources = [s for s in data.get("sources", []) if s.get("file") == "go2_description.urdf"]
    if len(sources) != 1 or sources[0].get("sha256") != URDF_SHA256:
        raise ValueError("Official profile source identity does not match the installed profile")
    if data.get("retained_application_guards", {}).get("tracking_error_rad") != .9:
        raise ValueError("Official profile tracking threshold is missing or changed")
    rows = data.get("limits", [])
    if len(rows) != 12:
        raise ValueError("Official profile must contain exactly 12 joint records")
    by_name = {}
    for index, name in enumerate(SDK_NAMES):
        row = rows[index]
        leg, kind = name.split("_")
        lower, upper = ((-math.radians(48), math.radians(48)) if kind == "hip" else
                        ((-1.5708, 3.4907) if leg in ("FR", "FL") else (-.5236, 4.5379))
                        if kind == "thigh" else (-2.7227, -.83776))
        expected = {"sdk_index": index, "joint": name, "position_lower_rad": lower,
                    "position_upper_rad": upper, "speed_limit_rad_s": 15.7 if kind == "calf" else 30.1,
                    "effort_limit_nm": 45.43 if kind == "calf" else 23.7}
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Official profile record for {name} does not match the installed limits")
        by_name[name] = row
    names = _names(MOE_NAMES if joint_names is None else joint_names)
    def vector(key, protocol=False):
        values = np.asarray([by_name[n][key] for n in names], dtype=np.float64)
        return _readonly(values.astype(np.float32).astype(np.float64) if protocol else values)
    return OfficialGuardProfile(names, vector("speed_limit_rad_s", True),
                                vector("position_lower_rad", True), vector("position_upper_rad", True),
                                vector("effort_limit_nm"), float(np.float32(.9)), str(path.resolve()),
                                hashlib.sha256(payload).hexdigest())


@dataclass(frozen=True)
class GuardSample:
    profile: OfficialGuardProfile
    values: dict[str, np.ndarray | None]
    ratios: dict[str, np.ndarray | None]
    excess: dict[str, np.ndarray | None]
    exceeded: dict[str, np.ndarray | None]

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.values["velocity"].shape[:-1]

    @property
    def hard_violation(self) -> np.ndarray:
        """Violation of any observed check; missing applied torque is NOT a pass."""
        return np.logical_or.reduce([np.any(x, axis=-1) for x in self.exceeded.values() if x is not None])


def evaluate_guard(profile: OfficialGuardProfile, q, dq, q_target, kp, kd, *,
                   dq_target=0., tau_ff=0., applied_torque=None) -> GuardSample:
    """Evaluate one state or a batch with trailing joint dimension 12.

    All required input arrays broadcast to a common (...,12) shape. Values are
    promoted to float64, not rounded to the boundary: tiny excesses stay visible.
    `applied_torque` must be actual simulator actuator effort if supplied; absent
    evidence is explicitly None. Requested PD is computed before any limiting.
    Caller must evaluate/record the failure's final physics substep before reset.
    """
    fields = dict(q=q, dq=dq, q_target=q_target, kp=kp, kd=kd, dq_target=dq_target, tau_ff=tau_ff)
    if applied_torque is not None:
        fields["applied_torque"] = applied_torque
    arrays = {key: np.asarray(value, dtype=np.float64) for key, value in fields.items()}
    for key in ("q", "dq", "q_target"):
        if arrays[key].ndim == 0 or arrays[key].shape[-1] != 12:
            raise ValueError(f"{key} must have trailing joint dimension 12")
    try:
        broadcast = np.broadcast_arrays(*arrays.values())
    except ValueError as exc:
        raise ValueError("Joint inputs cannot be broadcast to a common (...,12) shape") from exc
    arrays = dict(zip(arrays, broadcast))
    if not arrays["q"].size:
        raise ValueError("Joint evidence cannot contain an empty batch")
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError("Nonfinite joint evidence must not be turned into a guard pass")
    if np.any(arrays["kp"] < 0) or np.any(arrays["kd"] < 0):
        raise ValueError("PD gains must be nonnegative")
    with np.errstate(over="raise", invalid="raise"):
        requested = arrays["kp"] * (arrays["q_target"] - arrays["q"]) + arrays["kd"] * (
            arrays["dq_target"] - arrays["dq"]) + arrays["tau_ff"]
        values = {"velocity": arrays["dq"].copy(), "target_velocity": arrays["dq_target"].copy(),
                  "position": arrays["q"].copy(), "target_position": arrays["q_target"].copy(),
                  "tracking": np.abs(arrays["q_target"] - arrays["q"]),
                  "requested_torque": requested,
                  "applied_torque": arrays["applied_torque"].copy() if applied_torque is not None else None}
        ratios, excess, exceeded = {}, {}, {}
        limits = {"velocity": profile.velocity, "target_velocity": profile.velocity,
                  "tracking": profile.tracking, "requested_torque": profile.effort,
                  "applied_torque": profile.effort}
        for key, value in values.items():
            if value is None:
                ratios[key] = excess[key] = exceeded[key] = None
                continue
            if key in ("position", "target_position"):
                half_width = (profile.upper - profile.lower) / 2
                center = profile.lower + half_width
                ratios[key] = np.abs(value - center) / half_width
                excess[key] = np.maximum(np.maximum(profile.lower - value, value - profile.upper), 0.)
            else:
                ratios[key] = np.abs(value) / limits[key]
                excess[key] = np.maximum(np.abs(value) - limits[key], 0.)
            exceeded[key] = excess[key] > 0.
    return GuardSample(profile, values, ratios, excess, exceeded)


@dataclass(frozen=True)
class GuardCost:
    total: np.ndarray
    raw: dict[str, np.ndarray]
    bounded: dict[str, np.ndarray]
    ratios: dict[str, np.ndarray]
    scale: float
    budget_total: float


def compute_guard_cost(sample: GuardSample, *, budgets: Mapping[str, float] | None = None,
                       scale=1., velocity_soft_ratio=.90, torque_soft_ratio=.85,
                       tracking_soft_ratio=.85, position_overshoot_scale_rad=.10,
                       target_position_overshoot_scale_rad=.15, physics_axis=None) -> GuardCost:
    """Positive cost in reward units/second, at most 3; reward subtracts cost*dt.

    Reduces by the worst joint, optionally also the specified physics-window
    axis among batch axes (e.g. (T,N,12), physics_axis=0 -> N costs). It never
    clips audit evidence. Applied torque and target velocity are audited but not
    duplicated in the five-component cost. A small bounded cost is not a pass.
    """
    budgets = dict(DEFAULT_BUDGETS if budgets is None else budgets)
    if set(budgets) != set(COST_KINDS) or any(not math.isfinite(v) or v <= 0 for v in budgets.values()):
        raise ValueError("Each of the five guard budgets must be positive and finite")
    total_budget = sum(budgets.values())
    if total_budget > 3. + 1e-12 or not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError("Guard budget must be <=3 reward units/s and scale in (0,1]")
    soft = dict(velocity=velocity_soft_ratio, torque=torque_soft_ratio, tracking=tracking_soft_ratio)
    if any(not math.isfinite(v) or not 0 < v < 1 for v in soft.values()):
        raise ValueError("Soft margin ratios must be in (0,1)")
    if any(not math.isfinite(v) or v <= 0 for v in (position_overshoot_scale_rad, target_position_overshoot_scale_rad)):
        raise ValueError("Position overshoot normalization scales must be positive")
    if physics_axis is not None:
        if type(physics_axis) is not int or not -len(sample.batch_shape) <= physics_axis < len(sample.batch_shape):
            raise ValueError("physics_axis must index a batch axis, not the joint axis")
    def peak(x):
        result = np.max(x, axis=-1)
        return np.max(result, axis=physics_axis) if physics_axis is not None else result
    ratios = {"velocity": peak(sample.ratios["velocity"]),
              "torque": peak(sample.ratios["requested_torque"]),
              "tracking": peak(sample.ratios["tracking"]),
              "position": peak(sample.excess["position"]) / position_overshoot_scale_rad,
              "target_position": peak(sample.excess["target_position"]) / target_position_overshoot_scale_rad}
    with np.errstate(over="raise", invalid="raise"):
        raw = {}
        for key in COST_KINDS:
            if key in soft:
                margin = np.maximum(ratios[key] - soft[key], 0.) / (1. - soft[key])
                hard = np.maximum(ratios[key] - 1., 0.)
                raw[key] = .1 * margin**2 + 2. * hard + hard**2
            else:
                raw[key] = ratios[key] + ratios[key]**2
        if any(not np.isfinite(v).all() for v in raw.values()):
            raise FloatingPointError("Nonfinite guard costs are invalid training evidence")
        bounded = {key: budgets[key] * (raw[key] / (budgets[key] + raw[key])) for key in COST_KINDS}
        total = np.sum(np.stack(list(bounded.values())), axis=0) * scale
    return GuardCost(total, raw, bounded, ratios, float(scale), float(total_budget))


class OfficialGuardAudit:
    """Weighted all-physics-sample audit; record before resetting failed episodes.

    There is intentionally no warmup/startup exclusion and no success-only mask.
    Merging accumulates exact integer counts, never averages run percentages.
    """
    def __init__(self, profile: OfficialGuardProfile):
        self.profile = profile
        self.state_samples = 0
        self.terminal_samples = 0
        self.data = {key: {"observed": 0, "counts": np.zeros(12, dtype=np.int64),
                           "min": np.full(12, np.inf), "max": np.full(12, -np.inf),
                           "peak_abs": np.zeros(12), "ratio": np.zeros(12), "excess": np.zeros(12)}
                     for key in METRICS}

    def _check_profile(self, profile):
        if (self.profile.names != profile.names or self.profile.source_sha256 != profile.source_sha256
                or any(not np.array_equal(getattr(self.profile, key), getattr(profile, key))
                       for key in ("velocity", "lower", "upper", "effort"))
                or self.profile.tracking != profile.tracking):
            raise ValueError("Cannot mix joint orders or guard profiles in one audit")

    def record(self, sample: GuardSample, *, terminal=False):
        self._check_profile(sample.profile)
        n = math.prod(sample.batch_shape) if sample.batch_shape else 1
        terminal_array = np.asarray(terminal)
        if terminal_array.dtype != np.dtype(bool):
            raise ValueError("terminal must be a boolean or boolean batch mask")
        try:
            terminals = np.broadcast_to(terminal_array, sample.batch_shape)
        except ValueError as exc:
            raise ValueError("terminal must broadcast to the sample batch shape") from exc
        self.state_samples += n
        self.terminal_samples += int(np.count_nonzero(terminals))
        for key in METRICS:
            if sample.values[key] is None:
                continue
            value = sample.values[key].reshape(-1, 12)
            row = self.data[key]
            row["observed"] += n
            row["counts"] += sample.exceeded[key].reshape(-1, 12).sum(axis=0)
            row["min"] = np.minimum(row["min"], value.min(axis=0))
            row["max"] = np.maximum(row["max"], value.max(axis=0))
            row["peak_abs"] = np.maximum(row["peak_abs"], np.abs(value).max(axis=0))
            row["ratio"] = np.maximum(row["ratio"], sample.ratios[key].reshape(-1, 12).max(axis=0))
            row["excess"] = np.maximum(row["excess"], sample.excess[key].reshape(-1, 12).max(axis=0))

    def merge(self, other: "OfficialGuardAudit"):
        self._check_profile(other.profile)
        self.state_samples += other.state_samples
        self.terminal_samples += other.terminal_samples
        for key in METRICS:
            row, rhs = self.data[key], other.data[key]
            row["observed"] += rhs["observed"]
            row["counts"] += rhs["counts"]
            row["min"] = np.minimum(row["min"], rhs["min"])
            for metric in ("max", "peak_abs", "ratio", "excess"):
                row[metric] = np.maximum(row[metric], rhs[metric])
        return self

    def summary(self) -> dict:
        metrics = {}
        for key, row in self.data.items():
            n = row["observed"]
            count = int(row["counts"].sum())
            metrics[key] = {"value_and_excess_unit": METRIC_UNITS[key],
                            "observed_state_samples": n, "unobserved_state_samples": self.state_samples - n,
                            "sample_count": n * 12, "excess_count": count,
                            "excess_fraction": count / (n * 12) if n else None,
                            "max_ratio": float(row["ratio"].max()) if n else None,
                            "max_excess": float(row["excess"].max()) if n else None,
                            "peak_absolute_value": float(row["peak_abs"].max()) if n else None,
                            "per_joint": {name: {"sample_count": n, "excess_count": int(row["counts"][i]),
                                                 "excess_fraction": float(row["counts"][i] / n) if n else None,
                                                 "min_value": float(row["min"][i]) if n else None,
                                                 "max_value": float(row["max"][i]) if n else None,
                                                 "peak_absolute_value": float(row["peak_abs"][i]) if n else None,
                                                 "max_ratio": float(row["ratio"][i]) if n else None,
                                                 "max_excess": float(row["excess"][i]) if n else None}
                                          for i, name in enumerate(self.profile.names)}}
        complete = bool(self.state_samples) and all(row["observed"] == self.state_samples for row in self.data.values())
        observed_violation = any(np.any(row["counts"]) for row in self.data.values())
        passed = False if observed_violation else True if complete else None
        return {"profile": self.profile.metadata(), "physics_state_samples": self.state_samples,
                "terminal_physics_state_samples": self.terminal_samples,
                "startup_grace_samples": 0, "metrics": metrics, "all_metrics_observed": complete,
                "joint_and_torque_acceptance_passed": passed,
                "acceptance_scope": "sampled joint and torque checks only; excludes task success and runtime watchdogs"}
