"""Evaluation-only URDF velocity calibration; no simulator or tensor imports.

The profile supplies robot-description limits, not measured firmware cutoffs.
This module never changes observations, measured velocities, or audit bounds.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path


GO2_JOINT_NAMES = tuple(f"{leg}_{kind}_joint" for leg in ("FL", "FR", "RL", "RR")
                        for kind in ("hip", "thigh", "calf"))


def _distinct_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _joint_names(names):
    if (len(names) != 12 or any(not isinstance(name, str) for name in names)
            or len(set(names)) != 12 or set(names) != set(GO2_JOINT_NAMES)):
        raise ValueError("Expected exactly the 12 distinct Go2 joint names")


def _positive(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("Joint velocity limits must be positive finite numbers")
    return float(value)


@dataclass(frozen=True)
class EvaluationVelocityProfile:
    path: str
    sha256: str
    limits: dict[str, float]

    def apply(self, actuator_cfg):
        """Set only this evaluation's actuator speed caps, by exact joint name."""
        actuator_cfg.velocity_limit = dict(self.limits)
        actuator_cfg.velocity_limit_sim = dict(self.limits)

    def metadata(self):
        return {
            "profile_path": self.path,
            "profile_sha256": self.sha256,
            "scope": "evaluation_only_actuator_and_physics_velocity_limits",
            "source_field": "description_velocity_limit_rad_s",
            "interpretation": "URDF specification; not drive firmware readback or a real-world safety guarantee",
            "requested_limits_rad_s": dict(self.limits),
            "measured_velocity_clipping": False,
            "audit_velocity_limit_rad_s": 13.0,
        }


def load_evaluation_velocity_profile(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    data = json.loads(raw, object_pairs_hook=_distinct_object)
    if not isinstance(data, dict) or not isinstance(data.get("joints"), list):
        raise ValueError("Velocity profile must contain a joints array")
    rows = data["joints"]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Each joint profile entry must be an object")
    names = [row.get("joint_name") for row in rows]
    _joint_names(names)
    limits = {name: _positive(row.get("description_velocity_limit_rad_s"))
              for name, row in zip(names, rows)}
    return EvaluationVelocityProfile(str(path), hashlib.sha256(raw).hexdigest(), limits)


def _readback(profile, names, rows, label):
    """Check all environments, not merely the first instance's settings."""
    names = list(names)
    _joint_names(names)
    if not rows or any(not isinstance(row, (tuple, list)) or len(row) != 12 for row in rows):
        raise ValueError(f"{label} readback must have shape (num_envs, 12)")
    values = [[_positive(value) for value in row] for row in rows]
    result = {}
    for index, name in enumerate(names):
        column = [row[index] for row in values]
        if any(not math.isclose(value, profile.limits[name], rel_tol=1e-6, abs_tol=1e-6)
               for value in column):
            raise ValueError(f"{label} velocity cap mismatch for {name}")
        result[name] = {"min_rad_s": min(column), "max_rad_s": max(column)}
    return {"verified_environments": len(rows), "per_joint": result}


def verify_evaluation_velocity_readback(profile, robot, audit_velocity_limit):
    """Fail closed on unresolved PhysX/actuator caps or a changed 13 rad/s audit.

    Called after environment creation and again before saving the final report.
    PhysX data are read directly instead of relying on the requested config.
    """
    if audit_velocity_limit != 13.0:
        raise ValueError("Calibrated evaluation requires the existing 13 rad/s audit threshold")
    actuator = robot.actuators["legs"]
    sources = {
        "physics": (robot.joint_names, robot.root_physx_view.get_dof_max_velocities()),
        "actuator": (actuator.joint_names, actuator.velocity_limit),
        "actuator_sim": (actuator.joint_names, actuator.velocity_limit_sim),
    }
    result = {label: _readback(profile, names, values.detach().cpu().tolist(), label)
              for label, (names, values) in sources.items()}
    if len({row["verified_environments"] for row in result.values()}) != 1:
        raise ValueError("Physics and actuator velocity readback environment counts differ")
    return result
