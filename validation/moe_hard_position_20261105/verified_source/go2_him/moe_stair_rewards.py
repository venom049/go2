"""Task rewards for MoE-CTS stair fine-tuning, independent of joint guards.

This is a stateless, single-environment NumPy calculation. The environment
owns contacts, terrain measurements, stop commands and one-shot event edges.
Continuous terms are reward units/second; events are reward units/event.
Only ``rate_total`` is multiplied by the control timestep. Joint protection
costs belong to a separate, per-axis/substep audit and are NOT included here.

Reference ideas: saifahmadgit/go2-sim2real-locomotion-rl, commit
0830f3e3582acd5c9b3a5c09186d4f028f54d9dd. Coefficients are deliberately retuned
for the two-step, short-landing task rather than copied across simulators.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np


DEFAULT_CONFIG = {
    "tracking_sigma": 0.25,
    "command_deadzone": 0.10,
    "progress_rate": 5.0,
    "tracking_rate": 4.0,
    "yaw_tracking_rate": 0.4,
    "stall_cost_rate": 0.8,
    "roll_cost_rate": 5.0,
    "angular_xy_cost_rate": 0.03,
    "vertical_deadzone": 0.15,
    "vertical_cost_rate": 0.5,
    "base_height_target": 0.30,
    "base_height_cost_rate": 0.10,
    "foot_clearance_target": 0.17,
    "foot_clearance_cost_rate": 0.5,
    "foot_slip_cost_rate": 0.15,
    "action_change_cost_rate": 0.02,
    "stop_linear_cost_rate": 3.0,
    "stop_yaw_cost_rate": 1.5,
    "stop_contact_cost_rate": 0.15,
    "stop_hold_rate": 1.5,
    "stop_linear_threshold": 0.05,
    "stop_yaw_threshold": 0.10,
    "stop_linear_sigma": 0.0225,
    "stop_yaw_sigma": 0.0625,
    "platform_event_reward": 5.0,
    "success_event_reward": 20.0,
    "failure_event_cost": 10.0,
}

TERM_DESCRIPTIONS = {
    "progress": "Positive dx/dt capped at commanded forward speed, normalized by that speed; disabled in stop phase.",
    "tracking": "Velocity tracking multiplied by actual progress fraction; a stationary moving-command policy earns zero.",
    "yaw_tracking": "Yaw tracking rewarded only while actual forward progress is made.",
    "stall": "Moving-command penalty for a missing fraction of requested progress.",
    "roll": "Squared lateral projected gravity; pitch itself is not penalized.",
    "angular_xy": "Small roll/pitch angular-rate cost, not an orientation cost.",
    "vertical": "Squared body vertical-speed excess beyond the ascent deadzone.",
    "base_height": "Weak squared deviation from terrain-relative base clearance.",
    "foot_clearance": "Swing-foot terrain-relative clearance error weighted by horizontal foot speed and moving command.",
    "foot_slip": "Contact-foot horizontal speed squared; foot positions must refer to the actual feet.",
    "action_change": "Mean squared change of the caller's 12-dimensional action coordinates: raw in legacy modes, inverse-mapped submitted targets in target_clipped mode; no q-default or dq-squared penalty.",
    "stop_motion": "Zero-command horizontal/yaw velocity cost, also active before the platform.",
    "stop_contacts": "Zero-command cost for each foot off the ground.",
    "stop_hold": "Platform stop-phase reward requiring three contacts, zero command, speed <= 0.05 m/s and yaw rate <= 0.10 rad/s.",
    "platform": "One-time newly-reached-platform event supplied by the caller, not platform occupancy.",
    "success": "One-time completed ascent-and-stable-stop event supplied by the caller.",
    "failure": "One-time wall contact, fall or other task failure; suppresses simultaneous positive events.",
}


def _array(state, name, shape):
    value = np.asarray(state[name], dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite with shape {shape}, got {value.shape}")
    return value


def _scalar(state, name, default=None):
    value = state[name] if name in state else default
    if value is None or np.asarray(value).ndim != 0:
        raise ValueError(f"{name} must be a finite scalar")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite scalar")
    return value


def _flag(state, name):
    value = state.get(name, False)
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be boolean, not an event count or occupancy array")
    return bool(value)


def compute_stair_rewards(state: Mapping, config: Mapping | None = None) -> dict:
    """Calculate task rewards for one post-physics control transition.

    Required keys (SI units): ``body_vxy`` (2,), ``commands`` (vx, vy, yaw),
    ``body_w`` (3,), ``gravity`` (body-frame projected unit gravity, 3,),
    ``foot_clearance`` (4, bottom of each foot above its local terrain),
    ``foot_velocity`` (4, 3, world-frame linear velocity), ``foot_contact``
    (4, bool/0-or-1), ``actions``/``prev_actions`` (12, raw policy outputs),
    ``base_height`` (terrain-relative), ``dx`` (world-forward displacement
    over this transition), and positive ``dt``. ``body_vz`` defaults to 0.

    Optional boolean keys: ``stop_phase``, ``new_platform_event``,
    ``success_event``, ``failure_event``. All event flags MUST be rising edges:
    this pure function cannot deduplicate repeated flags or prove a hold time.
    The caller must derive success from actual landing/contact/velocity and
    hold-duration checks, not elapsed time alone. ``jointq``/``q0`` may be
    supplied together (12,) for diagnostics; they never add posture penalties.

    This reward assumes a forward stair corridor, as in the reference task.
    Neither a reward nor a small protection cost certifies a safe trajectory.
    """
    cfg = dict(DEFAULT_CONFIG)
    if config is not None:
        unknown = set(config) - set(cfg)
        if unknown:
            raise ValueError(f"unknown reward configuration keys: {sorted(unknown)}")
        cfg.update(config)
    for name, value in cfg.items():
        if np.asarray(value).ndim or not np.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{name} must be a finite nonnegative scalar")
        cfg[name] = float(value)
    for name in ("tracking_sigma", "command_deadzone", "stop_linear_sigma", "stop_yaw_sigma"):
        if cfg[name] <= 0:
            raise ValueError(f"{name} must be positive")

    dt = _scalar(state, "dt")
    if dt <= 0:
        raise ValueError("dt must be positive")
    vxy = _array(state, "body_vxy", (2,))
    command = _array(state, "commands", (3,))
    omega = _array(state, "body_w", (3,))
    gravity = _array(state, "gravity", (3,))
    feet_height = _array(state, "foot_clearance", (4,))
    feet_velocity = _array(state, "foot_velocity", (4, 3))
    contact_values = _array(state, "foot_contact", (4,))
    if not np.isin(contact_values, [0, 1]).all():
        raise ValueError("foot_contact must contain booleans or 0/1 values")
    contact = contact_values.astype(bool)
    action = _array(state, "actions", (12,))
    previous_action = _array(state, "prev_actions", (12,))
    height = _scalar(state, "base_height")
    vz = _scalar(state, "body_vz", 0.0)
    forward_speed = _scalar(state, "dx") / dt
    stop_phase = _flag(state, "stop_phase")
    failure = _flag(state, "failure_event")
    success = _flag(state, "success_event")
    platform = _flag(state, "new_platform_event")

    commanded_forward = float(command[0])
    moving = commanded_forward > cfg["command_deadzone"] and not stop_phase
    zero_command = np.linalg.norm(command) < cfg["command_deadzone"]
    fraction = float(np.clip(forward_speed / commanded_forward, 0., 1.)) if moving else 0.
    velocity_error = float(np.sum(np.square(vxy - command[:2])))
    yaw_error = float(np.square(omega[2] - command[2]))
    foot_xy_speed = np.linalg.norm(feet_velocity[:, :2], axis=1)
    stopped = (stop_phase and zero_command and np.sum(contact) >= 3
               and np.linalg.norm(vxy) <= cfg["stop_linear_threshold"]
               and abs(omega[2]) <= cfg["stop_yaw_threshold"])

    rates = {
        "progress": cfg["progress_rate"] * fraction,
        "tracking": cfg["tracking_rate"] * fraction * float(np.exp(-velocity_error / cfg["tracking_sigma"])),
        "yaw_tracking": cfg["yaw_tracking_rate"] * fraction * float(np.exp(-yaw_error / cfg["tracking_sigma"])),
        "stall": -cfg["stall_cost_rate"] * (1. - fraction) if moving else 0.,
        "roll": -cfg["roll_cost_rate"] * float(gravity[1] ** 2),
        "angular_xy": -cfg["angular_xy_cost_rate"] * float(np.sum(omega[:2] ** 2)),
        "vertical": -cfg["vertical_cost_rate"] * max(abs(vz) - cfg["vertical_deadzone"], 0.) ** 2,
        "base_height": -cfg["base_height_cost_rate"] * (height - cfg["base_height_target"]) ** 2,
        "foot_clearance": -cfg["foot_clearance_cost_rate"] * float(np.sum(
            (~contact) * (feet_height - cfg["foot_clearance_target"]) ** 2 * foot_xy_speed
        )) if moving else 0.,
        "foot_slip": -cfg["foot_slip_cost_rate"] * float(np.sum(contact * foot_xy_speed ** 2)),
        "action_change": -cfg["action_change_cost_rate"] * float(np.mean((action - previous_action) ** 2)),
        "stop_motion": -(cfg["stop_linear_cost_rate"] * float(vxy @ vxy)
                         + cfg["stop_yaw_cost_rate"] * float(omega[2] ** 2)) if zero_command else 0.,
        "stop_contacts": -cfg["stop_contact_cost_rate"] * int(np.sum(~contact)) if zero_command else 0.,
        "stop_hold": cfg["stop_hold_rate"] * float(np.exp(
            -float(vxy @ vxy) / cfg["stop_linear_sigma"] - float(omega[2] ** 2) / cfg["stop_yaw_sigma"]
        )) if stopped else 0.,
    }
    events = {
        "platform": cfg["platform_event_reward"] if platform and not failure else 0.,
        "success": cfg["success_event_reward"] if success and not failure else 0.,
        "failure": -cfg["failure_event_cost"] if failure else 0.,
    }
    diagnostics = {
        "forward_speed": forward_speed,
        "progress_fraction": fraction,
        "moving_command": bool(moving),
        "zero_command": bool(zero_command),
        "stop_phase": stop_phase,
        "stop_velocity_and_contact_check": bool(stopped),
        "contact_count": int(np.sum(contact)),
        "joint_guard_cost_included": False,
    }
    if "jointq" in state or "q0" in state:
        q = _array(state, "jointq", (12,))
        q0 = _array(state, "q0", (12,))
        diagnostics["default_pose_deviation_l1_diagnostic_only"] = float(np.sum(np.abs(q - q0)))

    rate_total = float(sum(rates.values()))
    event_total = float(sum(events.values()))
    step_reward = rate_total * dt + event_total
    if not np.isfinite(step_reward):
        raise ValueError("nonfinite reward; inspect the physical state instead of silently clipping it")
    return {
        "rate_terms": rates,
        "event_terms": events,
        "rate_total": rate_total,
        "event_total": event_total,
        "step_reward": step_reward,
        "diagnostics": diagnostics,
    }
