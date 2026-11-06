"""Simulator-free joint limits sampled before commands and after every physics step.

The 13 rad/s speed limit is the current deployment guard's software threshold,
not the URDF's assumed 30 rad/s motor limit. No target or measured state is
clipped here. Failed and terminal episodes remain part of the audit.
"""
from __future__ import annotations

import math
import torch


def joint_position_limits(joint_names, device="cpu"):
    """Return the model URDF's asymmetric [lower, upper] bounds by joint name."""
    names = list(joint_names)
    expected = {f"{leg}_{kind}_joint" for leg in ("FL", "FR", "RL", "RR")
                for kind in ("hip", "thigh", "calf")}
    if len(names) != 12 or len(set(names)) != 12 or set(names) != expected:
        raise ValueError("Expected the 12 distinct Go2 joint names; ordering is not assumed")
    values = []
    for name in names:
        leg, kind, _ = name.split("_")
        if kind == "hip":
            bounds = (-1.0472, 1.0472)
        elif kind == "calf":
            bounds = (-2.7227, -0.83776)
        elif leg.startswith("F"):
            bounds = (-1.5708, 3.4907)
        else:
            bounds = (-0.5236, 4.5379)
        values.append(bounds)
    return torch.tensor(values, dtype=torch.float32, device=device)


def _number(value, name, *, ratio=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if not (0 < value < 1 if ratio else value > 0):
        raise ValueError(f"{name} must be in (0, 1)" if ratio else f"{name} must be positive")
    return float(value)


class JointConstraintWindow:
    """Frozen control-interval evidence with one command/state pair per substep.

    ``record_command`` observes the final PD target before integration;
    ``record_state`` must run AFTER each physical integration and scene update.
    The latter includes the final and terminal substep, before automatic reset.
    Initialization and writes following a complete interval are ignored.
    """
    KINDS = ("velocity", "position", "target_position", "tracking_error")
    PENALTIES = ("velocity_margin", "velocity_excess", "position_margin",
                 "position_excess", "target_position_excess", "tracking_error")

    def __init__(self, num_envs, joint_names, device="cpu", velocity_limit=13.0,
                 velocity_soft_ratio=.8, position_soft_ratio=.9,
                 tracking_error_limit=.9, tracking_soft_ratio=.8):
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("Positive integer environment count required")
        self.names = list(joint_names)
        self.limits = joint_position_limits(self.names, device)
        self.lower, self.upper = self.limits.unbind(-1)
        self.center = (self.lower + self.upper) * .5
        self.half_range = (self.upper - self.lower) * .5
        self.velocity_limit = _number(velocity_limit, "velocity_limit")
        self.velocity_soft_ratio = _number(velocity_soft_ratio, "velocity_soft_ratio", ratio=True)
        self.position_soft_ratio = _number(position_soft_ratio, "position_soft_ratio", ratio=True)
        self.tracking_error_limit = _number(tracking_error_limit, "tracking_error_limit")
        self.tracking_soft_ratio = _number(tracking_soft_ratio, "tracking_soft_ratio", ratio=True)
        shape = (num_envs, len(self.names))
        self.counts = {kind: torch.zeros(shape, dtype=torch.long, device=device) for kind in self.KINDS}
        self.peaks = {kind: torch.zeros(shape, device=device) for kind in self.KINDS}
        self.peak_ratios = {kind: torch.zeros(shape, device=device) for kind in self.KINDS}
        self.penalty_sums = {kind: torch.zeros(num_envs, device=device) for kind in self.PENALTIES}
        self._pending_target = torch.zeros(shape, device=device)
        self._pending_tracking = torch.zeros_like(self._pending_target)
        self.startup = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.expected = self.command_samples = self.state_samples = 0
        self.invalid = False

    @property
    def samples(self):
        return self.state_samples

    def begin(self, substeps, startup_mask):
        if type(substeps) is not int or substeps < 1:
            raise ValueError("Positive integer physics substep count required")
        self._mask(startup_mask)
        if self.invalid:
            raise RuntimeError("Invalid joint constraint window cannot be rearmed")
        if self.expected:
            self.require_complete()
        for collection in (self.counts, self.peaks, self.peak_ratios, self.penalty_sums):
            for value in collection.values():
                value.zero_()
        self._pending_target.zero_()
        self._pending_tracking.zero_()
        self.startup.copy_(startup_mask)
        self.expected, self.command_samples, self.state_samples = substeps, 0, 0
        self.invalid = False

    def _mask(self, value):
        if (not isinstance(value, torch.Tensor) or value.shape != self.startup.shape
                or value.dtype != torch.bool or value.device != self.startup.device):
            raise ValueError("One boolean flag per environment on the window device required")

    def _check_sample(self, *values):
        if self.invalid:
            raise RuntimeError("Invalid joint constraint window cannot be certified")
        for value in values:
            if (not isinstance(value, torch.Tensor) or value.shape != self.peaks["velocity"].shape
                    or value.device != self.startup.device or not value.is_floating_point()):
                self.invalid = True
                raise ValueError("Joint sample must have shape (num_envs, 12), floating dtype and matching device")
        self._check_finite(*values)

    def _check_finite(self, *values):
        # A single host/device synchronization for the whole group, rather
        # than synchronizing once per joint tensor or penalty component.
        if not bool(torch.stack([torch.isfinite(value).all() for value in values]).all()):
            self.invalid = True
            raise FloatingPointError("Nonfinite joint data or constraint calculation; cannot certify interval")

    def _position(self, position):
        # Test in physical coordinates, so exact lower/upper boundaries remain
        # legal despite roundoff in the center/half-range normalization.
        overshoot = (self.lower - position).clamp_min(0) + (position - self.upper).clamp_min(0)
        ratio = ((position - self.center) / self.half_range).abs()
        return overshoot, ratio

    def _record(self, kind, metric, ratio, excess):
        self.peaks[kind].copy_(torch.maximum(self.peaks[kind], metric))
        self.peak_ratios[kind].copy_(torch.maximum(self.peak_ratios[kind], ratio))
        self.counts[kind].add_(excess)

    def _add_penalty(self, kind, value):
        self.penalty_sums[kind].add_(value)

    def record_command(self, target, position):
        if self.expected == 0 or (self.command_samples == self.expected and self.state_samples == self.expected):
            return
        self._check_sample(target, position)
        if self.command_samples != self.state_samples or self.command_samples >= self.expected:
            self.invalid = True
            raise RuntimeError("Each command sample must be followed by exactly one post-physics state sample")
        target, position = target.detach(), position.detach()
        overshoot, ratio = self._position(target)
        tracking = (target - position).abs()
        self._record("target_position", overshoot, ratio, overshoot > 0)
        # Hold the precise target for this physical substep. A manager/actuator
        # may later reuse its action tensor, so never retain an alias to it.
        self._pending_target.copy_(target)
        self._pending_tracking.copy_(tracking)
        self._add_penalty("target_position_excess", (overshoot / self.half_range).square().sum(-1))
        self._check_finite(ratio, self._pending_target, self._pending_tracking,
                           self.penalty_sums["target_position_excess"])
        self.command_samples += 1

    def record_state(self, position, velocity):
        if self.expected == 0 or (self.command_samples == self.expected and self.state_samples == self.expected):
            return
        self._check_sample(position, velocity)
        if self.command_samples != self.state_samples + 1 or self.state_samples >= self.expected:
            self.invalid = True
            raise RuntimeError("Post-physics state sample requires its preceding command sample")
        position, velocity = position.detach(), velocity.detach().abs()
        overshoot, position_ratio = self._position(position)
        velocity_ratio = velocity / self.velocity_limit
        # Tracking error can cross 0.9 rad only AFTER the final integration,
        # even when actual q and dq remain legal. Retain the larger endpoint
        # error and count it once per physical substep, including terminal ones.
        tracking = torch.maximum(self._pending_tracking, (self._pending_target - position).abs())
        tracking_ratio = tracking / self.tracking_error_limit
        self._record("position", overshoot, position_ratio, overshoot > 0)
        self._record("velocity", velocity, velocity_ratio, velocity > self.velocity_limit)
        self._record("tracking_error", tracking, tracking_ratio, tracking > self.tracking_error_limit)
        self._add_penalty("position_margin", (position_ratio - self.position_soft_ratio).clamp_min(0).square().sum(-1))
        self._add_penalty("position_excess", (overshoot / self.half_range).square().sum(-1))
        self._add_penalty("velocity_margin", (velocity_ratio - self.velocity_soft_ratio).clamp_min(0).square().sum(-1))
        self._add_penalty("velocity_excess", (velocity_ratio - 1).clamp_min(0).square().sum(-1))
        self._add_penalty("tracking_error", (tracking_ratio - self.tracking_soft_ratio).clamp_min(0).square().sum(-1))
        self._check_finite(position_ratio, velocity_ratio, tracking_ratio,
                           *(self.penalty_sums[kind] for kind in (
                               "position_margin", "position_excess", "velocity_margin", "velocity_excess", "tracking_error")))
        self.state_samples += 1

    def require_complete(self):
        if (self.invalid or self.expected < 1 or self.command_samples != self.expected
                or self.state_samples != self.expected):
            raise RuntimeError("Joint constraint window invalid/incomplete; missing substeps cannot be certified")

    def penalty(self, kind, startup_multiplier=1.):
        self.require_complete()
        if kind not in self.penalty_sums:
            raise ValueError("Unknown joint constraint penalty")
        multiplier = _number(startup_multiplier, "startup_multiplier")
        if multiplier < 1:
            raise ValueError("startup_multiplier must be >= 1")
        weight = torch.where(self.startup, multiplier, 1.)
        return self.penalty_sums[kind] / self.samples * weight

    def hard_violation(self):
        self.require_complete()
        return torch.stack([counts.any(-1) for counts in self.counts.values()]).any(0)

    def log_metrics(self):
        self.require_complete()
        result = {}
        for kind in self.KINDS:
            result[f"JointLimits/{kind}_excess_fraction"] = self.counts[kind].float().mean() / self.samples
            result[f"JointLimits/{kind}_peak_ratio"] = self.peak_ratios[kind].amax()
        result["JointLimits/violation_environment_fraction"] = self.hard_violation().float().mean()
        for i, name in enumerate(self.names):
            result[f"JointLimits/velocity_peak_rad_s/{name}"] = self.peaks["velocity"][:, i].amax()
            result[f"JointLimits/position_peak_overshoot_rad/{name}"] = self.peaks["position"][:, i].amax()
            result[f"JointLimits/target_position_peak_overshoot_rad/{name}"] = self.peaks["target_position"][:, i].amax()
            result[f"JointLimits/tracking_error_peak_rad/{name}"] = self.peaks["tracking_error"][:, i].amax()
        return {key: value.detach().clone() for key, value in result.items()}


class JointConstraintAudit:
    """Audit active first episodes, retaining the complete failing terminal step."""
    def __init__(self, window):
        self.names = list(window.names)
        self.limits = window.limits.detach().clone()
        self.velocity_limit = window.velocity_limit
        self.tracking_error_limit = window.tracking_error_limit
        self.counts = {kind: torch.zeros(len(self.names), dtype=torch.long, device=self.limits.device)
                       for kind in window.KINDS}
        self.startup_counts = {kind: torch.zeros_like(value) for kind, value in self.counts.items()}
        self.peaks = {kind: torch.zeros(len(self.names), device=self.limits.device) for kind in window.KINDS}
        self.peak_ratios = {kind: torch.zeros_like(value) for kind, value in self.peaks.items()}
        self.samples_per_joint = torch.zeros((), dtype=torch.long, device=self.limits.device)
        self.startup_samples_per_joint = torch.zeros_like(self.samples_per_joint)
        self.invalid = False

    def update(self, window, active_env_mask):
        try:
            window.require_complete()
            window._mask(active_env_mask)
            if (window.names != self.names or window.velocity_limit != self.velocity_limit
                    or window.tracking_error_limit != self.tracking_error_limit
                    or window.limits.device != self.limits.device or not torch.equal(window.limits, self.limits)):
                raise ValueError("Audit cannot mix different joint names, bounds or devices")
        except (ValueError, RuntimeError):
            self.invalid = True
            raise
        if self.invalid:
            raise RuntimeError("Invalid audit cannot be resumed")
        self.samples_per_joint += active_env_mask.sum() * window.samples
        startup = active_env_mask & window.startup
        self.startup_samples_per_joint += startup.sum() * window.samples
        selected = active_env_mask[:, None]
        for kind in self.counts:
            self.counts[kind] += (window.counts[kind] * selected).sum(0)
            self.startup_counts[kind] += (window.counts[kind] * startup[:, None]).sum(0)
            self.peaks[kind] = torch.maximum(self.peaks[kind], window.peaks[kind].masked_fill(~selected, 0).amax(0))
            self.peak_ratios[kind] = torch.maximum(self.peak_ratios[kind], window.peak_ratios[kind].masked_fill(~selected, 0).amax(0))

    def report(self):
        count = int(self.samples_per_joint.item())
        startup_count = int(self.startup_samples_per_joint.item())
        totals = {kind: int(value.sum().item()) for kind, value in self.counts.items()}
        result = {
            "scope": "Every physics substep in active episodes, including failures and terminal steps; commands before integration and measured state after integration",
            "physics_samples_per_joint": count,
            "startup_physics_samples_per_joint": startup_count,
            "joint_limits_passed": not self.invalid and count > 0 and startup_count > 0 and not any(totals.values()),
            "audit_invalid": self.invalid,
            "velocity_limit_rad_s": self.velocity_limit,
            "tracking_error_limit_rad": self.tracking_error_limit,
            "tracking_sampling": "maximum pre/post-integration target error per physical substep, counted once",
            "note": "Zero sampled violations is an evaluation result, not a guarantee for unseen real-world states; no clipping is performed by this audit.",
        }
        for kind, total in totals.items():
            startup_total = int(self.startup_counts[kind].sum().item())
            result[f"{kind}_excess_joint_samples"] = total
            result[f"{kind}_excess_fraction"] = total / (count * len(self.names)) if count else None
            result[f"startup_{kind}_excess_joint_samples"] = startup_total
            result[f"startup_{kind}_excess_fraction"] = startup_total / (startup_count * len(self.names)) if startup_count else None
            result[f"{kind}_peak_ratio"] = float(self.peak_ratios[kind].max()) if count else None
        result["joints"] = {name: {
            "position_lower_rad": float(self.limits[i, 0]),
            "position_upper_rad": float(self.limits[i, 1]),
            "velocity_peak_rad_s": float(self.peaks["velocity"][i]),
            "position_peak_overshoot_rad": float(self.peaks["position"][i]),
            "target_position_peak_overshoot_rad": float(self.peaks["target_position"][i]),
            "tracking_error_peak_rad": float(self.peaks["tracking_error"][i]),
            **{f"{kind}_excess_samples": int(self.counts[kind][i]) for kind in self.counts},
        } for i, name in enumerate(self.names)}
        return result
