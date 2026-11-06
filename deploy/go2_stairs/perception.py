"""Conservative forward stair geometry, with no transport or motor interface.

Input points are metres in an already gravity-aligned, heading-forward local
frame (x forward, y left, z up). The caller owns calibration and time alignment.
This prototype reports visible geometry, never permission to move.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class PerceptionConfig:
    x_min: float = 0.15
    x_max: float = 2.5
    y_half_width: float = 0.55
    z_min: float = -0.8
    z_max: float = 1.8
    lane_centres: tuple = (-0.30, 0.0, 0.30)
    lane_half_width: float = 0.10
    min_lane_span: float = 0.12
    bin_width: float = 0.04
    min_points_per_bin: int = 3
    min_plateau_width: float = 0.16
    max_plateau_slope: float = 0.10
    max_flat_slope: float = 0.02
    max_plateau_residual: float = 0.015
    max_bin_height_span: float = 0.055
    min_step_height: float = 0.07
    max_step_height: float = 0.24
    min_tread_depth: float = 0.18
    max_tread_depth: float = 0.60
    max_step_height_difference: float = 0.025
    max_tread_depth_difference: float = 0.08
    max_lane_edge_difference: float = 0.10
    min_steps: int = 3
    max_edge_missing_bins: int = 1
    min_stair_bin_coverage: float = 0.85
    max_input_points: int = 1_000_000

    def __post_init__(self):
        values = (self.x_min, self.x_max, self.y_half_width, self.z_min, self.z_max,
                  self.lane_half_width, self.min_lane_span, self.bin_width,
                  self.min_plateau_width, self.max_plateau_slope, self.max_flat_slope,
                  self.max_plateau_residual, self.max_bin_height_span,
                  self.min_step_height, self.max_step_height, self.min_tread_depth,
                  self.max_tread_depth, self.max_step_height_difference,
                  self.max_tread_depth_difference, self.max_lane_edge_difference,
                  self.min_stair_bin_coverage, *self.lane_centres)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Configuration must be finite")
        if not (0 <= self.x_min < self.x_max and self.z_min < self.z_max):
            raise ValueError("Invalid ROI")
        if not (0 < self.min_step_height < self.max_step_height
                and 0 < self.min_tread_depth < self.max_tread_depth):
            raise ValueError("Invalid step geometry ranges")
        positive = (self.y_half_width, self.lane_half_width, self.min_lane_span,
                    self.bin_width, self.min_plateau_width, self.max_plateau_slope, self.max_flat_slope,
                    self.max_plateau_residual, self.max_bin_height_span,
                    self.max_step_height_difference, self.max_tread_depth_difference,
                    self.max_lane_edge_difference)
        if min(positive) <= 0 or not 0 < self.min_stair_bin_coverage <= 1:
            raise ValueError("Invalid positive threshold or coverage")
        for name in ("min_points_per_bin", "min_steps", "max_input_points"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(name + " must be a positive integer")
        if self.min_steps < 3 or self.min_points_per_bin < 3:
            raise ValueError("At least three steps and three unique points per bin are required")
        if type(self.max_edge_missing_bins) is not int or not 0 <= self.max_edge_missing_bins <= 1:
            raise ValueError("At most one unobserved edge bin is supported")
        centres = self.lane_centres
        if (len(centres) < 3 or tuple(sorted(set(centres))) != tuple(centres)
                or min(centres) >= 0 or max(centres) <= 0
                or min(abs(v) for v in centres) > self.lane_half_width
                or any(abs(v) + self.lane_half_width > self.y_half_width for v in centres)
                or any(b-a < 2*self.lane_half_width for a, b in zip(centres, centres[1:]))):
            raise ValueError("Require separate left, centre and right lanes inside ROI")
        if self.min_lane_span > 2*self.lane_half_width or self.bin_width > self.min_plateau_width/3:
            raise ValueError("Insufficient lateral or longitudinal resolution")
        if math.ceil((self.x_max-self.x_min)/self.bin_width) > 1000:
            raise ValueError("At most 1000 longitudinal bins are supported")


def _plateau(indices, centres, heights, cfg):
    x, z = centres[indices], heights[indices]
    width = len(indices)*cfg.bin_width
    if width < cfg.min_plateau_width or len(indices) < 3:
        return None
    dx = x-x.mean()
    slope = float(np.dot(dx, z-z.mean()) / np.dot(dx, dx))
    residual = float(np.max(np.abs(z-(z.mean()+slope*dx))))
    if abs(slope) > cfg.max_plateau_slope or residual > cfg.max_plateau_residual:
        return None
    return {"start": int(indices[0]), "end": int(indices[-1]),
            "height": float(np.median(z)), "width": float(width), "slope": slope}


def _lane(points, lane_y, cfg, bins, centres):
    lane = points[np.abs(points[:, 1]-lane_y) <= cfg.lane_half_width]
    heights = np.full(bins, np.nan)
    observed_min = np.full(bins, np.nan)
    observed_max = np.full(bins, np.nan)
    support = np.zeros(bins, dtype=np.int64)
    bad_height_bins = 0
    tall_vertical_bins = 0
    if len(lane):
        ids = np.floor((lane[:, 0]-cfg.x_min)/cfg.bin_width).astype(np.int64)
        ids = np.minimum(ids, bins-1)
        for index in range(bins):
            sample = lane[ids == index]
            support[index] = len(sample)
            if len(sample):
                observed_min[index] = np.min(sample[:, 2])
                observed_max[index] = np.max(sample[:, 2])
                if np.ptp(sample[:, 2]) > cfg.max_step_height+cfg.max_step_height_difference:
                    tall_vertical_bins += 1
            if len(sample) < cfg.min_points_per_bin:
                continue
            if np.ptp(sample[:, 1]) < cfg.min_lane_span:
                continue
            # An arbitrary vertical return must not be hidden by a floor median.
            if np.ptp(sample[:, 2]) > cfg.max_bin_height_span:
                bad_height_bins += 1
                continue
            heights[index] = np.median(sample[:, 2])
    valid = np.flatnonzero(np.isfinite(heights))
    segments, current = [], []
    for index in valid:
        if current and (index != current[-1]+1 or
                        abs(heights[index]-heights[current[-1]]) >= cfg.min_step_height):
            segments.append(np.array(current, dtype=int))
            current = []
        current.append(int(index))
    if current:
        segments.append(np.array(current, dtype=int))
    plateaus = [_plateau(segment, centres, heights, cfg) for segment in segments]
    y_span = float(np.ptp(lane[:, 1])) if len(lane) else 0.0
    coverage = {"centre_y_m": float(lane_y), "unique_points": int(len(lane)),
                "valid_bins": int(len(valid)), "total_bins": bins,
                "valid_fraction": float(len(valid)/bins), "lateral_span_m": y_span,
                "vertical_or_mixed_bins": bad_height_bins,
                "tall_vertical_bins": tall_vertical_bins,
                "horizontal_plateaus": sum(p is not None for p in plateaus),
                "observed_x_min_m": float(centres[valid[0]]) if len(valid) else None,
                "observed_x_max_m": float(centres[valid[-1]]) if len(valid) else None}
    # Missing near or far footprint cannot be treated as a known level surface.
    covered = (len(valid) > 0 and valid[0] <= 1 and valid[-1] >= bins-2
               and y_span >= cfg.min_lane_span)
    flat = (covered and len(valid) == bins and len(plateaus) == 1
            and plateaus[0] is not None
            and abs(plateaus[0]["slope"]) <= cfg.max_flat_slope
            and np.ptp(heights[valid]) <= cfg.max_step_height_difference)
    candidate = None
    if (covered and tall_vertical_bins == 0 and len(valid)/bins >= cfg.min_stair_bin_coverage
            and len(plateaus) >= cfg.min_steps+1 and all(p is not None for p in plateaus)):
        gaps = [b["start"]-a["end"]-1 for a, b in zip(plateaus, plateaus[1:])]
        changes = np.diff([p["height"] for p in plateaus])
        edges = np.array([(centres[a["end"]]+centres[b["start"]])/2
                          for a, b in zip(plateaus, plateaus[1:])])
        depths = np.diff(edges)
        rises = np.abs(changes)
        same_direction = np.all(changes > 0) or np.all(changes < 0)
        edge_envelopes_valid = True
        for a, b in zip(plateaus, plateaus[1:]):
            for index in range(a["end"]+1, b["start"]):
                low = min(a["height"], b["height"])-cfg.max_step_height_difference
                high = max(a["height"], b["height"])+cfg.max_step_height_difference
                if np.isfinite(observed_min[index]) and (observed_min[index] < low or observed_max[index] > high):
                    edge_envelopes_valid = False
        if (max(gaps) <= cfg.max_edge_missing_bins and same_direction and edge_envelopes_valid
                and np.all(rises >= cfg.min_step_height) and np.all(rises <= cfg.max_step_height)
                and np.ptp(rises) <= cfg.max_step_height_difference
                and np.all(depths >= cfg.min_tread_depth) and np.all(depths <= cfg.max_tread_depth)
                and np.ptp(depths) <= cfg.max_tread_depth_difference):
            candidate = {"direction": 1 if changes[0] > 0 else -1,
                         "edges": edges, "heights": changes,
                         "levels": np.array([p["height"] for p in plateaus]),
                         "height": float(np.median(rises)), "depth": float(np.median(depths)),
                         "steps": len(changes), "support": int(support[np.isfinite(heights)].sum())}
    return coverage, candidate, flat, plateaus


def analyze_stairs(points_xyz, config: Optional[PerceptionConfig] = None):
    """Classify visible forward-aligned geometry; the return value is JSON-safe.

    Unknown bins are never filled. One missing bin is permitted only between
    two measured horizontal treads with an independently measured height jump.
    No slope, boxed platform, or isolated step is considered a stair flight.
    """
    cfg = config if config is not None else PerceptionConfig()
    if not isinstance(cfg, PerceptionConfig):
        raise TypeError("config must be PerceptionConfig")
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected an Nx3 point cloud")
    if len(points) > cfg.max_input_points:
        raise ValueError("Point cloud exceeds configured processing limit")
    finite = np.isfinite(points).all(axis=1)
    filtered = points[finite]
    roi = ((filtered[:, 0] >= cfg.x_min) & (filtered[:, 0] < cfg.x_max)
           & (np.abs(filtered[:, 1]) <= cfg.y_half_width)
           & (filtered[:, 2] >= cfg.z_min) & (filtered[:, 2] <= cfg.z_max))
    selected = filtered[roi]
    unique = np.unique(selected, axis=0)
    bins = int(math.ceil((cfg.x_max-cfg.x_min)/cfg.bin_width))
    centres = cfg.x_min+(np.arange(bins)+0.5)*cfg.bin_width
    lanes = [_lane(unique, y, cfg, bins, centres) for y in cfg.lane_centres]
    coverage = {"input_points": len(points), "nonfinite_points_removed": int((~finite).sum()),
                "roi_points": len(selected), "unique_roi_points": len(unique),
                "roi_m": {"x": [cfg.x_min, cfg.x_max], "y": [-cfg.y_half_width, cfg.y_half_width],
                          "z": [cfg.z_min, cfg.z_max]},
                "lanes": [lane[0] for lane in lanes],
                "required_lanes": len(lanes), "rear_footprint_observed": False}
    result = {"state": "unknown", "geometry": None, "score": 0.0,
              "score_definition": "Heuristic geometric support in [0,1], not a calibrated probability or safety confidence",
              "control_authorized": False, "coverage": coverage,
              "reason": "Insufficient or inconsistent visible horizontal footprint",
              "assumptions": ["Metres; x forward, y left, z up; already gravity aligned",
                              "Calibrated sensor-to-body transform and synchronized pose supplied by caller",
                              "Straight stair flight approximately aligned with forward direction"],
              "limits": ["Prototype geometry only; no collision clearance, load capacity or traversability guarantee",
                         "Required lanes must cover the full forward ROI; sparse or occluded scans can miss real stairs",
                         "No rear-foot or landing-clearance proof; flat means the visible forward footprint only",
                         "No freshness check here; caller must reject stale data and invalid transforms",
                         "The existing 1100 policy has not been validated for real stairs or descending stairs"]}
    candidates = [lane[1] for lane in lanes]
    if all(candidate is not None for candidate in candidates):
        first = candidates[0]
        consistent = all(c["direction"] == first["direction"] and c["steps"] == first["steps"]
                         for c in candidates)
        if consistent:
            edges = np.stack([c["edges"] for c in candidates])
            heights = np.stack([c["heights"] for c in candidates])
            levels = np.stack([c["levels"] for c in candidates])
            depths = np.array([c["depth"] for c in candidates])
            consistent = (np.max(np.ptp(edges, axis=0)) <= cfg.max_lane_edge_difference
                          and np.max(np.ptp(heights, axis=0)) <= cfg.max_step_height_difference
                          and np.max(np.ptp(levels, axis=0)) <= cfg.max_step_height_difference
                          and np.ptp(depths) <= cfg.max_tread_depth_difference)
        if consistent:
            result.update(state="stairs_up" if first["direction"] > 0 else "stairs_down",
                          score=float(min(lane[0]["valid_fraction"] for lane in lanes)),
                          reason="At least three consistent height changes with horizontal treads in every required lane",
                          geometry={"distance_m": float(np.median(edges[:, 0])),
                                    "distance_semantics": "First visible edge of this candidate, not guaranteed the first physical stair",
                                    "step_height_m": float(np.median(np.abs(heights))),
                                    "signed_step_height_m": float(np.median(heights)),
                                    "tread_depth_m": float(np.median(depths)),
                                    "visible_steps": int(first["steps"]),
                                    "supported_lanes": len(lanes),
                                    "supported_width_m": float(max(cfg.lane_centres)-min(cfg.lane_centres)
                                                               +min(lane[0]["lateral_span_m"] for lane in lanes)),
                                    "support_points": sum(c["support"] for c in candidates)})
            return result
        result["reason"] = "Different lanes disagree on stair edges or elevations"
        return result
    if all(lane[2] for lane in lanes):
        levels = [lane[3][0]["height"] for lane in lanes]
        if np.ptp(levels) <= cfg.max_step_height_difference:
            result.update(state="flat", score=1.0,
                          reason="Fully observed horizontal forward footprint in every required lane")
            return result
    if all(lane[0]["unique_points"] >= cfg.min_points_per_bin*3 for lane in lanes):
        if any(lane[0]["vertical_or_mixed_bins"] > 0 or lane[0]["tall_vertical_bins"] > 0 for lane in lanes):
            result.update(state="obstacle", reason="Vertical/mixed returns interrupt the footprint; stairs not established")
        elif all(lane[0]["horizontal_plateaus"] >= 2 for lane in lanes):
            result.update(state="obstacle", reason="Height discontinuities without a consistent stair flight")
    return result


class TemporalConfirmation:
    """Confirm independent fresh frames; never authorizes robot control.

    stamp_s and now_s must use the same clock. A caller must not substitute a
    new receipt timestamp for an old acquisition timestamp. Duplicate/backward
    stamps and unknown/obstacle/expired data immediately clear confirmation.
    Confirmation concerns the label, not stable geometry. The caller must also
    expire cached results when no new data arrives; this object has no timer.
    """
    def __init__(self, required_frames=3, min_duration_s=0.20, max_age_s=0.30,
                 max_gap_s=0.30, future_tolerance_s=0.02):
        if type(required_frames) is not int or required_frames < 2:
            raise ValueError("required_frames must be an integer >=2")
        vals = (min_duration_s, max_age_s, max_gap_s, future_tolerance_s)
        if not all(math.isfinite(v) for v in vals) or min(vals) < 0 or min(max_age_s, max_gap_s) <= 0:
            raise ValueError("Invalid timing configuration")
        self.required_frames, self.min_duration_s = required_frames, min_duration_s
        self.max_age_s, self.max_gap_s = max_age_s, max_gap_s
        self.future_tolerance_s = future_tolerance_s
        self.reset()

    def reset(self):
        self.last_stamp = self.last_now = self.started = None
        self.candidate = None
        self.count = 0

    def update(self, result, stamp_s, now_s):
        observed = result.get("state", "unknown")
        output = dict(result)
        output["observed_state"] = observed
        output["control_authorized"] = False
        reason = None
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (stamp_s, now_s)):
            reason = "Invalid timestamp"
        elif now_s-stamp_s > self.max_age_s or stamp_s-now_s > self.future_tolerance_s:
            reason = "Expired or future source timestamp"
        elif self.last_now is not None and now_s < self.last_now:
            reason = "Clock moved backward; reset required"
        elif self.last_stamp is not None and stamp_s <= self.last_stamp:
            reason = "Duplicate or backward source timestamp"
        if reason is not None or observed not in ("stairs_up", "stairs_down", "flat"):
            self.candidate, self.started, self.count = None, None, 0
            if reason is None:
                self.last_stamp, self.last_now = stamp_s, now_s
            output["state"] = "unknown" if reason is not None else observed
            output["confirmation"] = {"confirmed": False, "frames": 0, "duration_s": 0.0,
                                      "reason": reason or "Uncertain or obstacle geometry clears confirmation"}
            return output
        gap = self.last_stamp is not None and stamp_s-self.last_stamp > self.max_gap_s
        if self.candidate != observed or gap:
            self.candidate, self.started, self.count = observed, stamp_s, 0
        self.count += 1
        duration = stamp_s-self.started
        confirmed = self.count >= self.required_frames and duration+1e-9 >= self.min_duration_s
        self.last_stamp, self.last_now = stamp_s, now_s
        output["state"] = observed if confirmed else "unknown"
        output["confirmation"] = {"confirmed": bool(confirmed), "frames": self.count,
                                  "duration_s": float(duration),
                                  "reason": "Independent fresh frames confirmed" if confirmed else "Awaiting independent fresh frames"}
        return output
