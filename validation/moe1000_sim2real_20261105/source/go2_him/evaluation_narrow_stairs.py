"""Evaluation-only finite-width stairs; never modifies a training configuration.

The upper landing is an explicit simulation assumption, not a measured part of
the user's staircase. Geometric completion is not a real-robot safety approval.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from .lab_terrain import course_geometry


FOOT_NAMES = tuple(f"{leg}_foot" for leg in ("FL", "FR", "RL", "RR"))
FLIGHT_WIDTH_M = .60
WIDE_FLIGHT_WIDTH_M = .90


def ascending_narrow_stairs(difficulty, cfg):
    """Keep the entire tile floor, with an explicit 60 cm flight.

    Isaac Lab copies sub-terrain configs from their declared fields, dropping
    dynamically attached fields. Each supported width therefore has a named
    module-level function that survives cfg.copy() without extra config fields.
    """
    return _ascending_stairs_with_width(difficulty, cfg, FLIGHT_WIDTH_M)


def ascending_narrow_stairs_09(difficulty, cfg):
    """Explicit 90 cm counterpart, preserving every other terrain dimension."""
    return _ascending_stairs_with_width(difficulty, cfg, WIDE_FLIGHT_WIDTH_M)


def _ascending_stairs_with_width(difficulty, cfg, width):
    import trimesh

    course = course_geometry(difficulty, cfg)
    if not math.isfinite(width) or not 0 < width < course.width:
        raise ValueError("Flight width must be positive and narrower than its tile")
    transform = np.eye(4)
    transform[:3, 3] = (course.length / 2, course.width / 2, -.05)
    meshes = [trimesh.creation.box(extents=(course.length, course.width, .1), transform=transform)]
    for start, end, surface_z in course.segments():
        if surface_z <= 0:
            continue
        transform = np.eye(4)
        transform[:3, 3] = ((start + end) / 2, course.width / 2, surface_z / 2)
        meshes.append(trimesh.creation.box(extents=(end - start, width, surface_z), transform=transform))
    return meshes, course.origin


def upper_landing_occupancy(root, feet, vertical_contact_forces, geometry):
    """Vectorized geometric predicate, compatible with NumPy and torch arrays.

    All four foot centers must be over the assumed landing and near its height;
    at least two feet must exert positive vertical contact force. These checks
    reject lower-floor detours, but do not certify support balance or edge safety.
    Coordinates are relative to each environment's lower-approach spawn origin.
    """
    if (len(root.shape) != 2 or root.shape[1] != 3 or
            feet.shape != (root.shape[0], 4, 3) or
            vertical_contact_forces.shape != (root.shape[0], 4)):
        raise ValueError("Expected root[N,3], named feet[N,4,3], contact forces[N,4]")
    lo, hi = geometry["landing_x_relative_m"]
    half = geometry["flight_width_m"] / 2 - geometry["foot_center_edge_margin_m"]
    height = geometry["upper_height_m"]
    valid = ((root[:, 0] >= lo) & (root[:, 0] <= hi) &
             (abs(root[:, 1]) <= half) &
             (root[:, 2] >= height + geometry["root_height_above_landing_m"][0]) &
             (root[:, 2] <= height + geometry["root_height_above_landing_m"][1]))
    supported = vertical_contact_forces[:, 0] * 0
    for i in range(4):
        foot = feet[:, i]
        valid = (valid & (foot[:, 0] >= lo) & (foot[:, 0] <= hi) &
                 (abs(foot[:, 1]) <= half) &
                 (foot[:, 2] >= height + geometry["foot_center_height_above_landing_m"][0]) &
                 (foot[:, 2] <= height + geometry["foot_center_height_above_landing_m"][1]))
        force = vertical_contact_forces[:, i]
        # Explicit finite check without importing a tensor backend; NaN/Inf
        # comparisons cannot become accepted support samples.
        finite = (force == force) & (abs(force) < float("inf"))
        valid = valid & finite
        supported = supported + (force > geometry["contact_threshold_n"])
    return valid & (supported >= geometry["minimum_contact_feet"])


def narrow_stairs_success(env, original_success, original_params, geometry):
    """Retain every existing failure predicate, adding landing occupancy only."""
    robot = env.scene["robot"]
    sensor = env.scene["contact_forces"]
    # Resolve both independently by name: articulation and sensor orders differ.
    foot_ids = [robot.body_names.index(name) for name in FOOT_NAMES]
    contact_ids = [sensor.body_names.index(name) for name in FOOT_NAMES]
    origin = env.scene.env_origins
    root = robot.data.root_pos_w - origin
    feet = robot.data.body_pos_w[:, foot_ids, :] - origin[:, None, :]
    force_z = sensor.data.net_forces_w[:, contact_ids, 2]
    occupied = upper_landing_occupancy(root, feet, force_z, geometry)
    return original_success(env, **original_params) & occupied


def configure(cfg, flight_width_m=.6):
    """Apply only to an evaluation cfg and return serializable assumptions.

    The caller must have built cfg with evaluation=True. No guard, actuator,
    policy, original checkpoint settings, reward or failure threshold is edited.
    """
    if cfg.go2_settings["terrain"]["curriculum"] is not False:
        raise ValueError("Narrow staircase geometry requires an evaluation configuration")
    generator = cfg.scene.terrain.terrain_generator
    stairs = generator.sub_terrains["stairs"]
    if tuple(generator.size) != (8., 3.):
        raise ValueError("Expected the existing 8 x 3 m tile; do not narrow tile spacing")
    if (type(flight_width_m) not in (int, float) or not math.isfinite(flight_width_m)
            or flight_width_m not in (FLIGHT_WIDTH_M, WIDE_FLIGHT_WIDTH_M)):
        raise ValueError("This bounded evaluation supports only 0.60 m or 0.90 m flight widths")
    if tuple(stairs.step_height_range) != (.15, .15) or stairs.num_steps != 2 or not math.isclose(stairs.step_width, .30):
        raise ValueError("This bounded evaluation requires two 15 cm risers with 30 cm treads")
    if (stairs.approach_length, stairs.spawn_x, stairs.goal_offset) != (2., .8, .7):
        raise ValueError("Existing success/metric coordinates require approach 2 m, spawn .8 m, goal offset .7 m")
    if cfg.terminations.success.func is narrow_stairs_success:
        raise ValueError("Narrow staircase evaluation is already configured")
    geometry = {
        "scope": "evaluation_only_finite_width_geometry",
        "tile_size_m": [8., 3.], "flight_width_m": float(flight_width_m),
        "num_steps": 2, "step_height_m": .15, "tread_m": .30,
        "upper_height_m": .30,
        "flight_x_relative_m": [1.2, 1.8],
        "landing_x_relative_m": [1.8, 6.85],
        "goal_x_relative_m": 2.5,
        "ground_floor_width_m": 3.,
        "neighboring_flight_ground_gap_m": 3. - float(flight_width_m),
        "assumed_upper_landing": {
            "real_dimensions_known": False, "width_m": float(flight_width_m),
            "tile_x_start_m": 2.6, "tile_x_end_m": 8., "length_m": 5.4,
            "note": "Final 30 cm tread merges with an assumed long narrow upper landing. Its real width and depth are unknown; this is not an exact site replica.",
        },
        "foot_center_edge_margin_m": .025,
        "foot_center_height_above_landing_m": [-.025, .10],
        "root_height_above_landing_m": [.15, .80],
        "minimum_contact_feet": 2, "contact_threshold_n": 1.,
        "success_addition": "All four named foot centers inside assumed upper landing and near its height, at least two positive vertical contacts, centered elevated root, plus every original success/failure predicate.",
        "interpretation": "Sampled geometric completion only; not balance certification, contact-patch clearance certification, or authorization for real traversal onto an unmeasured landing.",
        "side_exit_diagnostic_added": False,
    }
    stairs.function = (ascending_narrow_stairs if flight_width_m == FLIGHT_WIDTH_M
                       else ascending_narrow_stairs_09)
    original = cfg.terminations.success
    original.params = {"original_success": original.func,
                       "original_params": copy.deepcopy(original.params),
                       "geometry": copy.deepcopy(geometry)}
    original.func = narrow_stairs_success
    generator.use_cache = False
    return geometry
