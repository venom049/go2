"""Isaac Lab terrain-generator adapter; importing this file needs only NumPy.

The callable follows the Isaac Lab SubTerrainBaseCfg function contract. Meshes
use [0, size_x] x [0, size_y] coordinates and the returned origin is on the LOWER
approach. Isaac Lab's terrain generator applies its own world translation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .terrain import StairCourse


def course_geometry(difficulty: float, cfg: Any) -> StairCourse:
    """Resolve the exact geometry shared by Lab meshes and course metadata.

    ``cfg`` is an Isaac Lab sub-terrain configuration or a simple namespace with
    size, step_height_range, step_width, num_steps, approach_length,
    landing_length, spawn_x, goal_offset, and boundary_margin attributes.
    """
    if not math.isfinite(difficulty) or not 0.0 <= difficulty <= 1.0:
        raise ValueError("difficulty must be finite and in [0, 1]")
    minimum, maximum = cfg.step_height_range
    if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum < 0 or maximum < minimum:
        raise ValueError("step_height_range must be finite, non-negative, and ascending")
    return StairCourse(
        length=cfg.size[0], width=cfg.size[1],
        step_height=minimum + difficulty * (maximum - minimum),
        step_width=cfg.step_width, num_steps=cfg.num_steps,
        approach_length=cfg.approach_length, landing_length=cfg.landing_length,
        spawn_x=cfg.spawn_x, goal_offset=cfg.goal_offset,
        boundary_margin=cfg.boundary_margin,
    )


def ascending_stairs(difficulty: float, cfg: Any) -> tuple[list[Any], np.ndarray]:
    """Build one flight of exact vertical risers followed by an upper landing.

    Each section is a watertight solid extending 10 cm below the lower ground
    plane. No heightfield interpolation is involved, so tread/riser dimensions
    are preserved even when the simulation height sensor uses a coarse grid.
    """
    import trimesh

    course = course_geometry(difficulty, cfg)
    meshes = []
    depth = 0.1
    for start, end, surface_z in course.segments():
        transform = np.eye(4)
        transform[:3, 3] = ((start + end) / 2.0, course.width / 2.0, (surface_z - depth) / 2.0)
        meshes.append(trimesh.creation.box(
            extents=(end - start, course.width, surface_z + depth),
            transform=transform,
        ))
    return meshes, course.origin
