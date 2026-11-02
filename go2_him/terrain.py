"""Backend-independent, forward-only stairs and optional Isaac Gym height fields.

All course coordinates are measured from the lower-left corner of a tile. Robot
origins are on the lower approach, with +X pointing up the stairs. Metadata used
by rewards and terminations is relative to that robot origin. No simulator is
imported unless ``StairsTerrain(..., mesh_type='trimesh')`` is explicitly used.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StairCourse:
    """Exact dimensions of one ascending flight, including its two landings."""

    length: float = 8.0
    width: float = 3.0
    step_height: float = 0.18
    step_width: float = 0.32
    num_steps: int = 8
    approach_length: float = 2.0
    landing_length: float = 2.0
    spawn_x: float = 0.8
    goal_offset: float = 0.7
    boundary_margin: float = 0.35

    def __post_init__(self) -> None:
        for name in (
            "length", "width", "step_width", "approach_length", "landing_length",
            "spawn_x", "goal_offset", "boundary_margin",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive; got {value!r}")
        if not math.isfinite(self.step_height) or self.step_height < 0:
            raise ValueError("step_height must be finite and non-negative")
        if isinstance(self.num_steps, bool) or not isinstance(self.num_steps, Integral) or self.num_steps < 1:
            raise ValueError("num_steps must be a positive integer")
        if self.stair_end > self.length - self.landing_length + 1e-9:
            raise ValueError("stairs do not leave the configured minimum upper landing")
        if not self.boundary_margin < self.spawn_x < self.approach_length - self.boundary_margin:
            raise ValueError("spawn_x must lie inside the lower approach with boundary clearance")
        if self.width <= 2.0 * self.boundary_margin:
            raise ValueError("width must exceed twice boundary_margin")
        if self.stair_end + self.goal_offset >= self.length - self.boundary_margin:
            raise ValueError("goal must lie on the upper landing before the safe tile boundary")

    @property
    def stair_end(self) -> float:
        """Start of the upper landing, after all full-width treads."""
        return self.approach_length + self.num_steps * self.step_width

    @property
    def upper_height(self) -> float:
        return self.num_steps * self.step_height

    @property
    def origin(self) -> np.ndarray:
        return np.array((self.spawn_x, self.width / 2.0, 0.0), dtype=np.float64)

    @property
    def goal_x(self) -> float:
        return self.stair_end + self.goal_offset - self.spawn_x

    @property
    def bounds(self) -> np.ndarray:
        """Safe [xmin, xmax, ymin, ymax] relative to the robot origin."""
        return np.array((
            self.boundary_margin - self.spawn_x,
            self.length - self.boundary_margin - self.spawn_x,
            -self.width / 2.0 + self.boundary_margin,
            self.width / 2.0 - self.boundary_margin,
        ), dtype=np.float64)

    def height_at(self, x: Any) -> np.ndarray:
        """Surface height at tile-local x, including either flat landing."""
        risers = self.approach_length + np.arange(self.num_steps) * self.step_width
        return np.searchsorted(risers, np.asarray(x), side="right") * self.step_height

    def segments(self) -> list[tuple[float, float, float]]:
        """Contiguous (x_start, x_end, surface_z) blocks for mesh generation."""
        if self.step_height == 0.0:
            return [(0.0, self.length, 0.0)]
        blocks = [(0.0, self.approach_length, 0.0)]
        for step in range(self.num_steps):
            start = self.approach_length + step * self.step_width
            # Merge the final tread and upper landing into one solid block.
            end = self.length if step == self.num_steps - 1 else start + self.step_width
            blocks.append((start, end, (step + 1) * self.step_height))
        return blocks


@dataclass
class StairsTerrainConfig:
    """Small standalone configuration; a legged_gym config class also works."""

    mesh_type: str = "heightfield"
    horizontal_scale: float = 0.05
    vertical_scale: float = 0.005
    border_size: float = 2.0
    terrain_length: float = 8.0
    terrain_width: float = 3.0
    num_rows: int = 10
    num_cols: int = 4
    stair_height_min: float = 0.0
    stair_height_max: float = 0.18
    stair_tread_min: float = 0.28
    stair_tread_max: float = 0.39
    stair_approach_length: float = 2.0
    stair_landing_length: float = 2.0
    stair_spawn_x: float = 0.8
    stair_num_steps: int = 8
    stair_goal_offset: float = 0.7
    stair_boundary_margin: float = 0.35
    slope_treshold: float = 0.75  # Isaac Gym uses this spelling.


def _grid_pixels(value: float, scale: float, name: str) -> int:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    result = round(value / scale)
    if not math.isclose(result * scale, value, abs_tol=1e-8):
        raise ValueError(f"{name}={value} must be an exact multiple of horizontal_scale={scale}")
    return result


class StairsTerrain:
    """Deterministic curriculum array with legged_gym-compatible attributes.

    Rows include both endpoints of the height range. Columns vary tread width.
    Heights and treads are rounded to the raster resolution, and metadata stores
    the actual raster dimensions. Tile boundaries can drop to the next tile;
    consumers must terminate at ``course_bounds`` or ``goal_x`` before them.
    """

    def __init__(self, cfg: Any, num_robots: int = 0) -> None:
        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type not in ("heightfield", "trimesh"):
            raise ValueError("StairsTerrain requires mesh_type 'heightfield' or 'trimesh'")
        for name in ("horizontal_scale", "vertical_scale"):
            if not math.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("num_rows", "num_cols"):
            value = getattr(cfg, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for low_name, high_name in (("stair_height_min", "stair_height_max"), ("stair_tread_min", "stair_tread_max")):
            low, high = getattr(cfg, low_name), getattr(cfg, high_name)
            if not math.isfinite(low) or not math.isfinite(high) or low < 0 or high < low:
                raise ValueError(f"invalid range: {low_name}, {high_name}")

        self.env_length, self.env_width = cfg.terrain_length, cfg.terrain_width
        self.length_per_env_pixels = _grid_pixels(self.env_length, cfg.horizontal_scale, "terrain_length")
        self.width_per_env_pixels = _grid_pixels(self.env_width, cfg.horizontal_scale, "terrain_width")
        self.border = _grid_pixels(cfg.border_size, cfg.horizontal_scale, "border_size")
        approach_pixels = _grid_pixels(cfg.stair_approach_length, cfg.horizontal_scale, "stair_approach_length")
        self.tot_rows = cfg.num_rows * self.length_per_env_pixels + 2 * self.border
        self.tot_cols = cfg.num_cols * self.width_per_env_pixels + 2 * self.border
        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        shape = (cfg.num_rows, cfg.num_cols)
        self.env_origins = np.zeros((*shape, 3), dtype=np.float64)
        self.step_heights = np.zeros(shape)
        self.stair_treads = np.zeros(shape)
        self.goal_x = np.zeros(shape)
        self.upper_heights = np.zeros(shape)
        self.stair_start_x = np.zeros(shape)
        self.stair_end_x = np.zeros(shape)
        self.course_bounds = np.zeros((*shape, 4))

        heights = np.linspace(cfg.stair_height_min, cfg.stair_height_max, cfg.num_rows)
        treads = np.linspace(cfg.stair_tread_min, cfg.stair_tread_max, cfg.num_cols)
        for row, height in enumerate(heights):
            height_units = int(round(height / cfg.vertical_scale))
            if height_units * cfg.stair_num_steps > np.iinfo(np.int16).max:
                raise ValueError("total stair elevation exceeds the int16 heightfield range")
            for col, tread in enumerate(treads):
                tread_pixels = int(round(tread / cfg.horizontal_scale))
                course = StairCourse(
                    length=self.env_length, width=self.env_width,
                    step_height=height_units * cfg.vertical_scale,
                    step_width=tread_pixels * cfg.horizontal_scale,
                    num_steps=cfg.stair_num_steps,
                    approach_length=cfg.stair_approach_length,
                    landing_length=cfg.stair_landing_length,
                    spawn_x=cfg.stair_spawn_x,
                    goal_offset=cfg.stair_goal_offset,
                    boundary_margin=cfg.stair_boundary_margin,
                )
                step_indices = np.clip(
                    (np.arange(self.length_per_env_pixels) - approach_pixels) // tread_pixels + 1,
                    0, course.num_steps,
                )
                start_x = self.border + row * self.length_per_env_pixels
                start_y = self.border + col * self.width_per_env_pixels
                self.height_field_raw[
                    start_x:start_x + self.length_per_env_pixels,
                    start_y:start_y + self.width_per_env_pixels,
                ] = (step_indices * height_units).astype(np.int16)[:, None]
                self.env_origins[row, col] = course.origin + np.array((row * self.env_length, col * self.env_width, 0.0))
                self.step_heights[row, col] = course.step_height
                self.stair_treads[row, col] = course.step_width
                self.goal_x[row, col] = course.goal_x
                self.upper_heights[row, col] = course.upper_height
                self.stair_start_x[row, col] = course.approach_length - course.spawn_x
                self.stair_end_x[row, col] = course.stair_end - course.spawn_x
                self.course_bounds[row, col] = course.bounds

        self.heightsamples = self.height_field_raw
        if self.type == "trimesh":
            # Gym is optional and is never imported for NumPy-only inspection.
            from isaacgym import terrain_utils

            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(
                self.height_field_raw, cfg.horizontal_scale, cfg.vertical_scale,
                getattr(cfg, "slope_treshold", 0.75),
            )
