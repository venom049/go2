"""Isaac Lab manager terms for a single, forward-only staircase.

Import this module after launching Isaac Sim. Course success is distinct from
failure, and episodic event rewards compensate for Lab's reward dt scaling.
"""

from __future__ import annotations

import torch

from isaaclab.envs.mdp import *  # noqa: F401,F403 -- expose the standard Lab terms.
from isaaclab.managers import SceneEntityCfg

from go2_him.lateral_geometry import side_sign_tensor, signed_foot_lateral_positions


def _relative_position(env) -> torch.Tensor:
    return env.scene["robot"].data.root_pos_w - env.scene.env_origins


def _goal_x(env) -> float:
    terrain = env.cfg.go2_settings["terrain"]
    # Matches StairCourse: approach + eight treads + landing clearance - spawn.
    return 2.0 + terrain["num_steps"] * terrain["tread"] + 0.7 - 0.8


def forward_progress(env, command_name: str = "base_velocity") -> torch.Tensor:
    """Positive world +X speed, capped at the requested forward speed."""
    actual = env.scene["robot"].data.root_lin_vel_w[:, 0].clamp(min=0.0)
    requested = env.command_manager.get_command(command_name)[:, 0].clamp(min=0.0)
    return torch.minimum(actual, requested)


def success(env) -> torch.Tensor:
    """Reach the upper landing without a simultaneous contact/fall/bounds failure.

    Re-evaluate the configured failure predicates: reading their cached manager
    buffers here would depend on the order of termination terms, while reading
    ``terminated`` would include success itself during reward computation.
    """
    position = _relative_position(env)
    upright = env.scene["robot"].data.projected_gravity_b[:, 2] < -0.65
    reached = (position[:, 0] >= _goal_x(env)) & (position[:, 1].abs() < 1.15) & upright
    for name in ("base_contact", "fall", "out_of_bounds"):
        term = env.termination_manager.get_term_cfg(name)
        reached &= ~term.func(env, **term.params)
    return reached


def failure(env) -> torch.Tensor:
    """True terminal failures, excluding successful completion and timeouts."""
    return env.termination_manager.terminated & ~success(env)


def success_reward(env) -> torch.Tensor:
    """One event worth the configured reward weight after RewardManager's dt."""
    return success(env).float() / env.step_dt


def failure_reward(env) -> torch.Tensor:
    return failure(env).float() / env.step_dt


def out_of_bounds(env) -> torch.Tensor:
    """Stop before a tile edge or adjacent lane can become a shortcut."""
    position = _relative_position(env)
    return (position[:, 0] < -0.45) | (position[:, 0] > 6.85) | (position[:, 1].abs() > 1.15)


def lateral_drift(env) -> torch.Tensor:
    return _relative_position(env)[:, 1].square()


def joint_power(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Absolute mechanical joint power, summed over the selected joints."""
    data = env.scene[asset_cfg.name].data
    return (data.applied_torque[:, asset_cfg.joint_ids] * data.joint_vel[:, asset_cfg.joint_ids]).abs().sum(dim=1)


def roll_orientation(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize sideways tilt while permitting the pitch required on stairs."""
    return env.scene[asset_cfg.name].data.projected_gravity_b[:, 1].square()


def roll_rate(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Body-x angular rate, without adding a pitch-rate constraint on stairs."""
    return env.scene[asset_cfg.name].data.root_ang_vel_b[:, 0].square()


def lateral_velocity(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Body-y translation; forward and stair-climbing vertical speed are free."""
    return env.scene[asset_cfg.name].data.root_lin_vel_b[:, 1].square()


def hip_abduction(
    env,
    asset_cfg: SceneEntityCfg,
    max_angle: float = 0.30,
    side_signs: tuple[float, ...] = (1, -1, 1, -1),
) -> torch.Tensor:
    """Squared excess outward hip angle, summed over FL, FR, RL, RR (rad²).

    Select only the four hip joints with ``preserve_order=True``. This uses
    measured joint positions, not policy actions or deviation from the default
    pose. The dead zone permits ordinary abduction for balance; configure a
    negative reward weight. Physical joint limits and motor targets are unchanged.
    """
    if max_angle < 0.0:
        raise ValueError("max_angle must be nonnegative")
    angles = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    if angles.ndim != 2 or angles.shape[1] != 4:
        raise ValueError("hip_abduction requires four hip joints in FL, FR, RL, RR order")
    signed_angles = angles * side_sign_tensor(angles, side_signs)
    return (signed_angles - max_angle).clamp(min=0.0).square().sum(dim=1)


def feet_lateral(
    env,
    asset_cfg: SceneEntityCfg,
    min_lateral: float = 0.10,
    max_lateral: float = 0.22,
    inner_weight: float = 1.0,
    side_signs: tuple[float, ...] = (1, -1, 1, -1),
) -> torch.Tensor:
    """Squared foot-center corridor violations in body coordinates (m²).

    Select feet explicitly in FL, FR, RL, RR order with ``preserve_order=True``.
    Both stance and swing feet contribute. The outer edge discourages excessive
    spreading while the inner edge prevents earning the reward by crossing legs.
    The full body-frame transform permits stair pitch and arbitrary world yaw.
    A finite-width dead zone permits natural variation; use a negative weight.
    """
    if not 0.0 <= min_lateral < max_lateral:
        raise ValueError("Expected 0 <= min_lateral < max_lateral")
    if inner_weight < 0.0:
        raise ValueError("inner_weight must be nonnegative")
    data = env.scene[asset_cfg.name].data
    lateral = signed_foot_lateral_positions(
        data.root_pos_w, data.root_quat_w, data.body_pos_w[:, asset_cfg.body_ids, :], side_signs
    )
    outer = (lateral - max_lateral).clamp(min=0.0)
    inner = (min_lateral - lateral).clamp(min=0.0)
    return (outer.square() + inner_weight * inner.square()).sum(dim=1)


def action_smoothness(env) -> torch.Tensor:
    """Squared second action difference with per-episode history boundaries.

    Lab increments both step counters before computing rewards, then resets
    terminated environments. Its action manager already supplies a_t and
    a_(t-1); retain the preceding call's a_(t-1) for a_(t-2). Exclude the first
    two actions after every reset. Repeated calls at the same environment step
    return the cached result and never advance history twice.

    The cache belongs to this environment, not to an imported module, so it is
    also independent across training and evaluation environment instances.
    """
    current = env.action_manager.action
    previous = env.action_manager.prev_action
    step = env.common_step_counter
    state = getattr(env, "_go2_action_smoothness_state", None)
    valid_episode = env.episode_length_buf > 2
    if state is not None and state["step"] == step:
        # A reset can occur between a reward calculation and a diagnostic read.
        return state["value"] * valid_episode

    value = torch.zeros(current.shape[0], dtype=current.dtype, device=current.device)
    if state is not None and state["step"] == step - 1:
        value = (current - 2.0 * previous + state["previous"]).square().sum(dim=1)
        value = value * valid_episode
    env._go2_action_smoothness_state = {
        "step": step,
        "previous": previous.detach().clone(),
        "value": value,
    }
    return value


def feet_air_time(
    env,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    target_air_time: float = 0.10,
    max_air_time: float = 0.35,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward completed swing duration on landing, without rewarding long hops.

    Requires ``track_air_time=True`` on the contact sensor. Read last_air_time,
    not current_air_time: the latter is already zero after the foot lands.
    Initial standing contacts (no preceding swing) receive no reward. A maximum
    duration bounds the incentive to stay airborne. No fixed leg phase is imposed.
    """
    if not 0.0 <= target_air_time < max_air_time:
        raise ValueError("Expected 0 <= target_air_time < max_air_time")
    sensor = env.scene.sensors[sensor_cfg.name]
    first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = sensor.data.last_air_time[:, sensor_cfg.body_ids]
    landed = first_contact & (last_air_time > 0.0)
    duration = last_air_time.clamp(min=0.0, max=max_air_time)
    reward = ((duration - target_air_time) * landed).sum(dim=1)
    moving = env.command_manager.get_command(command_name)[:, 0] > command_threshold
    return reward * moving


def feet_stumble(
    env,
    sensor_cfg: SceneEntityCfg,
    force_ratio: float = 5.0,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Detect foot impacts against stair risers over the sensor's force history.

    Compare planar and vertical force in each *same* physics sample, then reduce
    over history and feet. This retains short impacts without pairing unrelated
    horizontal and vertical peaks. Tiny numerical contact forces are ignored.
    Configure a negative reward weight. A force history covering the full control
    interval captures impacts that happen between policy actions.
    """
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    horizontal = torch.linalg.vector_norm(forces[..., :2], dim=-1)
    vertical = forces[..., 2].abs()
    stumble = (horizontal > force_ratio * vertical) & (horizontal > force_threshold)
    return stumble.any(dim=1).any(dim=1).to(forces.dtype)


def _foot_contacts(env, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids, :]
    return torch.linalg.vector_norm(forces, dim=-1) > threshold


def foot_slip(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Squared planar foot speed while that foot is in contact.

    The selected articulation bodies and contact sensor bodies must have the
    same foot ordering; configure explicit names and ``preserve_order=True``.
    """
    velocities = env.scene[asset_cfg.name].data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    contacts = _foot_contacts(env, sensor_cfg, threshold)
    return (velocities.square().sum(dim=-1) * contacts).sum(dim=1)


def foot_clearance(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    height_sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    target_height: float = 0.08,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Swing-foot clearance deficit relative to the nearest terrain ray hit.

    This is a penalty (configure a negative reward weight), gated by horizontal
    swing velocity. Missing rays and feet beyond the scanner footprint do not
    contribute. The ray-grid spacing should be <= 0.1 m for useful stair edges.
    Foot link centers are used, so the target includes any foot collision radius.
    """
    data = env.scene[asset_cfg.name].data
    feet = data.body_pos_w[:, asset_cfg.body_ids, :]
    speeds = torch.linalg.vector_norm(data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1)
    hits = env.scene.sensors[height_sensor_cfg.name].data.ray_hits_w
    valid_rays = torch.isfinite(hits).all(dim=-1)
    finite_hits = torch.nan_to_num(hits, nan=0.0, posinf=0.0, neginf=0.0)
    distance_sq = (feet[:, :, None, :2] - finite_hits[:, None, :, :2]).square().sum(dim=-1)
    distance_sq = distance_sq.masked_fill(~valid_rays[:, None, :], float("inf"))
    nearest_distance, nearest_index = distance_sq.min(dim=-1)
    ground_z = finite_hits[:, :, 2].gather(1, nearest_index)
    clearance = feet[:, :, 2] - ground_z
    deficit = (target_height - clearance).clamp(min=0.0)
    # Reject extrapolation beyond the scanner, especially when a foot leaves the tile.
    usable = nearest_distance < 0.15**2
    swinging = ~_foot_contacts(env, sensor_cfg, threshold)
    return (deficit.square() * speeds.clamp(max=2.0) * swinging * usable).sum(dim=1)


def base_height(
    env,
    target_height: float = 0.30,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    radius: float = 0.35,
) -> torch.Tensor:
    """Height penalty above the local ground under the central base footprint."""
    position = env.scene["robot"].data.root_pos_w
    hits = env.scene.sensors[sensor_cfg.name].data.ray_hits_w
    finite = torch.isfinite(hits).all(dim=-1)
    clean = torch.nan_to_num(hits, nan=0.0, posinf=0.0, neginf=0.0)
    distance_sq = (clean[:, :, :2] - position[:, None, :2]).square().sum(dim=-1)
    central = finite & (distance_sq <= radius**2)
    count = central.sum(dim=1)
    ground_z = (clean[:, :, 2] * central).sum(dim=1) / count.clamp(min=1)
    error = (position[:, 2] - ground_z - target_height).square()
    return torch.where(count > 0, error, torch.zeros_like(error))


def command_speed_curriculum(env, env_ids) -> dict[str, float]:
    """Ramp the command sampling range using this run's control-step counter.

    This is independent of episode lengths, success counts and checkpoint PPO
    update numbers. A warm-start begins at the configured start range. Called
    before command reset, so newly sampled commands use the updated bounds;
    running episodes keep their command until the ordinary command resample.
    With zero ramp updates the term is disabled and the original command ranges
    are untouched. Evaluation disables it in ``build_env_cfg``.
    """
    settings = env.cfg.go2_settings
    commands = settings["commands"]
    updates = commands["speed_ramp_updates"]
    if updates <= 0:
        return {}
    duration = updates * settings["training"]["num_steps_per_env"]
    fraction = min(max(env.common_step_counter / duration, 0.0), 1.0)
    low = commands["start_forward_min"] + fraction * (commands["forward_min"] - commands["start_forward_min"])
    high = commands["start_forward_max"] + fraction * (commands["forward_max"] - commands["start_forward_max"])
    term = env.command_manager.get_term("base_velocity")
    term.cfg.ranges.lin_vel_x = (low, high)
    return {"fraction": fraction, "forward_min": low, "forward_max": high}


def curriculum(env, env_ids) -> dict[str, torch.Tensor]:
    """Promote successful climbs, demote failures, and log completed episodes.

    Called before Lab resets the robot and termination buffers. Initial resets
    (episode length zero) never update levels. Successful robots at the hardest
    level stay there rather than triggering the importer's random-level wrap.
    Keep this term enabled to retain metrics even when terrain.curriculum=false.
    """
    terrain = env.scene.terrain
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    ids = ids[env.episode_length_buf[ids] > 0]
    level = terrain.terrain_levels.float().mean()
    if ids.numel() == 0:
        zero = torch.zeros((), device=env.device)
        return {"success_rate": zero, "forward_distance": zero, "terrain_level": level}

    position_x = _relative_position(env)[ids, 0]
    completed = success(env)[ids]
    failed = failure(env)[ids]
    command_x = env.command_manager.get_command("base_velocity")[ids, 0].clamp(min=0.0)
    elapsed = env.episode_length_buf[ids].float() * env.step_dt
    sufficient_distance = torch.minimum(0.5 * command_x * elapsed, torch.full_like(position_x, 0.8 * _goal_x(env)))
    stalled_timeout = env.termination_manager.time_outs[ids] & (position_x < sufficient_distance)
    move_up = completed & (terrain.terrain_levels[ids] < terrain.max_terrain_level - 1)
    move_down = ~completed & (failed | stalled_timeout)
    if env.cfg.go2_settings["terrain"]["curriculum"]:
        terrain.update_env_origins(ids, move_up, move_down)

    return {
        "success_rate": completed.float().mean(),
        "forward_distance": position_x.mean(),
        "terrain_level": terrain.terrain_levels.float().mean(),
    }
