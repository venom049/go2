"""Independent rollout accounting and tensor-only gait measurements.

Measurements use the state BEFORE each simulator step. The first two frames of
each episode are excluded, and differences additionally require consecutive
valid frames. No metric calls ``item()`` until the final report is requested.
"""
import math

import torch

from go2_him.lateral_geometry import signed_foot_lateral_positions


class EpisodeCounter:
    """Count completed episodes, optionally limiting each environment equally."""

    def __init__(self, num_envs, device, episodes_per_env=0):
        if episodes_per_env < 0:
            raise ValueError("episodes_per_env must be non-negative")
        self.target = episodes_per_env
        self.completed = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.successes = torch.zeros_like(self.completed)
        self.timeouts = torch.zeros_like(self.completed)
        self.failures = torch.zeros_like(self.completed)

    @property
    def active(self):
        return self.completed < self.target if self.target else torch.ones_like(self.completed, dtype=torch.bool)

    def update(self, done, success, time_out, terminated):
        finished = done.bool() & self.active
        succeeded = success.bool() & finished
        timed_out = time_out.bool() & ~terminated.bool() & ~succeeded & finished
        self.completed += finished
        self.successes += succeeded
        self.timeouts += timed_out
        self.failures += finished & ~succeeded & ~timed_out

    def all_finished(self):
        # One synchronization per step is necessary only for a finite quota.
        return bool((self.completed >= self.target).all()) if self.target else False

    def report(self):
        completed = self.completed.cpu().tolist()
        episodes = sum(completed)
        successes = int(self.successes.sum().item())
        return {
            "episodes": episodes,
            "successes": successes,
            "failures": int(self.failures.sum().item()),
            "timeouts": int(self.timeouts.sum().item()),
            "success_rate": successes / episodes if episodes else None,
            "episodes_per_env_target": self.target,
            "completed_episodes_per_env": completed,
            "incomplete_env_ids": [i for i, n in enumerate(completed) if self.target and n < self.target],
            "envs_incomplete": sum(n < self.target for n in completed) if self.target else None,
        }


class GaitMetrics:
    """Sample-weighted diagnostics, separate from training reward.

    ``raw_actions`` passed to ``update_pre_step`` must be the previous policy
    output, which produced the current pre-step state. The actual normalized
    action is read from the manager, allowing later actuator filters to be
    measured instead of silently treating policy output as applied action.
    """

    def __init__(self, env, foot_names, action_scale, action_clip, contact_threshold=1.0):
        self.env = env
        self.scale = float(action_scale)
        self.clip = float(action_clip)
        self.threshold = float(contact_threshold)
        self.robot = env.scene["robot"]
        self.sensor = env.scene["contact_forces"]
        # Resolve the two name lists independently: PhysX and the sensor need
        # not expose bodies in the same order.
        self.robot_foot_ids = [self.robot.body_names.index(name) for name in foot_names]
        self.sensor_foot_ids = [self.sensor.body_names.index(name) for name in foot_names]
        self.leg_names = ("FL", "FR", "RL", "RR")
        if list(foot_names) != [f"{leg}_foot" for leg in self.leg_names]:
            raise ValueError("lateral metrics require foot_names ordered FL, FR, RL, RR")
        # Joint and rigid-body order are independent. Hip angles are measured
        # from the URDF zero pose, not the reward's default standing pose.
        self.hip_ids = [self.robot.joint_names.index(f"{leg}_hip_joint") for leg in self.leg_names]
        self.side_signs = torch.tensor((1., -1., 1., -1.), device=env.device)
        settings = getattr(getattr(env, "cfg", None), "go2_settings", {})
        targets = settings.get("targets", {})
        self.hip_abduction_max = float(targets.get("hip_abduction_max", .30))
        self.feet_lateral_min = float(targets.get("feet_lateral_min", .10))
        self.feet_lateral_max = float(targets.get("feet_lateral_max", .22))
        terrain = settings.get("terrain", {})
        # StairCourse uses a 2 m approach and an origin 0.8 m along it.
        self.stairs_start_x = 1.2
        self.stairs_end_x = self.stairs_start_x + int(terrain.get("num_steps", 8)) * float(terrain.get("tread", .32))
        self.sums = {}
        self.counts = {}
        self.rms = set()
        self.previous = None
        self.previous_previous_action = None
        self.previous_consecutive = None
        self.previous_position = None
        num_envs = env.episode_length_buf.numel()
        self.raw_max = torch.zeros(num_envs, device=env.device, dtype=torch.float64)
        self.samples = torch.zeros(num_envs, device=env.device, dtype=torch.long)
        self.stairs_samples = torch.zeros_like(self.samples)
        self.nonfinite = torch.zeros(num_envs, device=env.device, dtype=torch.long)
        self.stairs_entry_time = torch.full((num_envs,), float("nan"), device=env.device, dtype=torch.float64)
        self.stairs_crossing_finished = torch.zeros(num_envs, device=env.device, dtype=torch.bool)

    def _record(self, name, values, mask, rms=False):
        while mask.ndim < values.ndim:
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(values)
        finite = torch.isfinite(values)
        self.nonfinite += (mask & ~finite).reshape(values.shape[0], -1).sum(dim=1)
        usable = mask & finite
        values = torch.where(usable, values, 0).to(torch.float64)
        total = (values.square() if rms else values).reshape(values.shape[0], -1).sum(dim=1)
        count = usable.reshape(values.shape[0], -1).sum(dim=1)
        if name not in self.sums:
            self.sums[name] = torch.zeros_like(total)
            self.counts[name] = torch.zeros_like(count)
        self.sums[name] += total
        self.counts[name] += count
        if rms:
            self.rms.add(name)

    def _update_stair_crossings(self, relative_x, lengths, valid):
        """Time the first complete forward flight in each observed episode.

        Crossing times interpolate between consecutive pre-step samples. A
        retreat after entry remains part of the elapsed time. Spawns inside the
        flight, resets, skipped samples and incomplete flights cannot produce a
        deceptively short traversal. Completion here is geometric, not success;
        callers can additionally select completed successful first episodes.
        """
        valid = valid & torch.isfinite(relative_x)
        crossed = torch.zeros_like(valid)
        elapsed = torch.zeros_like(self.stairs_entry_time)
        if self.previous_position is not None:
            old_valid, old_lengths, old_x = self.previous_position
            consecutive = valid & old_valid & (lengths == old_lengths + 1)
            # Any gap makes the time of entry ambiguous; do not bridge it.
            self.stairs_entry_time = torch.where(consecutive, self.stairs_entry_time, float("nan"))
            new_episode = lengths <= old_lengths
            self.stairs_crossing_finished &= ~new_episode
            dx = relative_x - old_x
            forward = consecutive & torch.isfinite(relative_x) & torch.isfinite(old_x) & (dx > 0)
            enters = forward & (old_x < self.stairs_start_x) & (relative_x >= self.stairs_start_x)
            enters &= ~torch.isfinite(self.stairs_entry_time) & ~self.stairs_crossing_finished
            old_time = old_lengths.to(torch.float64) * self.env.step_dt
            safe_dx = torch.where(forward, dx, 1).to(torch.float64)
            entry_time = old_time + (self.stairs_start_x - old_x.to(torch.float64)) / safe_dx * self.env.step_dt
            self.stairs_entry_time = torch.where(enters, entry_time, self.stairs_entry_time)
            crossed = forward & (old_x < self.stairs_end_x) & (relative_x >= self.stairs_end_x)
            crossed &= torch.isfinite(self.stairs_entry_time) & ~self.stairs_crossing_finished
            end_time = old_time + (self.stairs_end_x - old_x.to(torch.float64)) / safe_dx * self.env.step_dt
            elapsed = end_time - self.stairs_entry_time
            crossed &= torch.isfinite(elapsed) & (elapsed > 0)
            self.stairs_crossing_finished |= crossed
        self._record("stairs_traversal_time_mean_s", elapsed, crossed)
        safe_elapsed = torch.where(crossed, elapsed, 1)
        self._record("stairs_traversal_speed_mean_m_s", (self.stairs_end_x - self.stairs_start_x) / safe_elapsed, crossed)
        self.previous_position = (valid.clone(), lengths.clone(), relative_x.clone())

    def update_pre_step(self, raw_actions, active):
        data = self.robot.data
        lengths = self.env.episode_length_buf
        valid = active.bool() & (lengths > 1)
        self.samples += valid
        command = self.env.command_manager.get_command("base_velocity")
        self._record("planar_velocity_mae_m_s", (data.root_lin_vel_b[:, :2] - command[:, :2]).abs(), valid)
        gravity = data.projected_gravity_b
        roll = torch.atan2(-gravity[:, 1], -gravity[:, 2])
        pitch = torch.atan2(gravity[:, 0], torch.linalg.vector_norm(gravity[:, 1:], dim=-1))
        self._record("joint_velocity_rms_rad_s", data.joint_vel, valid, rms=True)
        self._record("mechanical_power_mean_w", (data.applied_torque * data.joint_vel).abs().sum(dim=-1), valid)

        hip_abduction = data.joint_pos[:, self.hip_ids] * self.side_signs
        lateral = signed_foot_lateral_positions(
            data.root_pos_w, data.root_quat_w, data.body_pos_w[:, self.robot_foot_ids]
        )
        # Keep non-finite geometry observable instead of reporting it as a
        # valid in-range or out-of-range sample through a boolean comparison.
        hip_over = torch.where(torch.isfinite(hip_abduction),
            (hip_abduction > self.hip_abduction_max).to(torch.float32), float("nan"))
        feet_outer = torch.where(torch.isfinite(lateral),
            (lateral > self.feet_lateral_max).to(torch.float32), float("nan"))
        feet_inner = torch.where(torch.isfinite(lateral),
            (lateral < self.feet_lateral_min).to(torch.float32), float("nan"))
        hip_inward = torch.where(torch.isfinite(hip_abduction),
            (hip_abduction < 0).to(torch.float32), float("nan"))
        hip_outward = torch.where(torch.isfinite(hip_abduction), hip_abduction.clamp(min=0), float("nan"))
        feet_crossed = torch.where(torch.isfinite(lateral),
            (lateral < 0).to(torch.float32), float("nan"))
        relative_x = data.root_pos_w[:, 0] - self.env.scene.env_origins[:, 0]
        stairs_valid = valid & (relative_x >= self.stairs_start_x) & (relative_x < self.stairs_end_x)
        self.stairs_samples += stairs_valid
        self._update_stair_crossings(relative_x, lengths, valid)
        for prefix, mask in (("", valid), ("stairs_", stairs_valid)):
            # The course ascends in world +X. Body vx alone can overstate
            # useful progress when the robot yaws or sways sideways.
            self._record(prefix + "course_forward_velocity_mean_m_s", data.root_lin_vel_w[:, 0], mask)
            self._record(prefix + "body_forward_velocity_mean_m_s", data.root_lin_vel_b[:, 0], mask)
            self._record(prefix + "forward_command_mean_m_s", command[:, 0], mask)
            self._record(prefix + "forward_velocity_mae_m_s", (data.root_lin_vel_b[:, 0] - command[:, 0]).abs(), mask)
            self._record(prefix + "body_roll_rms_rad", roll, mask, rms=True)
            self._record(prefix + "body_pitch_rms_rad", pitch, mask, rms=True)
            for i, axis in enumerate(("roll", "pitch", "yaw")):
                self._record(f"{prefix}body_{axis}_rate_rms_rad_s", data.root_ang_vel_b[:, i], mask, rms=True)
            self._record(prefix + "hip_abduction_mean_rad", hip_abduction, mask)
            self._record(prefix + "hip_outward_component_mean_rad", hip_outward, mask)
            self._record(prefix + "hip_adduction_fraction", hip_inward, mask)
            self._record(prefix + "hip_abduction_rms_rad", hip_abduction, mask, rms=True)
            self._record(prefix + "hip_abduction_over_limit_fraction", hip_over, mask)
            self._record(prefix + "feet_lateral_mean_m", lateral, mask)
            self._record(prefix + "feet_lateral_outer_fraction", feet_outer, mask)
            self._record(prefix + "feet_lateral_inner_fraction", feet_inner, mask)
            self._record(prefix + "feet_cross_body_center_fraction", feet_crossed, mask)
            for i, leg in enumerate(self.leg_names):
                self._record(f"{prefix}{leg}_hip_abduction_mean_rad", hip_abduction[:, i], mask)
                self._record(f"{prefix}{leg}_hip_outward_component_mean_rad", hip_outward[:, i], mask)
                self._record(f"{prefix}{leg}_hip_adduction_fraction", hip_inward[:, i], mask)
                self._record(f"{prefix}{leg}_hip_abduction_rms_rad", hip_abduction[:, i], mask, rms=True)
                self._record(f"{prefix}{leg}_hip_abduction_over_limit_fraction", hip_over[:, i], mask)
                self._record(f"{prefix}{leg}_foot_lateral_mean_m", lateral[:, i], mask)
                self._record(f"{prefix}{leg}_foot_lateral_outer_fraction", feet_outer[:, i], mask)
                self._record(f"{prefix}{leg}_foot_lateral_inner_fraction", feet_inner[:, i], mask)
                self._record(f"{prefix}{leg}_foot_cross_body_center_fraction", feet_crossed[:, i], mask)
            self._record(prefix + "front_foot_lateral_width_mean_m", lateral[:, 0] + lateral[:, 1], mask)
            self._record(prefix + "rear_foot_lateral_width_mean_m", lateral[:, 2] + lateral[:, 3], mask)

        self._record("raw_action_clip_fraction", (raw_actions.abs() > self.clip).to(torch.float32), valid)
        self._record("raw_action_abs_mean", raw_actions.abs(), valid)
        finite_raw = torch.where(valid[:, None] & torch.isfinite(raw_actions), raw_actions.abs(), 0)
        self.raw_max = torch.maximum(self.raw_max, finite_raw.max(dim=1).values.to(torch.float64))
        applied = self.env.action_manager.action
        self._record("applied_vs_clipped_raw_action_rms", applied - raw_actions.clamp(-self.clip, self.clip), valid, rms=True)

        contact_force = self.sensor.data.net_forces_w[:, self.sensor_foot_ids]
        contact = torch.linalg.vector_norm(contact_force, dim=-1) > self.threshold
        foot_speed = torch.linalg.vector_norm(data.body_lin_vel_w[:, self.robot_foot_ids, :2], dim=-1)
        self._record("stance_foot_slip_rms_m_s", foot_speed, valid[:, None] & contact, rms=True)
        self._record("stairs_stance_foot_slip_rms_m_s", foot_speed, stairs_valid[:, None] & contact, rms=True)
        for n in range(5):
            self._record(f"contact_{n}_feet_fraction", (contact.sum(dim=1) == n).to(torch.float32), valid)

        consecutive = torch.zeros_like(valid)
        if self.previous is not None:
            old_valid, old_lengths, old_action, old_joint_vel = self.previous
            consecutive = valid & old_valid & (lengths == old_lengths + 1)
            delta = applied - old_action
            self._record("applied_action_delta_rms_rad", delta * self.scale, consecutive, rms=True)
            self._record("joint_acceleration_rms_rad_s2", (data.joint_vel - old_joint_vel) / self.env.step_dt, consecutive, rms=True)
            if self.previous_previous_action is not None:
                twice = consecutive & self.previous_consecutive
                self._record("applied_action_second_delta_rms_rad", (applied - 2 * old_action + self.previous_previous_action) * self.scale, twice, rms=True)
            self.previous_previous_action = old_action
        self.previous = (valid.clone(), lengths.clone(), applied.clone(), data.joint_vel.clone())
        self.previous_consecutive = consecutive

    def report(self, env_mask=None):
        """Reduce all samples, or only selected environments' samples.

        With one episode per environment, selecting successful environments
        produces diagnostics over completed successful first episodes. This
        avoids conflating early failures with smooth successful traversals.
        """
        selected = torch.ones_like(self.samples, dtype=torch.bool) if env_mask is None else env_mask.to(device=self.samples.device, dtype=torch.bool)
        if selected.shape != self.samples.shape:
            raise ValueError("env_mask must have one boolean per environment")
        values = {}
        counts = {}
        for name, total in self.sums.items():
            count = int(self.counts[name][selected].sum().item())
            counts[name] = count
            value = float(total[selected].sum().item()) / count if count else None
            values[name] = math.sqrt(value) if value is not None and name in self.rms else value
        # A rollout too short to form differences still emits the same schema.
        for name in ("applied_action_delta_rms_rad", "applied_action_second_delta_rms_rad", "joint_acceleration_rms_rad_s2"):
            values.setdefault(name, None)
            counts.setdefault(name, 0)
        sample_count = int(self.samples[selected].sum().item())
        values["raw_action_abs_max"] = float(self.raw_max[selected].max().item()) if sample_count else None
        return {
            **values,
            "valid_env_samples": sample_count,
            "stairs_valid_env_samples": int(self.stairs_samples[selected].sum().item()),
            "nonfinite_metric_elements": int(self.nonfinite[selected].sum().item()),
            "metric_element_counts": counts,
            "selection": "all_environments" if env_mask is None else "environment_mask",
            "selected_env_count": int(selected.sum().item()),
            "sampling": "pre-step; active environments with episode_length > 1; differences require consecutive valid frames",
            "planar_velocity_mae_definition": "mean absolute error across vx and vy components in body frame",
            "forward_velocity_definition": "course_forward_velocity uses world +X (stair ascent direction); body_forward_velocity and forward_velocity_mae use body vx; means weight valid control samples",
            "stairs_traversal_definition": "first complete forward crossing from stair start to end in each episode; boundaries linearly interpolated between consecutive valid pre-step samples; elapsed time includes retreats; one value per completed crossing; spawned-inside, interrupted and incomplete crossings excluded; geometric completion is not episode success",
            "stairs_completed_traversals": counts.get("stairs_traversal_time_mean_s", 0),
            "power_definition": "mean sum(abs(applied_torque * joint_velocity)), not electrical power",
            "joint_acceleration_definition": "finite difference at control step interval",
            "lateral_metric_thresholds": {
                "hip_abduction_max_rad": self.hip_abduction_max,
                "feet_lateral_min_m": self.feet_lateral_min,
                "feet_lateral_max_m": self.feet_lateral_max,
            },
            "hip_abduction_definition": "URDF hip angle multiplied by left +1 / right -1; positive is outward; over-limit fraction uses signed angle > maximum, not absolute angle",
            "hip_outward_component_definition": "mean(max(signed hip angle, 0)); unlike signed mean this cannot be reduced by inward angles; adduction fraction separately counts signed angles below zero",
            "feet_lateral_definition": "foot position relative to root, rotated into full body frame (including roll/pitch), with left +y / right -y positive outward; all gait phases; pair width sums left and right signed lateral positions",
            "feet_cross_body_center_definition": "signed foot lateral coordinate below zero; distinct from merely being inside the configured positive minimum width",
            "stairs_lateral_sampling": {
                "definition": "stairs_ metrics include only valid pre-step frames whose base center is within the stair flight horizontal projection; approach and upper platform excluded; this does not classify individual foot contacts",
                "relative_x_definition": "root_pos_w.x minus current scene.env_origins.x",
                "start_x_inclusive_m": self.stairs_start_x,
                "end_x_exclusive_m": self.stairs_end_x,
            },
        }
