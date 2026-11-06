"""Sensor-driven walk/stairs routing and explicit PD handover for simulation.

The gate uses terrain ray hits, never episode outcome or tile x coordinates.
This is a simulation height-sensor interface, not a real-robot perception stack.
"""
from __future__ import annotations

import math
import torch


class TerrainPolicyGate:
    """Hysteretic terrain gate with delayed exit and smooth controller weight.

    The window spans rear x=-0.45 m through configurable forward lookahead
    (default 0.70 m). Any >5 cm variation requests stairs; <2.5 cm for 0.4 s requests
    walking. Missing rays retain the previous mode. Reset starts in walking,
    with an immediate stairs request on the first valid elevated scan.
    """
    def __init__(self, num_envs, device, dt=.02, blend_seconds=.30,
                 enter_height=.05, exit_height=.025, flat_dwell_seconds=.40,
                 lookahead_m=.70):
        if not 0 <= exit_height < enter_height or min(dt, blend_seconds, flat_dwell_seconds, lookahead_m) <= 0:
            raise ValueError("Invalid gate thresholds or durations")
        if not all(math.isfinite(x) for x in (dt, blend_seconds, enter_height, exit_height, flat_dwell_seconds, lookahead_m)):
            raise ValueError("Gate parameters must be finite")
        self.dt, self.blend_seconds = dt, blend_seconds
        self.enter_height, self.exit_height = enter_height, exit_height
        self.lookahead_m = float(lookahead_m)
        self.boundary_tolerance_m = 1e-5
        self.flat_dwell_seconds = flat_dwell_seconds
        self.flat_dwell_steps = math.ceil(flat_dwell_seconds / dt)
        self.stairs = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.weight = torch.zeros(num_envs, device=device)
        self.flat_frames = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.transitions = torch.zeros_like(self.flat_frames)
        self.last_span = torch.zeros_like(self.weight)
        self.valid_scan = torch.zeros_like(self.stairs)

    def reset(self, ids):
        self.stairs[ids] = False
        self.weight[ids] = 0
        self.flat_frames[ids] = 0
        self.valid_scan[ids] = False
        self.last_span[ids] = float("nan")

    def update(self, root_pos_w, root_quat_w, ray_hits_w):
        # RayCaster grid is yaw-aligned. Undo yaw only: pitch/roll must not make
        # horizontal floor rays look like stairs or remove the rear-foot area.
        w, x, y, z = root_quat_w.unbind(-1)
        yaw = torch.atan2(2 * (w*z + x*y), 1 - 2*(y*y + z*z))
        delta = ray_hits_w - root_pos_w[:, None, :]
        cosine, sine = yaw.cos()[:, None], yaw.sin()[:, None]
        local_x = cosine * delta[..., 0] + sine * delta[..., 1]
        local_y = -sine * delta[..., 0] + cosine * delta[..., 1]
        tolerance = self.boundary_tolerance_m
        usable = (torch.isfinite(ray_hits_w).all(-1) & (local_x >= -.45-tolerance)
                  & (local_x <= self.lookahead_m+tolerance) & (local_y.abs() <= .30+tolerance))
        z_hits = ray_hits_w[..., 2]
        z_max = z_hits.masked_fill(~usable, -torch.inf).amax(-1)
        z_min = z_hits.masked_fill(~usable, torch.inf).amin(-1)
        # A cluster of surviving center/front rays is not evidence that the
        # rear feet and the upcoming footholds are both on flat ground.
        coverage = ((usable & (local_x <= -.20+tolerance)).any(-1)
                    & (usable & (local_x >= self.lookahead_m-.20-tolerance)).any(-1)
                    & (usable & (local_y <= -.10+tolerance)).any(-1)
                    & (usable & (local_y >= .10-tolerance)).any(-1))
        self.valid_scan = (usable.sum(-1) >= 12) & coverage
        span = z_max - z_min
        self.last_span = torch.where(self.valid_scan, span, float("nan"))
        is_flat = self.valid_scan & (span < self.exit_height)
        self.flat_frames = torch.where(is_flat, self.flat_frames + 1, 0)
        old_stairs = self.stairs.clone()
        self.stairs = torch.where(self.valid_scan & (span > self.enter_height), True, self.stairs)
        self.stairs = torch.where(self.flat_frames >= self.flat_dwell_steps, False, self.stairs)
        self.transitions += self.stairs != old_stairs
        target = self.stairs.to(self.weight.dtype)
        self.weight += (target - self.weight).clamp(-self.dt/self.blend_seconds, self.dt/self.blend_seconds)
        return self.weight


class HybridLocomotionController:
    """Run actors with their respective histories and blend target + PD.

    HIM actions are clipped with their original bound before conversion. Walking
    retains the upstream target slew and all four stiffness actions. The caller
    must disable the wrapper's second action clip (clip_actions=None); returned
    actions are joint-target encoding relative to the existing HIM default pose.
    This preserves HIM observations of the *applied* joint target on flat ground.
    The walk actor retains its upstream 16-component clipped output history,
    which differs from the mixed applied target during a handover.
    """
    def __init__(self, env, walk_policy, joint_names, mode="hybrid", action_scale=.25,
                 action_clip=4., stairs_kp=40., stairs_kd=1., gate_lookahead_m=.70,
                 height_sensor_name="height_scanner"):
        if mode not in ("walk", "hybrid"):
            raise ValueError("Expected walk or hybrid controller")
        if not isinstance(height_sensor_name, str) or not height_sensor_name:
            raise ValueError("height_sensor_name must be a non-empty scene sensor name")
        self.env, self.walk, self.mode = env, walk_policy, mode
        self.height_sensor_name = height_sensor_name
        self.robot = env.scene["robot"]
        self.joint_ids = [self.robot.joint_names.index(name) for name in joint_names]
        self.actuator = self.robot.actuators["legs"]
        self.to_actuator = [joint_names.index(name) for name in self.actuator.joint_names]
        # Retain the training actuator's original per-joint clipping buffers;
        # they are already in actuator order, which may differ from actions.
        self.stairs_effort_limit = self.actuator.effort_limit.clone()
        self.default = self.robot.data.default_joint_pos[:, self.joint_ids]
        self.scale, self.clip = action_scale, action_clip
        self.stairs_kp, self.stairs_kd = stairs_kp, stairs_kd
        self.gate = TerrainPolicyGate(env.num_envs, env.device, dt=env.step_dt, lookahead_m=gate_lookahead_m)
        self.previous_target = self.robot.data.joint_pos[:, self.joint_ids].clone()
        self.walk.reset(joint_position=self.previous_target)
        self.last_weights = torch.zeros(env.num_envs, device=env.device)
        self.sample_counts = torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.long)
        self.max_target_delta = torch.zeros(env.num_envs, device=env.device)
        self.active_mode_transitions = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.transition_sequence_phase = torch.zeros_like(self.active_mode_transitions)
        self.completed_transition_sequences = torch.zeros_like(self.active_mode_transitions)

    def action(self, stairs_actions, active=None):
        data = self.robot.data
        active_mask = torch.ones_like(self.last_weights, dtype=torch.bool) if active is None else active.bool()
        old_stairs = self.gate.stairs.clone()
        if self.mode == "walk":
            weights = torch.zeros_like(self.last_weights)
        else:
            weights = self.gate.update(data.root_pos_w, data.root_quat_w,
                                      self.env.scene[self.height_sensor_name].data.ray_hits_w)
        self.active_mode_transitions += (self.gate.stairs != old_stairs) & active_mask
        # Reset walking's action memory once when leaving a pure stair interval,
        # anchoring the upstream slew limit to the actual previous joint target.
        returning = (self.last_weights >= 1-1e-6) & (weights < 1-1e-6)
        self.walk.reset(returning, joint_position=self.previous_target[returning])
        walk = self.walk.step(data.root_ang_vel_b, data.root_quat_w,
                              self.env.command_manager.get_command("base_velocity"),
                              data.joint_pos[:, self.joint_ids], data.joint_vel[:, self.joint_ids])
        alpha = weights[:, None]
        stairs_target = self.default + self.scale * stairs_actions.clamp(-self.clip, self.clip)
        def handover(walking, stair):
            # Select endpoints explicitly: IEEE 0*NaN is NaN, so arithmetic
            # weighting alone allows an unselected actor to poison flat gait.
            mixed = (1-alpha)*walking + alpha*stair
            return torch.where(alpha <= 0, walking, torch.where(alpha >= 1, stair, mixed))
        target = handover(walk.joint_target, stairs_target)
        kp = handover(walk.stiffness, torch.full_like(walk.stiffness, self.stairs_kp))
        kd = handover(walk.damping, torch.full_like(walk.damping, self.stairs_kd))
        self.actuator.stiffness.copy_(kp[:, self.to_actuator])
        self.actuator.damping.copy_(kd[:, self.to_actuator])
        self.actuator.effort_limit.copy_(handover(
            walk.effort_limit[:, self.to_actuator], self.stairs_effort_limit))
        # Reset poses are excluded from target-jump measurements.
        valid = active_mask & (self.env.episode_length_buf > 1)
        self.max_target_delta = torch.maximum(self.max_target_delta,
            torch.where(valid, (target-self.previous_target).abs().amax(-1), 0))
        categories = torch.stack((weights <= 1e-6, (weights > 1e-6) & (weights < 1-1e-6), weights >= 1-1e-6), -1)
        self.sample_counts += categories & valid[:, None]
        old_phase = self.transition_sequence_phase.clone()
        starts = (old_phase == 0) & categories[:, 0] & valid
        reaches_stairs = (old_phase == 1) & categories[:, 2] & valid
        completes = (old_phase == 2) & categories[:, 0] & valid
        self.transition_sequence_phase += starts | reaches_stairs | completes
        self.completed_transition_sequences += completes
        self.previous_target.copy_(target)
        self.last_weights.copy_(weights)
        return (target - self.default) / self.scale

    def reset(self, done):
        ids = done.bool()
        self.previous_target[ids] = self.robot.data.joint_pos[ids][:, self.joint_ids]
        self.walk.reset(ids, joint_position=self.previous_target[ids])
        self.gate.reset(ids)
        self.last_weights[ids] = 0
        self.transition_sequence_phase[ids] = 0

    def report(self):
        return {
            "mode": self.mode,
            "gate": f"{self.height_sensor_name} rays in yaw frame x=[-0.45,{self.gate.lookahead_m:g}] m, |y|<=0.30 m; height span enter>{self.gate.enter_height:g} m, exit<{self.gate.exit_height:g} m for {self.gate.flat_dwell_steps*self.gate.dt:g} s; blend {self.gate.blend_seconds:g} s in both directions",
            "gate_parameters": {
                "height_sensor_name": self.height_sensor_name,
                "lookahead_m": self.gate.lookahead_m,
                "rear_boundary_m": -.45,
                "half_width_m": .30,
                "boundary_tolerance_m": self.gate.boundary_tolerance_m,
                "front_coverage_min_x_m": self.gate.lookahead_m-.20,
                "minimum_valid_rays": 12,
                "enter_height_m": self.gate.enter_height,
                "exit_height_m": self.gate.exit_height,
                "flat_dwell_seconds": self.gate.flat_dwell_seconds,
                "flat_dwell_steps": self.gate.flat_dwell_steps,
                "control_dt_s": self.gate.dt,
                "entry_blend_seconds": self.gate.blend_seconds,
                "exit_blend_seconds": self.gate.blend_seconds,
            },
            "walk_transition_stairs_samples": self.sample_counts.sum(0).cpu().tolist(),
            "mode_transitions_per_env": self.active_mode_transitions.cpu().tolist(),
            "flat_stairs_flat_completed_per_env": (self.completed_transition_sequences > 0).cpu().tolist(),
            "flat_stairs_flat_completion_counts_per_env": self.completed_transition_sequences.cpu().tolist(),
            "transition_metric_note": "only active environments are counted; completed sequence requires pure walking, pure stairs, then pure walking within one episode, excluding reset frames; a sequence is not itself course success",
            "max_joint_target_delta_rad": float(self.max_target_delta.max().item()),
            "walk_interface": "upstream 49 observations / 16 actions, 50 Hz, dynamic PD gains and 0.1 rad target slew",
            "action_metric_note": "manager actions encode executed mixed joint targets in HIM units; raw-action clip metrics are not upstream network clipping metrics",
        }
