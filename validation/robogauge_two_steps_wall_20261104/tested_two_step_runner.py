#!/usr/bin/env python3
"""Offline MoE-CTS test: two 15 cm steps, 90 cm upper tread and terminal wall.

Reuses the audited original-policy adapter from test_robogauge_stairs. There is
no robot transport, network interface or motor command in this program.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import test_robogauge_stairs as original

ORIGINAL_MAKE_SCENE = original.make_scene
PLATFORM_START = .30
PLATFORM_END = 1.20
TOP_Z = .30
WALL_THICKNESS = .10
WALL_WIDTH = 3.
WALL_HEIGHT = 2.


def make_scene(robot_xml, args):
    root = ET.fromstring(ORIGINAL_MAKE_SCENE(robot_xml, args))
    world = root.find("worldbody")
    # The 90 cm includes the entire second-level tread, not another 30 cm tread
    # followed by a 90 cm landing: x=[.30,1.20], z=.30.
    world.remove(world.find("geom[@name='stair_2']"))
    platform = world.find("geom[@name='top_platform']")
    platform.set("size", ".45 .45 .15")
    platform.set("pos", ".75 0 .15")
    ET.SubElement(world, "geom", name="terminal_wall", type="box",
                  size=".05 1.5 1", pos="1.25 0 1",
                  rgba=".65 .44 .37 1", friction="0.8 0.02 0.01")
    return ET.tostring(root, encoding="unicode")


class Experiment(original.Experiment):
    def __init__(self, args):
        self.command_vx = args.vx
        # Only this single-threaded constructor's scene factory is replaced;
        # the shared runner on disk and its policy/control code stay unchanged.
        previous_factory = original.make_scene
        try:
            original.make_scene = make_scene
            super().__init__(args)
        finally:
            original.make_scene = previous_factory
        mj, m = self.mj, self.model
        self.wall_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "terminal_wall")
        self.robot_geoms = [i for i in range(m.ngeom)
                            if m.geom_bodyid[i] != 0 and (m.geom_contype[i] or m.geom_conaffinity[i])]
        allowed = {int(mj.mjtGeom.mjGEOM_SPHERE), int(mj.mjtGeom.mjGEOM_BOX),
                   int(mj.mjtGeom.mjGEOM_CAPSULE), int(mj.mjtGeom.mjGEOM_CYLINDER)}
        if any(int(m.geom_type[i]) not in allowed for i in self.robot_geoms):
            raise ValueError("Unimplemented robot collision-geometry bound")
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 2.7, 135., -23.
        if self.viewer is not None:
            self.viewer.cam.distance = self.camera.distance
            self.viewer.cam.azimuth = self.camera.azimuth
            self.viewer.cam.elevation = self.camera.elevation

    def observation(self):
        obs = super().observation()
        obs["velocity_commands"] = self.np.array([self.command_vx, 0., 0.])
        return obs

    def geom_description(self, gid):
        mj, m = self.mj, self.model
        bid = int(m.geom_bodyid[gid])
        return {"geom_id": int(gid), "geom_name": mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, int(gid)),
                "body_id": bid, "body_name": mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, bid)}

    def wall_clearance(self):
        """Exact world-x extents for every effective robot collision primitive."""
        np, mj, m, d = self.np, self.mj, self.model, self.data
        furthest_x, furthest_gid = -math.inf, None
        for gid in self.robot_geoms:
            kind = int(m.geom_type[gid])
            size = m.geom_size[gid]
            rotation = d.geom_xmat[gid].reshape(3, 3)
            if kind == int(mj.mjtGeom.mjGEOM_SPHERE):
                extent = float(size[0])
            elif kind == int(mj.mjtGeom.mjGEOM_BOX):
                extent = float(np.abs(rotation[0]) @ size)
            elif kind == int(mj.mjtGeom.mjGEOM_CAPSULE):
                extent = float(size[0] + size[1]*abs(rotation[0, 2]))
            elif kind == int(mj.mjtGeom.mjGEOM_CYLINDER):
                extent = float(size[0]*math.hypot(rotation[0, 0], rotation[0, 1]) + size[1]*abs(rotation[0, 2]))
            else:
                raise AssertionError("Unvalidated collision primitive")
            front = float(d.geom_xpos[gid, 0])+extent
            if front > furthest_x:
                furthest_x, furthest_gid = front, gid
        return {"signed_clearance_m": PLATFORM_END-furthest_x,
                "robot_frontmost_collision_x_m": furthest_x,
                "closest_geometry": self.geom_description(furthest_gid)}

    def wall_contacts(self, t):
        np, mj, m, d = self.np, self.mj, self.model, self.data
        contacts, max_force = [], 0.
        for i in range(d.ncon):
            c = d.contact[i]
            a, b = int(c.geom1), int(c.geom2)
            if a != self.wall_id and b != self.wall_id:
                continue
            robot = b if a == self.wall_id else a
            if robot not in self.robot_geoms:
                continue
            force = np.zeros(6)
            mj.mj_contactForce(m, d, i, force)
            force_norm = float(np.linalg.norm(force[:3]))
            max_force = max(max_force, force_norm)
            if force_norm > 1. or float(c.dist) < 0.:
                contacts.append({"time_s": t, **self.geom_description(robot),
                                 "contact_distance_m": float(c.dist), "penetrating": bool(c.dist < 0.),
                                 "contact_force_norm_n": force_norm, "normal_force_n": abs(float(force[0])),
                                 "position_m": c.pos.tolist()})
        return contacts, max_force

    def episode(self, sample, label):
        np, m, d, a = self.np, self.model, self.data, self.args
        self.command_vx = a.vx
        start = self.initialize(sample)
        audit = original.GuardAudit(np, m.jnt_range[self.jids].copy())
        initial_clearance = self.wall_clearance()
        min_clearance = {"time_s": 0., **initial_clearance}
        if a.video:
            import imageio.v2 as imageio
            self.writer = imageio.get_writer(str(a.outdir/f"{label}.mp4"), fps=a.video_fps,
                                            codec="libx264", quality=7)
        target = self.policy.q0.copy()
        start_x = float(d.qpos[0])
        previous_x = start_x
        max_x = start_x
        climb_entry_t = climb_exit_t = None
        collision_time = fall_time = side_time = success_time = 0.
        roll_sq = 0.
        frames, next_video, next_viewer = 0, 0., 0.
        status = "timeout"
        first_wall_contact = stop_event = platform_event = None
        first_zero_inference = None
        max_wall_force = 0.
        numeric_failure = None
        recent_platform_contacts, progress = deque(), deque()
        stalled = False
        last_predicate = {}
        start_wall = time.monotonic()
        try:
            for step in range(math.ceil(a.duration/a.physics_dt)):
                t = (step+1)*a.physics_dt
                try:
                    if step % self.decimation == 0:
                        if self.command_vx == 0. and first_zero_inference is None:
                            first_zero_inference = step*a.physics_dt
                        target = self.policy.act(self.observation())
                    self.physics(target, audit, t)
                except FloatingPointError as exc:
                    status = "numeric_failure"
                    numeric_failure = {"error": str(exc), "attempted_physics_time_s": t,
                                       "invalid_sample_excluded_from_audit": True,
                                       "valid_physics_samples_retained_per_joint": audit.n}
                    break
                # Inherited physics records the failed substep before this
                # adjudication and refreshes contacts/geometry with mj_forward.
                position = d.qpos[:3]
                current_x = float(position[0])
                if climb_entry_t is None and previous_x < 0. <= current_x:
                    climb_entry_t = t-a.physics_dt+a.physics_dt*(-previous_x)/(current_x-previous_x)
                if climb_entry_t is not None and climb_exit_t is None and previous_x < PLATFORM_START <= current_x:
                    climb_exit_t = t-a.physics_dt+a.physics_dt*(PLATFORM_START-previous_x)/(current_x-previous_x)
                previous_x = current_x
                max_x = max(max_x, current_x)
                rotation = d.xmat[self.base].reshape(3, 3)
                tilt = math.acos(float(np.clip(rotation[2, 2], -1., 1.)))
                roll = math.atan2(rotation[2, 1], rotation[2, 2])
                roll_sq += roll*roll
                foot = d.geom_xpos[self.feet]
                touching, platform_feet, trunk, side_floor = self.contacts()
                recent_platform_contacts.append((t, platform_feet.copy()))
                while recent_platform_contacts and recent_platform_contacts[0][0] < t-.10:
                    recent_platform_contacts.popleft()
                supported_feet = set().union(*(v for _, v in recent_platform_contacts))
                feet_inside = bool(np.all(foot[:, 0]-self.radii >= PLATFORM_START+.003) and
                                   np.all(foot[:, 0]+self.radii <= PLATFORM_END-.003) and
                                   np.all(np.abs(foot[:, 1])+self.radii <= a.width/2-.003))
                feet_height = bool(np.all(foot[:, 2]-self.radii >= TOP_Z-.025) and
                                   np.all(foot[:, 2]-self.radii <= TOP_Z+.20))
                base_ok = bool(PLATFORM_START <= current_x <= PLATFORM_END and
                               .18 <= float(position[2])-TOP_Z <= .60 and tilt < math.pi/4)
                on_platform = feet_inside and feet_height and len(supported_feet) >= 2 and base_ok
                clearance = self.wall_clearance()
                if clearance["signed_clearance_m"] < min_clearance["signed_clearance_m"]:
                    min_clearance = {"time_s": t, **clearance}
                wall_hits, wall_force = self.wall_contacts(t)
                max_wall_force = max(max_wall_force, wall_force)
                if wall_hits and first_wall_contact is None:
                    first_wall_contact = {"time_s": t, "contacts": wall_hits, "clearance": clearance}
                horizontal_speed = float(np.linalg.norm(d.qvel[:2]))
                # Free-joint angular velocity is body-local; rotate it into the
                # world frame for the yaw-axis angular-speed stop criterion.
                world_angular_velocity = rotation @ d.qvel[3:6]
                world_yaw_rate = float(world_angular_velocity[2])
                vertical_speed = float(d.qvel[2])
                event_state = {"time_s": t, "base_position_m": position.tolist(),
                               "foot_centers_m": foot.tolist(), "horizontal_speed_m_s": horizontal_speed,
                               "vertical_speed_m_s": vertical_speed, "tilt_rad": tilt,
                               "world_angular_velocity_rad_s": world_angular_velocity.tolist(),
                               "world_yaw_rate_rad_s": world_yaw_rate, "wall_clearance": clearance}
                if on_platform and platform_event is None:
                    platform_event = event_state
                if on_platform and a.command_mode == "stop_on_platform" and stop_event is None:
                    stop_event = {**event_state, "command_before_m_s": self.command_vx,
                                  "command_after_m_s": 0., "source": "external simulation-ground-truth platform trigger"}
                    self.command_vx = 0.
                stopped = bool(horizontal_speed < .05 and abs(vertical_speed) < .05 and
                               np.all(np.abs(world_angular_velocity) < .1) and tilt < math.pi/12)
                successful = on_platform and stopped and first_wall_contact is None
                success_time = success_time+a.physics_dt if successful else 0.
                collision_time = collision_time+a.physics_dt if trunk else 0.
                fall_time = fall_time+a.physics_dt if tilt > math.pi/3 or position[2] < .12 else 0.
                side = bool(abs(position[1]) > a.width/2 or side_floor)
                side_time = side_time+a.physics_dt if side else 0.
                last_predicate = {"all_four_feet_inside_upper_platform": feet_inside,
                                  "feet_near_platform_height": feet_height, "base_height_pose_ok": base_ok,
                                  "different_support_feet_in_last_100ms": len(supported_feet),
                                  "horizontal_speed_m_s": horizontal_speed, "world_yaw_rate_rad_s": world_yaw_rate,
                                  "vertical_speed_m_s": vertical_speed, "tilt_rad": tilt,
                                  "world_angular_velocity_rad_s": world_angular_velocity.tolist(),
                                  "stopped_speed_thresholds_passed": stopped, "never_touched_wall": first_wall_contact is None,
                                  "continuous_stopped_supported_s": success_time}
                progress.append((t, current_x))
                while progress and progress[0][0] < t-5.:
                    progress.popleft()
                if t > 5. and self.command_vx > 0. and current_x-progress[0][1] < .05:
                    stalled = True
                if self.renderer is not None and t+1e-9 >= next_video:
                    self.renderer.update_scene(d, camera=self.camera)
                    self.writer.append_data(self.renderer.render())
                    frames += 1
                    next_video += 1/a.video_fps
                if self.viewer is not None:
                    if not self.viewer.is_running():
                        status = "viewer_closed"
                        break
                    if t >= next_viewer:
                        self.viewer.sync()
                        next_viewer += 1/60
                    if a.realtime:
                        delay = start_wall+t-time.monotonic()
                        if delay > 0.:
                            time.sleep(delay)
                if first_wall_contact is not None:
                    status = "wall_contact"
                    break
                if collision_time >= .05:
                    status = "trunk_contact"
                    break
                if fall_time >= .1:
                    status = "fall"
                    break
                if side_time >= .1:
                    status = "side_exit"
                    break
                if success_time >= 2.-1e-9:
                    status = "success"
                    break
            guard = audit.report()
            if numeric_failure:
                guard["local_guard_audit_passed"] = False
                guard["incomplete_due_to_numeric_failure"] = True
            report = {"label": label, "status": status, "task_success": status == "success",
                      "reached_upper_platform": platform_event is not None, "platform_event": platform_event,
                      "stop_event": stop_event, "first_zero_command_policy_inference_time_s": first_zero_inference,
                      "command_mode": a.command_mode, "initial_command_m_s": a.vx, "final_command_m_s": self.command_vx,
                      "elapsed_s": float(d.time), "initial_state": start,
                      "final_base_position_m": d.qpos[:3].tolist(), "final_foot_centers_m": d.geom_xpos[self.feet].tolist(),
                      "forward_displacement_m": float(d.qpos[0]-start_x), "max_forward_displacement_m": max_x-start_x,
                      "mean_world_forward_speed_m_s": float((d.qpos[0]-start_x)/max(d.time, 1e-9)),
                      "climb_horizontal_speed_m_s": (PLATFORM_START/(climb_exit_t-climb_entry_t)
                                                       if climb_exit_t is not None and climb_exit_t > climb_entry_t else None),
                      "climb_speed_window": {"entry_time_s": climb_entry_t, "exit_time_s": climb_exit_t,
                                             "definition": "Interpolated base first crossings from first riser x=0 to upper-platform front x=.30; not a full-body ascent-success measure"},
                      "roll_rms_rad": math.sqrt(roll_sq/max(audit.n, 1)),
                      "stalled_while_forward_command_5s_window_seen": stalled,
                      "initial_wall_clearance": initial_clearance, "wall_min_clearance": min_clearance,
                      "final_wall_clearance": self.wall_clearance(), "first_wall_contact": first_wall_contact,
                      "max_wall_contact_force_n": max_wall_force, "last_success_predicate": last_predicate,
                      "guard_audit": guard, "numeric_failure": numeric_failure,
                      "torque_limit_mode": a.torque_limit_mode, "video_frames": frames,
                      "real_robot_driven": False}
            original.write_json(a.outdir/f"{label}.json", report)
            print(json.dumps({key: report[key] for key in ("label", "status", "elapsed_s", "reached_upper_platform", "forward_displacement_m")}), flush=True)
            return report
        finally:
            if self.writer is not None:
                self.writer.close()
                self.writer = None


def aggregate(reports):
    result = original.aggregate(reports)
    result.update({"reached_upper_platform": sum(r["reached_upper_platform"] for r in reports),
                   "stop_commands_triggered": sum(r["stop_event"] is not None for r in reports),
                   "wall_contacts": sum(r["first_wall_contact"] is not None for r in reports),
                   "wall_min_clearance_m": min(r["wall_min_clearance"]["signed_clearance_m"] for r in reports),
                   "max_wall_contact_force_n": max(r["max_wall_contact_force_n"] for r in reports),
                   "episodes_passing_local_guard_audit": sum(r["guard_audit"]["local_guard_audit_passed"] for r in reports),
                   "requested_torque_peak_ratio": max(r["guard_audit"]["requested_torque_peak_ratio"] for r in reports),
                   "applied_torque_peak_ratio": max(r["guard_audit"]["applied_torque_peak_ratio"] for r in reports),
                   "episode_files": [r["label"]+".json" for r in reports]})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", type=Path, default=original.POLICY_DIR/"exported/policy.onnx")
    p.add_argument("--config", type=Path, default=original.POLICY_DIR/"params/deploy.yaml")
    p.add_argument("--robot-xml", type=Path, default=original.UPSTREAM/"RoboGauge/resources/robots/go2/go2.xml")
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--episodes", type=int, default=8, help="Perturbed starts per seed, plus deterministic baseline")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    p.add_argument("--no-baseline", action="store_true")
    p.add_argument("--vx", type=float, default=.6)
    p.add_argument("--duration", type=float, default=25.)
    p.add_argument("--settle-seconds", type=float, default=2.)
    p.add_argument("--torque-limit-mode", choices=("author", "physical"), default="author")
    p.add_argument("--command-mode", choices=("constant", "stop_on_platform"), default="constant")
    view = p.add_mutually_exclusive_group()
    view.add_argument("--headless", action="store_true")
    view.add_argument("--viewer", action="store_true")
    p.add_argument("--hold-final", action="store_true")
    p.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--video", action="store_true")
    p.add_argument("--video-fps", type=int, default=30)
    p.add_argument("--video-width", type=int, default=640)
    p.add_argument("--video-height", type=int, default=480)
    args = p.parse_args()
    args.steps, args.height, args.tread, args.width, args.platform = 2, .15, .30, .90, .90
    args.start_distance, args.physics_dt = .50, .002
    if args.outdir.exists():
        p.error("Output directory already exists; choose a new run directory")
    if args.episodes < 0 or (args.no_baseline and not args.episodes):
        p.error("Invalid empty/negative episode schedule")
    if args.hold_final and not args.viewer:
        p.error("--hold-final requires --viewer")
    for name in ("vx", "duration", "settle_seconds"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0.:
            p.error(name+" must be finite and positive")
    if args.duration < 25.:
        p.error("This two-step/wall benchmark requires at least 25 s maximum episode duration")
    if min(args.video_fps, args.video_width, args.video_height) <= 0:
        p.error("Video dimensions and fps must be positive")
    if not args.viewer:
        os.environ.setdefault("MUJOCO_GL", "egl")
    args.outdir.mkdir(parents=True)
    experiment, reports = None, []
    try:
        experiment = Experiment(args)
        np = experiment.np
        samples = [] if args.no_baseline else [{"kind": "deterministic", "seed": None, "episode": 0, "y_m": 0., "yaw_rad": 0.}]
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            for i in range(args.episodes):
                samples.append({"kind": "perturbed", "seed": seed, "episode": i,
                                "y_m": float(rng.uniform(-.02, .02)), "yaw_rad": float(rng.uniform(-math.pi/90, math.pi/90))})
        (args.outdir/"scene.xml").write_text(experiment.xml)
        manifest = {"upstream": "RoboGauge f116b00a934262348f41cbfd0d07cc4cb6fbf8d8",
                    "model_sha256": original.digest(args.policy), "deploy_yaml_sha256": original.digest(args.config),
                    "robot_xml_sha256": original.digest(args.robot_xml), "runner_sha256": original.digest(__file__),
                    "shared_runner_sha256": original.digest(original.__file__),
                    "source_files": {"policy": str(args.policy.resolve()), "config": str(args.config.resolve()), "robot_xml": str(args.robot_xml.resolve())},
                    "geometry": {"width_m": .90, "risers": 2, "rise_m": .15,
                                 "first_tread": {"x_min_m": 0., "x_max_m": .30, "top_z_m": .15},
                                 "second_tread_and_upper_platform": {"x_min_m": .30, "x_max_m": 1.20, "depth_m": .90, "top_z_m": .30},
                                 "wall": {"inner_face_x_m": 1.20, "thickness_m": .10, "width_m": 3., "height_m": 2.},
                                 "start_front_foot_sphere_leading_edge_distance_m": .50},
                    "geometry_assumptions": "Uppermost depth 90 cm includes the second tread; wall dimensions (3 m wide, 2 m tall, 10 cm thick) are simulation assumptions. No side railings.",
                    "command_mode": args.command_mode, "initial_command_m_s": args.vx,
                    "stop_command_definition": "stop_on_platform alone latches zero vx at first supported full-four-foot upper-platform entry using simulation ground truth. Original ONNX/history/PD unchanged. Not autonomous wall recognition.",
                    "success_definition": "Full four foot spheres within upper platform with 3 mm edge margin, feet surface clearance [-.025,.20] m, at least two different supporting feet over last .10 s, base .18-.60 m above platform; world planar base speed <.05 m/s, absolute vertical speed <.05 m/s, absolute angular velocity about EACH world x/y/z axis <.1 rad/s, and tilt <15 degrees continuously 2 seconds; no wall contact ever. Platform-entry/stop-command trigger retains tilt <45 degrees and does not require these stopped-state limits.",
                    "stop_success_thresholds": {"horizontal_speed_m_s": .05, "absolute_vertical_speed_m_s": .05,
                                                "absolute_each_world_angular_velocity_rad_s": .1, "tilt_rad": math.pi/12,
                                                "continuous_duration_s": 2.},
                    "wall_failure_definition": "Any effective robot collision geom touching wall with contact-force vector norm >1 N or negative signed contact distance; fails immediately after the audited terminal substep.",
                    "wall_clearance_definition": "Signed x gap to wall inner plane from frontmost effective robot sphere/box/capsule/cylinder exact world-x extent, including head and legs. This infinite-plane envelope diagnostic is not itself the wall-contact criterion.",
                    "other_failures": "Inherited trunk/terrain contact 50 ms, >60 degree tilt or base below .12 m 100 ms, or lateral exit 100 ms; timeout; numerical failure.",
                    "physics_dt": args.physics_dt, "control_dt": experiment.control_dt,
                    "kp": experiment.kp.tolist(), "kd": experiment.kd.tolist(),
                    "torque_limit_mode": args.torque_limit_mode, "author_actuator_limits": experiment.author_actuator_limits,
                    "physical_mode_effort_limit_nm": experiment.physical_effort.tolist(),
                    "torque_note": "author preserves original unlimited motors; physical explicitly clips PD requested torque and is a separate comparison, not an alteration of original weights.",
                    "observation_contract": "Shared audited adapter: term-major 5-frame history, 225 floats, raw-term clipping then scaling, raw last action.",
                    "action_contract": "Shared audited adapter: FL/FR/RL/RR joint mapping, upstream target scale/offset then target clipping, 50 Hz inference and 500 Hz PD.",
                    "samples": samples, "randomization": "y +/-2 cm and yaw +/-2 degrees only; no dynamics randomization or noise",
                    "maximum_episode_duration_s": args.duration, "real_robot_driven": False, "network_interfaces_used": False}
        original.write_json(args.outdir/"manifest.json", manifest)
        for i, sample in enumerate(samples):
            reports.append(experiment.episode(sample, f"episode_{i:03d}"))
            if reports[-1]["status"] == "viewer_closed":
                break
        original.write_json(args.outdir/"per_episode.json", {"episodes": reports})
        summary = aggregate(reports)
        summary.update({"command_mode": args.command_mode, "torque_limit_mode": args.torque_limit_mode,
                        "limitations": "This finite MuJoCo test is not real-world validation or permission to actuate the robot; stop_on_platform depends on external ground truth."})
        original.write_json(args.outdir/"summary.json", summary)
        print("ROBOGAUGE_TWO_STEP_WALL_COMPLETE "+json.dumps(summary), flush=True)
        if args.hold_final:
            print("ROBOGAUGE_VIEWER_HOLD_FINAL: results saved; physics stopped", flush=True)
            while experiment.viewer.is_running():
                experiment.viewer.sync()
                time.sleep(1/30)
    except BaseException as exc:
        original.write_json(args.outdir/"failure.json", {"error": type(exc).__name__+": "+str(exc),
                                                        "completed_episodes": len(reports), "real_robot_driven": False})
        raise
    finally:
        if experiment is not None:
            experiment.close()


if __name__ == "__main__":
    main()
