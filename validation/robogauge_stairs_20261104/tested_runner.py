#!/usr/bin/env python3
"""Offline RoboGauge ONNX/MuJoCo staircase check; no DDS or robot connection.

Defaults assume six 15 cm rises, 30 cm treads, 90 cm clear width and a 1.5 m
top landing. One deterministic start plus eight starts per seed are evaluated.
The closest front-foot collision sphere edge starts exactly 0.50 m from the
first riser, after PD settling. Model physics and deployment PD/action/observation
contracts are retained; the local robot's guards are diagnostics, not early stops.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream/robogauge"
POLICY_DIR = UPSTREAM / "unitree_cpp_deploy/logs/rsl_rl/go2_moe_cts_expert_goal_137000_0.6745"
LEGS = ("FL", "FR", "RL", "RR")
KINDS = ("hip", "thigh", "calf")
JOINTS = tuple(f"{leg}_{kind}_joint" for leg in LEGS for kind in KINDS)
OBS_ORDER = ("base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos_rel", "joint_vel_rel", "last_action")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    def clean(item):
        if isinstance(item, dict):
            return {k: clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(v) for v in item]
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item
    with Path(path).open("x") as f:
        json.dump(clean(value), f, indent=2, allow_nan=False)
        f.write("\n")


def make_scene(robot_xml, args):
    """Copy XML in memory; retain original robot bodies, joints and motors."""
    tree = ET.parse(robot_xml)
    root = tree.getroot()
    compiler = root.find("compiler")
    compiler.set("meshdir", str((robot_xml.parent / compiler.get("meshdir", ".")).resolve()))
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", str(args.physics_dt))
    world = root.find("worldbody")
    ET.SubElement(world, "geom", name="floor", type="plane", size="15 10 0.1",
                  pos="0 0 0", rgba="0.30 0.34 0.38 1", friction="0.8 0.02 0.01")
    ET.SubElement(world, "light", pos="0 -2 5", dir="0 0 -1", diffuse="0.85 0.85 0.85")
    ET.SubElement(world, "light", pos="3 3 4", dir="-0.3 -0.3 -1", diffuse="0.6 0.6 0.6")
    for i in range(args.steps):
        height = (i + 1) * args.height
        ET.SubElement(world, "geom", name=f"stair_{i+1}", type="box",
                      size=f"{args.tread/2} {args.width/2} {height/2}",
                      pos=f"{(i+.5)*args.tread} 0 {height/2}",
                      rgba=f"{.45 + .035*(i%2)} 0.51 0.57 1", friction="0.8 0.02 0.01")
    if args.steps:
        top = args.steps * args.height
        ET.SubElement(world, "geom", name="top_platform", type="box",
                      size=f"{args.platform/2} {args.width/2} {top/2}",
                      pos=f"{args.steps*args.tread+args.platform/2} 0 {top/2}",
                      rgba="0.4 0.55 0.45 1", friction="0.8 0.02 0.01")
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_cfg = visual.find("global")
    if global_cfg is None:
        global_cfg = ET.SubElement(visual, "global")
    global_cfg.set("offwidth", str(args.video_width))
    global_cfg.set("offheight", str(args.video_height))
    return ET.tostring(root, encoding="unicode")


class GuardAudit:
    """Unclipped requests and post-integration states, including terminal steps."""
    def __init__(self, np, model_ranges):
        self.np = np
        self.n = 0
        self.startup_n = 0
        self.lower = np.array([(-1.0472 if k == "hip" else (-1.5708 if l[0] == "F" else -.5236) if k == "thigh" else -2.7227) for l in LEGS for k in KINDS])
        self.upper = np.array([(1.0472 if k == "hip" else (3.4907 if l[0] == "F" else 4.5379) if k == "thigh" else -.83776) for l in LEGS for k in KINDS])
        self.effort = np.array([23.7, 23.7, 45.43] * 4)
        self.nominal_speed = np.array([30.1, 30.1, 15.7] * 4)
        self.model_ranges = model_ranges
        self.keys = ("velocity", "nominal_velocity", "actual_position", "target_position", "tracking_error", "requested_torque", "applied_torque", "model_actual_position", "model_target_position")
        self.counts = {k: np.zeros(12, dtype=np.int64) for k in self.keys}
        self.startup_counts = {k: np.zeros(12, dtype=np.int64) for k in self.keys}
        self.peaks = {k: np.zeros(12) for k in self.keys}
        self.first = {}

    def add(self, t, q_pre, q, dq, target, requested, applied):
        np = self.np
        values = (q_pre, q, dq, target, requested, applied)
        if not all(np.isfinite(v).all() for v in values):
            raise FloatingPointError("Nonfinite physics/command sample")
        over = lambda x, lo, hi: np.maximum(lo-x, 0) + np.maximum(x-hi, 0)
        metrics = {
            "velocity": np.abs(dq), "nominal_velocity": np.abs(dq),
            "actual_position": over(q, self.lower, self.upper),
            "target_position": over(target, self.lower, self.upper),
            "tracking_error": np.maximum(np.abs(target-q_pre), np.abs(target-q)),
            "requested_torque": np.abs(requested), "applied_torque": np.abs(applied),
            "model_actual_position": over(q, self.model_ranges[:, 0], self.model_ranges[:, 1]),
            "model_target_position": over(target, self.model_ranges[:, 0], self.model_ranges[:, 1]),
        }
        thresholds = {"velocity": 13., "nominal_velocity": self.nominal_speed,
                      "tracking_error": .9, "requested_torque": self.effort, "applied_torque": self.effort}
        self.n += 1
        startup = t <= 2.0 + 1e-9
        self.startup_n += int(startup)
        for key, value in metrics.items():
            excess = value > thresholds.get(key, 0.)
            self.counts[key] += excess
            if startup:
                self.startup_counts[key] += excess
            self.peaks[key] = np.maximum(self.peaks[key], value)
            if key not in self.first and excess.any():
                j = int(np.flatnonzero(excess)[0])
                threshold = np.broadcast_to(thresholds.get(key, 0.), (12,))[j]
                self.first[key] = {"time_s": float(t), "joint": JOINTS[j], "value": float(value[j]), "threshold": float(threshold)}

    def report(self):
        result = {"physics_samples_per_joint": self.n, "startup_samples_per_joint": self.startup_n,
                  "scope": "Every policy physics substep, including failed task-terminal samples; settling is separate; numerical-corruption samples are flagged and excluded",
                  "speed_limit_rad_s": 13., "tracking_limit_rad": .9, "first_violations": self.first,
                  "joint_order": list(JOINTS), "joints": {}}
        for key in self.keys:
            result[key] = {"excess_samples": int(self.counts[key].sum()),
                           "excess_fraction": float(self.counts[key].sum() / (12*self.n)) if self.n else None,
                           "peak": float(self.peaks[key].max()),
                           "startup_excess_fraction": float(self.startup_counts[key].sum()/(12*self.startup_n)) if self.startup_n else None}
        for j, name in enumerate(JOINTS):
            result["joints"][name] = {
                "local_guard_position_range_rad": [float(self.lower[j]), float(self.upper[j])],
                "model_position_range_rad": self.model_ranges[j].tolist(),
                "effort_limit_nm": float(self.effort[j]), "nominal_description_speed_rad_s": float(self.nominal_speed[j]),
                **{key: {"peak": float(self.peaks[key][j]), "excess_samples": int(self.counts[key][j]),
                         "excess_fraction": float(self.counts[key][j]/self.n) if self.n else None} for key in self.keys},
            }
        guard_keys = ("velocity", "actual_position", "target_position", "tracking_error", "requested_torque", "applied_torque")
        result["local_guard_audit_passed"] = self.n > 0 and all(not self.counts[k].any() for k in guard_keys)
        result["requested_torque_peak_ratio"] = float((self.peaks["requested_torque"] / self.effort).max())
        result["applied_torque_peak_ratio"] = float((self.peaks["applied_torque"] / self.effort).max())
        return result


class Policy:
    def __init__(self, np, ort, cfg, path):
        self.np, self.cfg = np, cfg
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        assert len(inputs) == 1 and inputs[0].name == "obs" and inputs[0].shape == [1, 225], [(i.name, i.shape) for i in inputs]
        assert any(o.name == "actions" and o.shape == [1, 12] for o in self.session.get_outputs())
        assert tuple(cfg["observations"]) == OBS_ORDER
        assert cfg["joint_ids_map"] == [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        self.q0 = np.asarray(cfg["default_joint_pos"], dtype=np.float64)
        action_cfg = cfg["actions"]["JointPositionAction"]
        self.scale = np.asarray(action_cfg["scale"])
        self.offset = np.asarray(action_cfg["offset"])
        self.clip = np.asarray(action_cfg["clip"])
        self.last_action = np.zeros(12, dtype=np.float32)
        self.history = {}

    def reset(self, values):
        self.last_action.fill(0)
        self.history = {name: deque([value.copy() for _ in range(self.cfg["observations"][name]["history_length"])],
                                    maxlen=self.cfg["observations"][name]["history_length"])
                        for name, value in values.items()}

    def act(self, values):
        np = self.np
        chunks = []
        for name in OBS_ORDER:
            self.history[name].append(values[name].copy())
            term = self.cfg["observations"][name]
            chunks.extend(np.clip(x, term["clip"][0], term["clip"][1]) * np.asarray(term["scale"]) for x in self.history[name])
        obs = np.concatenate(chunks).astype(np.float32)[None, :]
        assert obs.shape == (1, 225)
        if not np.isfinite(obs).all():
            raise FloatingPointError("Nonfinite observation")
        raw = self.session.run(["actions"], {"obs": obs})[0][0]
        if raw.shape != (12,) or not np.isfinite(raw).all():
            raise FloatingPointError("Invalid policy action")
        # This is the upstream C++ contract: clip the processed target, retain
        # the UNCLIPPED network action for the next observation.
        self.last_action = raw.copy()
        return np.clip(raw*self.scale + self.offset, self.clip[:, 0], self.clip[:, 1])


class Experiment:
    def __init__(self, args):
        import numpy as np
        import mujoco
        import onnxruntime as ort
        import yaml
        self.np, self.mj, self.args = np, mujoco, args
        cfg = yaml.safe_load(args.config.read_text())
        self.cfg = cfg
        self.xml = make_scene(args.robot_xml, args)
        self.model = model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(model)
        self.jids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINTS])
        self.aids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n.removesuffix("_joint")) for n in JOINTS])
        if (self.jids < 0).any() or (self.aids < 0).any():
            raise ValueError("Missing named joints/actuators")
        assert model.nu == 12 and model.nq == 19 and model.nv == 18
        assert np.array_equal(model.actuator_trnid[self.aids, 0], self.jids)
        assert np.allclose(model.actuator_gear[self.aids, 0], 1)
        self.author_actuator_limits = {
            "ctrllimited": model.actuator_ctrllimited[self.aids].astype(bool).tolist(),
            "ctrlrange": model.actuator_ctrlrange[self.aids].tolist(),
            "forcelimited": model.actuator_forcelimited[self.aids].astype(bool).tolist(),
            "forcerange": model.actuator_forcerange[self.aids].tolist(),
            "gear": model.actuator_gear[self.aids].tolist(),
        }
        self.physical_effort = np.array([23.7, 23.7, 45.43]*4)
        self.qadr, self.vadr = model.jnt_qposadr[self.jids], model.jnt_dofadr[self.jids]
        self.base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.feet = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, leg) for leg in LEGS])
        assert (self.feet >= 0).all()
        assert (model.geom_type[self.feet] == mujoco.mjtGeom.mjGEOM_SPHERE).all()
        self.radii = model.geom_size[self.feet, 0].copy()
        self.gyro = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
        self.gyro_adr = int(model.sensor_adr[self.gyro])
        self.platform_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "top_platform" if args.steps else "floor")
        self.terrain_ids = {i for i in range(model.ngeom) if model.geom_bodyid[i] == 0}
        self.trunk_ids = {i for i in range(model.ngeom) if model.geom_bodyid[i] == self.base and (model.geom_contype[i] or model.geom_conaffinity[i])}
        self.policy = Policy(np, ort, cfg, args.policy)
        self.numeric_warnings = {name: int(getattr(mujoco.mjtWarning, name)) for name in
                                 ("mjWARN_BADQPOS", "mjWARN_BADQVEL", "mjWARN_BADQACC", "mjWARN_BADCTRL")}
        self.kp, self.kd = np.asarray(cfg["stiffness"]), np.asarray(cfg["damping"])
        self.control_dt = float(cfg["step_dt"])
        self.decimation = round(self.control_dt / args.physics_dt)
        if not math.isclose(self.decimation*args.physics_dt, self.control_dt):
            raise ValueError("Control dt must divide physics dt exactly")
        self.viewer = self.renderer = self.writer = None
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.camera.trackbodyid = self.base
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 3.1, 135., -24.
        if args.viewer:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(model, self.data)
            self.viewer.cam.type = self.camera.type
            self.viewer.cam.trackbodyid = self.base
            self.viewer.cam.distance = self.camera.distance
            self.viewer.cam.azimuth = self.camera.azimuth
            self.viewer.cam.elevation = self.camera.elevation
        if args.video:
            self.renderer = mujoco.Renderer(model, height=args.video_height, width=args.video_width)

    def observation(self):
        np, d = self.np, self.data
        rotation = d.xmat[self.base].reshape(3, 3)
        return {"base_ang_vel": d.sensordata[self.gyro_adr:self.gyro_adr+3].copy(),
                "projected_gravity": rotation.T @ np.array([0., 0., -1.]),
                "velocity_commands": np.array([self.args.vx, 0., 0.]),
                "joint_pos_rel": d.qpos[self.qadr] - self.policy.q0,
                "joint_vel_rel": d.qvel[self.vadr].copy(), "last_action": self.policy.last_action.copy()}

    def physics(self, target, audit, t):
        np, mj, m, d = self.np, self.mj, self.model, self.data
        q_pre = d.qpos[self.qadr].copy()
        requested = self.kp*(target-q_pre) - self.kd*d.qvel[self.vadr]
        if not np.isfinite(requested).all():
            raise FloatingPointError("Nonfinite PD torque")
        # The author's XML currently has NO motor limits. Preserve that in
        # author mode; physical mode is an explicitly different comparison.
        d.ctrl[self.aids] = (np.clip(requested, -self.physical_effort, self.physical_effort)
                             if self.args.torque_limit_mode == "physical" else requested)
        warnings_before = {name: int(d.warning[index].number) for name, index in self.numeric_warnings.items()}
        previous_time = float(d.time)
        mj.mj_step(m, d)
        triggered = {name: int(d.warning[index].number) for name, index in self.numeric_warnings.items()
                     if int(d.warning[index].number) > warnings_before[name]}
        if triggered or not math.isclose(float(d.time), previous_time+self.args.physics_dt, abs_tol=1e-8):
            raise FloatingPointError("MuJoCo invalid-state warning or automatic reset: " + json.dumps(
                {"warnings": triggered, "time_before": previous_time, "time_after": float(d.time)}))
        applied = d.actuator_force[self.aids].copy()
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            raise FloatingPointError("Nonfinite state")
        audit.add(t, q_pre, d.qpos[self.qadr].copy(), d.qvel[self.vadr].copy(), target, requested, applied)
        # mj_step leaves Cartesian position caches at the previous integration
        # endpoint. Refresh geometry/sensors for start distances and adjudication.
        mj.mj_forward(m, d)

    def contacts(self):
        np, mj, m, d = self.np, self.mj, self.model, self.data
        touching = set()
        platform_feet = set()
        trunk = False
        side_floor = False
        floor_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "floor")
        for i in range(d.ncon):
            c = d.contact[i]
            a, b = int(c.geom1), int(c.geom2)
            if c.dist > .002:
                continue
            force = np.zeros(6)
            mj.mj_contactForce(m, d, i, force)
            if abs(force[0]) <= 1.:
                continue
            pair = {a, b}
            if pair & self.trunk_ids and pair & self.terrain_ids:
                trunk = True
            for j, gid in enumerate(self.feet):
                if int(gid) not in pair:
                    continue
                other = b if a == gid else a
                if other in self.terrain_ids:
                    touching.add(j)
                if other == self.platform_id and abs(float(c.frame[2])) > .7:
                    platform_feet.add(j)
                if self.args.steps and other == floor_id and d.geom_xpos[gid, 0] > .05 and abs(d.geom_xpos[gid, 1]) > self.args.width/2:
                    side_floor = True
        return touching, platform_feet, trunk, side_floor

    def initialize(self, sample):
        np, mj, m, d, a = self.np, self.mj, self.model, self.data, self.args
        mj.mj_resetData(m, d)
        d.qpos[:3] = [-3., 0., .445]
        d.qpos[3:7] = [1., 0., 0., 0.]
        d.qpos[self.qadr] = self.policy.q0
        mj.mj_forward(m, d)
        # Begin just above ground, then allow the original PD/physics to settle.
        d.qpos[2] -= float((d.geom_xpos[self.feet, 2] - self.radii).min()) - .001
        mj.mj_forward(m, d)
        settle = GuardAudit(np, m.jnt_range[self.jids].copy())
        for i in range(round(a.settle_seconds/a.physics_dt)):
            self.physics(self.policy.q0, settle, (i+1)*a.physics_dt)
        rotation = d.xmat[self.base].reshape(3, 3)
        tilt = math.acos(float(np.clip(rotation[2, 2], -1, 1)))
        if tilt > .4 or len(self.contacts()[0]) < 2:
            raise RuntimeError("PD settling did not produce a supported upright start")
        # Apply requested global yaw about the free body while retaining settled
        # roll/pitch; qpos quaternion order is MuJoCo w,x,y,z.
        actual_yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        yaw_delta = sample["yaw_rad"] - actual_yaw
        turn = np.array([math.cos(yaw_delta/2), 0., 0., math.sin(yaw_delta/2)])
        quat = np.empty(4)
        mj.mju_mulQuat(quat, turn, d.qpos[3:7].copy())
        d.qpos[3:7] = quat
        d.qpos[1] = sample["y_m"]
        d.qvel[:] = 0.
        mj.mj_forward(m, d)
        front_edge = float((d.geom_xpos[self.feet[:2], 0] + self.radii[:2]).max())
        d.qpos[0] += -a.start_distance - front_edge
        d.qvel[:] = 0.
        d.qacc_warmstart[:] = 0.
        d.time = 0.
        mj.mj_forward(m, d)
        edges = d.geom_xpos[self.feet[:2], 0] + self.radii[:2]
        distances = -edges
        if not math.isclose(float(distances.min()), a.start_distance, abs_tol=1e-8):
            raise AssertionError("Front collision-sphere distance is not exact")
        self.policy.last_action.fill(0)
        self.policy.reset(self.observation())
        return {"requested_perturbation": sample, "base_position_m": d.qpos[:3].tolist(),
                "base_quaternion_wxyz": d.qpos[3:7].tolist(), "joint_position_rad": d.qpos[self.qadr].tolist(),
                "front_feet": {leg: {"collision_sphere_center_m": d.geom_xpos[self.feet[j]].tolist(),
                                     "radius_m": float(self.radii[j]), "front_edge_distance_to_first_riser_m": float(distances[j])}
                               for j, leg in enumerate(LEGS[:2])},
                "minimum_front_edge_distance_m": float(distances.min()),
                "settle_seconds": a.settle_seconds, "settling_guard_audit": settle.report()}

    def episode(self, sample, label):
        np, mj, m, d, a = self.np, self.mj, self.model, self.data, self.args
        start = self.initialize(sample)
        audit = GuardAudit(np, m.jnt_range[self.jids].copy())
        if a.video:
            import imageio.v2 as imageio
            self.writer = imageio.get_writer(str(a.outdir / f"{label}.mp4"), fps=a.video_fps, codec="libx264", quality=7)
        target = self.policy.q0.copy()
        start_x = float(d.qpos[0])
        previous_x = start_x
        climb_entry_t = climb_exit_t = None
        max_x = start_x
        success_time = collision_time = fall_time = side_time = 0.
        roll_sq = 0.
        frames = 0
        next_video = 0.
        next_viewer = 0.
        progress = deque()
        recent_platform_contacts = deque()
        stalled = False
        status = "timeout"
        wall_start = time.monotonic()
        last_success_predicate = {}
        numeric_failure = None
        try:
            for step in range(math.ceil(a.duration/a.physics_dt)):
                t = (step+1)*a.physics_dt
                try:
                    if step % self.decimation == 0:
                        target = self.policy.act(self.observation())
                    self.physics(target, audit, t)
                except FloatingPointError as exc:
                    status = "numeric_failure"
                    numeric_failure = {"error": str(exc), "attempted_physics_time_s": t,
                                       "invalid_sample_excluded_from_audit": True,
                                       "valid_physics_samples_retained_per_joint": audit.n}
                    break
                position = d.qpos[:3]
                if a.steps:
                    current_x = float(position[0])
                    if climb_entry_t is None and previous_x < 0 <= current_x:
                        climb_entry_t = t-a.physics_dt + a.physics_dt*(-previous_x)/(current_x-previous_x)
                    platform_boundary = a.steps*a.tread
                    if climb_entry_t is not None and climb_exit_t is None and previous_x < platform_boundary <= current_x:
                        climb_exit_t = t-a.physics_dt + a.physics_dt*(platform_boundary-previous_x)/(current_x-previous_x)
                    previous_x = current_x
                foot = d.geom_xpos[self.feet]
                rotation = d.xmat[self.base].reshape(3, 3)
                tilt = math.acos(float(np.clip(rotation[2, 2], -1, 1)))
                roll = math.atan2(rotation[2, 1], rotation[2, 2])
                roll_sq += roll*roll
                max_x = max(max_x, float(position[0]))
                touching, platform_feet, trunk, side_floor = self.contacts()
                recent_platform_contacts.append((t, platform_feet.copy()))
                while recent_platform_contacts and recent_platform_contacts[0][0] < t-.10:
                    recent_platform_contacts.popleft()
                recent_supported_feet = set().union(*(v for _, v in recent_platform_contacts))
                platform_start = a.steps*a.tread if a.steps else 2.
                platform_end = platform_start+a.platform
                top_z = a.steps*a.height
                feet_inside = bool(np.all(foot[:, 0]-self.radii >= platform_start+.003) and
                                   np.all(foot[:, 0]+self.radii <= platform_end-.003) and
                                   np.all(np.abs(foot[:, 1])+self.radii <= a.width/2-.003))
                feet_height = bool(np.all(foot[:, 2]-self.radii >= top_z-.025) and np.all(foot[:, 2]-self.radii <= top_z+.20))
                supported = len(recent_supported_feet) >= 2
                base_ok = bool(platform_start <= position[0] <= platform_end and .18 <= position[2]-top_z <= .60 and tilt < math.pi/4)
                successful = feet_inside and feet_height and supported and base_ok
                success_time = success_time + a.physics_dt if successful else 0.
                collision_time = collision_time + a.physics_dt if trunk else 0.
                fall_time = fall_time + a.physics_dt if tilt > math.pi/3 or position[2] < .12 else 0.
                side = bool(abs(position[1]) > a.width/2 or side_floor)
                side_time = side_time + a.physics_dt if side else 0.
                last_success_predicate = {"all_four_feet_inside_platform": feet_inside, "feet_near_platform_height": feet_height,
                                          "platform_contacting_feet": len(platform_feet), "base_height_pose_ok": base_ok,
                                          "different_support_feet_in_last_100ms": len(recent_supported_feet),
                                          "continuous_s": success_time}
                progress.append((t, float(position[0])))
                while progress and progress[0][0] < t-5:
                    progress.popleft()
                if t > 5 and progress and position[0]-progress[0][1] < .05:
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
                        delay = wall_start+t-time.monotonic()
                        if delay > 0:
                            time.sleep(delay)
                # All guard samples were counted before adjudicating failure.
                if collision_time >= .05:
                    status = "trunk_contact"
                    break
                if fall_time >= .1:
                    status = "fall"
                    break
                if side_time >= .1:
                    status = "side_exit"
                    break
                if success_time >= .2 - 1e-9:
                    status = "success"
                    break
            report = {"label": label, "status": status, "task_success": status == "success",
                      "elapsed_s": float(d.time), "initial_state": start,
                      "final_base_position_m": d.qpos[:3].tolist(), "final_foot_centers_m": d.geom_xpos[self.feet].tolist(),
                      "forward_displacement_m": float(d.qpos[0]-start_x), "max_forward_displacement_m": max_x-start_x,
                      "mean_world_forward_speed_m_s": float((d.qpos[0]-start_x)/max(d.time, 1e-9)),
                      "climb_horizontal_speed_m_s": (a.steps*a.tread/(climb_exit_t-climb_entry_t)
                                                       if climb_exit_t is not None and climb_exit_t > climb_entry_t else None),
                      "climb_speed_window": {"definition": "First forward base crossing from x=0 to x=steps*tread, interpolated between physics endpoints; null until both are reached",
                                             "entry_time_s": climb_entry_t, "exit_time_s": climb_exit_t,
                                             "horizontal_distance_m": a.steps*a.tread if a.steps else None},
                      "roll_rms_rad": math.sqrt(roll_sq/max(audit.n, 1)), "stalled_5s_window_seen": stalled,
                      "last_success_predicate": last_success_predicate, "guard_audit": audit.report(),
                      "command_m_s": a.vx, "video_frames": frames, "real_robot_driven": False}
            report["torque_limit_mode"] = a.torque_limit_mode
            report["numeric_failure"] = numeric_failure
            if numeric_failure:
                report["guard_audit"]["local_guard_audit_passed"] = False
                report["guard_audit"]["incomplete_due_to_numeric_failure"] = True
            write_json(a.outdir / f"{label}.json", report)
            print(json.dumps({k: report[k] for k in ("label", "status", "elapsed_s", "forward_displacement_m")}), flush=True)
            return report
        finally:
            if self.writer is not None:
                self.writer.close()
                self.writer = None

    def close(self):
        if self.writer is not None:
            self.writer.close()
        if self.renderer is not None:
            self.renderer.close()
        if self.viewer is not None:
            self.viewer.close()


def aggregate(reports):
    total = sum(r["guard_audit"]["physics_samples_per_joint"] for r in reports)
    metrics = {}
    for key in ("velocity", "nominal_velocity", "actual_position", "target_position", "tracking_error", "requested_torque", "applied_torque", "model_actual_position", "model_target_position"):
        excess = sum(r["guard_audit"][key]["excess_samples"] for r in reports)
        metrics[key] = {"excess_samples": excess, "excess_fraction": excess/(12*total) if total else None,
                        "peak": max((r["guard_audit"][key]["peak"] for r in reports), default=None)}
    return {"episodes": len(reports), "successes": sum(r["task_success"] for r in reports),
            "physics_samples_per_joint": total, "weighted_guard_metrics": metrics,
            "aggregation": "Excess counts divided by pooled physics-joint samples; peaks are maxima, never averaged",
            "status_counts": {s: sum(r["status"] == s for r in reports) for s in sorted({r["status"] for r in reports})}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", type=Path, default=POLICY_DIR / "exported/policy.onnx")
    p.add_argument("--config", type=Path, default=POLICY_DIR / "params/deploy.yaml")
    p.add_argument("--robot-xml", type=Path, default=UPSTREAM / "RoboGauge/resources/robots/go2/go2.xml")
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=8, help="Randomized episodes PER seed, in addition to deterministic baseline")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    p.add_argument("--no-baseline", action="store_true")
    p.add_argument("--vx", type=float, default=.6)
    p.add_argument("--duration", type=float, default=25.)
    p.add_argument("--steps", type=int, default=6, help="Stair risers; zero selects a flat sanity course")
    p.add_argument("--height", type=float, default=.15)
    p.add_argument("--tread", type=float, default=.30)
    p.add_argument("--width", type=float, default=.90)
    p.add_argument("--platform", type=float, default=1.5)
    p.add_argument("--start-distance", type=float, default=.50)
    p.add_argument("--settle-seconds", type=float, default=2.)
    p.add_argument("--physics-dt", type=float, default=.002)
    p.add_argument("--torque-limit-mode", choices=("author", "physical"), default="author",
                   help="author preserves original unlimited motors; physical explicitly clips requested motor torque to 23.7/23.7/45.43 Nm")
    view = p.add_mutually_exclusive_group()
    view.add_argument("--headless", action="store_true")
    view.add_argument("--viewer", action="store_true")
    p.add_argument("--loop", action="store_true", help="Repeat episode schedule in viewer until window closes")
    p.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--video", action="store_true")
    p.add_argument("--video-fps", type=int, default=30)
    p.add_argument("--video-width", type=int, default=640)
    p.add_argument("--video-height", type=int, default=480)
    args = p.parse_args()
    if args.outdir.exists():
        p.error("Output directory already exists; choose a new run directory")
    if args.steps < 0 or args.episodes < 0 or (args.no_baseline and not args.episodes):
        p.error("Invalid empty/negative episode schedule")
    if args.loop and not args.viewer:
        p.error("--loop requires --viewer")
    for key in ("vx", "duration", "height", "tread", "width", "platform", "start_distance", "settle_seconds", "physics_dt"):
        value = getattr(args, key)
        if not math.isfinite(value) or value <= 0:
            p.error(key+" must be finite and positive")
    if min(args.video_fps, args.video_width, args.video_height) <= 0:
        p.error("Video dimensions and fps must be positive")
    # Set before importing MuJoCo. GUI uses GLFW; offscreen video uses EGL.
    if not args.viewer:
        os.environ.setdefault("MUJOCO_GL", "egl")
    args.outdir.mkdir(parents=True)
    experiment = None
    reports = []
    try:
        experiment = Experiment(args)
        np = experiment.np
        samples = []
        if not args.no_baseline:
            samples.append({"kind": "deterministic", "seed": None, "episode": 0, "y_m": 0., "yaw_rad": 0.})
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            for i in range(args.episodes):
                samples.append({"kind": "perturbed", "seed": seed, "episode": i,
                                "y_m": float(rng.uniform(-.02, .02)), "yaw_rad": float(rng.uniform(-math.pi/90, math.pi/90))})
        (args.outdir / "scene.xml").write_text(experiment.xml)
        manifest = {"upstream": "RoboGauge f116b00a", "model_sha256": digest(args.policy),
                    "deploy_yaml_sha256": digest(args.config), "robot_xml_sha256": digest(args.robot_xml),
                    "runner_sha256": digest(__file__), "source_files": {"policy": str(args.policy.resolve()), "config": str(args.config.resolve()), "robot_xml": str(args.robot_xml.resolve())},
                    "geometry": {k: getattr(args, k) for k in ("steps", "height", "tread", "width", "platform", "start_distance")},
                    "geometry_assumptions": "Six risers and 1.5 m top platform assumed unless overridden; width/height/tread/start gap are explicit inputs",
                    "success_definition": "All four foot spheres inside top platform, feet within [-.025,+.20] m of surface, at least two different feet with upward platform contacts >1 N in the last .10 s, base clearance .18-.60 m and tilt <45 deg continuously .20 s",
                    "flat_success_definition": "Same supported test on floor beyond x=2 m when steps=0",
                    "physics_dt": args.physics_dt, "control_dt": experiment.control_dt,
                    "kp": experiment.kp.tolist(), "kd": experiment.kd.tolist(), "command_m_s": args.vx,
                    "observation_contract": "225 floats, term-major 5 oldest-to-newest frames; clip raw term then scale; initial current observations repeated; last raw network action",
                    "action_contract": "12 FL/FR/RL/RR position offsets; upstream scale/offset then target clip; no new action or joint-position clipping",
                    "torque_limit_mode": args.torque_limit_mode,
                    "author_actuator_limits": experiment.author_actuator_limits,
                    "physical_mode_effort_limit_nm": experiment.physical_effort.tolist(),
                    "torque_note": "Author mode retains original unlimited motors. Physical mode is a separate requested-PD-torque clipping comparison, not the original author's controller and not proof of real hardware equivalence.",
                    "model_joint_ranges": {JOINTS[j]: experiment.model.jnt_range[experiment.jids[j]].tolist() for j in range(12)},
                    "randomization": "Only y +/- .02 m and yaw +/- 2 deg; no noise, mass/friction randomization or action delay in this initial benchmark",
                    "local_guard_note": "13 rad/s/.9 rad and local application angle/effort limits are independent diagnostics, not claimed RoboGauge built-in protections; no early termination for these violations",
                    "samples": samples, "real_robot_driven": False, "network_interfaces_used": False}
        write_json(args.outdir / "manifest.json", manifest)
        cycle = 0
        while True:
            for i, sample in enumerate(samples):
                report = experiment.episode(sample, f"episode_{cycle:03d}_{i:03d}")
                reports.append(report)
                if report["status"] == "viewer_closed":
                    break
            if not args.loop or reports[-1]["status"] == "viewer_closed":
                break
            cycle += 1
        summary = aggregate(reports)
        summary["episode_files"] = [r["label"]+".json" for r in reports]
        summary["limits"] = "Finite simulated outcomes do not establish real-world safety or hardware compatibility"
        write_json(args.outdir / "summary.json", summary)
        print("ROBOGAUGE_OFFLINE_COMPLETE " + json.dumps(summary), flush=True)
    except BaseException as exc:
        write_json(args.outdir / "failure.json", {"error": type(exc).__name__+": "+str(exc), "completed_episodes": len(reports), "real_robot_driven": False})
        raise
    finally:
        if experiment is not None:
            experiment.close()


if __name__ == "__main__":
    main()
