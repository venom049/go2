"""Isaac Lab 2.3 / Isaac Sim 5.1 configuration. Import after AppLauncher."""
import math
from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp as lab_mdp
from isaaclab.managers import (ObservationGroupCfg, ObservationTermCfg as Obs,
    RewardTermCfg as Reward, TerminationTermCfg as Done, EventTermCfg as Event,
    CurriculumTermCfg as Curriculum, SceneEntityCfg)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg, SubTerrainBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Noise
from go2_him.lab_terrain import ascending_stairs
from go2_him import mdp
from go2_him.settings import ROOT

FOOT_NAMES = [f"{leg}_foot" for leg in ("FL", "FR", "RL", "RR")]
JOINT_NAMES = [f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR")
               for joint in ("hip", "thigh", "calf")]


@configclass
class StairsTerrainCfg(SubTerrainBaseCfg):
    function = ascending_stairs
    step_height_range = (0.0, 0.18)
    step_width = 0.32
    num_steps = 8
    approach_length = 2.0
    landing_length = 2.0
    spawn_x = 0.8
    goal_offset = 0.7
    boundary_margin = 0.35


@configclass
class SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="generator", collision_group=-1,
        terrain_generator=TerrainGeneratorCfg(
            size=(8.0, 3.0), border_width=2.0, num_rows=10, num_cols=4,
            curriculum=True, use_cache=False, sub_terrains={"stairs": StairsTerrainCfg()}),
        max_init_terrain_level=1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0,
            friction_combine_mode="multiply", restitution_combine_mode="multiply"),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.32, 0.38, 0.43)),
    )
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(ROOT / "assets/go2/urdf/go2.urdf"),
            usd_dir=str(ROOT / "assets/go2/usd"), usd_file_name="go2.usd",
            fix_base=False, merge_fixed_joints=False, replace_cylinders_with_capsules=False,
            activate_contact_sensors=True,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False,
                linear_damping=0.0, angular_damping=0.0, max_depenetration_velocity=1.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False,
                solver_position_iteration_count=4, solver_velocity_iteration_count=1)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, 0.40), joint_pos={
            "[FR]L_hip_joint": 0.1, "[FR]R_hip_joint": -0.1,
            "F[LR]_thigh_joint": 0.8, "R[LR]_thigh_joint": 1.0, ".*_calf_joint": -1.5},
            joint_vel={".*": 0.0}),
        soft_joint_pos_limit_factor=0.95,
        actuators={"legs": IdealPDActuatorCfg(joint_names_expr=[".*_joint"], stiffness=40.0, damping=1.0,
            armature=0.01, effort_limit={".*_hip_joint": 23.7, ".*_thigh_joint": 23.7, ".*_calf_joint": 45.43},
            effort_limit_sim={".*_hip_joint": 23.7, ".*_thigh_joint": 23.7, ".*_calf_joint": 45.43},
            velocity_limit=30.0, velocity_limit_sim=30.0)},
    )
    height_scanner = RayCasterCfg(prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0, 0, 20)), ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        mesh_prim_paths=["/World/ground"], update_period=0.02)
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3,
        track_air_time=True, update_period=0.005)
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=1000.0))


def ordered_joints():
    return SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)


@configclass
class ObservationsCfg:
    @configclass
    class Policy(ObservationGroupCfg):
        # HIM estimator relies on this exact order and the critic's identical prefix.
        commands = Obs(func=lab_mdp.generated_commands, params={"command_name": "base_velocity"})
        angular_velocity = Obs(func=lab_mdp.base_ang_vel, scale=0.25, noise=Noise(n_min=-0.2, n_max=0.2))
        gravity = Obs(func=lab_mdp.projected_gravity, noise=Noise(n_min=-0.05, n_max=0.05))
        joint_pos = Obs(func=lab_mdp.joint_pos_rel, params={"asset_cfg": ordered_joints()}, noise=Noise(n_min=-0.01, n_max=0.01))
        joint_vel = Obs(func=lab_mdp.joint_vel_rel, params={"asset_cfg": ordered_joints()}, scale=0.05, noise=Noise(n_min=-1.5, n_max=1.5))
        previous_action = Obs(func=lab_mdp.last_action)
        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = True

    @configclass
    class Critic(Policy):
        linear_velocity = Obs(func=lab_mdp.base_lin_vel, scale=2.0)
        terrain_heights = Obs(func=lab_mdp.height_scan, params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0), scale=5.0)
        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False
    policy = Policy()
    critic = Critic()


@configclass
class ActionsCfg:
    joints = lab_mdp.JointPositionActionCfg(asset_name="robot", joint_names=JOINT_NAMES,
        preserve_order=True, scale=0.25, use_default_offset=True, clip=None)


@configclass
class CommandsCfg:
    base_velocity = lab_mdp.UniformVelocityCommandCfg(asset_name="robot", resampling_time_range=(5.0, 10.0),
        rel_standing_envs=0.0, rel_heading_envs=1.0, heading_command=True, heading_control_stiffness=1.0,
        debug_vis=False, ranges=lab_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.3, 0.6), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.6, 0.6), heading=(0.0, 0.0)))


@configclass
class RewardsCfg:
    tracking_lin_vel = Reward(func=lab_mdp.track_lin_vel_xy_exp, weight=1.5, params={"command_name":"base_velocity", "std":0.5})
    tracking_ang_vel = Reward(func=lab_mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name":"base_velocity", "std":0.5})
    forward_progress = Reward(func=mdp.forward_progress, weight=0.3)
    success = Reward(func=mdp.success_reward, weight=5.0)
    failure = Reward(func=mdp.failure_reward, weight=-2.0)
    lin_vel_z = Reward(func=lab_mdp.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy = Reward(func=lab_mdp.ang_vel_xy_l2, weight=-0.05)
    orientation = Reward(func=lab_mdp.flat_orientation_l2, weight=-0.15)
    base_height = Reward(func=mdp.base_height, weight=-1.0, params={"target_height":0.30})
    joint_acc = Reward(func=lab_mdp.joint_acc_l2, weight=-2.5e-7)
    joint_power = Reward(func=mdp.joint_power, weight=-2e-5)
    action_rate = Reward(func=lab_mdp.action_rate_l2, weight=-0.01)
    action_smoothness = Reward(func=mdp.action_smoothness, weight=0.0)
    roll_orientation = Reward(func=mdp.roll_orientation, weight=0.0)
    roll_rate = Reward(func=mdp.roll_rate, weight=0.0)
    lateral_velocity = Reward(func=mdp.lateral_velocity, weight=0.0)
    joint_deviation = Reward(func=lab_mdp.joint_deviation_l1, weight=0.0,
        params={"asset_cfg": ordered_joints()})
    hip_deviation = Reward(func=lab_mdp.joint_deviation_l1, weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])})
    hip_abduction = Reward(func=mdp.hip_abduction, weight=0.0, params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=[f"{leg}_hip_joint" for leg in ("FL", "FR", "RL", "RR")], preserve_order=True),
        "max_angle": 0.30})
    feet_lateral = Reward(func=mdp.feet_lateral, weight=0.0, params={
        "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_NAMES, preserve_order=True),
        "min_lateral": 0.10, "max_lateral": 0.22, "inner_weight": 1.0})
    joint_vel = Reward(func=lab_mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": ordered_joints()})
    joint_limits = Reward(func=lab_mdp.joint_pos_limits, weight=-1.0)
    undesired_contacts = Reward(func=lab_mdp.undesired_contacts, weight=-0.5,
        params={"sensor_cfg":SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"]), "threshold":1.0})
    foot_slip = Reward(func=mdp.foot_slip, weight=-0.1, params={
        "asset_cfg":SceneEntityCfg("robot", body_names=FOOT_NAMES, preserve_order=True),
        "sensor_cfg":SceneEntityCfg("contact_forces", body_names=FOOT_NAMES, preserve_order=True)})
    lateral_drift = Reward(func=mdp.lateral_drift, weight=-0.3)
    foot_clearance = Reward(func=mdp.foot_clearance, weight=-0.1, params={
        "asset_cfg":SceneEntityCfg("robot", body_names=FOOT_NAMES, preserve_order=True),
        "sensor_cfg":SceneEntityCfg("contact_forces", body_names=FOOT_NAMES, preserve_order=True), "target_height":0.08})
    feet_air_time = Reward(func=mdp.feet_air_time, weight=0.0, params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_NAMES, preserve_order=True),
        "target_air_time": 0.12, "max_air_time": 0.40})
    feet_stumble = Reward(func=mdp.feet_stumble, weight=0.0, params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_NAMES, preserve_order=True)})


@configclass
class TerminationsCfg:
    time_out = Done(func=lab_mdp.time_out, time_out=True)
    base_contact = Done(func=lab_mdp.illegal_contact,
        params={"sensor_cfg":SceneEntityCfg("contact_forces", body_names="base_link"), "threshold":1.0})
    fall = Done(func=lab_mdp.bad_orientation, params={"limit_angle":1.1})
    out_of_bounds = Done(func=mdp.out_of_bounds)
    success = Done(func=mdp.success)


@configclass
class EventsCfg:
    material = Event(func=lab_mdp.randomize_rigid_body_material, mode="startup", params={
        "asset_cfg":SceneEntityCfg("robot", body_names=".*"), "static_friction_range":(0.6,1.2),
        "dynamic_friction_range":(0.6,1.2), "restitution_range":(0.0,0.0), "num_buckets":64})
    payload = Event(func=lab_mdp.randomize_rigid_body_mass, mode="startup", params={
        "asset_cfg":SceneEntityCfg("robot", body_names="base_link"), "mass_distribution_params":(-0.5,1.0), "operation":"add"})
    reset_base = Event(func=lab_mdp.reset_root_state_uniform, mode="reset", params={
        "pose_range":{"x":(-0.10,0.10), "y":(-0.10,0.10), "yaw":(-0.05,0.05)},
        "velocity_range":{}})
    reset_joints = Event(func=lab_mdp.reset_joints_by_scale, mode="reset", params={
        "position_range":(0.9,1.1), "velocity_range":(0.0,0.0)})


@configclass
class CurriculumCfg:
    stairs = Curriculum(func=mdp.curriculum)
    command_speed = Curriculum(func=mdp.command_speed_curriculum)


@configclass
class Go2StairsCfg(ManagerBasedRLEnvCfg):
    scene = SceneCfg(num_envs=1024, env_spacing=3.0)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    commands = CommandsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventsCfg()
    curriculum = CurriculumCfg()
    go2_settings: dict = {}

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 25.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_contact_count = 2**23
        self.viewer.eye = (5.0, 5.0, 3.0)
        self.viewer.lookat = (2.0, 0.0, 0.5)


def build_env_cfg(s, device="cuda:0", evaluation=False):
    cfg = Go2StairsCfg()
    cfg.go2_settings = s
    cfg.seed = s["seed"]
    cfg.sim.device = device
    cfg.scene.num_envs = s["num_envs"]
    cfg.episode_length_s = s["episode_length_s"]
    terrain = cfg.scene.terrain.terrain_generator
    terrain.seed = s["seed"]
    terrain.num_rows = s["terrain"]["rows"]
    terrain.num_cols = s["terrain"]["cols"]
    terrain.curriculum = True  # deterministic row-to-difficulty ordering even in evaluation
    stairs = terrain.sub_terrains["stairs"]
    stairs.step_height_range = (s["terrain"]["height_min"], s["terrain"]["height_max"])
    stairs.step_width = s["terrain"]["tread"]
    stairs.num_steps = s["terrain"]["num_steps"]
    cfg.scene.terrain.max_init_terrain_level = s["terrain"]["initial_level"]
    if evaluation:
        import copy
        cfg.go2_settings = copy.deepcopy(s)
        cfg.go2_settings["terrain"]["curriculum"] = False
        cfg.go2_settings["commands"]["speed_ramp_updates"] = 0
    if not evaluation and s["commands"]["speed_ramp_updates"] > 0:
        cfg.commands.base_velocity.ranges.lin_vel_x = (s["commands"]["start_forward_min"], s["commands"]["start_forward_max"])
    else:
        cfg.commands.base_velocity.ranges.lin_vel_x = (s["commands"]["forward_min"], s["commands"]["forward_max"])
    cfg.scene.robot.actuators["legs"].stiffness = s["control"]["stiffness"]
    cfg.scene.robot.actuators["legs"].damping = s["control"]["damping"]
    cfg.actions.joints.scale = s["control"]["action_scale"]
    # Raw normalized actions are clipped in the wrapper before scale and offset.
    cfg.actions.joints.clip = None
    for name, weight in s["rewards"].items():
        getattr(cfg.rewards, name).weight = weight
    # Evaluation must use the same failure detector for every checkpoint:
    # illegal_contact also reduces this history, so differing reward settings
    # must not change the observation window used to detect a base collision.
    if evaluation or s["rewards"]["feet_stumble"] != 0:
        cfg.scene.contact_forces.history_length = max(cfg.scene.contact_forces.history_length, cfg.decimation)
    cfg.rewards.base_height.params["target_height"] = s["targets"]["base_height"]
    cfg.rewards.foot_clearance.params["target_height"] = s["targets"]["foot_clearance"]
    cfg.rewards.feet_air_time.params["target_air_time"] = s["targets"]["air_time"]
    cfg.rewards.feet_air_time.params["max_air_time"] = s["targets"]["max_air_time"]
    cfg.rewards.hip_abduction.params["max_angle"] = s["targets"]["hip_abduction_max"]
    cfg.rewards.feet_lateral.params["min_lateral"] = s["targets"]["feet_lateral_min"]
    cfg.rewards.feet_lateral.params["max_lateral"] = s["targets"]["feet_lateral_max"]
    cfg.rewards.feet_lateral.params["inner_weight"] = s["targets"]["feet_lateral_inner_weight"]
    rand = s["randomization"]
    if rand["enabled"] and not evaluation:
        cfg.events.material.params["static_friction_range"] = (rand["friction_min"], rand["friction_max"])
        cfg.events.material.params["dynamic_friction_range"] = (rand["friction_min"], rand["friction_max"])
        cfg.events.payload.params["mass_distribution_params"] = (rand["payload_min"], rand["payload_max"])
    else:
        cfg.events.material = None
        cfg.events.payload = None
        cfg.observations.policy.enable_corruption = False
    return cfg


def agent_config(s, device="cuda:0"):
    t = s["training"]
    return dict(policy_class_name="HIMActorCritic", algorithm_class_name="HIMPPO", logger="tensorboard",
        num_steps_per_env=t["num_steps_per_env"], save_interval=t["save_interval"], seed=s["seed"], device=device,
        policy=dict(actor_hidden_dims=[512,256,128], critic_hidden_dims=[512,256,128], activation="elu",
            init_noise_std=t["init_noise_std"], min_noise_std=t["min_noise_std"],
            max_noise_std=t["max_noise_std"] or None),
        algorithm=dict(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
            entropy_coef=t["entropy_coef"], num_learning_epochs=t["num_learning_epochs"],
            num_mini_batches=t["num_mini_batches"], learning_rate=t["learning_rate"], schedule="adaptive",
            min_learning_rate=t["min_learning_rate"], max_learning_rate=t["max_learning_rate"],
            action_clip=s["control"]["action_clip"], action_mean_penalty_coef=t["action_mean_penalty_coef"],
            gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0))
