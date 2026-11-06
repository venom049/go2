"""Runtime checks executed with real PhysX tensors before learning."""
def validate_runtime(env):
    import torch
    from go2_him.lab_config import JOINT_NAMES, FOOT_NAMES
    robot = env.unwrapped.scene["robot"]
    if robot.num_joints != 12: raise RuntimeError("Expected 12 Go2 joints")
    if set(robot.joint_names) != set(JOINT_NAMES): raise RuntimeError("Unexpected Go2 joint names")
    for name in FOOT_NAMES:
        if name not in robot.body_names: raise RuntimeError("Missing independent foot link: " + name)
    expected_ids, ordered_names = robot.find_joints(JOINT_NAMES, preserve_order=True)
    action_term = env.unwrapped.action_manager.get_term("joints")
    # Isaac Lab 2.3.2 stores the resolved order on the action term.
    action_names = list(action_term._joint_names)
    if action_names != JOINT_NAMES or list(action_term._joint_ids) != list(expected_ids):
        raise RuntimeError("Action joint ordering differs from observation convention")
    if env.num_obs != 270 or env.num_actions != 12 or env.num_privileged_obs != 235:
        raise RuntimeError(f"Unexpected HIM interface {env.num_obs}, {env.num_actions}, {env.num_privileged_obs}")
    initial = robot.data.root_pos_w - env.unwrapped.scene.env_origins
    if not torch.all(initial[:,2] > 0.2): raise RuntimeError("Robot spawned below terrain")
    for _ in range(16):
        result = env.step(torch.zeros(env.num_envs, 12, device=env.device))
        for tensor in result[:3]:
            if not torch.isfinite(tensor).all(): raise RuntimeError("NaN/Inf in simulated observation or reward")
    report = {"num_envs":env.num_envs, "actor_obs":env.num_obs, "critic_obs":env.num_privileged_obs,
        "actions":env.num_actions, "joint_names":list(robot.joint_names), "action_joint_names":action_names,
        "action_joint_ids":list(expected_ids), "body_names":list(robot.body_names),
        "root_relative_z_mean":float((robot.data.root_pos_w - env.unwrapped.scene.env_origins)[:,2].mean()),
        "finite_zero_action_steps":16}
    env.reset()
    return report
