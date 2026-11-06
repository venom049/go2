"""CPU integration checks of the real HIM stepping order and limit hooks.

Isaac's physics backend/managers are fakes; the vendored environment step,
monitored actuator, constraint windows, termination terms and runner are real.
This does not certify a PhysX run or robot deployment.
"""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
NAMES = [f"{leg}_{kind}_joint" for leg in ("FL", "FR", "RL", "RR")
         for kind in ("hip", "thigh", "calf")]


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_integration():
    modules = {name: ModuleType(name) for name in (
        "isaaclab", "isaaclab.actuators", "isaaclab.utils", "isaaclab.envs",
        "isaaclab.envs.mdp", "isaaclab.managers", "himloco_lab", "himloco_lab.envs",
        "himloco_lab.rsl_rl", "himloco_lab.rsl_rl.runners",
        "himloco_lab.rsl_rl.runners.him_on_policy_runner")}

    class PhysicsBase:
        def _reset_idx(self, ids):
            self.events.append(("reset", self.sim_count))
            self.scene["robot"].data.joint_pos[ids] = self.rest_pose[:, self.robot_order][ids]
            self.scene["robot"].data.joint_vel[ids] = 0
            self.scene["robot"].data.root_pos_w[ids] = 0
            self.episode_length_buf[ids] = 0

    class IdealPD:
        def __init__(self, cfg):
            self.joint_names = NAMES
            self.stiffness = torch.ones((2, 12))

        def compute(self, action, joint_pos, joint_vel):
            self.events.append(("pd_compute", self.env.sim_count))
            self.computed_effort = torch.zeros_like(joint_pos)
            self.applied_effort = self.computed_effort.clone()
            return action

    class ParentRunner:
        def log(self, locs, width=80, pad=35):
            self.parent_ep_infos = locs["ep_infos"]
            for key in {key for row in locs["ep_infos"] for key in row}:
                values = [torch.as_tensor(row[key]).reshape(-1)
                          for row in locs["ep_infos"] if key in row]
                self.writer.add_scalar(key, torch.cat(values).mean(), locs["it"])

    class Entity:
        def __init__(self, name):
            self.name = name
            self.joint_ids = self.body_ids = slice(None)

    modules["isaaclab.envs"].ManagerBasedRLEnv = PhysicsBase
    modules["isaaclab.actuators"].IdealPDActuator = IdealPD
    modules["isaaclab.actuators"].IdealPDActuatorCfg = type("IdealPDCfg", (), {})
    modules["isaaclab.utils"].configclass = lambda cls: cls
    modules["isaaclab.managers"].SceneEntityCfg = Entity
    modules["himloco_lab.rsl_rl.runners.him_on_policy_runner"].HIMOnPolicyRunner = ParentRunner
    with patch.dict(sys.modules, modules):
        him = _load("_joint_test_him", "upstream/himloco_lab/source/himloco_lab/himloco_lab/envs/him_manager_based_rl_env.py")
        modules["himloco_lab.envs"].HimlocoManagerBasedRLEnv = him.HimlocoManagerBasedRLEnv
        monitored = _load("_joint_test_lab", "go2_him/lab_torque.py")
        mdp = _load("_joint_test_mdp", "go2_him/mdp.py")
    return monitored, mdp


class _Scene(dict):
    pass


@unittest.skipIf(torch is None, "PyTorch required; use .venv-deploy/bin/python")
class JointIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lab, cls.mdp = _load_integration()

    def make_env(self, enabled=True, spike=13.313):
        from go2_him.joint_constraints import joint_position_limits

        env = self.lab.TorqueHimlocoEnv.__new__(self.lab.TorqueHimlocoEnv)
        env.events, env.sim_count = [], 0
        env.device, env.num_envs = "cpu", 2
        env.physics_dt, env.step_dt = .005, .02
        env._sim_step_counter = env.common_step_counter = 0
        env.episode_length_buf = torch.zeros(2, dtype=torch.long)
        settings = {"terrain": {"num_steps": 8, "tread": .32},
                    "torque": {"startup_seconds": 2.}}
        if enabled:
            settings["joint_constraints"] = {
                "enabled": True, "terminate_on_violation": True,
                "startup_seconds": 2., "startup_multiplier": 2.}
        env.cfg = SimpleNamespace(decimation=4, sim=SimpleNamespace(render_interval=4),
                                  num_rerenders_on_reset=0, go2_settings=settings)
        actuator_cfg = SimpleNamespace(
            torque_soft_ratio=.8, joint_constraints_enabled=enabled,
            joint_velocity_limit=13., joint_velocity_soft_ratio=.8,
            joint_position_soft_ratio=.9, joint_tracking_error_limit=.9,
            joint_tracking_soft_ratio=.8)
        actuator = self.lab.MonitoredPDActuator(actuator_cfg)
        actuator.events, actuator.env = env.events, env
        env.rest_pose = joint_position_limits(NAMES).mean(-1).repeat(2, 1)
        # Deliberately reverse articulation order to exercise name-based mapping.
        robot_names = NAMES[::-1]
        env.robot_order = [NAMES.index(name) for name in robot_names]
        actuator_ids = [robot_names.index(name) for name in NAMES]
        data = SimpleNamespace(joint_pos=env.rest_pose[:, env.robot_order].clone(),
                               joint_vel=torch.zeros(2, 12),
                               root_pos_w=torch.tensor([[4.6, 0., 0.], [4.6, 0., 0.]]),
                               projected_gravity_b=torch.tensor([[0., 0., -1.]]).repeat(2, 1))
        env.scene = _Scene(robot=SimpleNamespace(data=data, joint_names=robot_names,
                                                actuators={"legs": actuator}))
        env.scene.env_origins = torch.zeros(2, 3)

        def write_data():
            actuator.compute(SimpleNamespace(joint_positions=env.rest_pose.clone()),
                             data.joint_pos[:, actuator_ids], data.joint_vel[:, actuator_ids])

        def physics_step(render=False):
            env.sim_count += 1
            env.events.append(("physics", env.sim_count))

        def update(dt):
            env.events.append(("scene_update", env.sim_count))
            data.joint_vel.zero_()
            if env.sim_count == 4:
                data.joint_vel[0, robot_names.index("RL_calf_joint")] = spike

        env.scene.write_data_to_sim, env.scene.update = write_data, update
        env.sim = SimpleNamespace(has_gui=lambda: False, has_rtx_sensors=lambda: False,
                                  step=physics_step, forward=lambda: None)
        env.action_manager = SimpleNamespace(prev_action=torch.zeros(2, 12),
                                             process_action=lambda action: None,
                                             apply_action=lambda: None)
        env.recorder_manager = SimpleNamespace(
            active_terms=[], record_pre_step=lambda: None,
            record_post_physics_decimation_step=lambda: None,
            record_pre_reset=lambda ids: None, record_post_reset=lambda ids: None)
        env.observation_manager = SimpleNamespace(
            compute=lambda **kwargs: {"dq": data.joint_vel.clone()})
        env.command_manager = SimpleNamespace(compute=lambda **kwargs: None)
        env.event_manager = SimpleNamespace(available_modes=[])
        env.extras = {}
        env.termination_manager = SimpleNamespace(
            terminated=torch.zeros(2, dtype=torch.bool),
            time_outs=torch.zeros(2, dtype=torch.bool),
            get_term_cfg=lambda name: SimpleNamespace(
                func=lambda current: torch.zeros(2, dtype=torch.bool), params={}))

        def terminations():
            violated = (self.mdp.joint_constraint_violation(env) if enabled
                        else torch.zeros(2, dtype=torch.bool))
            env.success_before_reset = self.mdp.success(env).clone()
            env.termination_manager.terminated = violated | env.success_before_reset
            return env.termination_manager.terminated

        env.termination_manager.compute = terminations
        env.reward_manager = SimpleNamespace(compute=lambda dt: self.mdp.success_reward(env) * dt)
        if enabled:
            window = actuator.joint_window
            record_command, record_state = window.record_command, window.record_state

            def command(target, position):
                env.events.append(("record_command", env.sim_count))
                return record_command(target, position)

            def state(position, velocity):
                env.events.append(("record_state", env.sim_count))
                return record_state(position, velocity)

            window.record_command, window.record_state = command, state
        return env

    def test_terminal_last_substep_is_observed_before_reset_and_disqualifies_goal(self):
        env = self.make_env()
        result = env.step(torch.zeros(2, 12))
        window = env.scene["robot"].actuators["legs"].joint_window
        self.assertEqual((window.command_samples, window.state_samples), (4, 4))
        for substep in range(4):
            command = env.events.index(("record_command", substep))
            compute = env.events.index(("pd_compute", substep))
            physics = env.events.index(("physics", substep + 1))
            update = env.events.index(("scene_update", substep + 1))
            state = env.events.index(("record_state", substep + 1))
            self.assertLess(command, compute)
            self.assertLess(compute, physics)
            self.assertLess(physics, update)
            self.assertLess(update, state)
        self.assertLess(env.events.index(("record_state", 4)), env.events.index(("reset", 4)))
        self.assertIn(("record_command", 4), env.events)  # Extra reset write was ignored.
        self.assertEqual(window.hard_violation().tolist(), [True, False])
        self.assertAlmostEqual(float(window.peaks["velocity"][0, NAMES.index("RL_calf_joint")]), 13.313, places=5)
        self.assertEqual(env.success_before_reset.tolist(), [False, True])
        torch.testing.assert_close(result[2], torch.tensor([0., 1.]))
        self.assertAlmostEqual(float(result[1]["dq"].abs().max()), 13.313, places=5)
        self.assertEqual(float(result[0]["dq"].abs().max()), 0.)
        self.assertAlmostEqual(float(env.extras["log"]["JointLimits/velocity_peak_ratio"]), 13.313 / 13., places=6)
        self.assertGreater(float(self.mdp.joint_constraint_penalty(env, "velocity_excess")[0]), 0.)

    def test_disabled_old_settings_preserve_goal_reward_without_new_termination(self):
        env = self.make_env(enabled=False)
        result = env.step(torch.zeros(2, 12))
        self.assertIsNone(env.scene["robot"].actuators["legs"].joint_window)
        self.assertEqual(env.success_before_reset.tolist(), [True, True])
        torch.testing.assert_close(result[2], torch.ones(2))
        self.assertFalse(any(key.startswith("JointLimits/") for key in env.extras["log"]))

    def test_safe_enabled_interval_has_no_penalty_or_false_failure(self):
        env = self.make_env(spike=10.)
        result = env.step(torch.zeros(2, 12))
        window = env.scene["robot"].actuators["legs"].joint_window
        self.assertFalse(bool(window.hard_violation().any()))
        self.assertEqual(env.success_before_reset.tolist(), [True, True])
        torch.testing.assert_close(result[2], torch.ones(2))
        for kind in window.PENALTIES:
            torch.testing.assert_close(self.mdp.joint_constraint_penalty(env, kind), torch.zeros(2))

    def test_evaluation_can_observe_violations_without_early_reset_hiding_them(self):
        env = self.make_env()
        env.cfg.go2_settings["joint_constraints"]["terminate_on_violation"] = False
        env.scene["robot"].data.root_pos_w[:, 0] = 0.
        result = env.step(torch.zeros(2, 12))
        window = env.scene["robot"].actuators["legs"].joint_window
        self.assertFalse(bool(result[3].any()))
        self.assertFalse(any(event == "reset" for event, _ in env.events))
        self.assertEqual(window.hard_violation().tolist(), [True, False])
        self.assertAlmostEqual(float(result[0]["dq"].abs().max()), 13.313, places=5)
        self.assertGreater(float(self.mdp.joint_constraint_penalty(env, "velocity_excess")[0]), 0.)

    def test_runner_preserves_brief_joint_spike_using_max_not_parent_mean(self):
        runner = self.lab.TorqueHIMRunner.__new__(self.lab.TorqueHIMRunner)
        runner.device = "cpu"
        logged = {}
        runner.writer = SimpleNamespace(add_scalar=lambda key, value, iteration: logged.update({key: float(value)}))
        rows = [
            {"JointLimits/velocity_peak_ratio": torch.tensor(.2), "JointLimits/velocity_excess_fraction": torch.tensor(.0)},
            {"JointLimits/velocity_peak_ratio": torch.tensor(1.024), "JointLimits/velocity_excess_fraction": torch.tensor(.2)},
        ]
        runner.log({"ep_infos": rows, "it": 17})
        self.assertAlmostEqual(logged["JointLimits/velocity_peak_ratio"], 1.024, places=6)
        self.assertAlmostEqual(logged["JointLimits/velocity_excess_fraction"], .1, places=6)
        self.assertTrue(all("JointLimits/velocity_peak_ratio" not in row for row in runner.parent_ep_infos))
        self.assertTrue(all("JointLimits/velocity_peak_ratio" in row for row in rows))


if __name__ == "__main__":
    unittest.main()
