"""Tensor-only manager-term checks; no Isaac Sim process is needed.

Run with the training environment's Python to enable these tests. Only the Lab
configuration class is stubbed; calculations use real PyTorch tensors.
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


class _Entity:
    def __init__(self, name):
        self.name = name
        self.body_ids = slice(None)
        self.joint_ids = slice(None)


def _load_terms():
    modules = {name: ModuleType(name) for name in ("isaaclab", "isaaclab.envs", "isaaclab.envs.mdp", "isaaclab.managers")}
    modules["isaaclab.managers"].SceneEntityCfg = _Entity
    path = Path(__file__).resolve().parents[1] / "go2_him" / "mdp.py"
    spec = importlib.util.spec_from_file_location("_go2_mdp_tensor_tests", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class _Scene(dict):
    pass


@unittest.skipIf(torch is None, "PyTorch is required; use the training environment Python")
class ManagerTermTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mdp = _load_terms()

    def env(self, count=4):
        scene = _Scene()
        scene.env_origins = torch.zeros(count, 3)
        data = SimpleNamespace(
            root_pos_w=torch.zeros(count, 3),
            root_lin_vel_w=torch.zeros(count, 3),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(count, 1),
        )
        scene["robot"] = SimpleNamespace(data=data)
        command = torch.tensor([[0.5, 0.0, 0.0]]).repeat(count, 1)
        env = SimpleNamespace(
            scene=scene, device="cpu", step_dt=0.02,
            cfg=SimpleNamespace(go2_settings={"terrain": {"num_steps": 8, "tread": .32, "curriculum": True}}),
            command_manager=SimpleNamespace(get_command=lambda name: command),
            termination_manager=SimpleNamespace(terminated=torch.zeros(count, dtype=torch.bool), time_outs=torch.zeros(count, dtype=torch.bool)),
            episode_length_buf=torch.zeros(count, dtype=torch.long),
        )
        env.failure_flags = {name: torch.zeros(count, dtype=torch.bool) for name in ("base_contact", "fall")}
        def term_cfg(name):
            predicate = self.mdp.out_of_bounds if name == "out_of_bounds" else lambda current: current.failure_flags[name]
            return SimpleNamespace(func=predicate, params={})
        env.termination_manager.get_term_cfg = term_cfg
        return env

    def test_success_failure_are_exclusive_and_event_weight_is_dt_invariant(self):
        env = self.env()
        env.scene["robot"].data.root_pos_w[:, 0] = torch.tensor([4.6, 4.6, 1.0, 4.6])
        env.scene["robot"].data.root_pos_w[1, 1] = 1.2
        env.scene["robot"].data.projected_gravity_b[3, 2] = -.4
        env.termination_manager.terminated[:] = True
        torch.testing.assert_close(self.mdp.success(env), torch.tensor([True, False, False, False]))
        torch.testing.assert_close(self.mdp.failure(env), torch.tensor([False, True, True, True]))
        for dt in (.01, .02, .04):
            env.step_dt = dt
            torch.testing.assert_close(self.mdp.success_reward(env) * dt, torch.tensor([1., 0., 0., 0.]))

    def test_fatal_contact_or_bounds_on_goal_frame_cannot_count_as_success(self):
        env = self.env()
        env.scene["robot"].data.root_pos_w[:, 0] = torch.tensor([4.6, 4.6, 7.0, 4.6])
        env.failure_flags["base_contact"][0] = True
        env.failure_flags["fall"][1] = True
        env.termination_manager.terminated[:] = True
        torch.testing.assert_close(self.mdp.success(env), torch.tensor([False, False, False, True]))
        torch.testing.assert_close(self.mdp.success_reward(env) * env.step_dt, torch.tensor([0., 0., 0., 1.]))
        torch.testing.assert_close(self.mdp.failure_reward(env) * env.step_dt, torch.tensor([1., 1., 1., 0.]))

    def test_progress_uses_world_x_and_caps_at_command(self):
        env = self.env()
        env.scene["robot"].data.root_lin_vel_w[:, 0] = torch.tensor([-.2, .2, .8, .5])
        env.command_manager.get_command("base_velocity")[:, 0] = torch.tensor([.5, .1, .3, .6])
        torch.testing.assert_close(self.mdp.forward_progress(env), torch.tensor([0., .1, .3, .5]))

    def test_roll_regularization_allows_pitch_but_penalizes_sideways_tilt(self):
        env = self.env(3)
        env.scene["robot"].data.projected_gravity_b[:] = torch.tensor(
            [[0., 0., -1.], [.6, 0., -.8], [0., .6, -.8]]
        )
        torch.testing.assert_close(self.mdp.roll_orientation(env), torch.tensor([0., 0., .36]))

    def test_stability_terms_allow_forward_vertical_motion_and_pitch_rate(self):
        env = self.env(3)
        env.scene["chosen"] = SimpleNamespace(data=SimpleNamespace(
            root_ang_vel_b=torch.tensor([[0., 5., 2.], [.5, 0., 0.], [-.5, 7., 9.]]),
            root_lin_vel_b=torch.tensor([[.6, 0., .4], [0., .2, 0.], [.6, -.2, .5]]),
        ))
        cfg = _Entity("chosen")
        torch.testing.assert_close(self.mdp.roll_rate(env, cfg), torch.tensor([0., .25, .25]))
        torch.testing.assert_close(self.mdp.lateral_velocity(env, cfg), torch.tensor([0., .04, .04]))

    def test_hip_abduction_dead_zone_joint_selection_and_gradient(self):
        env = self.env(2)
        # A differently named articulation and non-contiguous explicit indexes
        # ensure the term cannot accidentally select the base robot or thighs.
        positions = torch.tensor([[99., .2, -.2, 99., .3, -.3],
                                  [99., .5, -.6, 99., -.8, .9]], requires_grad=True)
        env.scene["chosen"] = SimpleNamespace(data=SimpleNamespace(joint_pos=positions))
        cfg = _Entity("chosen")
        cfg.joint_ids = [1, 2, 4, 5]
        value = self.mdp.hip_abduction(env, cfg)
        torch.testing.assert_close(value, torch.tensor([0., .2**2 + .3**2]))
        value.sum().backward()
        torch.testing.assert_close(positions.grad[0], torch.zeros(6))
        # Gradient descent decreases the excessive left angle and increases the
        # excessive negative right angle; ignored joints have zero gradient.
        torch.testing.assert_close(positions.grad[1], torch.tensor([0., .4, -.6, 0., 0., 0.]))
        with self.assertRaises(ValueError):
            self.mdp.hip_abduction(env, cfg, max_angle=-.1)
        cfg.joint_ids = [1, 2]
        with self.assertRaises(ValueError):
            self.mdp.hip_abduction(env, cfg)

    def test_feet_lateral_dead_zone_inner_crossing_selection_and_gradient(self):
        env = self.env(2)
        positions = torch.zeros(2, 6, 3)
        positions[:, [0, 5], 1] = 99.  # unselected bodies must be ignored
        positions[0, [1, 2, 3, 4], 1] = torch.tensor([.1, -.22, .16, -.15])
        positions[1, [1, 2, 3, 4], 1] = torch.tensor([.32, -.32, -.02, .02])
        positions.requires_grad_()
        env.scene["chosen"] = SimpleNamespace(data=SimpleNamespace(
            root_pos_w=torch.zeros(2, 3), root_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1),
            body_pos_w=positions,
        ))
        cfg = _Entity("chosen")
        cfg.body_ids = [1, 2, 3, 4]
        value = self.mdp.feet_lateral(env, cfg)
        torch.testing.assert_close(value, torch.tensor([0., 2*.1**2 + 2*.12**2]))
        value.sum().backward()
        torch.testing.assert_close(positions.grad[0], torch.zeros(6, 3))
        torch.testing.assert_close(positions.grad[1, :, 1], torch.tensor([0., .2, -.2, -.24, .24, 0.]))
        torch.testing.assert_close(positions.grad[:, :, [0, 2]], torch.zeros(2, 6, 2))
        for bounds in ((-.1, .2), (.2, .2), (.3, .2)):
            with self.assertRaises(ValueError):
                self.mdp.feet_lateral(env, cfg, min_lateral=bounds[0], max_lateral=bounds[1])
        # Extra inner protection must not make the outer corridor stricter.
        torch.testing.assert_close(self.mdp.feet_lateral(env, cfg, inner_weight=2.0),
                                   torch.tensor([0., 2*.1**2 + 4*.12**2]))
        with self.assertRaises(ValueError):
            self.mdp.feet_lateral(env, cfg, inner_weight=-1)

    def test_speed_curriculum_uses_this_runs_control_steps_and_does_not_clobber_current_command(self):
        env = self.env()
        ranges = SimpleNamespace(lin_vel_x=(.35, .5))
        term = SimpleNamespace(cfg=SimpleNamespace(ranges=ranges))
        env.command_manager.get_term = lambda name: term
        current_commands = env.command_manager.get_command("base_velocity").clone()
        env.cfg.go2_settings.update({
            "commands": {"forward_min": .45, "forward_max": .65,
                         "start_forward_min": .35, "start_forward_max": .5,
                         "speed_ramp_updates": 300},
            "training": {"num_steps_per_env": 24},
        })
        # Even many completed episodes cannot jump a fresh warm-start ahead.
        env.episode_length_buf[:] = 1250
        for step, fraction, low, high in ((0, 0., .35, .5), (3600, .5, .4, .575),
                                         (7200, 1., .45, .65), (10000, 1., .45, .65)):
            env.common_step_counter = step
            metrics = self.mdp.command_speed_curriculum(env, torch.tensor([0, 2]))
            self.assertAlmostEqual(metrics["fraction"], fraction)
            self.assertAlmostEqual(ranges.lin_vel_x[0], low)
            self.assertAlmostEqual(ranges.lin_vel_x[1], high)
        torch.testing.assert_close(env.command_manager.get_command("base_velocity"), current_commands)
        # Evaluation / legacy configs preserve their exact command range.
        env.cfg.go2_settings["commands"]["speed_ramp_updates"] = 0
        ranges.lin_vel_x = (.4, .4)
        self.assertEqual(self.mdp.command_speed_curriculum(env, torch.tensor([0])), {})
        self.assertEqual(ranges.lin_vel_x, (.4, .4))

    def test_smoothness_is_second_difference_idempotent_and_masks_partial_resets(self):
        env = self.env(2)
        env.action_manager = SimpleNamespace(action=torch.zeros(2, 2), prev_action=torch.zeros(2, 2))
        env.common_step_counter = 0

        def advance(values):
            env.action_manager.prev_action.copy_(env.action_manager.action)
            env.action_manager.action.copy_(torch.tensor(values))
            env.common_step_counter += 1
            env.episode_length_buf += 1
            value = self.mdp.action_smoothness(env)
            torch.testing.assert_close(self.mdp.action_smoothness(env), value)
            return value

        # Both a constant action and a linear ramp have no second difference.
        torch.testing.assert_close(advance([[1., 1.], [1., 1.]]), torch.zeros(2))
        torch.testing.assert_close(advance([[1., 1.], [2., 2.]]), torch.zeros(2))
        torch.testing.assert_close(advance([[1., 1.], [3., 3.]]), torch.zeros(2))
        torch.testing.assert_close(advance([[2., 2.], [5., 5.]]), torch.tensor([2., 2.]))

        # A single env resets after reward computation, as in ManagerBasedRLEnv.
        env.episode_length_buf[0] = 0
        env.action_manager.action[0] = 0.
        env.action_manager.prev_action[0] = 0.
        torch.testing.assert_close(self.mdp.action_smoothness(env), torch.tensor([0., 2.]))
        torch.testing.assert_close(advance([[9., 9.], [7., 7.]]), torch.zeros(2))
        torch.testing.assert_close(advance([[8., 8.], [9., 9.]]), torch.zeros(2))
        torch.testing.assert_close(advance([[7., 7.], [11., 11.]]), torch.zeros(2))
        torch.testing.assert_close(advance([[7., 7.], [13., 13.]]), torch.tensor([2., 0.]))

        # A disabled/missed term must not reuse stale history when enabled again.
        env.common_step_counter += 5
        env.episode_length_buf += 5
        torch.testing.assert_close(self.mdp.action_smoothness(env), torch.zeros(2))

    def test_air_time_uses_completed_swing_and_caps_long_hops_without_changing_sensor(self):
        env = self.env(4)
        air_time = torch.tensor([[.30, .05], [2., .2], [.3, .3], [0., 0.]])
        contact = torch.tensor([[True, True], [True, False], [True, True], [True, True]])
        calls = []
        def first_contact(dt):
            calls.append(dt)
            return contact
        env.scene.sensors = {"contacts": SimpleNamespace(
            data=SimpleNamespace(last_air_time=air_time, current_air_time=torch.zeros_like(air_time)),
            compute_first_contact=first_contact,
        )}
        env.command_manager.get_command("base_velocity")[2, 0] = 0.
        expected = torch.tensor([.15, .25, 0., 0.])
        torch.testing.assert_close(self.mdp.feet_air_time(env, _Entity("contacts")), expected)
        torch.testing.assert_close(self.mdp.feet_air_time(env, _Entity("contacts")), expected)
        self.assertEqual(calls, [env.step_dt, env.step_dt])
        torch.testing.assert_close(air_time, torch.tensor([[.30, .05], [2., .2], [.3, .3], [0., 0.]]))
        # The landing reward disappears once contact persists; no repeat credit.
        contact.zero_()
        torch.testing.assert_close(self.mdp.feet_air_time(env, _Entity("contacts")), torch.zeros(4))

    def test_stumble_uses_same_history_sample_and_ignores_noise_and_unselected_bodies(self):
        env = self.env(4)
        forces = torch.zeros(4, 2, 2, 3)
        forces[0, 0, 0] = torch.tensor([10., 0., 1.])  # brief riser impact
        forces[0, 1, 0] = torch.tensor([0., 0., 100.])  # unrelated later support peak
        forces[1, :, 0] = torch.tensor([1., 0., 10.])  # ordinary support
        forces[2, :, 0] = torch.tensor([.01, 0., 0.])  # numerical noise
        forces[3, :, 1] = torch.tensor([10., 0., 0.])  # ignored body
        env.scene.sensors = {"contacts": SimpleNamespace(data=SimpleNamespace(net_forces_w_history=forces))}
        cfg = _Entity("contacts")
        cfg.body_ids = [0]
        torch.testing.assert_close(self.mdp.feet_stumble(env, cfg), torch.tensor([1., 0., 0., 0.]))

    def foot_env(self):
        env = self.env(1)
        data = env.scene["robot"].data
        data.body_pos_w = torch.tensor([[[0., 0., 2.04], [.2, 0., 2.04]]])
        data.body_lin_vel_w = torch.tensor([[[.5, 0., 0.], [.5, 0., 0.]]])
        env.scene.sensors = {
            "height_scanner": SimpleNamespace(data=SimpleNamespace(ray_hits_w=torch.tensor([[[0., 0., 2.], [.2, 0., 2.]]]))),
            "contact_forces": SimpleNamespace(data=SimpleNamespace(net_forces_w=torch.tensor([[[0., 0., 0.], [0., 0., 10.]]]))),
        }
        return env

    def test_clearance_uses_local_step_top_and_ignores_stance_foot(self):
        env = self.foot_env()
        penalty = self.mdp.foot_clearance(env, _Entity("robot"), _Entity("contact_forces"))
        torch.testing.assert_close(penalty, torch.tensor([.04**2 * .5]), atol=1e-7, rtol=1e-5)
        torch.testing.assert_close(self.mdp.foot_slip(env, _Entity("robot"), _Entity("contact_forces")), torch.tensor([.25]))

    def test_base_height_tracks_step_and_invalid_rays_do_not_make_nan_rewards(self):
        env = self.foot_env()
        env.scene["robot"].data.root_pos_w[:, 2] = 2.3
        torch.testing.assert_close(self.mdp.base_height(env), torch.zeros(1), atol=1e-12, rtol=0)
        env.scene.sensors["height_scanner"].data.ray_hits_w[:] = float("inf")
        torch.testing.assert_close(self.mdp.base_height(env), torch.zeros(1))
        torch.testing.assert_close(self.mdp.foot_clearance(env, _Entity("robot"), _Entity("contact_forces")), torch.zeros(1))

    def test_curriculum_uses_success_not_distance_and_preserves_top_level(self):
        env = self.env(5)
        env.scene["robot"].data.root_pos_w[:, 0] = torch.tensor([4.6, 1., 4.6, 1., 4.3])
        env.termination_manager.terminated[:] = torch.tensor([True, True, True, False, False])
        env.termination_manager.time_outs[:] = torch.tensor([False, False, False, True, True])
        env.episode_length_buf[:] = 1250
        updates = []
        env.scene.terrain = SimpleNamespace(
            terrain_levels=torch.tensor([0, 1, 2, 1, 1]), max_terrain_level=3,
            update_env_origins=lambda ids, up, down: updates.append((ids, up, down)),
        )
        metrics = self.mdp.curriculum(env, torch.arange(5))
        self.assertEqual(len(updates), 1)
        torch.testing.assert_close(updates[0][1], torch.tensor([True, False, False, False, False]))
        torch.testing.assert_close(updates[0][2], torch.tensor([False, True, False, True, False]))
        torch.testing.assert_close(metrics["success_rate"], torch.tensor(.4))
        env.episode_length_buf.zero_()
        self.mdp.curriculum(env, torch.arange(5))
        self.assertEqual(len(updates), 1)


if __name__ == "__main__":
    unittest.main()
