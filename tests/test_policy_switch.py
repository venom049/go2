"""Sensor routing and physical-target handover without an Isaac Sim process."""
import math
from types import SimpleNamespace
import unittest

try:
    import torch
    from go2_him.policy_switch import TerrainPolicyGate, HybridLocomotionController
except ImportError:
    torch = None


def quaternion(roll=0., pitch=0., yaw=0.):
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    return torch.tensor([cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy,
                         cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy])


def scan(span=0., count=1, yaw=0., roll=0., pitch=0.):
    """A world-horizontal ray grid, yaw aligned as the real sensor is."""
    xs, ys = torch.meshgrid(torch.tensor([-.4, -.25, 0., .25, .5, .65]),
                            torch.tensor([-.2, -.1, 0., .1, .2]), indexing="ij")
    x, y = xs.flatten(), ys.flatten()
    z = torch.where(x >= .25, span, 0.)
    points = torch.stack((x*math.cos(yaw) - y*math.sin(yaw),
                          x*math.sin(yaw) + y*math.cos(yaw), z), -1)
    root = torch.tensor([[3., -4., .45]]).repeat(count, 1)
    points = points[None].repeat(count, 1, 1)
    points[..., :2] += root[:, None, :2]
    q = quaternion(roll, pitch, yaw)[None].repeat(count, 1)
    return root, q, points


class FakeWalk:
    def __init__(self, target, stiffness, damping, effort):
        self.command = SimpleNamespace(joint_target=target, stiffness=stiffness,
                                       damping=damping, effort_limit=effort)
        self.reset_calls = []
        self.step_calls = []

    def reset(self, env_ids=None, joint_position=None):
        self.reset_calls.append((None if env_ids is None else env_ids.clone(), joint_position.clone()))

    def step(self, angular, quat, command, position, velocity):
        self.step_calls.append((position.clone(), velocity.clone()))
        return self.command


@unittest.skipIf(torch is None, "PyTorch is required; use the training environment Python")
class TerrainPolicyGateTests(unittest.TestCase):
    def test_floor_is_flat_under_yaw_pitch_roll_and_rays_outside_window_are_ignored(self):
        gate = TerrainPolicyGate(1, "cpu")
        for yaw, pitch, roll in ((0., 0., 0.), (1.2, .6, -.4), (-2.1, -.7, .3)):
            root, q, hits = scan(yaw=yaw, pitch=pitch, roll=roll)
            outside = torch.tensor([[[100., 100., 100.], [-100., -100., -100.]]])
            weight = gate.update(root, q, torch.cat((hits, outside), 1))
            self.assertTrue(gate.valid_scan.item())
            self.assertEqual(gate.last_span.item(), 0.)
            self.assertFalse(gate.stairs.item())
            self.assertEqual(weight.item(), 0.)

    def test_real_height_step_is_detected_in_yaw_aligned_scan_with_inclined_body(self):
        gate = TerrainPolicyGate(1, "cpu", dt=.1, blend_seconds=.5)
        root, q, hits = scan(.18, yaw=1.7, pitch=.5, roll=.3)
        self.assertAlmostEqual(gate.update(root, q, hits).item(), .2, places=6)
        self.assertTrue(gate.valid_scan.item())
        self.assertTrue(gate.stairs.item())
        self.assertAlmostEqual(gate.last_span.item(), .18, places=6)

    def test_parameterized_lookahead_detects_only_steps_inside_requested_window(self):
        root, q, base_hits = scan()
        x, y = torch.meshgrid(torch.tensor([.75, .95, 1.1]), torch.tensor([-.2, 0., .2]), indexing="ij")
        for riser, expected in ((.75, (False, True, True)), (.90, (False, False, True))):
            extra = torch.stack((x.flatten(), y.flatten(), torch.where(x.flatten() >= riser, .18, 0.)), -1)[None]
            extra[..., :2] += root[:, None, :2]
            hits = torch.cat((base_hits, extra), 1)
            for lookahead, detected in zip((.70, .80, 1.10), expected):
                gate = TerrainPolicyGate(1, "cpu", lookahead_m=lookahead)
                gate.update(root, q, hits)
                self.assertTrue(gate.valid_scan.item())
                self.assertEqual(gate.stairs.item(), detected)
                self.assertEqual(gate.blend_seconds, .30)

    def test_larger_lookahead_requires_ray_coverage_near_its_front_boundary(self):
        gate = TerrainPolicyGate(1, "cpu", lookahead_m=1.10)
        gate.stairs[:] = True
        gate.weight[:] = 1.
        # The original 0.65 m test scan is valid at the default lookahead but
        # cannot establish flat terrain through a requested 1.10 m window.
        for _ in range(25):
            gate.update(*scan())
        self.assertFalse(gate.valid_scan.item())
        self.assertTrue(gate.stairs.item())
        self.assertEqual(gate.weight.item(), 1.)

    def test_boundary_tolerance_keeps_grid_edge_hits_but_excludes_distant_points(self):
        gate = TerrainPolicyGate(1, "cpu")
        x, y = torch.meshgrid(torch.tensor([-.450005, -.199995, .2, .499995, .700005]),
                             torch.tensor([-.300005, 0., .300005]), indexing="ij")
        hits = torch.stack((x.flatten(), y.flatten(), torch.where(x.flatten() > .6, .18, 0.)), -1)[None]
        outside = torch.tensor([[[.70003, 0., 100.], [0., .30003, -100.], [-.45003, 0., -100.]]])
        gate.update(torch.zeros(1, 3), quaternion()[None], torch.cat((hits, outside), 1))
        self.assertTrue(gate.valid_scan.item())
        self.assertTrue(gate.stairs.item())
        self.assertAlmostEqual(gate.last_span.item(), .18, places=6)

    def test_hysteresis_dwell_and_blend_reject_threshold_noise(self):
        gate = TerrainPolicyGate(1, "cpu", dt=.1, blend_seconds=.5, flat_dwell_seconds=.3)
        gate.update(*scan(.05))  # equality does not enter
        self.assertFalse(gate.stairs.item())
        for _ in range(5):
            gate.update(*scan(.06))
        self.assertEqual(gate.weight.item(), 1.)
        for _ in range(5):
            gate.update(*scan(.04))  # between thresholds retains stairs
        self.assertTrue(gate.stairs.item())
        gate.update(*scan(.0))
        gate.update(*scan(.0))
        gate.update(*scan(.025))  # equality is not flat, resets dwell
        self.assertEqual(gate.flat_frames.item(), 0)
        for _ in range(2):
            gate.update(*scan(.0))
            self.assertTrue(gate.stairs.item())
        gate.update(*scan(.0))
        self.assertFalse(gate.stairs.item())
        self.assertAlmostEqual(gate.weight.item(), .8, places=6)
        for _ in range(5):
            gate.update(*scan(.0))
        self.assertEqual(gate.weight.item(), 0.)
        self.assertEqual(gate.transitions.item(), 2)

    def test_missing_or_spatially_incomplete_rays_retain_mode_and_break_exit_dwell(self):
        gate = TerrainPolicyGate(1, "cpu", dt=.1, blend_seconds=.1, flat_dwell_seconds=.3)
        gate.update(*scan(.18))
        gate.update(*scan(.0))
        self.assertEqual(gate.flat_frames.item(), 1)
        root, q, hits = scan()
        hits[:] = float("nan")
        gate.update(root, q, hits)
        self.assertFalse(gate.valid_scan.item())
        self.assertTrue(gate.stairs.item())
        self.assertEqual(gate.weight.item(), 1.)
        self.assertEqual(gate.flat_frames.item(), 0)
        # Twenty finite front/center rays exceed the old count threshold, but
        # cannot establish that the unseen rear feet have cleared the stairs.
        root, q, hits = scan()
        hits[:, :10] = float("inf")
        self.assertEqual(torch.isfinite(hits).all(-1).sum().item(), 20)
        for _ in range(5):
            gate.update(root, q, hits)
        self.assertFalse(gate.valid_scan.item())
        self.assertTrue(gate.stairs.item())
        for _ in range(3):
            gate.update(*scan())
        self.assertFalse(gate.stairs.item())

    def test_reset_only_selected_environments_clears_stale_scan_and_dwell(self):
        gate = TerrainPolicyGate(2, "cpu", dt=.1, blend_seconds=.1)
        gate.update(*scan(.18, count=2))
        gate.reset(torch.tensor([True, False]))
        self.assertEqual(gate.stairs.tolist(), [False, True])
        self.assertEqual(gate.weight.tolist(), [0., 1.])
        self.assertEqual(gate.valid_scan.tolist(), [False, True])
        self.assertTrue(math.isnan(gate.last_span[0].item()))
        self.assertEqual(gate.transitions.tolist(), [1, 1])  # cumulative diagnostic

    def test_invalid_thresholds_and_durations_fail_before_running(self):
        for kwargs in ({"dt": 0}, {"blend_seconds": 0}, {"flat_dwell_seconds": 0},
                       {"enter_height": .01, "exit_height": .02}, {"dt": float("nan")},
                       {"enter_height": float("inf")}, {"exit_height": -.01},
                       {"lookahead_m": 0}, {"lookahead_m": -.1},
                       {"lookahead_m": float("nan")}, {"lookahead_m": float("inf")}):
            with self.assertRaises(ValueError):
                TerrainPolicyGate(1, "cpu", **kwargs)


@unittest.skipIf(torch is None, "PyTorch is required; use the training environment Python")
class HybridControllerTests(unittest.TestCase):
    def make_controller(self, count=3, mode="hybrid", **controller_options):
        names = [f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR")
                 for joint in ("hip", "thigh", "calf")]
        # Each ordering is deliberately different to expose accidental index
        # agreement between the action term, articulation and actuator.
        robot_names = names[5:] + names[:5]
        actuator_names = names[::-1]
        root, q, hits = scan(count=count)
        data = SimpleNamespace(
            default_joint_pos=torch.arange(12).float()[None].repeat(count, 1) / 10,
            joint_pos=torch.arange(12).float()[None].repeat(count, 1) / 10 + .01,
            joint_vel=torch.arange(12).float()[None].repeat(count, 1),
            root_pos_w=root, root_quat_w=q, root_ang_vel_b=torch.zeros(count, 3))
        actuator = SimpleNamespace(joint_names=actuator_names,
            stiffness=torch.zeros(count, 12), damping=torch.zeros(count, 12),
            effort_limit=torch.tensor([45.43 if "calf" in name else 23.7 for name in actuator_names])[None].repeat(count, 1))
        robot = SimpleNamespace(joint_names=robot_names, data=data, actuators={"legs": actuator})
        env = SimpleNamespace(num_envs=count, device="cpu", step_dt=.02,
            episode_length_buf=torch.full((count,), 2),
            scene={"robot": robot, "height_scanner": SimpleNamespace(data=SimpleNamespace(ray_hits_w=hits))},
            command_manager=SimpleNamespace(get_command=lambda _: torch.zeros(count, 3)))
        default_caller = data.default_joint_pos[:, [robot_names.index(name) for name in names]]
        walk = FakeWalk(default_caller + 2., torch.arange(12).float()[None].repeat(count, 1)+20,
                        torch.arange(12).float()[None].repeat(count, 1)/10+1,
                        torch.tensor([45. if "calf" in name else 23.7 for name in names])[None].repeat(count, 1))
        controller = HybridLocomotionController(env, walk, names, mode=mode, **controller_options)
        return env, walk, controller

    def test_controller_uses_named_transition_scanner_and_reports_actual_gate_parameters(self):
        env, walk, controller = self.make_controller(count=1, gate_lookahead_m=.8,
                                                     height_sensor_name="transition_scanner")
        # The observation scanner remains flat. Only the independent routing
        # scanner sees the upcoming riser, so choosing the wrong name fails.
        root, q, hits = scan(.18)
        env.scene["transition_scanner"] = SimpleNamespace(data=SimpleNamespace(ray_hits_w=hits))
        observation_hits = env.scene["height_scanner"].data.ray_hits_w.clone()
        controller.action(torch.zeros(1, 12))
        self.assertTrue(controller.gate.stairs.item())
        torch.testing.assert_close(env.scene["height_scanner"].data.ray_hits_w, observation_hits)
        report = controller.report()
        self.assertIn("transition_scanner", report["gate"])
        parameters = report["gate_parameters"]
        self.assertEqual(parameters["height_sensor_name"], "transition_scanner")
        self.assertEqual(parameters["lookahead_m"], .8)
        self.assertAlmostEqual(parameters["front_coverage_min_x_m"], .6)
        self.assertEqual(parameters["boundary_tolerance_m"], 1e-5)
        self.assertEqual(parameters["entry_blend_seconds"], .3)
        self.assertEqual(parameters["exit_blend_seconds"], .3)
        self.assertEqual(parameters["control_dt_s"], env.step_dt)

    def test_pure_walk_has_no_stairs_target_or_sensor_dependency_and_no_second_him_clip(self):
        env, walk, controller = self.make_controller(mode="walk")
        del env.scene["height_scanner"]
        action = controller.action(torch.full((3, 12), float("nan")))
        # Physical walking targets exceed the HIM ±4 normalized action bound.
        # A second wrapper clip would silently alter the upstream walk policy.
        torch.testing.assert_close(action, torch.full((3, 12), 8.))
        torch.testing.assert_close(controller.default + .25*action, walk.command.joint_target)
        robot = env.scene["robot"]
        torch.testing.assert_close(robot.actuators["legs"].stiffness, walk.command.stiffness[:, controller.to_actuator])
        torch.testing.assert_close(robot.actuators["legs"].damping, walk.command.damping[:, controller.to_actuator])
        torch.testing.assert_close(robot.actuators["legs"].effort_limit, walk.command.effort_limit[:, controller.to_actuator])
        expected_positions = robot.data.joint_pos[:, controller.joint_ids]
        torch.testing.assert_close(walk.step_calls[-1][0], expected_positions)
        self.assertEqual(controller.sample_counts.tolist(), [[1, 0, 0]]*3)

    def test_mixed_physical_targets_pd_and_effort_order_and_stairs_clipping(self):
        env, walk, controller = self.make_controller()
        controller.gate.update = lambda *args: torch.tensor([0., .5, 1.])
        raw = torch.tensor([[-100., 100.]*6]*3)
        original_caps = controller.stairs_effort_limit.clone()
        action = controller.action(raw)
        stairs_target = controller.default + .25*raw.clamp(-4., 4.)
        alpha = torch.tensor([[0.], [.5], [1.]])
        expected_target = (1-alpha)*walk.command.joint_target + alpha*stairs_target
        torch.testing.assert_close(controller.default + .25*action, expected_target)
        actuator = env.scene["robot"].actuators["legs"]
        torch.testing.assert_close(actuator.stiffness, ((1-alpha)*walk.command.stiffness + alpha*40.)[:, controller.to_actuator])
        torch.testing.assert_close(actuator.damping, ((1-alpha)*walk.command.damping + alpha)[:, controller.to_actuator])
        torch.testing.assert_close(actuator.effort_limit, (1-alpha)*walk.command.effort_limit[:, controller.to_actuator] + alpha*original_caps)
        torch.testing.assert_close(controller.stairs_effort_limit, original_caps)
        torch.testing.assert_close(action[2], raw[2].clamp(-4., 4.))
        self.assertEqual(controller.sample_counts.tolist(), [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_pure_stairs_does_not_use_nonfinite_unselected_walking_output(self):
        env, walk, controller = self.make_controller()
        controller.gate.update = lambda *args: torch.ones(3)
        for tensor in vars(walk.command).values():
            tensor[:] = float("nan")
        action = controller.action(torch.full((3, 12), 2.))
        torch.testing.assert_close(action, torch.full((3, 12), 2.))
        actuator = env.scene["robot"].actuators["legs"]
        torch.testing.assert_close(actuator.stiffness, torch.full((3, 12), 40.))
        torch.testing.assert_close(actuator.effort_limit, controller.stairs_effort_limit)

    def test_return_from_stairs_anchors_only_returning_walk_history_to_previous_target(self):
        env, walk, controller = self.make_controller()
        previous = torch.arange(36).reshape(3, 12).float()/100
        controller.previous_target.copy_(previous)
        controller.last_weights[:] = torch.tensor([1., 1., .5])
        controller.gate.update = lambda *args: torch.tensor([.9, 1., .4])
        controller.action(torch.zeros(3, 12))
        mask, position = walk.reset_calls[-1]
        self.assertEqual(mask.tolist(), [True, False, False])
        torch.testing.assert_close(position, previous[:1])

    def test_reset_restores_selected_history_without_erasing_accumulated_metrics(self):
        env, walk, controller = self.make_controller()
        controller.previous_target[:] = -100.
        controller.last_weights[:] = 1.
        controller.gate.stairs[:] = True
        controller.gate.weight[:] = 1.
        controller.transition_sequence_phase[:] = 2
        controller.completed_transition_sequences[:] = 1
        controller.sample_counts[:] = 7
        controller.reset(torch.tensor([False, True, False]))
        torch.testing.assert_close(controller.previous_target[1], env.scene["robot"].data.joint_pos[1, controller.joint_ids])
        self.assertEqual(controller.previous_target[0, 0].item(), -100.)
        self.assertEqual(controller.last_weights.tolist(), [1., 0., 1.])
        self.assertEqual(controller.gate.stairs.tolist(), [True, False, True])
        self.assertEqual(controller.transition_sequence_phase.tolist(), [2, 0, 2])
        self.assertEqual(controller.completed_transition_sequences.tolist(), [1, 1, 1])
        self.assertEqual(controller.sample_counts.tolist(), [[7, 7, 7]]*3)

    def test_inactive_retries_cannot_inflate_transitions_samples_or_full_sequence(self):
        env, walk, controller = self.make_controller(count=2)
        active = torch.tensor([True, False])
        def choose(weight):
            def update(*args):
                controller.gate.stairs[:] = weight > 0
                return torch.full((2,), float(weight))
            controller.gate.update = update
        for weight in (0, 1, 0):
            choose(weight)
            controller.action(torch.zeros(2, 12), active=active)
        result = controller.report()
        self.assertEqual(result["mode_transitions_per_env"], [2, 0])
        self.assertEqual(result["flat_stairs_flat_completed_per_env"], [True, False])
        self.assertEqual(result["flat_stairs_flat_completion_counts_per_env"], [1, 0])
        self.assertEqual(controller.sample_counts.tolist(), [[2, 0, 1], [0, 0, 0]])
        self.assertEqual(controller.max_target_delta[1].item(), 0.)
        # A new reset episode cannot turn the first episode's partial sequence
        # into a false completed flat->stairs->flat transition.
        controller.reset(torch.tensor([True, True]))
        for weight in (0, 1, 0):
            choose(weight)
            controller.action(torch.zeros(2, 12), active=torch.tensor([False, False]))
        self.assertEqual(controller.report()["mode_transitions_per_env"], [2, 0])
        self.assertEqual(controller.completed_transition_sequences.tolist(), [1, 0])

    def test_transition_sequence_cannot_bridge_episodes_or_count_reset_frames(self):
        env, walk, controller = self.make_controller(count=1)
        for weight, length in ((0, 0), (1, 2), (0, 3)):
            env.episode_length_buf[:] = length
            controller.gate.update = lambda *args, w=weight: torch.tensor([float(w)])
            controller.action(torch.zeros(1, 12))
        self.assertEqual(controller.completed_transition_sequences.item(), 0)
        controller.gate.update = lambda *args: torch.ones(1)
        controller.action(torch.zeros(1, 12))
        self.assertEqual(controller.transition_sequence_phase.item(), 2)
        controller.reset(torch.tensor([True]))
        controller.gate.update = lambda *args: torch.zeros(1)
        controller.action(torch.zeros(1, 12))
        self.assertEqual(controller.completed_transition_sequences.item(), 0)


if __name__ == "__main__":
    unittest.main()
