"""Upstream walking interface regression tests; no simulation dependency."""

import math
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required; use the training environment Python")
class FlatPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from go2_him import flat_policy
        cls.fp = flat_policy

    def actor(self, output):
        class ConstantActor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("output", output)
                self.observation = None

            def forward(self, observation):
                self.observation = observation.clone()
                return self.output.expand(len(observation), -1)
        return ConstantActor()

    def states(self, n=1):
        return (torch.zeros(n, 3), torch.tensor([[1., 0., 0., 0.]]).repeat(n, 1),
                torch.tensor([[.4, -.2, .8]]).repeat(n, 1),
                torch.tensor([self.fp.DEFAULT_DOF_POS]).repeat(n, 1), torch.zeros(n, 12))

    def test_observation_matches_upstream_order_scaling_and_all_action_history(self):
        w, quat, cmd, q, dq = self.states()
        w[:] = torch.tensor([4., -8., 12.])
        q += torch.arange(12) / 10
        dq[:] = torch.arange(12) * 20
        history = torch.arange(16).unsqueeze(0).float()
        result = self.fp.build_walk_observation(w, quat, cmd, q, dq, history)
        torch.testing.assert_close(result[0, :9], torch.tensor([1., -2., 3., 0., 0., -1., .8, -.4, .2]))
        torch.testing.assert_close(result[0, 9:21], torch.arange(12) / 10)
        torch.testing.assert_close(result[0, 21:33], torch.arange(12).float())
        torch.testing.assert_close(result[0, 33:], history[0])

    def test_inverse_quaternion_gravity_uses_wxyz(self):
        w, quat, cmd, q, dq = self.states()
        # 90-degree roll: world downward gravity points toward body -Y.
        quat[:] = torch.tensor([math.sqrt(.5), math.sqrt(.5), 0., 0.])
        obs = self.fp.build_walk_observation(w, quat, cmd, q, dq, torch.zeros(1, 16))
        torch.testing.assert_close(obs[0, 3:6], torch.tensor([0., -1., 0.]), atol=1e-6, rtol=0)
        mirrored = self.fp.build_walk_observation(w, -quat, cmd, q, dq, torch.zeros(1, 16))
        torch.testing.assert_close(mirrored, obs)

    def test_joint_mapping_preserves_physical_legs_and_variable_gains(self):
        names = tuple(f"{leg}_{joint}_joint" for joint in ("hip", "thigh", "calf")
                      for leg in ("FL", "FR", "RL", "RR"))
        action = torch.tensor([[1., -1., .2, -1., 1., -.2, 1., 1., 1., -1., -1., -1.,
                                -3., -.5, .5, 3.]])
        actor = self.actor(action)
        adapter = self.fp.UpstreamWalkPolicy(actor, names, 1)
        w, quat, cmd, q, dq = self.states()
        caller_order = [self.fp.POLICY_JOINT_NAMES.index(name) for name in names]
        q = q[:, caller_order]
        result = adapter.step(w, quat, cmd, q, dq)
        # Native interleaved-by-joint ordering must not swap FL and FR policies.
        expected_policy_target = torch.tensor([self.fp.DEFAULT_DOF_POS]) + torch.tensor(
            [[.1, -.1, .05, -.1, .1, -.05, .1, .1, .1, -.1, -.1, -.1]])
        expected_kp = torch.tensor([[20., 20., 20., 30., 30., 30., 50., 50., 50., 60., 60., 60.]])
        torch.testing.assert_close(result.joint_target, expected_policy_target[:, caller_order])
        torch.testing.assert_close(result.stiffness, expected_kp[:, caller_order])
        torch.testing.assert_close(result.damping, .2 * expected_kp[:, caller_order].sqrt())
        torch.testing.assert_close(result.effort_limit, torch.tensor([[23.7, 23.7, 45.0] * 4])[:, caller_order])
        torch.testing.assert_close(actor.observation[0, 9:21], torch.zeros(12))

    def test_clipped_raw_history_is_not_slew_limited_action(self):
        actor = self.actor(torch.full((1, 16), 200.))
        adapter = self.fp.UpstreamWalkPolicy(actor, self.fp.POLICY_JOINT_NAMES, 1)
        states = self.states()
        first = adapter.step(*states)
        second = adapter.step(*states)
        torch.testing.assert_close(first.raw_action, torch.full((1, 16), 100.))
        torch.testing.assert_close(actor.observation[:, 33:], torch.full((1, 16), 100.))
        torch.testing.assert_close(second.joint_target - first.joint_target, torch.full((1, 12), .1))
        torch.testing.assert_close(second.stiffness, torch.full((1, 12), 60.))

    def test_partial_reset_avoids_cross_episode_memory_and_seeds_handover_slew(self):
        actor = self.actor(torch.full((1, 16), 2.))
        adapter = self.fp.UpstreamWalkPolicy(actor, self.fp.POLICY_JOINT_NAMES, 2)
        states = self.states(2)
        before = adapter.step(*states)
        target = torch.tensor([self.fp.DEFAULT_DOF_POS]) - .7
        adapter.reset(torch.tensor([True, False]), target)
        result = adapter.step(*states)
        torch.testing.assert_close(actor.observation[0, 33:], torch.zeros(16))
        torch.testing.assert_close(actor.observation[1, 33:], torch.full((16,), 2.))
        torch.testing.assert_close(result.joint_target[0], target[0] + .1)
        torch.testing.assert_close(result.joint_target[1], before.joint_target[1] + .1)

    def test_gain_scaling_does_not_double_scale_damping(self):
        actor = self.actor(torch.zeros(1, 16))
        adapter = self.fp.UpstreamWalkPolicy(actor, self.fp.POLICY_JOINT_NAMES, 1, kp_factor=.5, kd_factor=2.)
        result = adapter.step(*self.states())
        torch.testing.assert_close(result.stiffness, torch.full((1, 12), 20.))
        torch.testing.assert_close(result.damping, torch.full((1, 12), .4 * math.sqrt(40.)))

    def test_configuration_rejects_wrong_joint_set_frequency_and_invalid_gains(self):
        actor = self.actor(torch.zeros(1, 16))
        for kwargs in ({"policy_dt": .01}, {"kp_factor": 0.}, {"kd_factor": float("nan")}, {"num_envs": 0}):
            args = dict(actor=actor, joint_names=self.fp.POLICY_JOINT_NAMES, num_envs=1)
            args.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.fp.UpstreamWalkPolicy(**args)
        with self.assertRaises(ValueError):
            self.fp.UpstreamWalkPolicy(actor, ["FR_hip_joint"] * 12, 1)

    def test_strict_safe_load_rejects_malformed_or_nonfinite_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            actor = self.fp.WalkActor()
            torch.save({"model_state_dict": actor.state_dict()}, path)
            loaded = self.fp.load_walk_actor(path)
            torch.testing.assert_close(loaded(torch.zeros(1, 49)), actor(torch.zeros(1, 49)))
            self.assertFalse(any(p.requires_grad for p in loaded.parameters()))
            state = actor.state_dict()
            state["actor.0.weight"][0, 0] = float("nan")
            torch.save({"model_state_dict": state}, path)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                self.fp.load_walk_actor(path)
            torch.save({"model_state_dict": {}}, path)
            with self.assertRaises(ValueError):
                self.fp.load_walk_actor(path)


if __name__ == "__main__":
    unittest.main()
