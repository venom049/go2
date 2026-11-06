"""Integration checks for bounded MoE targets and executed-action history.

Most checks run the production environment against deterministic physics. One
short MuJoCo check exercises the real observation/history and reset pipeline.
No policy training or robot transport is involved.
"""
from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_moe_strict_guard import environment
from go2_him.moe_stair_env import JointGuardStop, MoEStairEnv
from go2_him.moe_stair_rewards import compute_stair_rewards


def clipped_environment(*, mode="audit", decimation=4, after_step=None):
    env = environment(mode=mode, decimation=decimation, after_step=after_step)
    env.guard_objective = "target_clipped"
    return env


def calf_action(env, excess=.2):
    raw = np.zeros(12, dtype=np.float32)
    index = env.profile.names.index("RL_calf")
    raw[index] = (env.profile.upper[index] + excess - env.exp.policy.offset[index]) / .25
    return raw, index


def expected_targets(env, raw):
    # Retain the original environment's affine arithmetic, then exact bounds.
    affine = (np.asarray(raw, dtype=np.float32) * env.exp.policy.scale
              + env.exp.policy.offset).astype(np.float64)
    sent = np.clip(affine, env.profile.lower, env.profile.upper)
    return affine, sent


class TargetClipEnvironmentTest(unittest.TestCase):
    def test_float32_reference_anchor_bounds_decode_inside_every_joint_limit(self):
        from train_moe_stairs import target_action_contract
        contract = target_action_contract()
        scale, offset = np.asarray(contract["scale"]), np.asarray(contract["offset"])
        lower, upper = np.asarray(contract["lower_rad"]), np.asarray(contract["upper_rad"])
        # These anchors are stored and clamped as float32 by PPO, while the
        # simulator applies float64 configuration scale/offset. No tolerance:
        # a small outward roundoff is still an illegal command for the guard.
        probes = np.stack((np.asarray(contract["raw_action_lower"], dtype=np.float32),
                           np.zeros(12, dtype=np.float32),
                           np.asarray(contract["raw_action_upper"], dtype=np.float32)))
        decoded = probes * scale + offset
        for i, name in enumerate(contract["joint_names"]):
            with self.subTest(joint=name):
                self.assertTrue(np.all(decoded[:, i] >= lower[i]))
                self.assertTrue(np.all(decoded[:, i] <= upper[i]))
                self.assertLessEqual(probes[0, i], 0.)
                self.assertGreaterEqual(probes[2, i], 0.)
        np.testing.assert_array_equal(decoded[1], offset)

    def test_nonzero_affine_targets_preserve_original_configuration_precision(self):
        env = clipped_environment()
        env.exp.policy.offset = np.array([.1, .8, -1.5, -.1, .8, -1.5,
                                         .1, 1., -1.5, -.1, 1., -1.5], dtype=np.float64)
        raw = np.array([.123456789, -.234567891, .345678912] * 4, dtype=np.float32)
        affine, sent = expected_targets(env, raw)
        _, _, _, info = env.step(raw)
        np.testing.assert_array_equal(info["target_step"]["raw_target_rad"], affine)
        np.testing.assert_array_equal(info["target_step"]["sent_target_rad"], sent)

    def test_zero_keeps_default_pose_and_has_no_raw_target_penalty(self):
        env = clipped_environment()
        _, reward, done, info = env.step(np.zeros(12, dtype=np.float32))
        self.assertFalse(done)
        self.assertTrue(np.isfinite(reward))
        np.testing.assert_array_equal(info["target_step"]["raw_target_rad"], env.exp.policy.offset)
        np.testing.assert_array_equal(info["target_step"]["sent_target_rad"], env.exp.policy.offset)
        np.testing.assert_array_equal(env.last_action, np.zeros(12, dtype=np.float32))
        np.testing.assert_array_equal(env.exp.data.ctrl, np.zeros(12))
        self.assertEqual(info["raw_target_cost_rate"], 0.)
        np.testing.assert_array_equal(info["target_violation_rad"], np.zeros(12))
        self.assertEqual(env.audit.summary()["metrics"]["target_position"]["excess_count"], 0)

    def test_outside_raw_target_is_penalized_while_pd_uses_clipped_target(self):
        env = clipped_environment()
        env.ramp = .25
        raw, index = calf_action(env)
        raw_before = raw.copy()
        q_before = env.exp.data.qpos[env.exp.qadr].copy()
        affine, sent = expected_targets(env, raw)
        expected_c = np.maximum(env.profile.lower - affine, 0.) + np.maximum(affine - env.profile.upper, 0.)
        _, reward, done, info = env.step(raw)
        self.assertFalse(done)
        np.testing.assert_array_equal(raw, raw_before)
        np.testing.assert_array_equal(info["target_step"]["raw_target_rad"], affine)
        np.testing.assert_array_equal(info["target_step"]["sent_target_rad"], sent)
        self.assertGreater(info["target_step"]["raw_target_rad"][index], env.profile.upper[index])
        self.assertEqual(info["target_step"]["sent_target_rad"][index], env.profile.upper[index])
        np.testing.assert_array_equal(info["target_violation_rad"], expected_c)
        self.assertGreater(info["raw_target_cost_rate"], 0.)
        self.assertAlmostEqual(info["raw_target_cost_rate"], 20. * expected_c.sum())
        expected_torque = np.clip(env.exp.kp * (sent - q_before),
                                  -env.exp.physical_effort, env.exp.physical_effort)
        np.testing.assert_allclose(env.exp.data.ctrl, expected_torque, atol=1e-12, rtol=0)
        self.assertLess(env.exp.data.ctrl[index], env.exp.kp[index] * (affine[index] - q_before[index]))
        np.testing.assert_array_equal(env.exp.data.qpos[env.exp.qadr], q_before)
        # The penalty is a rate: apply ramp and elapsed time exactly once.
        self.assertAlmostEqual(reward, info["reward_terms"]["step_reward"]
                               - env.ramp * info["guard_cost"] * env.elapsed)
        self.assertGreaterEqual(info["guard_cost"], info["raw_target_cost_rate"])
        guard = env.audit.summary()
        self.assertEqual(guard["physics_state_samples"], 2 * env.exp.decimation)
        self.assertEqual(guard["metrics"]["target_position"]["excess_count"], 0)
        self.assertIsNotNone(env.target_commands)
        command = env.target_commands.summary()
        self.assertEqual(command["control_steps"], 1)
        self.assertEqual(command["raw"]["sample_count"], 12)
        self.assertEqual(command["sent"]["sample_count"], 12)
        self.assertEqual(command["raw"]["excess_count"], 1)
        self.assertEqual(command["sent"]["excess_count"], 0)
        self.assertEqual(command["raw"]["per_joint"]["RL_calf"]["excess_count"], 1)
        self.assertEqual(command["clipping"]["control_steps"], 1)
        self.assertAlmostEqual(command["raw_cost_rate"]["sum"], info["raw_target_cost_rate"])

    def test_history_and_both_smoothness_operands_use_sent_actions(self):
        env = clipped_environment()
        observed_states = []

        def inspect_reward(state):
            observed_states.append({key: np.asarray(state[key]).copy() for key in ("actions", "prev_actions")})
            return compute_stair_rewards(state)

        raw1, index = calf_action(env, .2)
        raw2, _ = calf_action(env, 1.2)
        with patch("go2_him.moe_stair_env.compute_stair_rewards", side_effect=inspect_reward):
            env.step(raw1)
            first_sent_action = env.last_action.copy()
            _, _, _, info = env.step(raw2)
        self.assertGreater(raw2[index], raw1[index])
        self.assertLess(first_sent_action[index], raw1[index])
        np.testing.assert_array_equal(observed_states[0]["prev_actions"], np.zeros(12))
        np.testing.assert_array_equal(observed_states[0]["actions"], first_sent_action)
        np.testing.assert_array_equal(observed_states[1]["prev_actions"], first_sent_action)
        np.testing.assert_array_equal(observed_states[1]["actions"], first_sent_action)
        np.testing.assert_array_equal(env.last_action, first_sent_action)
        np.testing.assert_array_equal(env.prev_action, first_sent_action)
        self.assertEqual(info["reward_terms"]["rate_terms"]["action_change"], 0.)
        self.assertGreater(info["raw_target_cost_rate"], 20.)
        command = env.target_commands.summary()
        self.assertEqual(command["control_steps"], 2)
        self.assertEqual(command["raw"]["excess_count"], 2)
        self.assertEqual(command["sent"]["excess_count"], 0)

    def test_clipping_does_not_change_measured_state_or_hide_other_violations(self):
        def violate_on_second_step(number, data):
            if number == 2:
                data.qpos[7 + 2] = -2.9
                data.qvel[6 + 2] = 16.

        env = clipped_environment(after_step=violate_on_second_step)
        raw, _ = calf_action(env)
        env.step(raw)
        self.assertEqual(env.exp.data.qpos[7 + 2], -2.9)
        self.assertEqual(env.exp.data.qvel[6 + 2], 16.)
        guard = env.audit.summary()
        self.assertEqual(guard["metrics"]["target_position"]["excess_count"], 0)
        self.assertGreater(guard["metrics"]["position"]["excess_count"], 0)
        self.assertGreater(guard["metrics"]["velocity"]["excess_count"], 0)
        self.assertGreater(guard["metrics"]["tracking"]["excess_count"], 0)
        self.assertFalse(guard["joint_and_torque_acceptance_passed"])
        self.assertTrue(guard["all_metrics_observed"])

    def test_strict_guard_can_still_reject_legal_target_for_tracking(self):
        env = clipped_environment(mode="strict")
        raw, index = calf_action(env)
        env.exp.data.qpos[env.exp.qadr[index]] = -2.
        with self.assertRaises(JointGuardStop) as caught:
            env.step(raw)
        self.assertEqual(caught.exception.details["boundary"], "pre_command")
        self.assertIn("RL_calf", caught.exception.details["violations"]["tracking"])
        self.assertNotIn("target_position", caught.exception.details["violations"])
        self.assertEqual(env.exp.mj.steps, 0)
        guard = env.audit.summary()
        self.assertEqual(guard["metrics"]["target_position"]["excess_count"], 0)
        self.assertEqual(guard["metrics"]["applied_torque"]["sample_count"], 0)
        self.assertFalse(guard["joint_and_torque_acceptance_passed"])
        command = env.target_commands.summary()
        self.assertEqual(command["control_steps"], 1)
        self.assertEqual(command["raw"]["excess_count"], 1)
        self.assertEqual(command["sent"]["excess_count"], 0)

    def test_real_history_uses_sent_action_and_reset_clears_it(self):
        env = MoEStairEnv(seed=42, evaluation=True, guard_objective="target_clipped", guard_mode="audit")
        try:
            initial = env.reset(sample={"yaw_rad": 0., "y_m": 0.}, height=.15, vx=.6)
            np.testing.assert_array_equal(initial[-60:], np.zeros(60, dtype=np.float32))
            first_raw, index = calf_action(env, .2)
            obs, _, done, info = env.step(first_raw)
            self.assertFalse(done)
            expected = ((info["target_step"]["sent_target_rad"] - env.exp.policy.offset)
                        / env.exp.policy.scale).astype(np.float32)
            np.testing.assert_allclose(env.last_action, expected, atol=1e-7, rtol=0)
            np.testing.assert_array_equal(obs[-12:], env.last_action)
            self.assertLess(obs[-12 + index], first_raw[index])
            old_sent = env.last_action.copy()
            obs, _, done, _ = env.step(np.zeros(12, dtype=np.float32))
            self.assertFalse(done)
            np.testing.assert_array_equal(obs[-24:-12], old_sent)
            np.testing.assert_array_equal(obs[-12:], np.zeros(12, dtype=np.float32))
            old_accumulator = env.target_commands
            obs = env.reset(sample={"yaw_rad": 0., "y_m": 0.}, height=.15, vx=.6)
            self.assertIsNot(env.target_commands, old_accumulator)
            self.assertEqual(env.target_commands.summary()["control_steps"], 0)
            np.testing.assert_array_equal(obs[-60:], np.zeros(60, dtype=np.float32))
            np.testing.assert_array_equal(env.last_action, np.zeros(12, dtype=np.float32))
        finally:
            env.exp.close()


if __name__ == "__main__":
    unittest.main()
