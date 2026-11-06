import unittest
import numpy as np
from test_moe_target_clip_env import clipped_environment, calf_action, expected_targets
from go2_him.target_clip import transform_target_action
from go2_him.official_guard import load_official_profile, SDK_NAMES


class SDKTargetOnlyTest(unittest.TestCase):
    def test_previous_rear_calf_failure_is_bounded_without_new_penalty(self):
        env = clipped_environment()
        env.guard_objective = 'balanced'
        env.target_mode = 'bounded'
        raw, index = calf_action(env)
        raw[index] = (-.7929051518-env.exp.policy.offset[index])/.25
        before = env.exp.data.qpos[env.exp.qadr].copy()
        affine, sent = expected_targets(env, raw)
        _, _, _, info = env.step(raw)
        self.assertGreater(affine[index], env.profile.upper[index])
        self.assertEqual(sent[index], env.profile.upper[index])
        self.assertEqual(info['raw_target_cost_rate'], 0.)
        np.testing.assert_array_equal(info['target_step']['sent_target_rad'], sent)
        np.testing.assert_array_equal(env.exp.data.ctrl,
            np.clip(env.exp.kp*(sent-before),-env.exp.physical_effort,env.exp.physical_effort))
        np.testing.assert_array_equal(env.last_action,
            ((sent-env.exp.policy.offset)/env.exp.policy.scale).astype(np.float32))
        self.assertEqual(env.audit.summary()['metrics']['target_position']['excess_count'],0)

    def test_extreme_actions_and_sdk_order_remain_inside_exact_limits(self):
        profile=load_official_profile()
        sdk=load_official_profile(joint_names=SDK_NAMES)
        env=clipped_environment()
        rng=np.random.default_rng(42)
        actions=rng.normal(0,1000,(10000,12)).astype(np.float32)
        out=transform_target_action(actions,env.exp.policy.scale,env.exp.policy.offset,
                                    profile.lower,profile.upper)
        mapping=[profile.names.index(n) for n in sdk.names]
        sent=out.sent_target.astype(np.float32)[:,mapping]
        self.assertTrue(np.all(sent>=sdk.lower))
        self.assertTrue(np.all(sent<=sdk.upper))
        self.assertEqual(sdk.lower[1],np.float32(-1.5708))
        self.assertEqual(sdk.lower[7],np.float32(-.5236))

    def test_nonfinite_action_is_rejected_before_physics(self):
        env=clipped_environment();env.guard_objective='balanced';env.target_mode='bounded'
        for invalid in (np.nan,np.inf,-np.inf):
            a=np.zeros(12,np.float32);a[11]=invalid
            with self.assertRaises(FloatingPointError):env.step(a)
        self.assertEqual(env.exp.mj.steps,0)


if __name__=='__main__':unittest.main()
