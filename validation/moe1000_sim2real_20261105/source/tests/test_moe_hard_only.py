import sys
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_moe_hard_only_training import training_command
from train_moe_stairs import ROOT as TRAIN_ROOT
from go2_him.moe_stair_env import MoEStairEnv


class HardOnlyTest(unittest.TestCase):
    def test_command_preserves_original_protocol_and_uses_full_resume(self):
        cmd=training_command(Path('original'),Path('new'))
        expected={'--updates':'1000','--guard-objective':'balanced','--critic-warmup':'30',
                  '--guard-ramp-start':'.2','--guard-ramp-updates':'300','--anchor-coef':'.2',
                  '--steps':'24','--epochs':'3','--minibatches':'4','--std':'.03',
                  '--target-kl':'.005','--stair-steps':'6','--position-constraint':'hard'}
        for key,value in expected.items():
            self.assertEqual(cmd[cmd.index(key)+1],value)
        self.assertIn('--resume-training',cmd)
        self.assertNotIn('target_clipped',cmd)

    def test_actual_stop_does_not_change_legacy_target_history_or_add_penalty(self):
        env=MoEStairEnv(seed=42,evaluation=True,guard_objective='balanced',
                        stair_steps=6,position_constraint='hard')
        action=np.zeros(12,dtype=np.float32); action[11]=5.4441
        _,_,_,info=env.step(action)
        target=action*env.exp.policy.scale+env.exp.policy.offset
        np.testing.assert_array_equal(info['target_step']['raw_target_rad'],target)
        np.testing.assert_array_equal(info['target_step']['sent_target_rad'],target)
        np.testing.assert_array_equal(env.last_action,action)
        self.assertGreater(target[11],env.profile.upper[11])
        self.assertEqual(info['raw_target_cost_rate'],0.)
        self.assertEqual(info['actual_position']['excess_count'],0)
        self.assertTrue(np.all(env.exp.data.qpos[env.exp.qadr]>=env.profile.lower))
        self.assertTrue(np.all(env.exp.data.qpos[env.exp.qadr]<=env.profile.upper))


if __name__=='__main__': unittest.main()
