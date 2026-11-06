"""Full PPO continuation preserves state and counts the requested new updates."""
import contextlib
import hashlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import train_moe_stairs as training

ACTOR=ROOT/'validation/moe_cts_actor_conversion_20261104/actor.pt'

class SyntheticVec:
    objectives=[]
    def __init__(self,n,workers,seed,stair_steps=2,position_constraint='hard'):
        self.n=n;self.rng=np.random.default_rng(seed)
        self.obs=self.rng.normal(0,.15,(n,225)).astype(np.float32)
    def settings(self,frontier,ramp):
        self.objectives.append((self.guard_objective,ramp))
    def step(self,actions):
        self.obs=self.rng.normal(0,.15,(self.n,225)).astype(np.float32)
        reward=(.1*actions[:,0]+np.arange(self.n)*.05).astype(np.float32)
        return self.obs.copy(),reward,np.zeros(self.n,bool),[{'guard_cost':.1} for _ in range(self.n)]
    def close(self):pass

def args(actor,out,**kw):
    defaults=dict(actor=actor,out=out,updates=2,num_envs=4,workers=1,steps=4,epochs=1,
                  minibatches=1,critic_warmup=0,actor_lr=3e-6,std=.03,target_kl=1.,
                  anchor_coef=.2,seed=42,device='cpu',resume_training=False,
                  guard_objective='balanced',guard_ramp_start=.2,guard_ramp_updates=300)
    defaults.update(kw)
    return SimpleNamespace(**defaults)

@unittest.skipUnless(ACTOR.is_file(),'Requires verified actor conversion')
class ResumeTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def make_source(self,root):
        with patch.object(training,'Vec',SyntheticVec),contextlib.redirect_stdout(io.StringIO()):
            training.train(args(ACTOR,root/'warm'))
        data=torch.load(root/'warm/model_2.pt',map_location='cpu',weights_only=True)
        data['update']=data['cumulative_update']=1000
        path=root/'source1000.pt';torch.save(data,path)
        return path,data

    def test_full_resume_restores_actor_critic_adam_and_reference_exactly(self):
        with tempfile.TemporaryDirectory() as temp:
            source,data=self.make_source(Path(temp))
            a=args(source,Path(temp)/'unused',resume_training=True,guard_objective='calf_velocity')
            actor,critic,opt,anchor,meta=training.initialize_training_state(a,'cpu')
            self.assertEqual(meta['source_cumulative_update'],1000)
            self.assertTrue(meta['optimizer_restored'])
            self.assertFalse(meta['environment_and_rng_restored'])
            for name,v in data['actor_state_dict'].items():
                torch.testing.assert_close(actor.state_dict()[name],v,rtol=0,atol=0)
                torch.testing.assert_close(anchor.state_dict()[name],v,rtol=0,atol=0)
            for name,v in data['critic_state_dict'].items():
                torch.testing.assert_close(critic.state_dict()[name],v,rtol=0,atol=0)
            restored=opt.state_dict()
            self.assertEqual(restored['param_groups'],data['optimizer_state_dict']['param_groups'])
            for index,state in data['optimizer_state_dict']['state'].items():
                for key,value in state.items():
                    torch.testing.assert_close(restored['state'][index][key],value,rtol=0,atol=0)
            self.assertTrue(all(not p.requires_grad for p in actor.student_moe_encoder.parameters()))
            self.assertTrue(all(not p.requires_grad for p in anchor.parameters()))

    def test_additional_updates_count_from_source_and_do_not_mutate_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source,data=self.make_source(root)
            before=hashlib.sha256(source.read_bytes()).hexdigest()
            SyntheticVec.objectives=[]
            with patch.object(training,'Vec',SyntheticVec),contextlib.redirect_stdout(io.StringIO()):
                training.train(args(source,root/'continued',resume_training=True,
                                    guard_objective='calf_velocity',guard_ramp_start=.5,guard_ramp_updates=100))
            c=torch.load(root/'continued/model_2.pt',map_location='cpu',weights_only=True)
            self.assertEqual(c['update'],2);self.assertEqual(c['cumulative_update'],1002)
            self.assertEqual(c['guard_objective'],'calf_velocity')
            self.assertEqual(c['training_record']['cumulative_update'],1002)
            self.assertEqual(SyntheticVec.objectives,[('calf_velocity',.5),('calf_velocity',.505)])
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),before)
            self.assertFalse((root/'continued/model_3.pt').exists())
            self.assertTrue(any(not torch.equal(v,c['actor_state_dict'][name])
                                for name,v in data['actor_state_dict'].items() if name.startswith('actor.')))
            for name,v in data['actor_state_dict'].items():
                if name.startswith('student_moe_encoder.'):
                    torch.testing.assert_close(v,c['actor_state_dict'][name],rtol=0,atol=0)
            old=data['optimizer_state_dict'];new=c['optimizer_state_dict']
            for group in old['param_groups']:
                for index in group['params']:
                    self.assertGreater(float(new['state'][index]['step']),float(old['state'][index]['step']))

    def test_full_resume_rejects_changed_std_or_guard_profile(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source,data=self.make_source(root)
            with self.assertRaisesRegex(ValueError,'exploration std'):
                training.initialize_training_state(args(source,root/'unused',resume_training=True,std=.04),'cpu')
            data['profile']['source_sha256']='wrong';torch.save(data,source)
            with self.assertRaisesRegex(ValueError,'guard profile'):
                training.initialize_training_state(args(source,root/'unused',resume_training=True),'cpu')

if __name__=='__main__':unittest.main()
