import unittest,sys,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from go2_him.moe_stair_env import MoEStairEnv
from go2_him.sim2real_dr import Curriculum,PRIVILEGED_EXTRA

class Sim2RealTests(unittest.TestCase):
    def env(self,seed=1,enabled=True):
        e=MoEStairEnv(seed=seed,evaluation=True,position_constraint='native',target_mode='bounded',domain_randomization=enabled)
        self.addCleanup(e.exp.close);return e
    def test_official_asset_and_mapping(self):
        path=ROOT/'assets/unitree_go2/go2.xml'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),'2014a3d76e30f17ab9447d8a67bd015291f74fa4d71ae30d005f1a32bd693d4b')
        e=self.env(enabled=False);m=e.exp.model
        self.assertAlmostEqual(m.body_mass.sum(),15.206408)
        np.testing.assert_allclose(m.jnt_range[e.exp.jids[1::3],1],[3.4907,3.4907,4.5379,4.5379])
        self.assertTrue(np.all(m.actuator_ctrllimited[e.exp.aids]))
        np.testing.assert_array_equal(m.actuator_trnid[e.exp.aids,0],e.exp.jids)
    def test_randomization_effect_restore_and_reproducibility(self):
        a,b=self.env(17),self.env(17)
        for e in (a,b):e.dr.fixed_level=1.;e.reset()
        self.assertEqual(a.dr.summary(),b.dr.summary())
        self.assertTrue(np.any(a.exp.model.body_mass!=a.dr.nominal['body_mass']))
        for body in (a.exp.base,*a.dr.hips):
            np.testing.assert_allclose(a.exp.model.body_inertia[body]/a.dr.nominal['body_inertia'][body],
                np.repeat(a.exp.model.body_mass[body]/a.dr.nominal['body_mass'][body],3))
        self.assertTrue(np.all(a.exp.model.geom_friction[a.exp.feet,0]==a.dr.mu))
        a.dr.enabled=False;a.reset()
        for k,v in a.dr.nominal.items():np.testing.assert_array_equal(getattr(a.exp.model,k),v)
        np.testing.assert_array_equal(a.exp.kp,a.dr.kp)
    def test_downstream_noise_delay_and_bounds(self):
        e=self.env();e.dr.fixed_level=1.;e.reset();dr=e.dr
        dr.delay=1;dr.active=False
        first=e.profile.lower.copy();second=e.profile.upper.copy()
        q=dr.actuator_target(first);np.testing.assert_allclose(q,e.exp.policy.offset)
        q=dr.actuator_target(second);np.testing.assert_array_equal(q,first)
        dr.active=True
        for i in range(1000):
            q=dr.actuator_target(first if i%2 else second)
            self.assertTrue(np.all(q>=e.profile.lower)&np.all(q<=e.profile.upper))
        self.assertGreater(dr.stats['noisy_target_reclipped_joints'],0)
        self.assertEqual(dr.stats['applied_target_excess'],0)
    def test_noise_history_and_privileged_separation(self):
        e=self.env();e.dr.fixed_level=1.;obs=e.reset();raw=e.raw_observation()
        noisy=e.dr.noisy_observation(raw)
        np.testing.assert_array_equal(noisy['last_action'],raw['last_action'])
        np.testing.assert_array_equal(noisy['velocity_commands'],raw['velocity_commands'])
        self.assertTrue(np.any(noisy['joint_pos_rel']!=raw['joint_pos_rel']))
        before=obs.copy();priv=e.dr.privileged(obs)
        self.assertEqual(priv.shape,(225+PRIVILEGED_EXTRA,));np.testing.assert_array_equal(obs,before)
        obs,reward,done,info=e.step(np.full(12,100.,np.float32))
        q=info['target_step']['sent_target_rad']
        self.assertTrue(np.all(q>=e.profile.lower)&np.all(q<=e.profile.upper))
        self.assertTrue(np.isfinite(reward));self.assertTrue(np.isfinite(info['critic_observation']).all())
    def test_push_force_onset_and_expiration(self):
        e=self.env();e.dr.fixed_level=1.;e.reset();e.dr.next_push=0.
        e.dr.apply_push();self.assertEqual(e.dr.stats['push_events'],1)
        self.assertGreater(np.linalg.norm(e.exp.data.xfrc_applied[e.exp.base,:3]),0)
        self.assertLessEqual(np.max(np.abs(e.dr.push)),150.)
        e.elapsed=e.dr.push_until+.002;e.dr.apply_push()
        np.testing.assert_array_equal(e.exp.data.xfrc_applied[e.exp.base,:3],0)
    def test_curriculum_gating(self):
        c=Curriculum();good=[{'success':True,'tracking_score':1.,'status':'success'}]*32
        for _ in range(5):c.update(good,.15)
        self.assertAlmostEqual(c.level,.16)
        c.update(good,.10);self.assertLessEqual(c.level,.15)
        c=Curriculum();bad=[{'success':False,'tracking_score':0.,'status':'fall'}]*32
        for _ in range(2):c.update(bad,.15)
        self.assertAlmostEqual(c.level,.12)

if __name__=='__main__':unittest.main()
