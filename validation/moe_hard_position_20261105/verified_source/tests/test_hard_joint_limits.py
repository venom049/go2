"""Independent math, adversarial physical boundaries, and state rollback."""
import unittest
from unittest.mock import patch
import numpy as np

from go2_him.hard_joint_limits import mass_project_velocity, position_constraint_contract
from go2_him.moe_stair_env import MoEStairEnv


class ProjectionTest(unittest.TestCase):
    def test_coupled_impulse_moves_free_coordinate_and_dissipates_energy(self):
        mass=np.array([[2.,1.],[1.,2.]])
        v0=np.array([1.,0.])
        v,impulse=mass_project_velocity(mass,v0,[0],[-np.inf],[0.])
        np.testing.assert_allclose(v,[0.,.5],atol=1e-14)
        np.testing.assert_allclose(impulse,[-1.5,0.],atol=1e-14)
        self.assertLess(v@mass@v,v0@mass@v0)

    def test_random_coupled_qp_kkt_and_energy(self):
        rng=np.random.default_rng(20261105)
        for _ in range(120):
            a=rng.normal(size=(18,18));m=a.T@a+np.eye(18)*.2
            v0=rng.normal(size=18)*40
            lo=-rng.uniform(0,10,12);hi=rng.uniform(0,10,12)
            indices=np.arange(6,18)
            v,j=mass_project_velocity(m,v0,indices,lo,hi)
            self.assertTrue(np.all(v[indices]>=lo-1e-10))
            self.assertTrue(np.all(v[indices]<=hi+1e-10))
            np.testing.assert_allclose(j[:6],0.,atol=2e-10)
            free=(v[indices]>lo+1e-8)&(v[indices]<hi-1e-8)
            np.testing.assert_allclose(j[indices][free],0.,atol=2e-10)
            self.assertTrue(np.all(j[indices][v[indices]<=lo+1e-8]>=-2e-10))
            self.assertTrue(np.all(j[indices][v[indices]>=hi-1e-8]<=2e-10))
            self.assertLessEqual(v@m@v,v0@m@v0+1e-8)

    def test_feasible_velocity_is_bitwise_unchanged(self):
        v=np.array([.1234,-.321])
        result,j=mass_project_velocity(np.eye(2),v,[0,1],[-1,-1],[1,1])
        np.testing.assert_array_equal(result,v);np.testing.assert_array_equal(j,[0,0])

    def test_invalid_inputs_reject(self):
        for v in ([np.nan,0],[np.inf,0]):
            with self.assertRaises(ValueError):
                mass_project_velocity(np.eye(2),v,[0],[-1],[1])
        with self.assertRaises(ValueError): position_constraint_contract('unknown')


class PhysicalHardStopTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env=MoEStairEnv(seed=42,evaluation=True,stair_steps=6,
                          guard_objective='target_clipped',position_constraint='hard')

    @classmethod
    def tearDownClass(cls):cls.env.exp.close()

    def setUp(self):
        self.env.reset(sample={'yaw_rad':0.,'y_m':0.},height=.15,vx=.6)

    def step_direct(self):
        e=self.env.exp
        e.hard_joint_limits.step(e.data)
        e.mj.mj_forward(e.model,e.data)
        q=e.data.qpos[e.qadr]
        self.assertTrue(np.all(q>=self.env.profile.lower))
        self.assertTrue(np.all(q<=self.env.profile.upper))

    def test_model_bounds_and_startup_match_named_profile(self):
        e=self.env.exp;p=self.env.profile
        np.testing.assert_array_equal(e.model.jnt_range[e.jids,0],p.lower)
        np.testing.assert_array_equal(e.model.jnt_range[e.jids,1],p.upper)
        self.assertTrue(np.all(e.model.jnt_limited[e.jids]))
        self.assertEqual(self.env.startup_position_constraints['physics_steps'],1000)
        self.assertEqual(self.env.startup_position_constraints['accepted_excess_count'],0)

    def test_each_joint_both_stops_extreme_outward_velocity(self):
        e=self.env.exp;p=self.env.profile
        for i in range(12):
            for sign in (-1,1):
                with self.subTest(joint=p.names[i],sign=sign):
                    self.setUp()
                    e.data.qpos[2]=2.  # Isolate joint stops from floor contacts.
                    boundary=p.lower[i] if sign<0 else p.upper[i]
                    e.data.qpos[e.qadr[i]]=boundary-sign*1e-6
                    e.data.qvel[:]=0.;e.data.qvel[e.vadr[i]]=sign*100.
                    e.data.ctrl[:]=0.
                    e.mj.mj_forward(e.model,e.data)
                    self.step_direct()
                    self.assertLessEqual(sign*e.data.qvel[e.vadr[i]],1e-10)
                    self.assertGreater(e.hard_joint_limits.summary()['intervention_steps'],0)
                    self.assertGreater(e.hard_joint_limits.summary()['stop_generalized_impulse_max_abs'],0.)

    def test_simultaneous_limits_and_sustained_outward_torque(self):
        e=self.env.exp;p=self.env.profile
        e.data.qpos[2]=3.
        signs=np.array([1.,-1.]*6)
        e.data.qpos[e.qadr]=np.where(signs<0,p.lower+1e-6,p.upper-1e-6)
        e.data.qvel[:]=0.;e.data.qvel[e.vadr]=signs*80.
        e.data.ctrl[e.aids]=signs*e.physical_effort
        e.mj.mj_forward(e.model,e.data)
        for _ in range(300): self.step_direct()
        summary=e.hard_joint_limits.summary()
        self.assertGreater(summary['predictor_excess_count'],0)
        self.assertEqual(summary['physics_steps'],300)
        self.assertEqual(summary['accepted_excess_count'],0)

    def test_invalid_initial_state_is_rejected_not_silently_repaired(self):
        e=self.env.exp
        e.data.qpos[e.qadr[2]]=self.env.profile.upper[2]+.001
        q=e.data.qpos.copy();t=e.data.time
        with self.assertRaises(FloatingPointError):e.hard_joint_limits.step(e.data)
        np.testing.assert_array_equal(e.data.qpos,q);self.assertEqual(e.data.time,t)

    def test_failed_solver_restores_time_position_velocity(self):
        e=self.env.exp
        e.data.qpos[e.qadr[2]]=self.env.profile.upper[2]-1e-6
        e.data.qvel[e.vadr[2]]=100.
        e.mj.mj_forward(e.model,e.data)
        q,v,t=e.data.qpos.copy(),e.data.qvel.copy(),e.data.time
        with patch('go2_him.hard_joint_limits.mass_project_velocity',side_effect=FloatingPointError('test')):
            with self.assertRaises(FloatingPointError):e.hard_joint_limits.step(e.data)
        np.testing.assert_array_equal(e.data.qpos,q)
        np.testing.assert_array_equal(e.data.qvel,v)
        self.assertEqual(e.data.time,t)
        self.assertEqual(e.hard_joint_limits.summary()['physics_steps'],0)

    def test_policy_observation_and_physics_callback_only_see_legal_state(self):
        e=self.env.exp;seen=[]
        def inspect(env):
            q=e.data.qpos[e.qadr].copy();seen.append(q)
            self.assertTrue(np.all(q>=env.profile.lower))
            self.assertTrue(np.all(q<=env.profile.upper))
        self.env.physics_callback=inspect
        try:
            for _ in range(15):
                obs,_,done,info=self.env.step(np.full(12,10.,np.float32))
                self.assertTrue(np.isfinite(obs).all())
                if done:break
        finally:self.env.physics_callback=None
        self.assertGreater(len(seen),10)
        self.assertEqual(self.env.audit.summary()['metrics']['position']['excess_count'],0)


if __name__=='__main__':unittest.main()
