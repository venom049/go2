"""MuJoCo adaptation of saifahmadgit stair DR, pinned 0830f3e.

Each environment owns its model; reset-only parameter randomization avoids
Genesis's shared-model/global-reset restriction. Actor command history stays
observable: lag/noise are downstream actuator uncertainty, never leaked back.
All downstream targets are projected again before PD. No hardware transport.
"""
from __future__ import annotations
import copy
import numpy as np

REFERENCE = 'https://github.com/saifahmadgit/go2-sim2real-locomotion-rl/tree/0830f3e3582acd5c9b3a5c09186d4f028f54d9dd'
CONFIG = {
    'reference':REFERENCE, 'robot_variant':'Go2 EDU, no extra payload (user confirmed)',
    'official_model_commit':'ae6a8403e272733e9996ef59990880330496177f',
    'friction': [[.6,.8],[.3,1.25]],
    'kp_factor': [[.95,1.05],[.8,1.2]], 'kd_factor': [[.95,1.05],[.8,1.2]],
    'motor_strength': [[.97,1.03],[.9,1.1]],
    'base_mass_shift_kg':[[-.2,.5],[-1.,3.]],
    'base_com_shift_m':[[-.005,.005],[-.03,.03]],
    'hip_mass_shift_kg':[[-.1,.1],[-.5,.5]],
    'gravity_offset_m_s2':[[-.2,.2],[-1.,1.]],
    'observation_gaussian_std':{'base_ang_vel':.2,'projected_gravity':.05,'joint_pos_rel':.01,'joint_vel_rel':1.5},
    'actuator_target_noise_std_rad':.1,
    'actuator_delay_steps':[0,1], 'delay_one_step_probability':'.5 * episode_level',
    'push_xy_max_N':150.,'push_duration_s':[.05,.2],'push_start_level':.3,
    'nominal_episode_probability':.1,'current_level_probability_among_randomized':.8,
    'easy_level_upper':.5,
    'history':'bounded current policy command, before unobservable actuator lag/noise',
    'target_bounds':'clip command; delay and target noise; re-clip applied target',
    'inertia':'scale principal inertia with mass at fixed shape; shift COM in body inertial frame',
    'curriculum':{'initial_level':.15,'step_up':.01,'step_down':.03,'ema_alpha':.03,
                  'ready_success':.6,'ready_tracking':.45,'ready_fall':.35,'hard_fall':.4,
                  'ready_streak':5,'hard_streak':2,'cooldown_checks':5,
                  'minimum_new_episodes':32,'phase1_frontier_below_m':.125,'phase1_level_cap':.15},
    'differences_from_reference':['MuJoCo independent models and named joints',
        'fixed nominal PD 20/.5 with multiplicative uncertainty; no new stiffness action head',
        'success replaces survival timeout in curriculum because timeout is failure here',
        'delay probability ramps continuously rather than rounding maximum delay',
        'push phase sampled per episode so short stair episodes experience pushes',
        '10 percent nominal episodes; calibrated hardware parameters unavailable',
        'new asymmetric critic; original actor input/output and frozen encoder retained'],
}
PRIVILEGED_EXTRA = 145

class Curriculum:
    def __init__(self):
        self.level=.15; self.ema=None; self.ready=self.hard=self.cooldown=0; self.pending=[]; self.checks=0
    def update(self, episodes, frontier):
        self.pending.extend(episodes)
        if len(self.pending)<32:return
        rows=self.pending;self.pending=[];self.checks+=1
        value=np.array([np.mean([r['success'] for r in rows]),
            np.mean([r.get('tracking_score',0.) for r in rows]),
            np.mean([r['status'] in ('fall','trunk_contact','side_exit') for r in rows])])
        self.ema=value if self.ema is None else .97*self.ema+.03*value
        self.cooldown=max(0,self.cooldown-1)
        self.ready=self.ready+1 if self.ema[0]>=.6 and self.ema[1]>=.45 and self.ema[2]<=.35 else 0
        self.hard=self.hard+1 if self.ema[2]>=.4 else 0
        if self.hard>=2:
            self.level=max(0.,self.level-.03);self.ready=self.hard=0;self.cooldown=5
        elif self.ready>=5 and self.cooldown==0:
            self.level=min(1.,self.level+.01);self.ready=0;self.cooldown=5
        if frontier<.125:self.level=min(.15,self.level)
    def summary(self):
        return dict(level=self.level,checks=self.checks,ema=None if self.ema is None else self.ema.tolist(),
                    pending_episodes=len(self.pending),ready_streak=self.ready,hard_streak=self.hard)

class DomainRandomization:
    def __init__(self, env, enabled=False, level=.15):
        self.env=env;self.enabled=enabled;self.level=level;self.fixed_level=None
        self.rng=np.random.default_rng(env.seed+6700417)
        e=env.exp;m=e.model
        self.hips=np.array([e.mj.mj_name2id(m,e.mj.mjtObj.mjOBJ_BODY,f'{leg}_hip') for leg in ('FL','FR','RL','RR')])
        if np.any(self.hips<0):raise ValueError('Missing named hip bodies')
        self.nominal={k:getattr(m,k).copy() for k in ('body_mass','body_inertia','body_ipos','geom_friction')}
        self.gravity=m.opt.gravity.copy();self.kp=e.kp.copy();self.kd=e.kd.copy()
        self.episode={};self.push=np.zeros(3);self.previous_target=e.policy.offset.copy()
    def restore(self):
        e=self.env.exp
        for k,v in self.nominal.items():getattr(e.model,k)[:]=v
        e.model.opt.gravity[:]=self.gravity;e.kp=self.kp.copy();e.kd=self.kd.copy()
    def reset(self):
        self.restore();e=self.env.exp;m=e.model;g=self.rng
        active=self.enabled and (self.fixed_level is not None or g.random()>=.1)
        t=float(self.fixed_level if self.fixed_level is not None else self.level)
        if active and self.fixed_level is None and g.random()>=.8:t=float(g.uniform(0,min(t,.5)))
        if not active:t=0.
        def sample(name,size=None):
            bounds=np.array(CONFIG[name]);lo,hi=bounds[0]+t*(bounds[1]-bounds[0])
            return g.uniform(lo,hi,size)
        self.active=active;self.t=t
        self.kpf=sample('kp_factor',12) if active else np.ones(12)
        self.kdf=sample('kd_factor',12) if active else np.ones(12)
        self.strength=sample('motor_strength',12) if active else np.ones(12)
        self.mass=float(sample('base_mass_shift_kg')) if active else 0.
        self.com=sample('base_com_shift_m',3) if active else np.zeros(3)
        self.legmass=sample('hip_mass_shift_kg',4) if active else np.zeros(4)
        self.go=sample('gravity_offset_m_s2',3) if active else np.zeros(3)
        self.mu=float(sample('friction')) if active else float(m.geom_friction[e.feet[0],0])
        self.delay=int(g.random()<.5*t) if active else 0
        e.kp=self.kp*self.kpf;e.kd=self.kd*self.kdf
        if active:
            # Foot priority=1 means randomizing floor alone would not change contact mu.
            m.geom_friction[:,0]=self.mu
            for body,shift in [(e.base,self.mass),*zip(self.hips,self.legmass)]:
                old=self.nominal['body_mass'][body];new=old+shift
                if new<=0:raise ValueError('Invalid randomized mass')
                m.body_mass[body]=new;m.body_inertia[body]=self.nominal['body_inertia'][body]*(new/old)
            m.body_ipos[e.base]+=self.com;m.opt.gravity[:]=self.gravity+self.go
        e.mj.mj_setConst(m,e.data)
        self.previous_target=e.policy.offset.copy();self.push=np.zeros(3);self.push_until=0.
        self.push_scale=max(0.,(t-.3)/.7) if active else 0.
        self.push_interval=10.-5.*self.push_scale
        self.next_push=float(g.uniform(1.,self.push_interval))
        self.stats=dict(control_steps=0,delayed_steps=0,push_events=0,push_physics_steps=0,
                        noisy_target_reclipped_joints=0,applied_target_excess=0,max_push_N=0.)
        self.episode=dict(active=active,level=t,friction=self.mu,kp_factors=self.kpf.tolist(),
            kd_factors=self.kdf.tolist(),motor_strength=self.strength.tolist(),
            base_mass_shift_kg=self.mass,base_com_shift_m=self.com.tolist(),hip_mass_shift_kg=self.legmass.tolist(),
            gravity_offset_m_s2=self.go.tolist(),delay_steps=self.delay,total_mass_kg=float(m.body_mass.sum()))
    def noisy_observation(self, values):
        if not self.active:return values
        result={k:v.copy() for k,v in values.items()}
        for k,std in CONFIG['observation_gaussian_std'].items():
            result[k]+=self.rng.normal(0,std*self.t,result[k].shape)
        return result
    def actuator_target(self, command):
        self.stats['control_steps']+=1
        target=self.previous_target.copy() if self.delay else command.copy()
        self.previous_target=command.copy()
        if self.delay:self.stats['delayed_steps']+=1
        if self.active:target+=self.rng.normal(0,.1*self.t,12)
        p=self.env.profile
        self.stats['noisy_target_reclipped_joints']+=int(((target<p.lower)|(target>p.upper)).sum())
        target=np.clip(target,p.lower,p.upper)
        self.stats['applied_target_excess']+=int(((target<p.lower)|(target>p.upper)).sum())
        return target
    def apply_push(self):
        e=self.env.exp;t=self.env.elapsed
        if self.push_scale>0 and t>=self.next_push:
            self.push[:2]=self.rng.uniform(-150*self.push_scale,150*self.push_scale,2);self.push[2]=0.
            self.push_until=t+float(self.rng.uniform(.05,.2));self.next_push=t+self.push_interval
            self.stats['push_events']+=1
            self.stats['max_push_N']=max(self.stats['max_push_N'],float(np.linalg.norm(self.push)))
        if t>=self.push_until:self.push[:]=0.
        e.data.xfrc_applied[e.base,:3]=self.push
        if np.any(self.push):self.stats['push_physics_steps']+=1
    def summary(self):return {**copy.deepcopy(self.episode),'realized':dict(self.stats)}
    def privileged(self, actor_observation):
        e=self.env.exp;d=e.data;rot=d.xmat[e.base].reshape(3,3)
        # 11x7 terrain samples, aligned with body yaw; critic only.
        grid=np.array([(x,y) for x in np.linspace(-.5,.5,11) for y in np.linspace(-.3,.3,7)])
        yaw=np.arctan2(rot[1,0],rot[0,0]);c,s=np.cos(yaw),np.sin(yaw)
        xy=grid@np.array([[c,s],[-s,c]])+d.qpos[:2]
        heights=self.env.terrain_height(xy)-d.qpos[2]
        extra=np.concatenate([rot.T@d.qvel[:3],[self.mu],self.kpf-1,self.kdf-1,self.strength-1,
            [self.mass/3],self.com/.03,self.legmass/.5,self.go,self.push/150,[self.delay],
            (self.previous_target-e.policy.offset)/e.policy.scale,[self.t],heights])
        assert extra.shape==(PRIVILEGED_EXTRA,)
        return np.concatenate([actor_observation,extra]).astype(np.float32)
