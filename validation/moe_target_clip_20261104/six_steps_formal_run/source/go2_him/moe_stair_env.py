"""Offline MuJoCo environment for audited MoE actor fine-tuning (no robot I/O)."""
from __future__ import annotations
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import sys
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import test_robogauge_stairs as original
import test_robogauge_two_steps_wall as wall
from go2_him.official_guard import load_official_profile, evaluate_guard, compute_guard_cost, OfficialGuardAudit
from go2_him.moe_stair_rewards import compute_stair_rewards
from go2_him.calf_speed import compute_calf_speed_cost, calf_speed_summary
from go2_him.joint_recovery import compute_joint_recovery_cost
from go2_him.target_clip import TargetClipAccumulator, transform_target_action


def stair_environment_contract(stair_steps=2):
    """Geometry and unchanged task settings shared by training and evaluation.

    The final riser is the front edge of the upper platform, so N levels
    contain N-1 short treads followed by the original 90 cm landing.
    """
    if isinstance(stair_steps, (bool, np.bool_)) or not isinstance(stair_steps, (int, np.integer)) or stair_steps < 2:
        raise ValueError('stair_steps must be an integer >= 2')
    steps = int(stair_steps)
    start = (steps - 1) * .30
    end = start + .90
    return {
        'steps': steps, 'height': .15, 'tread': .30, 'width': .90,
        'platform': .90, 'start_distance': .50,
        'platform_start': start, 'platform_end': end, 'top_height': steps * .15,
        'wall_front': end, 'wall_thickness': .10, 'wall_width': 3., 'wall_height': 2.,
        'height_curriculum': {'evaluation_height': .15,
            'training_frontier_min': .10, 'training_frontier_max': .15,
            'multipliers': [1., .83, .67], 'probabilities': [.6, .2, .2],
            'rule': 'Existing frontier schedule and per-reset height sampling; every riser has the same sampled height.'},
        'physics_dt': .002, 'control_dt': .020, 'settle_seconds': 2.,
        'evaluation_vx': .6, 'training_vx_range': [.50, .60],
        'episode_timeout_seconds': 15., 'stable_stop_seconds': 2.,
        'platform_success': 'All feet inside the upper landing with 3 mm edge margin; foot surface clearance [-.025,.20] m; at least two supporting feet in .1 s; base clearance [.18,.60] m and tilt <45 deg.',
        'stable_stop': 'Platform success plus horizontal speed <.05 m/s, absolute vertical speed <.05 m/s, every world angular speed <.1 rad/s and tilt <15 deg for 2 s.',
    }


class JointGuardStop(RuntimeError):
    """Evaluation-only rejection before the next physical command is applied."""
    def __init__(self, boundary, elapsed, sample):
        self.details = {'boundary': boundary, 'seconds': elapsed, 'violations': {
            key: [sample.profile.names[i] for i in np.flatnonzero(value)]
            for key, value in sample.exceeded.items() if value is not None and np.any(value)}}
        super().__init__(str(self.details))


class MoEStairEnv:
    """Keep original 225/12 interface, original robot MJCF and 20/.5 PD.

    Legacy objectives retain their original motor-effort clipping. The optional
    target_clipped objective also bounds commanded angles and observes the
    submitted action; its additional cost measures the PRE-clipping request.
    Actual joint state is never clipped. The installed guard remains audited.
    """
    # Class defaults also describe the unchanged two-level legacy task.
    stair_steps = 2
    tread = .30
    width = .90
    platform = .90
    start_distance = .50

    @property
    def platform_start(self):
        return (self.stair_steps - 1) * self.tread

    @property
    def platform_end(self):
        return self.platform_start + self.platform

    def __init__(self, seed=0, evaluation=False, guard_objective='balanced', guard_mode='audit', stair_steps=2):
        if guard_objective not in ('balanced', 'calf_velocity', 'joint_recovery', 'target_clipped'):
            raise ValueError('Unknown guard objective')
        if guard_mode not in ('audit', 'strict') or (guard_mode == 'strict' and not evaluation):
            raise ValueError('Strict guard stopping is evaluation-only')
        self.environment_contract = stair_environment_contract(stair_steps)
        self.stair_steps = self.environment_contract['steps']
        self.guard_mode = guard_mode
        self.guard_objective = guard_objective
        self.physics_callback = None
        self.rng = np.random.default_rng(seed)
        self.evaluation = evaluation
        self.frontier = .15
        self.ramp = .2
        args = SimpleNamespace(
            config=original.POLICY_DIR/'params/deploy.yaml',
            policy=original.POLICY_DIR/'exported/policy.onnx',
            robot_xml=original.UPSTREAM/'RoboGauge/resources/robots/go2/go2.xml',
            physics_dt=.002, steps=self.stair_steps, height=.15, tread=self.tread, width=self.width,
            platform=self.platform, start_distance=self.start_distance, settle_seconds=2., vx=.6,
            torque_limit_mode='physical', video_width=640, video_height=480,
            viewer=False, video=False, camera_azimuth=315., camera_elevation=-30.)
        self.exp = wall.Experiment(args)
        self.exp.initialize({'yaw_rad':0., 'y_m':0.})
        self.initial_qpos = self.exp.data.qpos.copy()
        self.profile = load_official_profile(joint_names=original.JOINTS)
        self.calf_mask = np.array([name.endswith('_calf') for name in self.profile.names])
        self.stairs = [self.exp.mj.mj_name2id(self.exp.model,self.exp.mj.mjtObj.mjOBJ_GEOM,f'stair_{i}')
                       for i in range(1, self.stair_steps)]
        if any(gid < 0 for gid in self.stairs):
            raise ValueError('Scene is missing an intermediate stair geometry')
        self.stair = self.stairs[0]
        self.stop_phase = False
        self.reset()

    def terrain_height(self, xy):
        xy = np.asarray(xy)
        x, y = xy[...,0], xy[...,1]
        levels = np.searchsorted(np.arange(1, self.stair_steps) * self.tread, x, side='right') + 1
        return np.where((np.abs(y) <= self.width/2) & (x >= 0) & (x <= self.platform_end),
                        levels * self.height, 0.)

    def on_platform(self, foot, supported, tilt):
        """Original success geometry, translated to the final stair level."""
        e, d = self.exp, self.exp.data
        inside = bool(np.all(foot[:,0]-e.radii >= self.platform_start+.003)
                      and np.all(foot[:,0]+e.radii <= self.platform_end-.003)
                      and np.all(np.abs(foot[:,1])+e.radii <= self.width/2-.003))
        heights = bool(np.all(foot[:,2]-e.radii >= self.top-.025)
                       and np.all(foot[:,2]-e.radii <= self.top+.20))
        baseok = self.platform_start <= d.qpos[0] <= self.platform_end and .18 <= d.qpos[2]-self.top <= .60 and tilt < math.pi/4
        return inside and heights and len(supported) >= 2 and baseok

    def reset(self, sample=None, height=None, vx=None):
        e, m, d, mj = self.exp, self.exp.model, self.exp.data, self.exp.mj
        self.height = float(height if height is not None else
                            .15 if self.evaluation else self.frontier * self.rng.choice([1., .83, .67],p=[.6,.2,.2]))
        self.top = self.stair_steps*self.height
        for level, gid in enumerate(self.stairs, 1):
            m.geom_pos[gid,2] = m.geom_size[gid,2] = level*self.height/2
        m.geom_pos[e.platform_id,2] = m.geom_size[e.platform_id,2] = self.top/2
        e.top_z = self.top
        self.vx = float(vx if vx is not None else .6 if self.evaluation else self.rng.uniform(.50,.60))
        e.command_vx = self.vx
        mj.mj_resetData(m,d)
        d.qpos[:] = self.initial_qpos
        if sample is None:
            sample={'yaw_rad':float(self.rng.uniform(-.035,.035)), 'y_m':float(self.rng.uniform(-.02,.02))}
        rotation=d.xmat[e.base].reshape(3,3)
        # initial settled qpos is the reference; compose world yaw onto it.
        turn=np.array([math.cos(sample['yaw_rad']/2),0.,0.,math.sin(sample['yaw_rad']/2)])
        quat=np.empty(4)
        mj.mju_mulQuat(quat,turn,d.qpos[3:7].copy())
        d.qpos[3:7]=quat
        d.qpos[1]=sample['y_m']
        d.qvel[:]=0.
        mj.mj_forward(m,d)
        front=float((d.geom_xpos[e.feet[:2],0]+e.radii[:2]).max())
        d.qpos[0] += -self.start_distance-front
        mj.mj_forward(m,d)
        self.start_x=float(d.qpos[0]); self.last_x=self.start_x
        self.stop_phase=False
        self.platform_seen=False
        self.stable_time=0.; self.trunk_time=0.; self.side_time=0.; self.fall_time=0.
        self.elapsed=0.; self.episode_return=0.; self.roll_sq=0.; self.control_samples=0
        self.min_wall_clearance=math.inf
        self.audit=OfficialGuardAudit(self.profile)
        self.target_commands=TargetClipAccumulator(self.profile.names)
        self.recent=deque()
        self.last_action=np.zeros(12,dtype=np.float32)
        self.prev_action=self.last_action.copy()
        self.last_foot=d.geom_xpos[e.feet].copy()
        self.history={name:deque(maxlen=5) for name in original.OBS_ORDER}
        values=self.raw_observation()
        for name in self.history:
            self.history[name].extend([values[name].copy() for _ in range(5)])
        return self.observation(append=False)

    def raw_observation(self):
        e=self.exp
        e.policy.last_action=self.last_action.copy()
        e.command_vx=0. if self.stop_phase else self.vx
        return e.observation()

    def observation(self, append=True):
        values=self.raw_observation()
        chunks=[]
        for name in original.OBS_ORDER:
            if append: self.history[name].append(values[name].copy())
            cfg=self.exp.cfg['observations'][name]
            chunks.extend(np.clip(v,*cfg['clip'])*np.asarray(cfg['scale']) for v in self.history[name])
        return np.concatenate(chunks).astype(np.float32)

    def step(self, action):
        e,m,d,mj=self.exp,self.exp.model,self.exp.data,self.exp.mj
        action=np.asarray(action,dtype=np.float32)
        if action.shape!=(12,) or not np.isfinite(action).all():
            raise FloatingPointError('Nonfinite or invalid action')
        self.prev_action=self.last_action.copy()
        raw_target=action*e.policy.scale+e.policy.offset
        raw_excess=np.maximum(np.maximum(self.profile.lower-raw_target,
                                         raw_target-self.profile.upper),0.)
        raw_target_cost=0.
        if self.guard_objective == 'target_clipped':
            transformed=transform_target_action(action,e.policy.scale,e.policy.offset,
                                                self.profile.lower,self.profile.upper)
            raw_target=transformed.raw_target
            raw_excess=transformed.raw_excess
            target=transformed.sent_target
            self.last_action=transformed.sent_action.copy()
            raw_target_cost=float(transformed.cost_rate)
            reward_action=self.last_action
        else:
            self.last_action=action.copy()
            target=np.clip(raw_target,e.policy.clip[:,0],e.policy.clip[:,1])
            reward_action=action
        if not hasattr(self,'target_commands'):
            self.target_commands=TargetClipAccumulator(self.profile.names)
        # One record per control proposal, independent of decimation. In strict
        # evaluation a bounded proposal can still be rejected before application.
        self.target_commands.record(raw_target,target,self.profile.lower,self.profile.upper)
        guard_cost=0.; status=None; new_platform=False
        q_states=[];dq_states=[];applied_states=[]
        stop_before=self.stop_phase
        command_before=0. if stop_before else self.vx
        for substep in range(e.decimation):
            qpre=d.qpos[e.qadr].copy(); dqpre=d.qvel[e.vadr].copy()
            if self.guard_mode == 'strict':
                pre=evaluate_guard(self.profile,qpre,dqpre,target,e.kp,e.kd)
                if bool(pre.hard_violation):
                    # Preserve earlier substeps. The rejected boundary has no
                    # applied torque: it must be unknown, never fabricated zero.
                    if q_states:
                        self.audit.record(evaluate_guard(self.profile,np.stack(q_states),np.stack(dq_states),
                            target,e.kp,e.kd,applied_torque=np.stack(applied_states)))
                    self.audit.record(pre,terminal=True)
                    raise JointGuardStop('pre_command',self.elapsed,pre)
            requested=e.kp*(target-qpre)-e.kd*dqpre
            applied=np.clip(requested,-e.physical_effort,e.physical_effort)
            d.ctrl[e.aids]=applied
            oldtime=float(d.time)
            warnings_before={key:int(d.warning[idx].number) for key,idx in e.numeric_warnings.items()}
            mj.mj_step(m,d)
            badwarn=any(int(d.warning[idx].number)>warnings_before[key] for key,idx in e.numeric_warnings.items())
            if badwarn or not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or not math.isclose(float(d.time),oldtime+.002,abs_tol=1e-8):
                if q_states:
                    self.audit.record(evaluate_guard(self.profile,np.stack(q_states),np.stack(dq_states),
                        target,e.kp,e.kd,applied_torque=np.stack(applied_states)))
                # Retain the last known pre-boundary. Its failed integration
                # has unknown post-state and applied effort, never zero values.
                self.audit.record(evaluate_guard(self.profile,qpre,dqpre,target,e.kp,e.kd),terminal=True)
                raise FloatingPointError('MuJoCo numeric failure')
            actual=d.actuator_force[e.aids].copy()
            mj.mj_forward(m,d)
            self.elapsed+=.002
            if self.physics_callback is not None:
                self.physics_callback(self)
            # Collect both boundaries of EVERY physical step. Evaluate in one
            # vectorized call after the window, without subsampling evidence.
            q_states.extend([qpre,d.qpos[e.qadr].copy()])
            dq_states.extend([dqpre,d.qvel[e.vadr].copy()])
            applied_states.extend([actual,actual.copy()])
            if self.guard_mode == 'strict':
                post=evaluate_guard(self.profile,q_states[-1],dq_states[-1],target,e.kp,e.kd,
                                    applied_torque=actual)
                if bool(post.hard_violation):
                    terminal=np.zeros(len(q_states),dtype=bool); terminal[-1]=True
                    self.audit.record(evaluate_guard(self.profile,np.stack(q_states),np.stack(dq_states),
                        target,e.kp,e.kd,applied_torque=np.stack(applied_states)),terminal=terminal)
                    raise JointGuardStop('post_integration',self.elapsed,post)
            rot=d.xmat[e.base].reshape(3,3)
            tilt=math.acos(float(np.clip(rot[2,2],-1,1)))
            roll=math.atan2(rot[2,1],rot[2,2])
            self.roll_sq+=roll*roll;self.control_samples+=1
            touching,platform_feet,trunk,side_floor=e.contacts()
            self.recent.append((self.elapsed,platform_feet))
            while self.recent and self.recent[0][0] < self.elapsed-.1: self.recent.popleft()
            supported=set().union(*(v for _,v in self.recent))
            foot=d.geom_xpos[e.feet]
            on_platform=self.on_platform(foot,supported,tilt)
            if on_platform and not self.platform_seen:
                self.platform_seen=True;new_platform=True;self.stop_phase=True
            clearance=e.wall_clearance()['signed_clearance_m']
            self.min_wall_clearance=min(self.min_wall_clearance,clearance)
            hits,_=e.wall_contacts(self.elapsed)
            world_w=rot@d.qvel[3:6]
            stable=on_platform and np.linalg.norm(d.qvel[:2])<.05 and abs(d.qvel[2])<.05 and np.max(np.abs(world_w))<.1 and tilt<math.radians(15)
            self.stable_time=self.stable_time+.002 if stable else 0.
            self.trunk_time=self.trunk_time+.002 if trunk else 0.
            self.side_time=self.side_time+.002 if side_floor or abs(d.qpos[1])>self.width/2 else 0.
            self.fall_time=self.fall_time+.002 if tilt>math.pi/3 or d.qpos[2]<.12 else 0.
            if hits: status='wall_contact'
            elif self.trunk_time>=.05: status='trunk_contact'
            elif self.side_time>=.1: status='side_exit'
            elif self.fall_time>=.1: status='fall'
            elif self.stable_time>=2.-1e-8: status='success'
            elif self.elapsed>=15.: status='timeout'
            if status is not None: break
        evidence=evaluate_guard(self.profile,np.stack(q_states),np.stack(dq_states),target,e.kp,e.kd,
                                applied_torque=np.stack(applied_states))
        terminal=np.zeros(len(q_states),dtype=bool)
        terminal[-1]=status is not None
        self.audit.record(evidence,terminal=terminal)
        # Physical evidence uses the submitted target. The independent target
        # penalty below uses the original affine request, before ANY clipping.
        cost_fn = {'calf_velocity':compute_calf_speed_cost,'balanced':compute_guard_cost,
                   'joint_recovery':compute_joint_recovery_cost,
                   'target_clipped':compute_guard_cost}[self.guard_objective]
        cost=cost_fn(evidence,physics_axis=0)
        guard_cost=float(cost.total)+raw_target_cost
        elapsed_dt=(substep+1)*.002
        rot=d.xmat[e.base].reshape(3,3)
        foot=d.geom_xpos[e.feet].copy()
        foot_vel=(foot-self.last_foot)/elapsed_dt
        self.last_foot=foot.copy()
        touching,_,_,_=e.contacts()
        bodyvel=rot.T@d.qvel[:3]
        state={
            'dt':elapsed_dt,'dx':float(d.qpos[0])-self.last_x,
            'body_vxy':bodyvel[:2], 'body_w':d.sensordata[e.gyro_adr:e.gyro_adr+3].copy(),
            'commands':np.array([command_before,0.,0.]), 'gravity':rot.T@np.array([0.,0.,-1.]),
            'body_vz':float(bodyvel[2]),
            'foot_clearance':foot[:,2]-e.radii-self.terrain_height(foot[:,:2]),
            'foot_velocity':foot_vel, 'foot_contact':np.array([j in touching for j in range(4)]),
            'jointq':d.qpos[e.qadr].copy(),'q0':e.policy.q0,
            'actions':reward_action,'prev_actions':self.prev_action,
            'base_height':float(d.qpos[2]-self.terrain_height(d.qpos[:2])),
            'stop_phase':stop_before,'new_platform_event':new_platform,
            'success_event':status=='success','failure_event':status is not None and status!='success',
        }
        rewards=compute_stair_rewards(state)
        reward=float(rewards['step_reward'])-self.ramp*guard_cost*elapsed_dt
        self.last_x=float(d.qpos[0]);self.episode_return+=reward
        obs=self.observation()
        calf_exceeded=evidence.exceeded['velocity'][...,self.calf_mask]
        calf_values=np.abs(evidence.values['velocity'][...,self.calf_mask])
        calf_ratios=evidence.ratios['velocity'][...,self.calf_mask]
        calf_names=[name for name in self.profile.names if name.endswith('_calf')]
        info={'guard_cost':guard_cost,'reward_terms':rewards,
              'raw_target_cost_rate':raw_target_cost,'target_violation_rad':raw_excess.copy(),
              'target_step':{'raw_target_rad':raw_target.copy(),'sent_target_rad':target.copy()},
              'calf_speed':{'excess_count':int(calf_exceeded.sum()),'sample_count':int(calf_exceeded.size),
                            'peak_absolute_value':float(calf_values.max()),'max_ratio':float(calf_ratios.max()),
                            'per_joint_peak_rad_s':dict(zip(calf_names,calf_values.max(axis=0).tolist()))}}
        if self.guard_objective == 'joint_recovery':
            info['guard_cost_components']={key:float(value) for key,value in cost.bounded.items()}
            info['joint_guard_update']={key:{'excess_count':int(evidence.exceeded[key].sum()),
                'sample_count':int(evidence.exceeded[key].size),
                'max_excess':float(evidence.excess[key].max()),
                'peak_absolute_value':float(np.abs(evidence.values[key]).max()),
                'per_joint':{name:{'excess_count':int(evidence.exceeded[key][...,i].sum()),
                    'sample_count':int(evidence.exceeded[key][...,i].size),
                    'max_excess':float(evidence.excess[key][...,i].max()),
                    'min_value':float(evidence.values[key][...,i].min()),
                    'max_value':float(evidence.values[key][...,i].max())}
                    for i,name in enumerate(self.profile.names)}}
                for key in ('velocity','position','target_position','tracking','requested_torque','applied_torque')}
        if status:
            guard_summary=self.audit.summary()
            info['episode']={'status':status,'success':status=='success','return':self.episode_return,
                'seconds':self.elapsed,'height':self.height,'command_vx':self.vx,
                'stair_steps':self.stair_steps,
                'forward_m':self.last_x-self.start_x,'roll_rms_rad':math.sqrt(self.roll_sq/self.control_samples),
                'min_wall_clearance_m':self.min_wall_clearance,'guard':guard_summary,
                'target_commands':self.target_commands.summary(),
                'calf_speed':calf_speed_summary(guard_summary),'guard_objective':self.guard_objective,
                'audit_sampling':'pre-send and post-integration boundaries; two samples per physics step; post PD is the request implied by the new state, not an already sent command; applied is the actual actuator force copied before mj_forward',
                'external_stop_command':True}
            info['terminal_observation']=obs
            obs=self.reset()
        return obs,reward,status is not None,info


def env_worker(conn, seed, count, stair_steps=2):
    """One worker owns independent MuJoCo data/model pairs. No torch/CUDA here."""
    envs=[MoEStairEnv(seed+i, stair_steps=stair_steps) for i in range(count)]
    conn.send(np.stack([e.observation(append=False) for e in envs]))
    try:
        while True:
            operation,payload=conn.recv()
            if operation=='close': break
            if operation=='settings':
                objective=payload.get('guard_objective','balanced')
                if objective not in ('balanced','calf_velocity','joint_recovery','target_clipped'):raise ValueError('Unknown guard objective')
                for e in envs:
                    e.frontier=payload['frontier'];e.ramp=payload['ramp'];e.guard_objective=objective
                conn.send(True)
            elif operation=='step':
                results=[e.step(a) for e,a in zip(envs,payload)]
                conn.send((np.stack([r[0] for r in results]),np.array([r[1] for r in results],np.float32),
                           np.array([r[2] for r in results],bool),[r[3] for r in results]))
            else: raise ValueError(operation)
    except EOFError: pass
    finally: conn.close()
