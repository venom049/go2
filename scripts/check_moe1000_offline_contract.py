"""File-only contract audit and one nominal MuJoCo rollout; no robot imports/I/O."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import onnxruntime as ort
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MUJOCO_GL', 'egl')

CPP = r'''
#include "MoePolicy.h"
#include <iostream>
#include <iomanip>
int main() {
 std::cout<<std::setprecision(17); char mode;
 go2_moe::History history;
 while(std::cin>>mode) {
  if(mode=='a') {
   go2_moe::Action a{}; for(auto& x:a)std::cin>>x;
   auto c=go2_moe::decode(a); for(auto x:c.q)std::cout<<x<<' '; std::cout<<'\n';
  } else if(mode=='o') {
   go2_first_test::GuardInput s; s.state_available=true;
   for(auto& x:s.q)std::cin>>x; for(auto& x:s.dq)std::cin>>x;
   for(auto& x:s.quaternion_wxyz)std::cin>>x; for(auto& x:s.gyro)std::cin>>x;
   go2_moe::Action a;for(auto& x:a)std::cin>>x;float vx;std::cin>>vx;
   auto obs=history.append(go2_moe::frame(s,vx,a));
   for(auto x:obs)std::cout<<x<<' ';std::cout<<'\n';
  } else if(mode=='n') {
   int rejected=0;
   for(float bad:{NAN,INFINITY,-INFINITY}) {
    go2_moe::Action a{};a[0]=bad;
    try{go2_moe::decode(a);}catch(const std::exception&){++rejected;}
   }
   std::cout<<rejected<<'\n';
  }
 }
}
'''

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); out=args.out; out.mkdir(parents=True,exist_ok=False)
    bundle=ROOT/'_exports/go2-moe1000-20261104'
    cfg=yaml.safe_load((bundle/'params/deploy.yaml').read_text())
    ac=cfg['actions']['JointPositionAction']; offset=np.array(ac['offset'],dtype=np.float32)
    scale=np.array(ac['scale'],dtype=np.float32); mapping=np.array(cfg['joint_ids_map'])
    clip=np.array(ac['clip']); profile=json.loads((bundle/'limit_profile.json').read_text())
    limits=profile['limits']; names=[limits[i]['joint'] for i in mapping]
    lower=np.array([limits[i]['position_lower_rad'] for i in mapping],dtype=np.float32)
    upper=np.array([limits[i]['position_upper_rad'] for i in mapping],dtype=np.float32)
    decode=lambda a:np.clip(a*scale+offset,clip[:,0],clip[:,1])
    report={'model':'moe1000','model_sha256':hashlib.sha256((bundle/'exported/policy.onnx').read_bytes()).hexdigest(),
            'robot_connected':False,'weights_or_controller_modified':False,
            'scope':'Local source snapshot, offline fixtures and one nominal simulated first episode; no live firmware or installed-binary verification'}
    source=out/'contract_probe.cpp'; source.write_text(CPP); binary=out/'contract_probe'
    subprocess.run(['g++','-std=c++17','-O2','-I',str(ROOT/'deploy/moe1000_controller/overlay/include'),str(source),'-o',str(binary)],check=True)
    actions=np.vstack([np.zeros(12),np.eye(12)*.1,-np.eye(12)*.1,np.full(12,1000.),np.full(12,-1000.)]).astype(np.float32)
    lines=['a '+' '.join(map(str,a)) for a in actions]
    native=np.array([np.fromstring(x,sep=' ') for x in subprocess.check_output([str(binary)],input='\n'.join(lines)+'\n',text=True).splitlines()])
    expected=np.zeros_like(native);expected[:,mapping]=decode(actions)
    delta=native[1:13]-native[0]
    expected_delta=np.zeros((12,12));expected_delta[np.arange(12),mapping]=.025
    report['action_mapping']={'native_python_max_abs_error_rad':float(np.max(np.abs(native-expected))),
       'zero_returns_default':bool(np.allclose(native[0,mapping],offset,atol=1e-7,rtol=0)),
       'all_12_positive_and_negative_axes_pass':bool(np.allclose(delta,expected_delta,atol=1e-7,rtol=0) and np.allclose(native[13:25]-native[0],-expected_delta,atol=1e-7,rtol=0)),
       'nonfinite_cases_rejected':int(subprocess.check_output([str(binary)],input='n\n',text=True)),
       'zero_pose_inside_profile':bool(np.all((offset>=lower)&(offset<=upper))),
       'legal_raw_action_intervals':{n:[float(l),float(u)] for n,l,u in zip(names,(lower-offset)/scale,(upper-offset)/scale)}}
    # Exercise actual C++ observation builder/history with changing states.
    rng=np.random.default_rng(42); frames=[]; expected_obs=[]; lines=[]
    for t in range(8):
        q=offset+rng.normal(0,.05,12);dq=rng.normal(0,.3,12);quat=rng.normal(size=4);quat/=np.linalg.norm(quat)
        gyro=rng.normal(size=3);prev=rng.normal(size=12).astype(np.float32);vx=np.float32(.6)
        q_sdk=np.zeros(12);q_sdk[mapping]=q;dq_sdk=np.zeros(12);dq_sdk[mapping]=dq
        lines.append('o '+' '.join(map(str,np.r_[q_sdk,dq_sdk,quat,gyro,prev,vx])))
        w,x,y,z=quat;gravity=np.array([2*(w*y-x*z),-2*(w*x+y*z),-1+2*(x*x+y*y)])
        values=[gyro,gravity,np.array([vx,0,0]),q-offset,dq,prev]
        terms=list(cfg['observations'].values())
        frame=[(np.clip(v,*c['clip'])*np.array(c['scale'])).astype(np.float32) for v,c in zip(values,terms)]
        if not frames: frames=[frame]*5
        else: frames=(frames+[frame])[-5:]
        expected_obs.append(np.concatenate([f[i] for i in range(6) for f in frames]))
    native_obs=np.array([np.fromstring(x,sep=' ') for x in subprocess.check_output([str(binary)],input='\n'.join(lines)+'\n',text=True).splitlines()])
    report['observation_contract']={'cases':8,'native_python_max_abs_error':float(np.max(np.abs(native_obs-expected_obs))),
       'passed':bool(np.allclose(native_obs,expected_obs,atol=2e-6,rtol=0)),
       'history':'5 frames, term-major, oldest to newest; q-default; raw clip then scale'}
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(bundle/'exported/policy.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    fixtures=np.load(bundle/'parity_fixtures.npz');pred=np.concatenate([session.run(['actions'],{'obs':v[None]})[0] for v in fixtures['obs']])
    error=np.abs(pred-fixtures['actions']);tol=2e-4+2e-4*np.abs(fixtures['actions'])
    report['saved_export_fixtures']={'cases':len(pred),'all_finite':bool(np.isfinite(pred).all()),'max_absolute_error':float(error.max()),'within_atol_rtol_2e_4':bool(np.all(error<=tol)),
       'note':'Includes artificial stress observations; not a task success or violation-rate benchmark.'}
    from go2_him.moe_stair_env import MoEStairEnv
    env=MoEStairEnv(evaluation=True)
    report['training_export_config_equal']=env.exp.cfg==cfg
    report['consistency_findings']={
       'target_formula':'q_des = 0.25 * raw_action + default; absolute default-relative, not cumulative',
       'target_position_clip_rad':[-100,100],
       'physical_target_limiter_present':False,
       'target_slew_limiter_present':False,
       'dq_command_zero_is_not_target_finite_difference':True,
       'training_policy_period_s':float(env.exp.control_dt),
       'training_physics_period_s':.002,
       'deployment_source_policy_period_s':.020,
       'deployment_source_publication_period_s':.001,
       'pd_kp':env.exp.kp.tolist(),'pd_kd':env.exp.kd.tolist(),
       'training_calf_speed_limit_rad_s':15.7,
       'deployment_source_calf_speed_limit_rad_s':30.,
       'failure_behavior':'Training audit continues through joint violations with effort clipping; deployment source rejects entire proposal and latches damping.',
       'last_action':'Training retains raw network action; deployment inverse-maps last submitted target. Equivalent only while target clip is inactive and a policy command was submitted; initial handoff differs.',
       'initial_state':'Training starts from settled simulated pose and zero last_action; real deployment uses measured held pose and inverse of last submitted hold.',
       'mjcf_vs_application_position_bounds':{n:{'mjcf':r.tolist(),'application':[float(lo),float(hi)]} for n,r,lo,hi in zip(names,env.exp.model.jnt_range[env.exp.jids],lower,upper) if not np.allclose(r,[lo,hi],atol=1e-6,rtol=0)},
       'limits_are_application_profile_not_firmware_readback':True}
    obs=env.reset(sample={'yaw_rad':0.,'y_m':0.},height=.15,vx=.6)
    traces=[];prev=offset.copy();first=None;peak=None;done=False
    try:
        while not done:
            a=session.run(['actions'],{'obs':obs[None]})[0][0]
            e=env.exp;q=e.data.qpos[e.qadr].copy();dq=e.data.qvel[e.vadr].copy()
            target=decode(a);bounded=np.clip(target,lower,upper);diff=target-bounded
            trace={'seconds':env.elapsed,'observation':obs.copy(),'raw_action':a.copy(),'q':q,'dq':dq,'target':target,'hypothetical_bounded_target':bounded,'target_slew':(target-prev)/.02}
            bad=np.flatnonzero(diff)
            if len(bad):
                i=int(np.argmax(np.abs(diff)))
                event={'seconds':env.elapsed,'joint':names[i],'policy_index':i,'sdk_index':int(mapping[i]),'raw_action':float(a[i]),'offset':float(offset[i]),'scale':float(scale[i]),
                    'mapped_target_rad':float(target[i]),'actual_rad':float(q[i]),'actual_velocity_rad_s':float(dq[i]),'lower_rad':float(lower[i]),'upper_rad':float(upper[i]),
                    'excess_rad':float(abs(diff[i])),'hypothetical_clipped_target_rad':float(bounded[i]),
                    'pd_requested_nm':float(20*(target[i]-q[i])-.5*dq[i]),'hypothetical_clipped_pd_nm':float(20*(bounded[i]-q[i])-.5*dq[i]),
                    'real_final_send':'Not sent; no robot I/O. Publication guard would reject this proposed target.'}
                if first is None:first=event
                if peak is None or event['excess_rad']>peak['excess_rad']:peak=event
            traces.append(trace);prev=target.copy();obs,_,done,info=env.step(a)
        arrays={k:np.array([t[k] for t in traces]) for k in traces[0]}
        np.savez_compressed(out/'nominal_trace.npz',**arrays)
        diff=arrays['target']-arrays['hypothetical_bounded_target'];bad=diff!=0
        report['nominal_rollout']={'status':info['episode']['status'],'control_steps':len(traces),'control_dt':.02,'physics_dt':.002,
            'target_joint_sample_excess_count':int(bad.sum()),'target_joint_sample_count':int(bad.size),'target_joint_sample_excess_fraction':float(bad.mean()),
            'control_steps_with_any_target_excess':int(bad.any(axis=1).sum()),'control_step_target_excess_fraction':float(bad.any(axis=1).mean()),
            'max_hypothetical_clip_rad':float(abs(diff).max()),'first_target_excess':first,'worst_target_excess':peak,
            'per_joint':{n:{'target_excess_count':int(bad[:,i].sum()),'samples':len(traces),'max_clip_rad':float(abs(diff[:,i]).max()),'peak_target_slew_rad_s':float(abs(arrays['target_slew'][:,i]).max())} for i,n in enumerate(names)},
            'guard_pre_post_physics':info['episode']['guard'],
            'clipping_was_applied_to_simulation':False,'note':'Hypothetical clip statistics on original trajectory only; no claim about clipped-policy gait or success. Platform stop uses external geometry.'}
    finally:env.exp.close()
    log=ROOT/'validation/moe1000_controller_20261104/attempt3_stand03/controller.log'
    events=[]
    for line in log.read_text().splitlines():
        if 'GUARD_JOINT joint=RL_calf ' not in line:continue
        fields=dict(re.findall(r'(\w+)=([^ ]+)',line));q=float(fields['q']);target=float(fields['target_q']);hi=float(fields['q_upper_rad'])
        if target>hi:events.append({'joint':'RL_calf','actual_rad':q,'proposed_target_rad':target,'upper_rad':hi,'excess_rad':target-hi,
           'raw_action_reconstructed_not_logged':(target+1.5)/.25,'raw_action_was_logged':False,'target_was_rejected':True})
    report['historical_real_target_events']=events
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'go2_him/moe_stair_env.py',ROOT/'deploy/moe1000_controller/overlay/include/MoePolicy.h',ROOT/'deploy/moe1000_controller/overlay/include/FirstTestGuard.h',ROOT/'deploy/moe1000_controller/overlay/include/Go2JointLimits.h',bundle/'params/deploy.yaml']}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('nominal_rollout','source_sha256')},indent=2))
    print(json.dumps({k:v for k,v in report['nominal_rollout'].items() if k not in ('guard_pre_post_physics','per_joint')},indent=2))

if __name__=='__main__':main()
