#!/usr/bin/env python3
"""Offline ray-cast lidar -> supervisor -> frozen ONNX -> bounded targets.
No ROS, network, DDS or motor publication. The simulated ray pattern is a
forward test grid, not a vendor-accurate L1 beam/timing model.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,math,sys,hashlib
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'deploy/lidar_moe_20261106'))
import test_robogauge_two_steps_wall as sim
import test_robogauge_stairs as original
from lidar_moe.frames import Calibration,rpy_matrix
from lidar_moe.geometry import detect
from lidar_moe.supervisor import Supervisor

def run(root,out,yaw,offset,strict,height=.15,config_name='supervisor.json'):
 cfg=json.loads((root/'config'/config_name).read_text());cal=Calibration(root/'config/go2_calibrated.urdf');sup=Supervisor(cfg)
 args=SimpleNamespace(policy=root/'model/exported/policy.onnx',config=root/'model/params/deploy.yaml',robot_xml=ROOT/'validation/moe1000_sim2real_20261105/source/assets/unitree_go2/go2.xml',outdir=out,steps=2,height=height,tread=.3,width=.9,platform=.9,physics_dt=.002,video_width=640,video_height=480,viewer=False,video=False,camera_azimuth=135.,camera_elevation=-24.,vx=0.,torque_limit_mode='physical',settle_seconds=2.,start_distance=.5)
 e=sim.Experiment(args);m,d,mj=e.model,e.data,e.mj;start=e.initialize({'yaw_rad':yaw,'y_m':offset});limits=json.loads((root/'model/limit_profile.json').read_text())['limits'];perm=np.array([3,4,5,0,1,2,9,10,11,6,7,8]);lo=np.array([limits[i]['position_lower_rad'] for i in perm]);hi=np.array([limits[i]['position_upper_rad'] for i in perm]);vel=np.array([limits[i]['speed_limit_rad_s'] for i in perm]);eff=np.array([limits[i]['effort_limit_nm'] for i in perm]);
 for gid in range(m.ngeom):m.geom_group[gid]=0 if m.geom_bodyid[gid]==0 else 1
 az,el=np.meshgrid(np.linspace(-.70,.70,65),np.unique(np.r_[np.linspace(-1.25,.52,95),np.linspace(-.07,.01,55)]));rays=np.c_[np.cos(el.ravel())*np.cos(az.ravel()),np.cos(el.ravel())*np.sin(az.ravel()),np.sin(el.ravel())];groups=np.array([1,0,0,0,0,0],np.uint8);geom=np.zeros(1,np.int32)
 def cloud():
  R=d.xmat[e.base].reshape(3,3);yawnow=math.atan2(R[1,0],R[0,0]);Rz=rpy_matrix([0,0,yawnow]);origin=d.xpos[e.base]+R@cal.lidar[:3,3];hits=[]
  for ray in rays@Rz.T:
   distance=mj.mj_ray(m,d,origin,ray,groups,1,-1,geom)
   if .07<distance<3.5:hits.append(origin+ray*distance)
  return np.array(hits)
 history=deque(maxlen=3);command=np.zeros(3);target=e.policy.q0.copy();active=False;parked=False;park_at=None;hold=None;trace=[];geos=[];first_guard=None;wall_contact=None;min_wall=99.;audit=original.GuardAudit(np,m.jnt_range[e.jids]);status='timeout';active_at=None;last_out={'phase':'WAIT'};last_cloud=np.empty((0,3));target_excess=0
 riser_contact_count=0;first_riser_contact=None
 for step in range(13000):
  t=step*.002;R=d.xmat[e.base].reshape(3,3);position=d.xpos[e.base].copy();yn=math.atan2(R[1,0],R[0,0]);Rz=rpy_matrix([0,0,yn]);pose={'position':position.tolist(),'yaw':yn};speed=float(np.linalg.norm(d.qvel[:2]));tilt=math.acos(np.clip(R[2,2],-1,1))
  if step%50==0:
   last_cloud=cloud();history.append(last_cloud)
   if not active:
    g=detect((np.concatenate(history)-position)@Rz);
    if step==100:np.savez_compressed(out/f'cloud_h{height:.2f}_yaw{yaw:.2f}.npz',heading=(np.concatenate(history)-position)@Rz)
    sup.observe(g,pose,t);geos.append({'time_s':t,'geometry':g})
  if step%25==0:
   feet=cal.feet_body(d.qpos[e.qadr][perm])@R.T@Rz;points=(last_cloud-position)@Rz
   front=cal.support_extent(d.qpos[e.qadr][perm],R.T@np.r_[sup.map['n'],0.])+.04 if sup.map is not None else None
   q_sdk=d.qpos[e.qadr][perm];dq_sdk=d.qvel[e.vadr][perm];leg_body=cal.leg_probes_body(q_sdk)
   probes=leg_body@R.T@Rz
   velocities=(cal.leg_motion_body(q_sdk,dq_sdk)+np.cross(e.observation()['base_ang_vel'],leg_body))@R.T@Rz+d.qvel[:3]@Rz
   last_out=sup.update(pose,feet,points,t,fresh=True,active=active,speed=speed,tilt=tilt,front_extent=front,leg_probes=probes,leg_velocities=velocities,joint_speed=float(np.max(abs(dq_sdk))));command=np.array(last_out['command']);trace.append({'time_s':t,'position':position.tolist(),'speed':speed,'tilt':tilt,**last_out})
   if not active and last_out['phase']=='READY':
    active=True;active_at=t;e.policy.reset(e.observation());continue
   if last_out['mode']=='FAULT':status='supervisor_fault:'+last_out['reason'];break
   if last_out['mode']=='HOLD' and not parked:parked=True;park_at=t;hold=d.qpos[e.qadr].copy()
  if active and not parked and t-active_at>=20.:status='motion_budget_complete';break
  if step%e.decimation==0 and active and not parked:
   obs=e.observation();obs['velocity_commands']=command.copy();rawtarget=e.policy.act(obs);target=np.clip(rawtarget,lo,hi);e.policy.last_action=((target-e.policy.offset)/e.policy.scale).astype(np.float32)
   target_excess+=int(((target<lo)|(target>hi)).sum())
  if parked:
   target=hold;blend=min(1.,t-park_at);e.kp=np.full(12,20.+20.*blend);e.kd=np.full(12,.5+.5*blend)
  q=d.qpos[e.qadr];dq=d.qvel[e.vadr];requested=e.kp*(target-q)-e.kd*dq
  reason=('measured_position' if ((q<lo)|(q>hi)).any() else 'measured_speed' if (abs(dq)>vel).any() else 'tracking_above_2rad' if (abs(target-q)>2.).any() else 'estimated_pd_effort' if (abs(requested)>eff).any() else None)
  if active and reason and first_guard is None:
   first_guard={'time_s':t,'reason':reason,'q':q.tolist(),'dq':dq.tolist(),'target':target.tolist(),'estimated_pd_nm':requested.tolist()}
   if strict:status='publication_guard:'+reason;break
  e.physics(target,audit,t+.002);clear=e.wall_clearance()['signed_clearance_m'];min_wall=min(min_wall,clear);contacts,_=e.wall_contacts(t)
  for contact in d.contact[:d.ncon]:
   ids=[int(contact.geom1),int(contact.geom2)]
   names=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,i) or '' for i in ids]
   stair=any(name.startswith('stair_') or name=='top_platform' for name in names)
   if stair and any(m.geom_bodyid[i]!=0 for i in ids) and abs(contact.frame[2])<.5:
    riser_contact_count+=1
    if first_riser_contact is None:first_riser_contact={'time_s':t,'geom_names':names,'position':contact.pos.tolist(),'normal':contact.frame[:3].tolist()}
  if contacts:wall_contact=contacts[0];status='wall_contact';break
  if parked and t>trace[-1]['time_s']+.8:status=last_out['phase'];break
  if parked and sum(x['mode']=='HOLD' for x in trace)>20:status=last_out['phase'];break
  if not active and t>=3.:status='geometry_not_ready';break
 source_files=[root/'model/exported/policy.onnx',root/'config/go2_calibrated.urdf',root/'config'/config_name,root/'lidar_moe/geometry.py',root/'lidar_moe/supervisor.py',root/'lidar_moe/leg_clearance.py',Path(__file__),args.robot_xml]
 result={'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},'status':status,'strict_publication_guard':strict,'step_height_m':height,'real_robot_control':False,'start':start,'yaw_rad':yaw,'lateral_offset_m':offset,'active_at_s':active_at,'final_position_m':d.qpos[:3].tolist(),'wall_contact':wall_contact,'minimum_wall_clearance_m':min_wall,'target_joint_limit_violations':target_excess,'first_guard_violation':first_guard,'last_supervisor':last_out,'geometry_valid_frames':sum(g['geometry']['valid'] for g in geos),'geometry_frames':len(geos),'sensor_model':'forward ray grid from accepted lidar mounting position; exact MuJoCo visibility, no added noise; not vendor beam model','full_real_controller_equivalence':False}
 result.update(riser_contact_count=riser_contact_count,first_riser_contact=first_riser_contact)
 label=('strict' if strict else 'diagnostic')+f'_h{height:.2f}_yaw{yaw:+.2f}_y{offset:+.2f}';(out/(label+'.json')).write_text(json.dumps(result,indent=2));(out/(label+'_trace.json')).write_text(json.dumps(trace));(out/(label+'_geometry.json')).write_text(json.dumps(geos));e.close();print(json.dumps({k:result[k] for k in ('status','strict_publication_guard','yaw_rad','geometry_valid_frames','geometry_frames','minimum_wall_clearance_m','target_joint_limit_violations','active_at_s','riser_contact_count')}),flush=True);return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--independent',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);root=ROOT/'deploy/lidar_moe_20261106';results=[]
 cases=[(True,0.,0.,.15),(True,0.,0.,.10)] if a.independent else [(True,0.,0.,.15),(True,0.,0.,.10),(False,0.,0.,.10),(False,.12,.04,.10)]
 for strict,yaw,offset,height in cases:results.append(run(root,a.out,yaw,offset,strict,height,'supervisor_independent.json' if a.independent else 'supervisor.json'))
 (a.out/'summary.json').write_text(json.dumps(results,indent=2))
