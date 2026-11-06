"""LIO lever-arm/gravity transforms and conservative riser-stop behavior."""
import json,math,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];PACKAGE=ROOT/'deploy/lidar_moe_20261106';sys.path.insert(0,str(PACKAGE))
from lidar_moe.frames import Calibration,rpy_matrix,quaternion_matrix
from lidar_moe.lio_frames import initial_alignment,transform_pose,matrix_quaternion
from lidar_moe.leg_clearance import assess,stance_alignment
from lidar_moe.supervisor import Supervisor
from lidar_moe.ros_bridge import latest_pair
LIO=json.loads((PACKAGE/'config/independent_lio.json').read_text())
CFG=json.loads((PACKAGE/'config/supervisor_independent.json').read_text())

class Frames(unittest.TestCase):
 def test_atomic_pair_does_not_fail_on_new_half_or_retime_old_pair(self):
  a={'stamp':10.,'source_mono':8.};b={'stamp':10.065,'source_mono':8.065};p={'stamp':10.}
  cloud,pose=latest_pair([a,b],[p],.02);self.assertIs(cloud,a);self.assertIs(pose,p);self.assertEqual(cloud['source_mono'],8.)
  p2={'stamp':10.065};cloud,pose=latest_pair([a,b],[p,p2],.02);self.assertIs(cloud,b);self.assertIs(pose,p2)
  self.assertEqual(latest_pair([b],[p],.02),(None,None))
 def test_base_origin_and_gravity_alignment(self):
  p=np.array([1.,-2.,.3]);r=rpy_matrix([.1,-.2,.7]);imu=rpy_matrix([-.05,.12,-.6])
  a,t=initial_alignment(p,r,imu,LIO);bp,br,c=transform_pose(p,r,np.eye(6)*.001,a,t,LIO)
  np.testing.assert_allclose(bp,0,atol=1e-12);np.testing.assert_allclose(br[2],imu[2],atol=1e-12)
  self.assertGreater(np.linalg.eigvalsh(c).min(),0.)
 def test_world_cloud_and_base_pose_are_same_transform(self):
  p=np.array([.3,-.5,.8]);r=rpy_matrix([.1,.2,.3]);a,t=initial_alignment(p,r,rpy_matrix([0,.1,0]),LIO)
  bp,br,_=transform_pose(p,r,np.eye(6)*.001,a,t,LIO)
  raw=np.array([1.,2.,3.]);body=np.asarray(LIO['R_base_raw'])@raw+LIO['t_base_raw_m']
  old_world=r@(np.asarray(LIO['R_imu_raw'])@raw+LIO['t_imu_raw_m'])+p
  np.testing.assert_allclose(a@old_world+t,br@body+bp,atol=1e-10)
 def test_quaternion_roundtrip(self):
  for angles in ([.1,-.2,.8],[math.pi,0,0],[0,math.pi,0],[0,0,math.pi]):
   r=rpy_matrix(angles);np.testing.assert_allclose(quaternion_matrix(matrix_quaternion(r)),r,atol=1e-12)
 def test_bad_covariance_rejected(self):
  for c in (np.eye(6)*float('nan'),-np.eye(6)):
   with self.assertRaises(ValueError):transform_pose([0,0,0],np.eye(3),c,np.eye(3),np.zeros(3),LIO)
 def test_named_leg_motion_matches_foot_derivative(self):
  cal=Calibration(PACKAGE/'config/go2_calibrated.urdf');q=np.array([0,.8,-1.5]*4);dq=np.array([0,.5,-.7]*4)
  p=cal.leg_probes_body(q);self.assertEqual(p.shape,(4,17,3));np.testing.assert_allclose(p[:,-1],cal.feet_body(q))
  v=cal.leg_motion_body(q,dq);eps=1e-4
  np.testing.assert_allclose(v[:,-1],(cal.feet_body(q+eps*dq)-cal.feet_body(q-eps*dq))/(2*eps))

class Riser(unittest.TestCase):
 def test_ground_alignment_rejects_offset_and_accepts_rotated_plane(self):
  r=rpy_matrix([0,.04,.3]);n=r@np.array([0.,0.,1.]);feet=np.array([[.2,-.15,.022],[.2,.15,.022],[-.2,-.15,.022],[-.2,.15,.022]])@r.T+[1.,2.,.4];d=float(n@np.array([1.,2.,.4]))
  good=stance_alignment(feet,n,d,foot_radius=.022,max_error=.025);self.assertTrue(good['verified'])
  bad=stance_alignment(feet,n,d-.04,foot_radius=.022,max_error=.025);self.assertFalse(bad['verified']);np.testing.assert_allclose(bad['inferred_sole_gap_m'],.04)
 def test_recorded_grounded_capture_is_not_approved(self):
  recorded=json.loads((ROOT/'validation/lidar_moe_reliability_20261106/resume/feet_grounded_alignment_result.json').read_text())
  self.assertTrue(recorded['user_confirmed_all_feet_grounded'])
  for row in recorded['rows']:
   floor=row['geometry']['floor'];out=stance_alignment(row['feet'],floor['normal'],floor['distance'],foot_radius=.022,max_error=.025)
   self.assertFalse(out['verified'],out)
 def probes(self,x=.2,z=-.32,vx=.25,vz=0.):
  p=np.zeros((4,17,3));p[...,0]=x;p[...,2]=z;v=np.zeros_like(p);v[...,0]=vx;v[...,2]=vz;return p,v
 def risk(self,p,v):return assess(p,v,normal=[1,0],edges=[.64,.94],heights=[.15,.15],floor_z=0.,body_z=.32,speed=.25,config=CFG['leg_clearance'])
 def test_far_riser_clear(self):self.assertFalse(self.risk(*self.probes())['risk'])
 def test_low_toe_approach_detected(self):
  r=self.risk(*self.probes(x=.55));self.assertTrue(r['risk']);self.assertEqual(r['risks'][0]['riser'],1)
 def test_sufficient_lift_and_retreat(self):
  self.assertFalse(self.risk(*self.probes(x=.55,z=.1))['risk'])
  self.assertFalse(self.risk(*self.probes(x=.55,vx=-.2))['risk'])
 def test_second_riser_not_ignored(self):
  self.assertTrue(self.risk(*self.probes(x=.85,z=-.17))['risk'])
 def test_malformed_snapshot_rejected(self):
  p,v=self.probes();p[0,0,0]=float('nan')
  with self.assertRaises(ValueError):self.risk(p,v)
 def setup_supervisor(self):
  s=Supervisor(CFG);s.count=3;s.map={'n':np.array([1.,0.]),'edges':np.array([.64,.94,1.84]),'center':np.array([0.,0.]),'floor_z':0.,'floor_normal_world':np.array([0.,0.,1.]),'floor_distance_world':0.,'top_height':.3,'geometry':{'corridor_y_m':[-.5,.5],'step_heights_m':[.15,.15]}}
  p={'position':[0.,0.,.32],'yaw':0.};feet=np.array([[.2,-.15,-.32],[.2,.15,-.32],[-.2,-.15,-.32],[-.2,.15,-.32]])
  y,z=np.meshgrid(np.linspace(-.4,.4,10),np.linspace(0.,.5,10));points=np.c_[np.full(y.size,1.84),y.ravel(),z.ravel()]
  return s,p,feet,points
 def test_risk_decelerates_holds_and_cannot_auto_resume(self):
  s,p,f,pts=self.setup_supervisor();a,v=self.probes()
  before=s.update(p,f,pts,1.,fresh=True,active=True,leg_probes=a,leg_velocities=v,joint_speed=0.)
  a,v=self.probes(x=.55)
  out=s.update(p,f,pts,1.05,fresh=True,active=True,leg_probes=a,leg_velocities=v,joint_speed=0.)
  self.assertEqual(out['phase'],'OBSTACLE_STOPPING');self.assertGreaterEqual(out['command'][0],0.);self.assertLess(out['command'][0],before['command'][0])
  a,v=self.probes()
  for i in range(18):out=s.update(p,f,pts,1.1+i*.05,fresh=True,active=True,leg_probes=a,leg_velocities=v,joint_speed=0.)
  self.assertEqual(out['phase'],'OBSTACLE_HOLD');self.assertEqual(out['mode'],'HOLD')
  self.assertEqual(s.update(p,f,pts,3.,fresh=True,active=True)['command'],[0,0,0])
 def test_missing_leg_state_faults_in_motion(self):
  s,p,f,pts=self.setup_supervisor();out=s.update(p,f,pts,1.,fresh=True,active=True)
  self.assertEqual(out['mode'],'FAULT');self.assertEqual(out['reason'],'leg_clearance_state_missing')
 def test_initial_floor_offset_blocks_ready_without_recalibrating(self):
  s,p,f,pts=self.setup_supervisor();s.map['floor_normal_world']=np.array([0.,0.,1.]);s.map['floor_distance_world']=-.06
  before=s.map['floor_distance_world'];out=s.update(p,f,pts,1.,fresh=True)
  self.assertEqual(out['mode'],'WAIT');self.assertFalse(out['ready']);self.assertEqual(out['reason'],'floor_foot_alignment_unverified');self.assertEqual(out['command'],[0,0,0]);self.assertEqual(s.map['floor_distance_world'],before)
  out=s.update(p,f,pts,1.025,fresh=True,active=True);self.assertEqual(out['mode'],'WAIT');self.assertFalse(s.active)
  s.map['floor_distance_world']=-.022;out=s.update(p,f,pts,1.05,fresh=True);self.assertEqual(out['phase'],'READY',out)

if __name__=='__main__':unittest.main()
