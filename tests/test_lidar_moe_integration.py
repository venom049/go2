"""Geometry, frame, supervisor and actual captured-cloud regression checks."""
import json,math,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'deploy/lidar_moe_20261106'))
from lidar_moe.frames import Calibration,quaternion_matrix,rpy_matrix
from lidar_moe.geometry import detect
from lidar_moe.supervisor import Supervisor
from lidar_moe.transport import encode,PACKET,feedback,FEEDBACK
CFG=json.loads((ROOT/'deploy/lidar_moe_20261106/config/supervisor.json').read_text())
def scene(steps=2,wall=True,yaw=0.,offset=0.,noise=.002):
 rng=np.random.default_rng(43);parts=[]
 def horizontal(lo,hi,z):
  x,y=np.meshgrid(np.arange(lo,hi,.018),np.arange(-.55,.55,.025));parts.append(np.c_[x.ravel(),y.ravel(),np.full(x.size,z)])
 horizontal(.13,.64,-.32)
 for i in range(steps):
  x=.64+i*.30;y,z=np.meshgrid(np.arange(-.55,.55,.022),np.arange(-.32+i*.15,-.32+(i+1)*.15,.012));parts.append(np.c_[np.full(y.size,x),y.ravel(),z.ravel()]);horizontal(x+.025,x+(.30 if i<steps-1 else .88),-.32+(i+1)*.15)
 if wall:
  y,z=np.meshgrid(np.arange(-.55,.55,.022),np.arange(-.32+steps*.15,.75,.018));parts.append(np.c_[np.full(y.size,.64+(steps-1)*.30+.9),y.ravel(),z.ravel()])
 a=np.concatenate(parts);a+=rng.normal(0,noise,a.shape);a=a@rpy_matrix([0,0,yaw]).T;a[:,1]+=offset;return a
class Tests(unittest.TestCase):
 def test_geometry_and_rotation(self):
  for yaw in (0,.16,-.18):
   g=detect(scene(yaw=yaw));self.assertTrue(g['valid'],g);self.assertAlmostEqual(g['heading_error_rad'],yaw,delta=.04);self.assertAlmostEqual(g['tread_depth_m'],.3,delta=.035);self.assertAlmostEqual(sum(g['step_heights_m']),.3,delta=.035)
 def test_negative_geometries(self):
  for a in (scene(1),scene(2,False),scene(3),np.full((200,3),np.nan),scene()[::150]):self.assertFalse(detect(a)['valid'])
 def test_recorded_window(self):
  p=ROOT/'validation/lidar_moe_integration_20261106';m=json.loads((p/'frames_v2.json').read_text());d=np.load(p/'frames_v2.npz');good=[]
  for i in range(2,8):
   row=m['clouds']['cloud_deskewed'][i];pose=min(m['poses'],key=lambda k:abs(k[0]-row['stamp_s']));R=rpy_matrix([0,0,math.atan2(quaternion_matrix(pose[2])[1,0],quaternion_matrix(pose[2])[0,0])]);a=(np.concatenate([d[m['clouds']['cloud_deskewed'][j]['key']] for j in range(i-2,i+1)])-pose[1])@R;g=detect(a);good.append(g['valid'])
  self.assertGreaterEqual(sum(good),5,good)
 def setup_supervisor(self):
  s=Supervisor(CFG);g=detect(scene());p={'position':[0,0,.32],'yaw':0.};
  for i in range(3):s.observe(g,p,i*.2)
  feet=np.array([[.2,-.15,-.32],[.2,.15,-.32],[-.2,-.15,-.32],[-.2,.15,-.32]])
  return s,p,feet
 def test_confirmation_and_stale(self):
  s,p,f=self.setup_supervisor();o=s.update(p,f,scene(),1.,fresh=True);self.assertEqual(o['phase'],'READY');self.assertEqual(o['command'],[0,0,0]);o=s.update(p,f,scene(),1.05,fresh=True,active=True);self.assertGreater(o['command'][0],0);o=s.update(p,f,scene(),1.1,fresh=False,active=True);self.assertEqual(o['mode'],'FAULT');self.assertEqual(o['command'],[0,0,0]);self.assertEqual(s.update(p,f,scene(),1.2,fresh=True,active=True)['mode'],'FAULT')
 def test_park_and_no_reentry(self):
  s,p,f=self.setup_supervisor();s.update(p,f,scene(),1.,fresh=True,active=True)
  p={'position':[1.20,0,.62],'yaw':0.};pts=scene()-np.array([1.2,0,.3])
  for i in range(25):o=s.update(p,f,pts,1.05+i*.05,fresh=True,active=True,speed=0)
  self.assertEqual(o['phase'],'COMPLETE',o);self.assertEqual(o['mode'],'HOLD');self.assertEqual(s.update(p,f,pts,3.,fresh=False,active=False)['mode'],'HOLD')
 def test_wall_blocks_before_all_feet(self):
  s,p,f=self.setup_supervisor();s.update(p,f,scene(),1.,fresh=True,active=True)
  p={'position':[1.34,0,.32],'yaw':0.};pts=scene()-np.array([1.34,0,0])
  for i in range(25):o=s.update(p,f,pts,1.05+i*.05,fresh=True,active=True,speed=.03)
  self.assertEqual(o['phase'],'BLOCKED',o)
 def test_repeated_source_not_confirmation(self):
  s,p,f=self.setup_supervisor();s=Supervisor(CFG);g=detect(scene())
  for i in range(9):s.observe(g,p,10.)
  self.assertEqual(s.count,1)
 def test_calibration(self):
  c=Calibration(ROOT/'deploy/lidar_moe_20261106/config/go2_calibrated.urdf');q=np.array([0,.8,-1.5]*4);feet=c.feet_body(q);self.assertTrue((feet[:2,0]>.1).all());self.assertTrue((feet[2:,0]<-.1).all());np.testing.assert_allclose(c.lidar[:3,3],[.28945,0,-.046825])
  a=np.array([[1.,2.,3.]]);body=a@c.lidar[:3,:3].T+c.lidar[:3,3];np.testing.assert_allclose(c.cloud_heading(a,[1,0,0,0]),body);np.testing.assert_allclose(c.cloud_heading(body,[1,0,0,0],'base_link'),body)
 def test_full_collision_envelope(self):
  c=Calibration(ROOT/'deploy/lidar_moe_20261106/config/go2_calibrated.urdf');q=np.array([0,.8,-1.5]*4)
  extent=c.support_extent(q,[1,0,0]);self.assertGreater(extent,.33);self.assertLess(extent,.6)
  # A posture with a much larger collision envelope must not reuse foot-only room.
  s,p,f=self.setup_supervisor();o=s.update(p,f,scene(),1.,fresh=True,front_extent=.80)
  self.assertEqual(o['mode'],'WAIT');self.assertEqual(o['reason'],'landing_too_short_for_all_feet_and_stop')
 def test_wire(self):
  b=encode(4,10.,[.2,-.05,.1],'RUN');self.assertEqual(len(b),64);self.assertEqual(PACKET.unpack(b)[3],10000000000)
  for cmd in ([float('nan'),0,0],[.4,0,0],[.2,.2,0]):
   with self.assertRaises(ValueError):encode(1,10.,cmd,'RUN')
  self.assertEqual(feedback(FEEDBACK.pack(b'G2F2',2,10000000000,1,0),10.1),(1,0))
 def test_feedback_receive_time_and_rejection(self):
  # A frame sent after the loop's earlier sample is valid at receive time.
  packet=FEEDBACK.pack(b'G2F2',2,10001000000,0,100)
  with self.assertRaises(ValueError):feedback(packet,10.)
  self.assertEqual(feedback(packet,10.002),(0,100))
  for packet in (b'bad',FEEDBACK.pack(b'G2F2',2,9000000000,1,0),
                 FEEDBACK.pack(b'G2F2',2,11000000000,1,0),
                 FEEDBACK.pack(b'G2F2',2,10000000000,4,0)):
   with self.assertRaises(ValueError):feedback(packet,10.01)
if __name__=='__main__':unittest.main()
