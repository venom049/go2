import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'deploy/lidar_moe_20261106'))
from lidar_moe.translation_registration import SurfaceMap,interpolate_quaternions,rotate_points
from lidar_moe.frames import rpy_matrix
from lidar_moe.lio_frames import matrix_quaternion

class Registration(unittest.TestCase):
 def scene(self):
  a,b=np.meshgrid(np.linspace(.1,1.6,40),np.linspace(.1,1.6,40));x=a.ravel();y=b.ravel();z=np.zeros_like(x)
  return np.concatenate((np.c_[x,y,z],np.c_[x,z,y],np.c_[z,x,y]))
 def test_translation_with_outliers(self):
  scene=self.scene();surface=SurfaceMap(scene);delta=np.array([.035,-.025,.012]);rng=np.random.default_rng(5)
  measured=np.r_[scene-delta+rng.normal(0,.001,scene.shape),rng.uniform(2,3,(600,3))]
  correction,status=surface.register(measured)
  np.testing.assert_allclose(correction,delta,atol=.003);self.assertGreater(status['pose_information_ratio'],.01)
 def test_missing_translation_direction_rejected(self):
  scene=self.scene()[:1600];surface=SurfaceMap(scene)
  with self.assertRaisesRegex(ValueError,'degenerate'):surface.register(scene)
 def test_antipodal_quaternions_and_rotation(self):
  r=rpy_matrix([0.,0.,.2]);q=matrix_quaternion(r)
  sampled=interpolate_quaternions([0.,.02],np.array([q,-q]),np.array([.01]))
  np.testing.assert_allclose(rotate_points(sampled,np.array([[1.,2.,3.]])),np.array([[1.,2.,3.]])@r.T)
 def test_missing_imu_bracket_or_gap_rejected(self):
  q=np.array([[1.,0,0,0],[1.,0,0,0]])
  for t,query in [([0.,.1],[.05]),([0.,.02],[.03])]:
   with self.assertRaises(ValueError):interpolate_quaternions(t,q,np.array(query))

if __name__=='__main__':unittest.main()
