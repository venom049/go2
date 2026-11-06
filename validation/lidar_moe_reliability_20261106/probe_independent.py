"""Receive-only LIO metrics and bounded point-cloud evidence; no publishers."""
import collections,json,sys,time
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2,Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from unitree_go.msg import LowState
from lidar_moe.ros_bridge import stamp,xyz_cloud
rclpy.init();n=Node('moe_independent_probe',enable_rosout=False,start_parameter_services=False)
counts=collections.Counter();ages=collections.defaultdict(list);values=collections.defaultdict(list);reasons=collections.Counter();last={};regress=collections.Counter();arrays={};cloud_meta=[]
topics=[('/utlidar/cloud',PointCloud2),('/moe_lio/body_imu',Imu),('/moe_lio/quality',String),('/moe_lio/imu_health',String),('/moe_lio/health',String),('/moe_lio/internal/imu_odom',Odometry),('/moe_lio/internal/cloud_registered',PointCloud2),('/moe_lio/base_odom',Odometry),('/moe_lio/cloud_world',PointCloud2),('/utlidar/robot_odom',Odometry),('/lowstate',LowState)]
def cb(topic):
 def receive(m):
  counts[topic]+=1
  if hasattr(m,'header'):
   ts=stamp(m);ages[topic].append(time.time()-ts)
   if topic in last and ts<=last[topic]:regress[topic]+=1
   last[topic]=ts
  if isinstance(m,String):
   v=json.loads(m.data);values[topic].append(v)
   if 'reason' in v:reasons[v['reason']]+=1
  if isinstance(m,Imu) and counts[topic]%50==0:
   q=m.orientation;g=m.angular_velocity;a=m.linear_acceleration
   values[topic].append({'stamp':stamp(m),'q':[q.w,q.x,q.y,q.z],'gyro':[g.x,g.y,g.z],'acc':[a.x,a.y,a.z]})
  if isinstance(m,Odometry):
   p=m.pose.pose.position;q=m.pose.pose.orientation
   values[topic].append({'stamp':stamp(m),'p':[p.x,p.y,p.z],'q':[q.w,q.x,q.y,q.z],'cov':list(m.pose.covariance)})
  if isinstance(m,PointCloud2) and topic.endswith('cloud_world') and counts[topic]%3==0 and len(cloud_meta)<20:
   key='cloud_'+str(len(cloud_meta));arrays[key]=xyz_cloud(m,'moe_odom');cloud_meta.append({'key':key,'stamp':stamp(m),'frame':m.header.frame_id})
 return receive
subs=[n.create_subscription(t,p,cb(p),qos_profile_sensor_data) for p,t in topics]
start=time.monotonic();seconds=float(sys.argv[2]) if len(sys.argv)>2 else 35
try:
 while time.monotonic()-start<seconds:rclpy.spin_once(n,timeout_sec=.01)
 result={'duration_s':time.monotonic()-start,'counts':dict(counts),'source_stamp_regressions':dict(regress),'ages_s':{k:{'median':float(np.median(v)),'p95':float(np.quantile(v,.95)),'min':min(v),'max':max(v)} for k,v in ages.items()},'health_reasons':dict(reasons),'values':dict(values),'clouds':cloud_meta,'control_authorized':False}
 output=Path(sys.argv[1]);output.write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(output.with_suffix('.npz'),**arrays)
 print(json.dumps({k:result[k] for k in ('duration_s','counts','source_stamp_regressions','ages_s','health_reasons')}))
finally:n.destroy_node();rclpy.shutdown()
