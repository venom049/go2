import json,time,sys
from pathlib import Path
from collections import deque
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2,Imu
from nav_msgs.msg import Odometry
from unitree_go.msg import LowState
rclpy.init(args=[]);node=Node('moe_raw_calibration_capture',enable_rosout=False,start_parameter_services=False);clouds={};poses=deque(maxlen=1000);states=deque(maxlen=3500);imus=deque(maxlen=1000);types={4:'<u2',7:'<f4'}
def stamp(m):return m.header.stamp.sec+m.header.stamp.nanosec*1e-9
def cloud(topic):
 def f(m):
  fields={x.name:x for x in m.fields};names=[x for x in ['x','y','z','intensity','ring','time'] if x in fields];dtype=np.dtype({'names':names,'formats':[types[fields[x].datatype] for x in names],'offsets':[fields[x].offset for x in names],'itemsize':m.point_step});a=np.ndarray((m.height,m.width),dtype=dtype,buffer=memoryview(m.data),strides=(m.row_step,m.point_step)).copy().reshape(-1)
  clouds.setdefault(topic,deque(maxlen=8)).append((stamp(m),m.header.frame_id,a))
 return f
def odom(m):
 p=m.pose.pose.position;q=m.pose.pose.orientation;poses.append([stamp(m),p.x,p.y,p.z,q.w,q.x,q.y,q.z])
def low(m):states.append({'mono':time.monotonic(),'wall':time.time(),'dq':[x.dq for x in m.motor_state[:12]],'foot_force':[int(v) for v in m.foot_force],'q':[x.q for x in m.motor_state[:12]],'quat':[float(x) for x in m.imu_state.quaternion],'tick':m.tick})
def imu(m):imus.append([stamp(m),m.angular_velocity.x,m.angular_velocity.y,m.angular_velocity.z,m.linear_acceleration.x,m.linear_acceleration.y,m.linear_acceleration.z])
subs=[node.create_subscription(PointCloud2,'/utlidar/'+x,cloud(x),qos_profile_sensor_data) for x in ['cloud','cloud_base','cloud_deskewed']]+[node.create_subscription(Odometry,'/utlidar/robot_odom',odom,qos_profile_sensor_data),node.create_subscription(LowState,'/lowstate',low,qos_profile_sensor_data),node.create_subscription(Imu,'/utlidar/imu',imu,qos_profile_sensor_data)]
try:
 until=time.monotonic()+6
 while time.monotonic()<until:rclpy.spin_once(node,timeout_sec=.01)
 metadata={'clouds':{},'poses':list(poses),'states':list(states),'imus':list(imus)};arrays={}
 for topic,rows in clouds.items():
  metadata['clouds'][topic]=[]
  for i,(ts,frame,a) in enumerate(rows):
   k=topic+'_'+str(i);arrays[k]=a;metadata['clouds'][topic].append({'key':k,'stamp':ts,'frame':frame})
 base=Path(sys.argv[1]);base.with_suffix('.json').write_text(json.dumps(metadata)+'\n');np.savez_compressed(str(base)+'.npz',**arrays);print(json.dumps({'clouds':{k:len(v) for k,v in clouds.items()},'imu':len(imus),'odom':len(poses)}))
finally:node.destroy_node();rclpy.shutdown()
