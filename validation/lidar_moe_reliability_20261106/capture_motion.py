"""Receive-only, bounded hand-motion capture. Does not import a motor SDK."""
import json,time,sys
from pathlib import Path
from collections import deque
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2,Imu
from unitree_go.msg import LowState,SportModeState
rclpy.init(args=[]);n=Node('moe_manual_imu_alignment',enable_rosout=False,start_parameter_services=False)
low=[];imu=[];sport=[];clouds={};counts={};arrays={};types={4:'<u2',7:'<f4'}
def stamp(m):return m.header.stamp.sec+m.header.stamp.nanosec*1e-9
def lc(m):
 low.append([time.time(),time.monotonic(),m.tick,*map(float,m.imu_state.quaternion),*map(float,m.imu_state.gyroscope),*map(float,m.imu_state.accelerometer),*[float(s.q) for s in m.motor_state[:12]]])
def ic(m):
 imu.append([time.time(),stamp(m),m.angular_velocity.x,m.angular_velocity.y,m.angular_velocity.z,m.linear_acceleration.x,m.linear_acceleration.y,m.linear_acceleration.z,m.orientation.w,m.orientation.x,m.orientation.y,m.orientation.z])
def sc(m):sport.append([time.time(),*map(float,m.imu_state.quaternion),*map(float,m.position),*map(float,m.velocity)])
def cc(topic):
 def f(m):
  counts[topic]=counts.get(topic,0)+1
  if counts[topic]%40:return
  fields={x.name:x for x in m.fields};names=[x for x in ['x','y','z','intensity','ring','time'] if x in fields];dtype=np.dtype({'names':names,'formats':[types[fields[x].datatype] for x in names],'offsets':[fields[x].offset for x in names],'itemsize':m.point_step});a=np.ndarray((m.height,m.width),dtype=dtype,buffer=memoryview(m.data),strides=(m.row_step,m.point_step)).copy().reshape(-1);k=topic+'_'+str(counts[topic]);arrays[k]=a;clouds.setdefault(topic,[]).append({'key':k,'stamp':stamp(m),'frame':m.header.frame_id})
 return f
subs=[n.create_subscription(LowState,'/lowstate',lc,qos_profile_sensor_data),n.create_subscription(Imu,'/utlidar/imu',ic,qos_profile_sensor_data),n.create_subscription(SportModeState,'/sportmodestate',sc,qos_profile_sensor_data)]+[n.create_subscription(PointCloud2,'/utlidar/'+t,cc(t),qos_profile_sensor_data) for t in ['cloud','cloud_base']]
start=time.monotonic();last=start
try:
 print('READ_ONLY_RECORDING_STARTED duration_s=80',flush=True)
 while time.monotonic()-start<80:
  rclpy.spin_once(n,timeout_sec=.01)
  if time.monotonic()-last>=10:
   last=time.monotonic();print(json.dumps({'elapsed_s':round(last-start),'lowstate':len(low),'lidar_imu':len(imu),'cloud_counts':counts}),flush=True)
 p=Path(sys.argv[1]);p.with_suffix('.json').write_text(json.dumps({'clouds':clouds,'counts':counts,'duration_s':time.monotonic()-start,'motor_commands_sent':False})+'\n');np.savez_compressed(str(p)+'.npz',lowstate=np.array(low),lidar_imu=np.array(imu),sport=np.array(sport),**arrays);print('READ_ONLY_RECORDING_COMPLETE',flush=True)
finally:n.destroy_node();rclpy.shutdown()
