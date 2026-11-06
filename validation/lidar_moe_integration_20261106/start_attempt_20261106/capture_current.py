"""Sensor-only calibration-chain check; retains only bounded geometry/state samples."""
import json,time
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from unitree_go.msg import LowState

root=Path('/home/unitree/go2-lidar-start-check-20261106-1793943962282418774');clouds={};poses=[];state=None
def cloud(topic):
    def cb(m):
        fields={f.name:f for f in m.fields};dt=np.dtype({'names':['x','y','z'],'formats':['<f4']*3,
            'offsets':[fields[n].offset for n in ('x','y','z')],'itemsize':m.point_step})
        a=np.ndarray((m.height,m.width),dtype=dt,buffer=memoryview(m.data),strides=(m.row_step,m.point_step))
        points=np.stack([a[n].reshape(-1) for n in ('x','y','z')],axis=1)
        ts=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
        if topic not in clouds:clouds[topic]=[]
        clouds[topic].append((ts,m.header.frame_id,points.copy()))
        clouds[topic]=clouds[topic][-8:]
    return cb
def odom(m):
    p=m.pose.pose.position;q=m.pose.pose.orientation
    poses.append((m.header.stamp.sec+m.header.stamp.nanosec*1e-9,[p.x,p.y,p.z],[q.w,q.x,q.y,q.z]))
    if len(poses)>300:del poses[:-300]
def low(m):
    global state
    state={'received_mono':time.monotonic(),'tick':m.tick,'q':[s.q for s in m.motor_state[:12]],
           'quat_wxyz':[float(v) for v in m.imu_state.quaternion]}
rclpy.init(args=[]);node=Node('lidar_frame_chain_check',enable_rosout=False,start_parameter_services=False)
try:
    sub=[]
    for topic in ('cloud','cloud_base','cloud_deskewed'):
        sub.append(node.create_subscription(PointCloud2,'/utlidar/'+topic,cloud(topic),qos_profile_sensor_data))
    sub.append(node.create_subscription(Odometry,'/utlidar/robot_odom',odom,qos_profile_sensor_data))
    for topic in ('/lowstate','/lf/lowstate'):sub.append(node.create_subscription(LowState,topic,low,qos_profile_sensor_data))
    until=time.monotonic()+5
    while time.monotonic()<until:rclpy.spin_once(node,timeout_sec=.02)
    metadata={'state':state,'clouds':{},'poses':poses};arrays={}
    for name,rows in clouds.items():
        metadata['clouds'][name]=[]
        for i,(ts,frame,pts) in enumerate(rows):
            key=f'{name}_{i}';arrays[key]=pts
            metadata['clouds'][name].append({'key':key,'stamp_s':ts,'frame':frame})
    root.mkdir(exist_ok=True)
    encoded=json.dumps(metadata,indent=2)+'\n'
    with (root/'frames_v2.npz').open('xb') as f:np.savez_compressed(f,**arrays)
    (root/'frames_v2.json').write_text(encoded)
    print(json.dumps({'cloud_rows':{k:len(v) for k,v in clouds.items()},'state_received':state is not None}))
finally:node.destroy_node();rclpy.shutdown()
