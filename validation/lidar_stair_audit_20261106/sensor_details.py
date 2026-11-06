"""Bounded sensor-only capture of native lidar metadata and a few deskewed frames."""
import json,time
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from unitree_go.msg import LidarState,HeightMap
from go2_stairs.lidar_watch import parse_cloud

ROOT=Path('/home/unitree/go2-lidar-audit-20261106')
latest={};counts={};frames=[];poses=[];ticks=[]
def stamp(h):return h.stamp.sec+h.stamp.nanosec*1e-9
def cloud_callback(topic):
    def receive(m):
        counts[topic]=counts.get(topic,0)+1
        latest[topic]={'frame_id':m.header.frame_id,'stamp_s':stamp(m.header),'points':m.width*m.height,
            'fields':[{'name':f.name,'datatype':f.datatype,'offset':f.offset,'count':f.count} for f in m.fields]}
        if topic=='/utlidar/cloud_deskewed' and len(frames)<12 and (not ticks or stamp(m.header)-ticks[-1]>=.35):
            frames.append(parse_cloud(m));ticks.append(stamp(m.header))
    return receive
def odom(m):
    p=m.pose.pose.position;q=m.pose.pose.orientation
    poses.append((stamp(m.header),[p.x,p.y,p.z],[q.x,q.y,q.z,q.w]))
def lidar(m):
    keys=['stamp','firmware_version','software_version','sdk_version','sys_rotation_speed','com_rotation_speed',
          'error_state','cloud_frequency','cloud_packet_loss_rate','cloud_size','cloud_scan_num','imu_frequency','imu_packet_loss_rate']
    latest['lidar_state']={k:getattr(m,k) for k in keys}
def height(m):
    row={}
    for key in m.get_fields_and_field_types():
        value=getattr(m,key)
        if isinstance(value,(int,float,str,bool)):row[key]=value
        elif key=='header':row[key]={'frame_id':value.frame_id,'stamp_s':stamp(value)}
        else:
            try:
                a=np.asarray(value);row[key]={'shape':list(a.shape),'dtype':str(a.dtype)}
                if np.issubdtype(a.dtype,np.number) and a.size:
                    v=a[np.isfinite(a)];row[key].update(finite_count=int(v.size),min=float(v.min()) if v.size else None,max=float(v.max()) if v.size else None)
            except Exception:row[key]={'type':type(value).__name__}
    latest['height_map_array']=row
rclpy.init(args=[]);node=Node('go2_lidar_audit_details',enable_rosout=False,start_parameter_services=False)
try:
    subscriptions=[]
    for topic in ['/utlidar/cloud','/utlidar/cloud_base','/utlidar/cloud_deskewed','/utlidar/height_map']:
        subscriptions.append(node.create_subscription(PointCloud2,topic,cloud_callback(topic),qos_profile_sensor_data))
    subscriptions.append(node.create_subscription(Odometry,'/utlidar/robot_odom',odom,qos_profile_sensor_data))
    subscriptions.append(node.create_subscription(LidarState,'/utlidar/lidar_state',lidar,qos_profile_sensor_data))
    subscriptions.append(node.create_subscription(HeightMap,'/utlidar/height_map_array',height,qos_profile_sensor_data))
    until=time.monotonic()+8
    while time.monotonic()<until:rclpy.spin_once(node,timeout_sec=.02)
    out={'read_only':True,'motor_commands_sent':False,'cloud_counts':counts,'latest':latest,'capture_frames':len(frames)}
    if poses:
        positions=np.array([p[1] for p in poses]);out['odom_position_span_m']=np.ptp(positions,axis=0).tolist()
    if frames:
        kwargs={}
        for i,(points,ts) in enumerate(zip(frames,ticks)):
            pose=min(poses,key=lambda x:abs(x[0]-ts)) if poses else None
            kwargs[f'points_{i:02d}']=points
            kwargs[f'meta_{i:02d}']=np.array(json.dumps({'stamp_s':ts,'nearest_pose':pose}))
        with (ROOT/'detail_frames.npz').open('xb') as f:np.savez_compressed(f,**kwargs)
    (ROOT/'sensor_details.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2),flush=True)
finally:
    node.destroy_node();rclpy.shutdown()
