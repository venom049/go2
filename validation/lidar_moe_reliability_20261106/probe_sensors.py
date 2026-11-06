import json,time,collections,sys
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2,Imu
from nav_msgs.msg import Odometry
from unitree_go.msg import LowState,SportModeState,LidarState
from std_msgs.msg import String
rclpy.init(args=[]);n=Node('moe_dependency_probe',enable_rosout=False,start_parameter_services=False);counts=collections.Counter();first={};last={};meta={};dtypes={1:'i1',2:'u1',3:'<i2',4:'<u2',5:'<i4',6:'<u4',7:'<f4',8:'<f8'}
ps=[('/utlidar/'+t,PointCloud2) for t in ('cloud','cloud_base','cloud_deskewed')]+[('/utlidar/robot_odom',Odometry),('/utlidar/imu',Imu),('/utlidar/lidar_state',LidarState),('/sportmodestate',SportModeState),('/lf/sportmodestate',SportModeState),('/lowstate',LowState),('/utlidar/switch',String),('/utlidar/server_log',String)]
def cb(p):
 def f(m):
  now=time.monotonic();counts[p]+=1;first.setdefault(p,now);last[p]=now
  if p not in meta:
   d={'type':type(m).__name__}
   if hasattr(m,'header'):d.update(frame=m.header.frame_id,source_stamp=m.header.stamp.sec+m.header.stamp.nanosec*1e-9,age_s=time.time()-m.header.stamp.sec-m.header.stamp.nanosec*1e-9)
   if isinstance(m,PointCloud2):
    d.update(fields=[{'name':f.name,'offset':f.offset,'datatype':f.datatype,'count':f.count} for f in m.fields],point_step=m.point_step,width=m.width,height=m.height)
    for field in m.fields:
     if field.count==1:
      a=np.ndarray((m.height,m.width),dtype=dtypes[field.datatype],buffer=memoryview(m.data),offset=field.offset,strides=(m.row_step,m.point_step)).reshape(-1);v=a[np.isfinite(a)];d[field.name+'_range']=[float(v.min()),float(v.max())] if len(v) else []
   if isinstance(m,Imu):d.update(orientation=[m.orientation.w,m.orientation.x,m.orientation.y,m.orientation.z],angular_velocity=[m.angular_velocity.x,m.angular_velocity.y,m.angular_velocity.z],linear_acceleration=[m.linear_acceleration.x,m.linear_acceleration.y,m.linear_acceleration.z])
   if isinstance(m,Odometry):d.update(child=m.child_frame_id)
   if isinstance(m,(LidarState,String)):d['value']=str(m)
   meta[p]=d
 return f
subs=[n.create_subscription(t,p,cb(p),qos_profile_sensor_data) for p,t in ps];start=time.monotonic()
try:
 while time.monotonic()-start<6:rclpy.spin_once(n,timeout_sec=.01)
 endpoints={}
 for p,t in ps:
  endpoints[p]={}
  for k,fn in [('publishers',n.get_publishers_info_by_topic),('subscribers',n.get_subscriptions_info_by_topic)]:
   endpoints[p][k]=[{'node':x.node_name,'namespace':x.node_namespace,'gid':list(x.endpoint_gid),'type':x.topic_type,'reliability':str(x.qos_profile.reliability)} for x in fn(p)]
 result={'started_mono':start,'counts':dict(counts),'meta':meta,'endpoints':endpoints};Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'counts':dict(counts),'output':sys.argv[1]}))
finally:n.destroy_node();rclpy.shutdown()
