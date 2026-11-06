"""Five-second receipt/source timestamp audit; subscriptions only."""
import json,time,sys
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
rclpy.init(args=[]);node=Node('go2_lidar_sync_clock_audit',enable_rosout=False,start_parameter_services=False);rows={}
def callback(name):
 def receive(m):
  mono=time.monotonic();wall=time.time();ts=float(m.header.stamp.sec)+m.header.stamp.nanosec*1e-9
  rows.setdefault(name,[]).append({'received_mono':mono,'received_wall':wall,'source_stamp':ts,'age_s':wall-ts,'frame':m.header.frame_id})
 return receive
subs=[node.create_subscription(PointCloud2,'/utlidar/cloud_deskewed',callback('cloud_deskewed'),qos_profile_sensor_data),node.create_subscription(Odometry,'/utlidar/robot_odom',callback('robot_odom'),qos_profile_sensor_data)]
try:
 until=time.monotonic()+5.
 while time.monotonic()<until:rclpy.spin_once(node,timeout_sec=.01)
 summary={}
 for name,data in rows.items():
  summary[name]={'count':len(data),'age_quantiles_s':np.quantile([r['age_s'] for r in data],[0,.5,.95,1]).tolist(),'source_span_s':data[-1]['source_stamp']-data[0]['source_stamp'],'receipt_span_s':data[-1]['received_mono']-data[0]['received_mono'],'stamps_strictly_increasing':all(b['source_stamp']>a['source_stamp'] for a,b in zip(data,data[1:]))}
 out={'subscriptions_only':True,'summary':summary,'samples':rows};Path(sys.argv[1]).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summary,indent=2))
finally:node.destroy_node();rclpy.shutdown()
