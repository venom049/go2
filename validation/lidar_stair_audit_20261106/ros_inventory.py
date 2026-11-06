"""Receive-only ROS graph inventory, with no command publishers or clients."""
import json
import time
import rclpy
from rclpy.node import Node

rclpy.init(args=[])
node=Node('go2_lidar_audit_inventory',enable_rosout=False,start_parameter_services=False)
try:
    end=time.monotonic()+5
    while time.monotonic()<end:
        rclpy.spin_once(node,timeout_sec=.1)
    topics={}
    for name,types in node.get_topic_names_and_types():
        if any(key in name.lower() for key in ('lidar','cloud','odom','imu','height','range','scan')):
            pubs=[]
            for item in node.get_publishers_info_by_topic(name):
                pubs.append(dict(node_name=item.node_name,node_namespace=item.node_namespace,
                                 topic_type=item.topic_type,
                                 qos_reliability=str(item.qos_profile.reliability),
                                 qos_durability=str(item.qos_profile.durability)))
            topics[name]={'types':types,'publishers':pubs}
    print(json.dumps({'read_only':True,'topics':topics},indent=2),flush=True)
finally:
    node.destroy_node()
    rclpy.shutdown()
