"""Read-only observation of published TF and camera metadata; no services or commands."""
import json,time
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,QoSDurabilityPolicy,QoSReliabilityPolicy,qos_profile_sensor_data
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import CameraInfo

root=Path('/home/unitree/go2-calibration-audit-20261106')
transforms={};camera={};subscriptions=[]
def tf_callback(m):
    for t in m.transforms:
        a=t.transform.translation;q=t.transform.rotation
        transforms[t.header.frame_id+' -> '+t.child_frame_id]={
            'translation_m':[a.x,a.y,a.z],'quaternion_xyzw':[q.x,q.y,q.z,q.w]}
def camera_callback(topic):
    def receive(m):
        camera[topic]={'frame_id':m.header.frame_id,'width':m.width,'height':m.height,
            'distortion_model':m.distortion_model,'k':list(m.k),'d':list(m.d),'p':list(m.p)}
    return receive
rclpy.init(args=[]);node=Node('go2_calibration_audit',enable_rosout=False,start_parameter_services=False)
try:
    static_qos=QoSProfile(depth=100,durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                          reliability=QoSReliabilityPolicy.RELIABLE)
    subscriptions.append(node.create_subscription(TFMessage,'/tf_static',tf_callback,static_qos))
    start=time.monotonic()
    while time.monotonic()-start<2:rclpy.spin_once(node,timeout_sec=.1)
    topics=node.get_topic_names_and_types()
    info_topics=[n for n,t in topics if 'sensor_msgs/msg/CameraInfo' in t]
    for name in info_topics:
        subscriptions.append(node.create_subscription(CameraInfo,name,camera_callback(name),qos_profile_sensor_data))
    until=time.monotonic()+5
    while time.monotonic()<until:rclpy.spin_once(node,timeout_sec=.1)
    graph=[]
    for name,typ in topics:
        if name=='/tf_static' or name in info_topics:
            graph.append({'topic':name,'types':typ,'publishers':[
                {'node_name':p.node_name,'node_namespace':p.node_namespace}
                for p in node.get_publishers_info_by_topic(name)]})
    out={'read_only':True,'duration_s':time.monotonic()-start,'static_transforms_received':transforms,
         'camera_info_received':camera,'graph':graph,'note':'No transform received is not proof that sensor firmware has no calibration.'}
    (root/'live_frames.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2),flush=True)
finally:
    node.destroy_node();rclpy.shutdown()
