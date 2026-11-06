"""Receive-only command/state recorder; no publishers or motion SDK."""
import json,time,sys,signal
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from unitree_go.msg import LowState,LowCmd
running=True
signal.signal(signal.SIGTERM,lambda *_:globals().__setitem__('running',False))
signal.signal(signal.SIGINT,lambda *_:globals().__setitem__('running',False))
rclpy.init(args=[]);n=Node('original_moe_retest_observer',enable_rosout=False,start_parameter_services=False);path=Path(sys.argv[1]);path.parent.mkdir(parents=True,exist_ok=True);f=path.open('x');counts={'state':0,'command':0}
def record(event,row):
 counts[event]+=1;f.write(json.dumps({'event':event,'mono_s':time.monotonic(),'wall_s':time.time(),**row},separators=(',',':'))+'\n')
def state(m):record('state',{'tick':m.tick,'q':[float(x.q) for x in m.motor_state[:12]],'dq':[float(x.dq) for x in m.motor_state[:12]],'tau_est':[float(x.tau_est) for x in m.motor_state[:12]],'quat':[float(x) for x in m.imu_state.quaternion],'gyro':[float(x) for x in m.imu_state.gyroscope]})
def command(m):record('command',{key:[float(getattr(x,key)) for x in m.motor_cmd[:12]] for key in ['q','dq','kp','kd','tau']})
s=[n.create_subscription(LowState,'/lowstate',state,qos_profile_sensor_data),n.create_subscription(LowCmd,'/lowcmd',command,qos_profile_sensor_data)]
try:
 print('READ_ONLY_OBSERVER_READY seconds=180',flush=True);end=time.monotonic()+180;last=time.monotonic()
 while running and time.monotonic()<end:
  rclpy.spin_once(n,timeout_sec=.01)
  if time.monotonic()-last>1:f.flush();last=time.monotonic()
finally:f.close();n.destroy_node();rclpy.shutdown();print(json.dumps({'output':str(path),'counts':counts}),flush=True)
