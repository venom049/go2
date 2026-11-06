"""ROS2 sensor-only supervisor. Sends local command proposals, never DDS LowCmd.

The native deskewed cloud has ALREADY applied the factory extrinsic. Native
robot_odom is the base pose, verified by point correspondence. Never apply the
URDF lidar transform a second time and never use the navigation-adapter odom.
"""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import time
import numpy as np
from .frames import Calibration,quaternion_matrix,yaw_of,rpy_matrix,parse_cloud
from .geometry import detect
from .supervisor import Supervisor
from .transport import encode,feedback

def xyz_cloud(m,expected_frame='odom'):
    # Same validated XYZ parser, with the native world frame explicitly accepted.
    if m.header.frame_id!=expected_frame:raise ValueError('unexpected deskew world frame')
    class Header:frame_id='base_link'
    class View:pass
    v=View();v.header=Header()
    for name in ('width','height','point_step','fields','row_step','data','is_bigendian'):setattr(v,name,getattr(m,name))
    return parse_cloud(v)

def stamp(m):return float(m.header.stamp.sec)+float(m.header.stamp.nanosec)*1e-9

def latest_pair(clouds,poses,tolerance):
    """An in-flight newer half must not invalidate a still-fresh complete pair."""
    if not poses:return None,None
    for cloud in reversed(clouds):
        pose=min(poses,key=lambda p:abs(p['stamp']-cloud['stamp']))
        if abs(pose['stamp']-cloud['stamp'])<=tolerance:return cloud,pose
    return None,None

def json_line(f,row):
    f.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n');f.flush()

def detection_job(points,pose,source_stamp):
    before=time.monotonic()
    return detect(points),pose,source_stamp,time.monotonic()-before

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--seconds',type=float,default=30.);ap.add_argument('--send-local',action='store_true')
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--config',type=Path);args=ap.parse_args()
    if not math.isfinite(args.seconds) or not 1<=args.seconds<=3600:ap.error('seconds must be 1..3600')
    root=args.root.resolve();config_path=args.config or root/'config/supervisor.json';cfg=json.loads(config_path.read_text());cal=Calibration(root/cfg['calibration'])
    onnx=root/'model/exported/policy.onnx'
    if hashlib.sha256(onnx.read_bytes()).hexdigest()!=cfg['original_onnx_sha256']:raise RuntimeError('original ONNX identity mismatch')
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import PointCloud2
    from nav_msgs.msg import Odometry
    from unitree_go.msg import LowState
    rclpy.init(args=[]);node=Node('go2_lidar_moe_supervisor',enable_rosout=False,start_parameter_services=False)
    state=None;poses=deque(maxlen=200);clouds=deque(maxlen=5);last_cloud_stamp=None;last_tick=None;backend_health=None
    error='';invalid_at=0.;running=True;supervisor=Supervisor(cfg);future=None;pool=ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context("spawn"))
    # Preserve strictly increasing sequence IDs across a sensor-only restart on
    # this host. The controller still rejects replay and stale source stamps.
    latest_geometry=None;geometry_at=0.;last_detection=0.;last_cloud=None;last_control=0.;sequence=time.monotonic_ns();active_code=None;active_ack_at=0.
    session=sequence;controller_epoch=0;future_epoch=0
    def fail(reason):
        nonlocal error,invalid_at
        error=reason;invalid_at=time.monotonic()
    def cloud_cb(m):
        nonlocal last_cloud_stamp,last_cloud
        try:
            ts=stamp(m);now=time.monotonic();age=time.time()-ts
            if not -.03<=age<=cfg['max_stamp_age_s']:raise ValueError('cloud_source_clock_or_age')
            if last_cloud_stamp is not None and ts<=last_cloud_stamp:raise ValueError('cloud_source_stamp_not_advancing')
            a=xyz_cloud(m,cfg['cloud_frame']);a=a[np.isfinite(a).all(1)]
            if len(a)<100:raise ValueError('cloud_too_sparse')
            # Bound history by source time, retaining only measured world points.
            row={'stamp':ts,'received':now,'source_mono':now-max(age,0.),'points':a}
            clouds.append(row);last_cloud=row;last_cloud_stamp=ts
            while clouds and ts-clouds[0]['stamp']>cfg['geometry_window_s']:clouds.popleft()
        except Exception as e:fail('cloud:'+str(e))
    def odom_cb(m):
        try:
            if m.header.frame_id!=cfg['cloud_frame'] or m.child_frame_id!=cfg['odom_child']:raise ValueError('native_odom_frame')
            ts=stamp(m);now=time.monotonic();age=time.time()-ts
            if not -.03<=age<=cfg['max_stamp_age_s']:raise ValueError('odom_source_clock_or_age')
            if poses and ts<=poses[-1]['stamp']:raise ValueError('odom_stamp_not_advancing')
            p=m.pose.pose.position;q=m.pose.pose.orientation;R=quaternion_matrix([q.w,q.x,q.y,q.z]);position=[float(p.x),float(p.y),float(p.z)]
            if not np.isfinite(position).all():raise ValueError('nonfinite_odom')
            speed=0.
            if poses:
                dt=ts-poses[-1]['stamp'];distance=np.linalg.norm(np.array(position[:2])-np.array(poses[-1]['position'][:2]))
                if distance>.025+1.2*dt or abs(position[2]-poses[-1]['position'][2])>.025+.8*dt:raise ValueError('odom_jump')
                speed=distance/dt
            velocity=(np.asarray(position)-np.asarray(poses[-1]['position']))/(ts-poses[-1]['stamp']) if poses else np.zeros(3)
            poses.append({'stamp':ts,'received':now,'source_mono':now-max(age,0.),'position':position,'rotation':R,'yaw':yaw_of(R),'speed':float(speed),'velocity_world':velocity})
        except Exception as e:fail('odom:'+str(e))
    def state_cb(m):
        nonlocal state,last_tick
        try:
            tick=int(m.tick)
            if last_tick is not None and tick==last_tick:return
            if last_tick is not None and ((tick-last_tick)&0xffffffff)>1000:raise ValueError('lowstate_tick_jump')
            last_tick=tick;q=[float(s.q) for s in m.motor_state[:12]];dq=[float(s.dq) for s in m.motor_state[:12]]
            imu=[float(x) for x in m.imu_state.quaternion];R=quaternion_matrix(imu)
            gyro=[float(x) for x in m.imu_state.gyroscope]
            if len(q)!=12 or not np.isfinite(q+dq+gyro).all():raise ValueError('invalid_lowstate')
            state={'received':time.monotonic(),'q':q,'dq':dq,'gyro':gyro,'imu':imu,'tilt':math.acos(np.clip(R[2,2],-1,1))}
        except Exception as e:fail('state:'+str(e))
    def backend_health_cb(m):
        nonlocal backend_health
        try:
            v=json.loads(m.data);age=time.time()-v['stamp']
            backend_health={'received':time.monotonic(),'healthy':v.get('healthy') is True and -.03<=age<=cfg['max_stamp_age_s'],'reason':v.get('reason')}
        except Exception as e:fail('backend_health:'+str(e))
    subs=[node.create_subscription(PointCloud2,cfg['cloud_topic'],cloud_cb,qos_profile_sensor_data),
          node.create_subscription(Odometry,cfg['odom_topic'],odom_cb,qos_profile_sensor_data),
          node.create_subscription(LowState,cfg['lowstate_topic'],state_cb,qos_profile_sensor_data)]
    if cfg.get('health_topic'):
        from std_msgs.msg import String
        subs.append(node.create_subscription(String,cfg['health_topic'],backend_health_cb,10))
    sock=None;client_path=root/cfg['client_socket']
    if args.send_local:
        if client_path.exists():raise RuntimeError('client socket exists; inspect prior process')
        sock=socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM);sock.setblocking(False);sock.bind(str(client_path));os.chmod(client_path,0o600)
    args.output.parent.mkdir(parents=True,exist_ok=True);log=args.output.open('x');start=time.monotonic()
    signal.signal(signal.SIGTERM,lambda *_:stop());signal.signal(signal.SIGINT,lambda *_:stop())
    def stop():
        nonlocal running
        running=False
    try:
        json_line(log,{'event':'start','readonly_motor':True,'send_local':args.send_local,'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),'calibration_sha256':hashlib.sha256((root/cfg['calibration']).read_bytes()).hexdigest()})
        while running and time.monotonic()-start<args.seconds:
            rclpy.spin_once(node,timeout_sec=.005);now=time.monotonic()
            if sock:
                for _ in range(8):
                    try:packet=sock.recv(128)
                    except BlockingIOError:break
                    # The controller can send after this loop's earlier `now`.
                    # Validate against the receive time; rejected packets must
                    # neither terminate sensing nor refresh the watchdog.
                    received_at=time.monotonic()
                    try:new_active,new_epoch=feedback(packet,received_at)
                    except ValueError as e:
                        json_line(log,{'event':'feedback_rejected','mono_s':received_at,'reason':str(e)})
                        continue
                    active_code=new_active;active_ack_at=received_at
                    if new_epoch!=controller_epoch:
                        controller_epoch=new_epoch
                        supervisor=Supervisor(cfg);geometry_at=0.;latest_geometry=None
                        clouds.clear();last_cloud=None;last_detection=0.
                        json_line(log,{'event':'handoff_geometry_reset','mono_s':received_at,'epoch':controller_epoch})
                now=time.monotonic()
            if future is not None and future.done():
                try:
                    g,pose,ts,cost=future.result();latest_geometry=g;geometry_at=now-max(time.time()-ts,0.)
                    if now-geometry_at>cfg['geometry_timeout_s']:raise ValueError('geometry_result_source_stale')
                    if future_epoch==controller_epoch:
                        supervisor.observe(g,pose,ts)
                        json_line(log,{'event':'geometry','mono_s':now,'source_stamp':ts,'duration_s':cost,'geometry':g})
                    else:geometry_at=0.;latest_geometry=None
                except Exception as e:fail('geometry:'+str(e))
                future=None
            if now-last_control<.05:continue
            last_control=now
            control_cloud,control_pose=latest_pair(clouds,poses,cfg['max_odom_cloud_delta_s'])
            fresh=bool(state and control_pose and control_cloud and now-state['received']<=cfg['max_state_age_s'] and
                       now-control_pose['received']<=cfg['max_odom_age_s'] and now-control_cloud['received']<=cfg['max_cloud_age_s'] and
                       now-control_cloud['source_mono']<=.28 and now-control_pose['source_mono']<=.28 and now-invalid_at>.3)
            if cfg.get('health_topic'):
                fresh=fresh and bool(backend_health and backend_health['healthy'] and now-backend_health['received']<=.25)
            leg_probes=None;leg_velocities=None
            if fresh:
                pose=control_pose;yaw=pose['yaw'];Rz=rpy_matrix([0,0,yaw]);position=np.array(pose['position'])
                wall_cloud=np.concatenate([c['points'] for c in clouds if 0<=control_cloud['stamp']-c['stamp']<=.10])
                current_points=(wall_cloud-position)@Rz
                feet=cal.feet_body(state['q'])@pose['rotation'].T@Rz
                leg_body=cal.leg_probes_body(state['q'])
                leg_probes=leg_body@pose['rotation'].T@Rz
                joint_motion=cal.leg_motion_body(state['q'],state['dq'])
                leg_velocities=(joint_motion+np.cross(state['gyro'],leg_body))@pose['rotation'].T@Rz+pose['velocity_world']@Rz
                if not supervisor.active and future is None and now-last_detection>=cfg['geometry_period_s']:
                    points=(np.concatenate([c['points'] for c in clouds if 0<=control_cloud['stamp']-c['stamp']<=cfg['geometry_window_s']])-position)@Rz
                    gp={'position':pose['position'].copy(),'yaw':yaw};ts=control_cloud['stamp']
                    future=pool.submit(detection_job,points,gp,ts);future_epoch=controller_epoch;last_detection=now
            else:pose={'position':[0.,0.,0.],'yaw':0.,'speed':0.};feet=np.zeros((4,3));current_points=np.empty((0,3))
            if supervisor.active and (active_code==2 or now-active_ack_at>.25):fresh=False;error='controller_feedback_stale_or_stopped'
            sensor_fresh=fresh
            if not supervisor.active and now-geometry_at>cfg['geometry_timeout_s']:fresh=False
            front=None
            if fresh and supervisor.map is not None:
                direction=pose['rotation'].T@np.r_[supervisor.map['n'],0.]
                front=cal.support_extent(state['q'],direction)+.04
            out=supervisor.update(pose,feet,current_points,now,fresh=fresh,active=active_code==1,speed=pose['speed'],tilt=state['tilt'] if state else 0.,front_extent=front,
                                  leg_probes=leg_probes,leg_velocities=leg_velocities,joint_speed=max(map(abs,state['dq'])) if state else None)
            # Readonly capture reports proposals but does not pretend the controller was armed.
            sequence+=1;source=min(control_cloud['source_mono'],pose['source_mono']) if sensor_fresh and control_cloud else now
            ages={'cloud_s':now-control_cloud['received'] if control_cloud else None,'odom_s':now-control_pose['received'] if control_pose else None,'lowstate_s':now-state['received'] if state else None}
            row={'event':'command','mono_s':now,'sequence':sequence,'session':session,'controller_epoch':controller_epoch,'source_age_s':now-source,'controller_active':active_code,'sensor_error':error,'sensor_ages':ages,'backend_health':backend_health,'sensor_healthy':sensor_fresh,'fresh':fresh,**out}
            if sock:
                try:sock.sendto(encode(sequence,source,out['command'],out['mode'],healthy=sensor_fresh,epoch=controller_epoch,session=session),str(root/cfg['controller_socket']));row['local_delivery']=True
                except (FileNotFoundError,ConnectionRefusedError,BlockingIOError):row['local_delivery']=False
            json_line(log,row)
        print(json.dumps({'output':str(args.output),'phase':supervisor.phase,'commands':sequence,'control_authorized':False}))
    finally:
        # When active, the C++ socket watchdog expires within 300 ms. Do not
        # invent a fresh sensor stamp to keep a stopped supervisor alive.
        # Python 3.8 may close its process-pool wakeup pipe before the manager
        # thread exits with wait=False. Join the bounded geometry job and worker.
        pool.shutdown(wait=True);log.close()
        if sock:sock.close();client_path.unlink(missing_ok=True)
        node.destroy_node();rclpy.shutdown()
    return 0

if __name__=='__main__':raise SystemExit(main())
