"""Independent L1 deskew + measured-attitude, translation-only surface matching.

No SportModeState, factory odometry, raw lidar IMU convention or LowCmd is used.
The IMU quaternion constrains all three rotation axes. No acceleration is
double-integrated into an unobservable translation estimate.
"""
import argparse
from collections import deque
import json,math
from pathlib import Path
import time
import numpy as np
from .frames import quaternion_matrix,parse_cloud
from .lio_frames import matrix_quaternion
from .translation_registration import SurfaceMap,voxel,interpolate_quaternions,rotate_points
from .ros_bridge import stamp


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    cfg=json.loads((a.root/'config/independent_lio.json').read_text())
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data,QoSProfile,ReliabilityPolicy
    from sensor_msgs.msg import Imu,PointCloud2,PointField
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String
    rclpy.init();node=Node('moe_attitude_lio',enable_rosout=False,start_parameter_services=False)
    imus=deque(maxlen=1500);clouds=deque(maxlen=4);surface=None;keyframes=deque(maxlen=12);recent_scans=deque(maxlen=3)
    position=np.zeros(3);velocity=np.zeros(3);last_stamp=0.;last_cloud_stamp=0.;key_position=None;key_rotation=None;failed=False;count=0;initial_scans=[];initial_rotation=None;cached_cloud=None;cached_times=None;cached_end=None
    cloud_pub=node.create_publisher(PointCloud2,'/moe_lio/internal/cloud_registered',qos_profile_sensor_data)
    odom_pub=node.create_publisher(Odometry,'/moe_lio/internal/imu_odom',qos_profile_sensor_data)
    quality_pub=node.create_publisher(String,'/moe_lio/quality',10)
    def quality(ts,healthy,reason,**values):
        msg=String();msg.data=json.dumps({'stamp':ts,'healthy':healthy,'reason':reason,'algorithm':'body_attitude_translation_registration',**values});quality_pub.publish(msg)
    def imu_cb(m):
        nonlocal failed
        ts=stamp(m);q=m.orientation;quat=[q.w,q.x,q.y,q.z]
        if imus and (ts<=imus[-1][0] or ts-imus[-1][0]>.05):failed=True;quality(ts,False,'imu_discontinuity_restart_required')
        imus.append((ts,quat))
    subs=[node.create_subscription(Imu,'/moe_lio/body_imu',imu_cb,QoSProfile(depth=500,reliability=ReliabilityPolicy.BEST_EFFORT)),
          node.create_subscription(PointCloud2,cfg['raw_cloud_topic'],clouds.append,qos_profile_sensor_data)]
    def process():
        nonlocal surface,position,velocity,last_stamp,last_cloud_stamp,key_position,key_rotation,failed,count,initial_rotation,cached_cloud,cached_times,cached_end
        if failed or not clouds or len(imus)<2:return
        # Bound compute latency: discarded scans are never marked processed.
        # The accepted newest scan must still satisfy source-age and IMU gates.
        while len(clouds)>1:clouds.popleft()
        msg=clouds[0];ts=stamp(msg)
        try:
            age=time.time()-ts
            if msg.header.frame_id!='utlidar_lidar':raise ValueError('raw_source_frame')
            if not last_stamp and age>.25:
                clouds.popleft();last_cloud_stamp=ts
                quality(ts,False,'startup_discard_stale_scan');return
            if not -.03<=age<=.25:raise ValueError('raw_source_age')
            if ts<=last_cloud_stamp:raise ValueError('raw_source_regression')
            if cached_cloud is not msg:
                fields={f.name:f for f in msg.fields}
                if 'time' not in fields or fields['time'].datatype!=7 or fields['time'].count!=1:raise ValueError('l1_measured_point_time_missing')
                times=np.ndarray((msg.height,msg.width),dtype=('>' if msg.is_bigendian else '<')+'f4',buffer=memoryview(msg.data),offset=fields['time'].offset,strides=(msg.row_step,msg.point_step)).reshape(-1).astype(float)
                if not np.isfinite(times).all() or times.min()<0 or not .03<times.max()<.10:raise ValueError('l1_point_time_seconds')
                cached_cloud=msg;cached_times=times;cached_end=ts+float(times.max())
            times=cached_times;end=cached_end
            if ts<imus[0][0]:
                clouds.popleft();last_cloud_stamp=ts
                quality(ts,False,'waiting_full_imu_scan_coverage');return
            if imus[-1][0]<end:return
            clouds.popleft();last_cloud_stamp=ts
            raw=parse_cloud(msg);ranges=np.linalg.norm(raw,axis=1);valid=np.isfinite(raw).all(1)&(ranges>.30)&(ranges<12.)
            if valid.sum()<500 or valid.mean()<.20:raise ValueError('near_field_dominated_restart_required')
            raw=raw[valid];point_time=ts+times[valid]
            it=np.array([x[0] for x in imus]);iq=np.array([x[1] for x in imus]);q=interpolate_quaternions(it,iq,point_time)
            body=raw@np.asarray(cfg['R_base_raw']).T+cfg['t_base_raw_m']
            world=rotate_points(q,body)
            end_q=interpolate_quaternions(it,iq,np.array([end]))[0];rotation=quaternion_matrix(end_q)
            recent_scans.append((world,point_time))
            while recent_scans and end-float(recent_scans[0][1].max())>.22:recent_scans.popleft()
            if not last_stamp:
                # L1 scan phases observe different patches. Build a stationary
                # multi-scan reference instead of aligning every phase to one
                # incomplete first scan. The caller must hold a stable stand.
                if initial_rotation is None:initial_rotation=rotation.copy()
                if math.acos(float(np.clip((np.trace(initial_rotation.T@rotation)-1)/2,-1,1)))>.03:raise ValueError('initialization_requires_stable_stand')
                initial_scans.append(voxel(world))
                if len(initial_scans)<6:
                    quality(end,False,'stationary_multiscan_initialization');return
                reference=voxel(np.concatenate(initial_scans));keyframes.append(reference);surface=SurfaceMap(reference)
                key_position=position.copy();key_rotation=rotation.copy();last_stamp=end
                quality(end,False,'initial_map_waiting_for_registration');return
            dt=end-last_stamp
            if not 0<dt<=.25:raise ValueError('cloud_gap_restart_required')
            predicted=world+position+(point_time-last_stamp)[:,None]*velocity
            # Consecutive L1 phases do not each cover all surfaces. A sliding
            # three-scan window preserves measured times and uses a short
            # constant-translation-velocity trajectory with measured attitude.
            # Publish only the newest scan and its actual source end stamp.
            window=np.concatenate([points+position+(times-last_stamp)[:,None]*velocity for points,times in recent_scans])
            correction,status=surface.register(window)
            new_position=position+velocity*dt+correction
            if np.linalg.norm(new_position-position)>.015+1.2*dt:raise ValueError('registration_pose_jump')
            measured_velocity=(new_position-position)/dt
            velocity=.4*measured_velocity+.6*velocity
            aligned=predicted+correction;position=new_position;last_stamp=end;count+=1
            if np.linalg.norm(position-key_position)>.10 or math.acos(float(np.clip((np.trace(key_rotation.T@rotation)-1)/2,-1,1)))>.10:
                keyframes.append(voxel(aligned));surface=SurfaceMap(np.concatenate(keyframes));key_position=position.copy();key_rotation=rotation.copy()
            out=PointCloud2();out.header.frame_id='camera_init';ns=int(round(end*1e9));out.header.stamp.sec=ns//1000000000;out.header.stamp.nanosec=ns%1000000000
            out.height=1;out.width=len(aligned);out.point_step=12;out.row_step=12*len(aligned);out.is_dense=True
            out.fields=[PointField(name=n,offset=i*4,datatype=7,count=1) for i,n in enumerate(('x','y','z'))];out.data=np.asarray(aligned,dtype='<f4').tobytes()
            odom=Odometry();odom.header=out.header;odom.child_frame_id='body'
            ip=position+rotation@np.asarray(cfg['t_base_imu_m']);odom.pose.pose.position.x=float(ip[0]);odom.pose.pose.position.y=float(ip[1]);odom.pose.pose.position.z=float(ip[2])
            odom.pose.pose.orientation.w=float(end_q[0]);odom.pose.pose.orientation.x=float(end_q[1]);odom.pose.pose.orientation.y=float(end_q[2]);odom.pose.pose.orientation.z=float(end_q[3])
            cov=np.zeros((6,6));cov[:3,:3]=status.pop('covariance')+np.eye(3)*.02**2;cov[3:,3:]=np.eye(3)*.001;odom.pose.covariance=cov.reshape(-1).tolist()
            quality(end,True,'registered',**status);odom_pub.publish(odom);cloud_pub.publish(out)
        except (ValueError,KeyError) as e:
            failed=True;quality(ts,False,str(e));node.get_logger().error(str(e))
    try:
        while rclpy.ok():
            rclpy.spin_once(node,timeout_sec=.001);process()
            if failed:return 2
    except KeyboardInterrupt:pass
    finally:node.destroy_node();rclpy.shutdown()
    return 0


if __name__=='__main__':raise SystemExit(main())
