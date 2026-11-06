"""Sensor-only, source-time-preserving validated LIO output adapter."""
import argparse
from collections import deque
import copy
import json
import math
from pathlib import Path
import time
import numpy as np
from .frames import quaternion_matrix,rpy_matrix,yaw_of
from .lio_frames import initial_alignment, transform_pose, matrix_quaternion
from .ros_bridge import xyz_cloud, stamp


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    cfg=json.loads((a.root/'config/independent_lio.json').read_text())
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data,QoSProfile,ReliabilityPolicy
    from sensor_msgs.msg import PointCloud2,Imu,PointField
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String
    rclpy.init();node=Node('moe_lio_output_adapter',enable_rosout=False,start_parameter_services=False)
    imu=deque(maxlen=500);clouds=deque(maxlen=8);poses=deque(maxlen=8);qualities=deque(maxlen=8)
    cloud_pub=node.create_publisher(PointCloud2,'/moe_lio/cloud_world',qos_profile_sensor_data)
    odom_pub=node.create_publisher(Odometry,'/moe_lio/base_odom',qos_profile_sensor_data)
    health_pub=node.create_publisher(String,'/moe_lio/health',10)
    alignment=None;offset=None;imu_alignment=None;previous=None;last_source=0.;accepted=0;rejected=0;last_reason='initializing';fault=False
    def health(reason,ts=0.,healthy=False):
        nonlocal last_reason
        last_reason=reason
        msg=String();msg.data=json.dumps({'stamp':ts,'healthy':healthy,'reason':reason,'accepted':accepted,'rejected':rejected,'receipt_wall_s':time.time()});health_pub.publish(msg)
    def receive_imu(m):
        nonlocal fault
        ts=stamp(m)
        if imu and (ts<=imu[-1][0] or ts-imu[-1][0]>.05):fault=True
        imu.append((ts,m))
    imu_qos=QoSProfile(depth=500,reliability=ReliabilityPolicy.BEST_EFFORT)
    subs=[node.create_subscription(Imu,'/moe_lio/body_imu',receive_imu,imu_qos),
          node.create_subscription(PointCloud2,'/moe_lio/internal/cloud_registered',clouds.append,qos_profile_sensor_data),
          node.create_subscription(Odometry,'/moe_lio/internal/imu_odom',poses.append,qos_profile_sensor_data),
          node.create_subscription(String,'/moe_lio/quality',lambda m:qualities.append(json.loads(m.data)),10)]
    def process():
        nonlocal alignment,offset,imu_alignment,previous,last_source,accepted,rejected
        if not clouds or not poses or not qualities or not imu:return
        m=clouds[0];ts=stamp(m)
        odom=min(poses,key=lambda p:abs(stamp(p)-ts));quality=min(qualities,key=lambda q:abs(q['stamp']-ts))
        if abs(stamp(odom)-ts)>1e-5 or abs(quality['stamp']-ts)>1e-5:
            if ts < min(stamp(poses[-1]),qualities[-1]['stamp'])-.02:clouds.popleft()
            return
        clouds.popleft()
        try:
            if fault:raise ValueError('imu_discontinuity_restart_required')
            if not -.03<=time.time()-ts<=cfg['source_age_limit_s'] or ts<=last_source:raise ValueError('lio_source_age_or_regression')
            if not quality['healthy']:raise ValueError('lio_quality_degenerate')
            if m.header.frame_id!='camera_init' or odom.header.frame_id!='camera_init' or odom.child_frame_id!='body':raise ValueError('lio_frame')
            it,im=min(imu,key=lambda row:abs(row[0]-ts))
            if abs(it-ts)>.02:raise ValueError('imu_lidar_source_alignment')
            iq=im.orientation;ri=quaternion_matrix([iq.w,iq.x,iq.y,iq.z])
            op=odom.pose.pose.position;oq=odom.pose.pose.orientation
            p=np.array([op.x,op.y,op.z]);r=quaternion_matrix([oq.w,oq.x,oq.y,oq.z])
            if alignment is None:
                alignment,offset=initial_alignment(p,r,ri,cfg)
                imu_alignment=rpy_matrix([0.,0.,-yaw_of(ri)])
            bp,br,c=transform_pose(p,r,odom.pose.covariance,alignment,offset,cfg)
            if not np.isfinite(bp).all():raise ValueError('nonfinite_lio_pose')
            if max(np.diag(c)[:3])>cfg['max_position_covariance_m2'] or max(np.diag(c)[3:])>cfg['max_orientation_covariance_rad2']:raise ValueError('lio_covariance_excess')
            if math.acos(float(np.clip(np.dot(br[2],ri[2]),-1.,1.)))>.15:raise ValueError('lio_gravity_inconsistent_with_body_imu')
            orientation_delta=(imu_alignment@ri).T@br
            if math.acos(float(np.clip((np.trace(orientation_delta)-1)/2,-1.,1.)))>.10:raise ValueError('lio_orientation_inconsistent_with_body_imu')
            if previous:
                dt=ts-previous[0]
                if dt<=0 or dt>.25 or np.linalg.norm(bp-previous[1])>.025+1.2*dt:raise ValueError('lio_pose_discontinuity')
                relative=previous[2].T@br
                if math.acos(float(np.clip((np.trace(relative)-1)/2,-1.,1.)))>.04+2.*dt:raise ValueError('lio_rotation_discontinuity')
            points=xyz_cloud(m,'camera_init')@alignment.T+offset
            points=points[np.isfinite(points).all(1)]
            if len(points)<100:raise ValueError('lio_cloud_sparse')
            out=PointCloud2();out.header=copy.deepcopy(m.header);out.header.frame_id=cfg['world_frame']
            out.height=1;out.width=len(points);out.is_bigendian=False;out.is_dense=True;out.point_step=12;out.row_step=12*len(points)
            out.fields=[PointField(name=n,offset=i*4,datatype=7,count=1) for i,n in enumerate(('x','y','z'))]
            out.data=np.asarray(points,dtype='<f4').tobytes()
            pose=Odometry();pose.header=copy.deepcopy(out.header);pose.child_frame_id=cfg['body_frame']
            pose.pose.pose.position.x=float(bp[0]);pose.pose.pose.position.y=float(bp[1]);pose.pose.pose.position.z=float(bp[2])
            q=matrix_quaternion(br);pose.pose.pose.orientation.w=float(q[0]);pose.pose.pose.orientation.x=float(q[1]);pose.pose.pose.orientation.y=float(q[2]);pose.pose.pose.orientation.z=float(q[3]);pose.pose.covariance=c.reshape(-1).tolist()
            odom_pub.publish(pose);cloud_pub.publish(out);last_source=ts;previous=(ts,bp,br);accepted+=1;health('validated',ts,True)
        except (ValueError,KeyError) as e:rejected+=1;health(str(e),ts)
    last_health=0.
    try:
        while rclpy.ok():
            rclpy.spin_once(node,timeout_sec=.005);process()
            if time.monotonic()-last_health>.2:
                if time.time()-last_source>.25:health(last_reason if rejected else 'waiting_for_matching_lio_inputs')
                last_health=time.monotonic()
    except KeyboardInterrupt:pass
    finally:node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
