"""Explicit FAST-LIO IMU-pose -> gravity-aligned base pose transforms."""
import numpy as np
from .frames import rpy_matrix, yaw_of


def base_pose(position, rotation, cfg):
    rb = np.asarray(rotation) @ np.asarray(cfg['R_base_imu']).T
    pb = np.asarray(position) - rb @ np.asarray(cfg['t_base_imu_m'])
    return pb, rb


def initial_alignment(position, rotation, imu_rotation, cfg):
    pb, rb = base_pose(position, rotation, cfg)
    aligned_body = rpy_matrix([0., 0., -yaw_of(imu_rotation)]) @ imu_rotation
    transform = aligned_body @ rb.T
    return transform, -transform @ pb


def transform_pose(position, rotation, covariance, alignment, offset, cfg):
    pb, rb = base_pose(position, rotation, cfg)
    c = np.asarray(covariance, float).reshape(6, 6)
    if not np.isfinite(c).all() or np.max(abs(c-c.T)) > 1e-5 or np.linalg.eigvalsh((c+c.T)/2).min() < -1e-6:
        raise ValueError('invalid_lio_covariance')
    # Pose covariance is interpreted in the incoming ROS world convention.
    # Include the IMU-to-base lever arm instead of copying IMU covariance.
    lever = rb @ np.asarray(cfg['t_base_imu_m'])
    x, y, z = lever
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    j = np.zeros((6, 6));j[:3, :3] = alignment;j[3:, 3:] = alignment;j[:3, 3:] = alignment @ skew
    return alignment @ pb + offset, alignment @ rb, j @ c @ j.T


def matrix_quaternion(r):
    # Stable wxyz extraction, including rotations near 180 degrees.
    r = np.asarray(r, float)
    candidates = np.array([1+np.trace(r), 1+r[0,0]-r[1,1]-r[2,2],
                           1-r[0,0]+r[1,1]-r[2,2], 1-r[0,0]-r[1,1]+r[2,2]])
    k = int(np.argmax(candidates));a = np.sqrt(max(0., candidates[k]))/2
    if k == 0:q = np.array([a, (r[2,1]-r[1,2])/(4*a), (r[0,2]-r[2,0])/(4*a), (r[1,0]-r[0,1])/(4*a)])
    elif k == 1:q = np.array([(r[2,1]-r[1,2])/(4*a), a, (r[0,1]+r[1,0])/(4*a), (r[0,2]+r[2,0])/(4*a)])
    elif k == 2:q = np.array([(r[0,2]-r[2,0])/(4*a), (r[0,1]+r[1,0])/(4*a), a, (r[1,2]+r[2,1])/(4*a)])
    else:q = np.array([(r[1,0]-r[0,1])/(4*a), (r[0,2]+r[2,0])/(4*a), (r[1,2]+r[2,1])/(4*a), a])
    return q / np.linalg.norm(q)
