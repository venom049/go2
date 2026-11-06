"""Reject predicted lower-leg/riser contact; never invent a swing trajectory.

Constant-velocity extrapolation is a risk detector, not a contact-free planner.
The original blind MoE remains responsible for leg motion. Unknown clearance
or insufficient lift causes a stop rather than permission to push into a riser.
"""
import math
import numpy as np


def stance_alignment(feet,normal,distance,*,foot_radius,max_error):
    """Initial supported stance must agree with the observed ground plane.

    This is a rejection check, never an extrinsic estimator or a contact sensor.
    Coordinates are all world coordinates; the plane is n.dot(point) = distance.
    """
    p=np.asarray(feet,float);n=np.asarray(normal,float)
    if p.shape!=(4,3) or n.shape!=(3,) or not np.isfinite(np.r_[p.ravel(),n,distance,foot_radius,max_error]).all():
        raise ValueError('invalid_stance_alignment_inputs')
    size=float(np.linalg.norm(n))
    if not .99<size<1.01 or n[2]<.98 or foot_radius<=0 or max_error<=0:
        raise ValueError('invalid_stance_ground_plane')
    gaps=(p@n-float(distance))/size-float(foot_radius)
    return {'verified':bool(np.max(abs(gaps))<=max_error),
            'inferred_sole_gap_m':gaps.tolist(),'max_allowed_error_m':max_error,
            'method':'initial_urdf_foot_sphere_to_measured_ground_rejection_only'}


def assess(probes,velocities,*,normal,edges,heights,floor_z,body_z,speed,config):
    p=np.asarray(probes,float);v=np.asarray(velocities,float);n=np.asarray(normal,float)
    if p.shape!=(4,17,3) or v.shape!=p.shape or n.shape!=(2,) or not np.isfinite(np.r_[p.ravel(),v.ravel(),n,speed,body_z,floor_z]).all():
        raise ValueError('invalid_leg_clearance_inputs')
    radius=config['capsule_radius_m'];uncertainty=config['geometry_margin_m'];margin=radius+uncertainty
    horizon=config['prediction_horizon_s'];body_speed=max(0.,float(speed))
    stopping=config['reaction_s']*body_speed+body_speed**2/(2*config['deceleration_m_s2'])+margin
    projection=p[...,:2]@n;closing=v[...,:2]@n
    minimum=float('inf');risks=[]
    for index,(edge,top) in enumerate(zip(edges,np.cumsum(heights)+floor_z)):
        bottom=floor_z+sum(heights[:index]);distance=edge-projection
        height=p[...,2]+body_z
        overlap=(height+margin>bottom+.015)&(height-margin<top+config['toe_lift_margin_m'])
        ahead=distance>=-margin
        eligible=overlap&ahead
        if eligible.any():minimum=min(minimum,float(np.min(distance[eligible]-margin)))
        for leg,point in np.argwhere(ahead):
            d=float(distance[leg,point]-margin);vn=float(closing[leg,point]);z=float(height[leg,point]);vz=float(v[leg,point,2])
            crossing=d/vn if vn>.02 and d>=0. else (0. if d<0. else math.inf)
            cross_z=z+vz*crossing if math.isfinite(crossing) else z
            predicted=0<=crossing<=horizon and bottom-margin<cross_z<top+margin+config['toe_lift_margin_m']
            near=bool(eligible[leg,point] and d<stopping-margin and vn>.02 and z+max(vz,0.)*horizon<top+margin+config['toe_lift_margin_m'])
            if predicted or near:
                risks.append({'leg':('FR','FL','RR','RL')[leg],'probe':int(point),'riser':index+1,
                              'clearance_m':d,'closing_m_s':vn,'crossing_s':crossing if math.isfinite(crossing) else None,
                              'height_at_crossing_m':cross_z,'riser_top_z_m':float(top)})
    return {'risk':bool(risks),'min_leg_riser_clearance_m':minimum if math.isfinite(minimum) else None,
            'body_stopping_distance_m':stopping,'prediction_horizon_s':horizon,'risks':risks[:8],
            'method':'constant_velocity_capsule_riser_risk_only'}
