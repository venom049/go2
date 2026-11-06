"""Geometry-to-command supervisor; no robot transport or actuator calls."""
import math
import numpy as np
from .leg_clearance import assess,stance_alignment

def rotation(yaw):return np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
def wrap(x):return (x+math.pi)%(2*math.pi)-math.pi

class Supervisor:
    def __init__(self,config):
        self.cfg=config;self.map=None;self.count=0;self.last_geometry_stamp=None
        self.phase='WAIT';self.active=False;self.stopped=False;self.stop_reason=''
        self.previous=np.zeros(3);self.last_now=None;self.settled_since=None

    def observe(self,geometry,pose,source_stamp):
        if self.active:return
        if not geometry.get('valid'):
            self.count=0;self.map=None;return
        if self.last_geometry_stamp is not None and source_stamp<=self.last_geometry_stamp:return
        n=rotation(pose['yaw'])@np.array(geometry['normal_xy'])
        floor_normal=np.r_[rotation(pose['yaw'])@np.asarray(geometry['floor']['normal'][:2]),geometry['floor']['normal'][2]]
        p=np.asarray(pose['position'][:2]);candidate={
            'n':n,'edges':np.array([geometry['first_edge_m'],geometry['second_edge_m'],geometry['wall_distance_m']])+np.dot(n,p),
            'center':p+rotation(pose['yaw'])@np.array([0.,geometry['lateral_center_m']]),
            'floor_z':pose['position'][2]+geometry['floor']['distance']/geometry['floor']['normal'][2],
            'floor_normal_world':floor_normal,
            'floor_distance_world':float(geometry['floor']['distance']+floor_normal@np.asarray(pose['position'])),
            'top_height':sum(geometry['step_heights_m']),'geometry':geometry,'source_stamp':source_stamp}
        if self.map is not None:
            consistent=(np.dot(self.map['n'],n)>math.cos(.08) and
                        np.max(abs(self.map['edges']-candidate['edges']))<.07 and
                        source_stamp-self.last_geometry_stamp<1.5)
            self.count=self.count+1 if consistent else 1
        else:self.count=1
        self.map=candidate;self.last_geometry_stamp=source_stamp

    def _output(self,mode,command,reason,now,extra=None):
        dt=.1 if self.last_now is None else max(0.,min(.15,now-self.last_now));self.last_now=now
        target=np.array(command,float);rates=np.array([self.cfg['acceleration_m_s2'],.25,.8])
        self.previous+=np.clip(target-self.previous,-rates*dt,rates*dt)
        if mode in ('FAULT','HOLD'):self.previous[:]=0
        out={'mode':mode,'phase':self.phase,'reason':reason,'command':self.previous.tolist(),
             'confirmed_frames':self.count,'ready':mode=='RUN' and self.map is not None and self.count>=3 and not self.stopped}
        if extra:out.update(extra)
        return out

    def update(self,pose,feet,points,now,*,fresh,active=False,speed=0.,tilt=0.,front_extent=None,leg_probes=None,leg_velocities=None,joint_speed=None):
        if not math.isfinite(now) or (self.last_now is not None and now<self.last_now):
            self.stopped=True;self.stop_reason='clock_regression'
        if self.stopped:return self._output('FAULT',[0,0,0],self.stop_reason,now)
        if self.phase=='OBSTACLE_HOLD':
            return self._output('HOLD',[0,0,0],'riser_risk_stop_requires_manual_reset',now)
        if self.phase in ('COMPLETE','BLOCKED'):
            return self._output('HOLD',[0,0,0],'parked_on_platform' if self.phase=='COMPLETE' else 'wall_stop_before_platform_completion',now)
        if not fresh:
            if self.active:self.stopped=True;self.stop_reason='sensor_stale'
            self.phase='WAIT' if not self.active else 'FAULT'
            return self._output('FAULT' if self.active else 'WAIT',[0,0,0],'sensor_stale',now)
        if self.map is None or self.count<3:
            self.phase='WAIT';return self._output('WAIT',[0,0,0],'need_three_consistent_geometry_frames',now)
        if self.active and not active:
            self.stopped=True;self.stop_reason='controller_stopped'
            return self._output('FAULT',[0,0,0],self.stop_reason,now)
        n=rotation(-pose['yaw'])@self.map['n'];p=np.array(pose['position'][:2])
        distances=self.map['edges']-np.dot(self.map['n'],p)
        heading=math.atan2(n[1],n[0]);lateral=float((rotation(-pose['yaw'])@(self.map['center']-p))[1])
        feet=np.asarray(feet,float);foot_projection=feet[:,:2]@n
        first_clearance=float(distances[0]-max(foot_projection[:2])-.025)
        envelope=max(.37,float(max(foot_projection)+.035),float(front_extent) if front_extent is not None else 0.)
        wall_clearance=float(distances[2]-envelope)
        # Current unfilled point support must corroborate the stored wall.
        points=np.asarray(points,float);wall_support=points[abs(points[:,:2]@n-distances[2])<.06]
        wall_seen=len(wall_support)>=12 and np.ptp(wall_support[:,2])>.12
        extra={'first_edge_body_m':float(distances[0]),'first_edge_front_foot_m':first_clearance,
               'second_edge_body_m':float(distances[1]),'wall_body_m':float(distances[2]),
               'wall_clearance_m':wall_clearance,'heading_error_rad':heading,'lateral_error_m':lateral,
               'wall_support_points':len(wall_support)}
        if not wall_seen:
            if self.active:self.stopped=True;self.stop_reason='wall_support_lost'
            return self._output('FAULT' if self.active else 'WAIT',[0,0,0],'wall_support_lost',now,extra)
        if not self.active:
            alignment_cfg=self.cfg.get('initial_stance_alignment')
            if alignment_cfg and alignment_cfg.get('enabled'):
                if 'floor_normal_world' not in self.map or 'floor_distance_world' not in self.map:
                    return self._output('WAIT',[0,0,0],'floor_foot_alignment_missing',now,extra)
                world_feet=feet.copy();world_feet[:,:2]=feet[:,:2]@rotation(pose['yaw']).T
                world_feet+=np.asarray(pose['position'])
                check=stance_alignment(world_feet,self.map['floor_normal_world'],self.map['floor_distance_world'],
                                       foot_radius=alignment_cfg['foot_radius_m'],max_error=alignment_cfg['max_error_m'])
                extra['initial_stance_alignment']=check
                if not check['verified']:
                    self.phase='WAIT'
                    return self._output('WAIT',[0,0,0],'floor_foot_alignment_unverified',now,extra)
            room_required=envelope-float(np.min(foot_projection))+.03+.12
            if self.map['edges'][2]-self.map['edges'][1]<room_required:
                return self._output('WAIT',[0,0,0],'landing_too_short_for_all_feet_and_stop',now,extra)
            if first_clearance<.08 or wall_clearance<.25:
                return self._output('WAIT',[0,0,0],'initial_clearance_too_small',now,extra)
            if self.map['geometry']['corridor_y_m'][1]-self.map['geometry']['corridor_y_m'][0]<.38:
                return self._output('WAIT',[0,0,0],'stair_corridor_too_narrow',now,extra)
            if not active:
                self.phase='READY';return self._output('RUN',[0,0,0],'geometry_ready_waiting_for_G',now,extra)
            self.active=True
        if abs(heading)>.40 or abs(lateral)>.25 or abs(tilt)>.45:
            self.stopped=True;self.stop_reason='alignment_or_tilt_outside_corridor'
            return self._output('FAULT',[0,0,0],self.stop_reason,now,extra)
        if wall_clearance<.10:
            self.stopped=True;self.stop_reason='wall_emergency_clearance'
            return self._output('FAULT',[0,0,0],self.stop_reason,now,extra)
        clearance_cfg=self.cfg.get('leg_clearance')
        if clearance_cfg and clearance_cfg.get('enabled'):
            if leg_probes is None or leg_velocities is None or joint_speed is None:
                self.stopped=True;self.stop_reason='leg_clearance_state_missing'
                return self._output('FAULT',[0,0,0],self.stop_reason,now,extra)
            risk=assess(leg_probes,leg_velocities,normal=n,edges=distances[:2],heights=self.map['geometry']['step_heights_m'],
                        floor_z=self.map['floor_z'],body_z=pose['position'][2],speed=speed,config=clearance_cfg)
            extra['leg_clearance']=risk
            if risk['risk'] or self.phase=='OBSTACLE_STOPPING':
                self.phase='OBSTACLE_STOPPING'
                # This latch cannot be released by a later "clear" scan.
                if speed<.035 and joint_speed<.8 and abs(tilt)<.25 and np.max(abs(self.previous))<.025:
                    if self.settled_since is None:self.settled_since=now
                    if now-self.settled_since>=.6:
                        self.phase='OBSTACLE_HOLD';return self._output('HOLD',[0,0,0],'riser_risk_stop_requires_manual_reset',now,extra)
                else:self.settled_since=None
                return self._output('RUN',[0,0,0],'predicted_leg_riser_contact_decelerating',now,extra)
        all_up=bool(np.min(foot_projection)>distances[1]+.03 and
                    np.min(feet[:,2]+pose['position'][2])>=self.map['floor_z']+self.map['top_height']-.055)
        stopping_margin=.20+max(speed,0)*.60+max(speed,0)**2/(2*.25)
        must_stop=wall_clearance<stopping_margin
        if all_up or must_stop or self.phase=='PARKING':
            self.phase='PARKING'
            if speed<.055 and abs(tilt)<.25 and np.max(abs(self.previous))<.025:
                if self.settled_since is None:self.settled_since=now
                if now-self.settled_since>=.6:
                    if all_up:
                        self.phase='COMPLETE';return self._output('HOLD',[0,0,0],'parked_on_platform',now,extra)
                    self.phase='BLOCKED';return self._output('HOLD',[0,0,0],'wall_stop_before_platform_completion',now,extra)
            else:self.settled_since=None
            return self._output('RUN',[0,0,0],'decelerating',now,extra)
        self.settled_since=None
        if abs(heading)>.09 or abs(lateral)>.09:
            self.phase='ALIGN';command=[0,float(np.clip(.5*lateral,-.06,.06)),float(np.clip(1.2*heading,-.25,.25))]
        else:
            self.phase='APPROACH' if first_clearance>.18 else 'CLIMB'
            vx=min(self.cfg['approach_speed_m_s'],max(.20,.45*first_clearance)) if self.phase=='APPROACH' else self.cfg['climb_speed_m_s']
            command=[vx,float(np.clip(.3*lateral,-.04,.04)),float(np.clip(.8*heading,-.15,.15))]
        return self._output('RUN',command,'tracking_stair_corridor',now,extra)
