"""Named URDF calibration, foot kinematics and explicit coordinate conversions."""
import math
import xml.etree.ElementTree as ET
import numpy as np

SDK_NAMES=tuple(f'{leg}_{joint}_joint' for leg in ('FR','FL','RR','RL') for joint in ('hip','thigh','calf'))

def rpy_matrix(rpy):
    r,p,y=map(float,rpy);cr,sr,cp,sp,cy,sy=math.cos(r),math.sin(r),math.cos(p),math.sin(p),math.cos(y),math.sin(y)
    return np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
                     [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])

def quaternion_matrix(wxyz):
    q=np.asarray(wxyz,dtype=float)
    if q.shape!=(4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>.01:
        raise ValueError('invalid IMU quaternion')
    w,x,y,z=q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def yaw_of(matrix):return math.atan2(matrix[1,0],matrix[0,0])

def heading_rotation(wxyz):
    world_body=quaternion_matrix(wxyz)
    return rpy_matrix([0,0,-yaw_of(world_body)])@world_body

def axis_rotation(axis,angle):
    a=np.asarray(axis,float);a=a/np.linalg.norm(a)
    x,y,z=a;k=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+math.sin(angle)*k+(1-math.cos(angle))*(k@k)

class Calibration:
    def __init__(self,urdf):
        root=ET.parse(urdf).getroot();self.joints={}
        for j in root.findall('joint'):
            origin=j.find('origin');axis=j.find('axis')
            self.joints[j.find('child').get('link')]={
                'name':j.get('name'),'type':j.get('type'),'parent':j.find('parent').get('link'),
                'xyz':np.fromstring(origin.get('xyz','0 0 0') if origin is not None else '0 0 0',sep=' '),
                'rotation':rpy_matrix(np.fromstring(origin.get('rpy','0 0 0') if origin is not None else '0 0 0',sep=' ')),
                'axis':np.fromstring(axis.get('xyz','1 0 0') if axis is not None else '1 0 0',sep=' ')}
        self.collisions=[]
        for link in root.findall('link'):
            for collision in link.findall('collision'):
                origin=collision.find('origin');T=np.eye(4)
                if origin is not None:
                    T[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
                    T[:3,:3]=rpy_matrix(np.fromstring(origin.get('rpy','0 0 0'),sep=' '))
                geometry=collision.find('geometry');shape=list(geometry)[0]
                if shape.tag=='box':size=np.fromstring(shape.get('size'),sep=' ')/2
                elif shape.tag=='sphere':size=[float(shape.get('radius'))]
                elif shape.tag=='cylinder':size=[float(shape.get('radius')),float(shape.get('length'))/2]
                else:raise ValueError('unsupported collision primitive '+shape.tag)
                self.collisions.append((link.get('name'),T,shape.tag,np.asarray(size)))
        self.lidar=self.transform('utlidar_lidar',{})
        self.foot_links=[]
        for leg in ('FR','FL','RR','RL'):
            matches=[n for n in self.joints if n.lower() in (leg.lower()+'_foot',leg.lower()+'_foot_link')]
            if len(matches)!=1:raise ValueError('missing named foot '+leg)
            self.foot_links.append(matches[0])

    def transform(self,link,positions):
        if link=='base_link':return np.eye(4)
        j=self.joints[link];t=np.eye(4);t[:3,:3]=j['rotation'];t[:3,3]=j['xyz']
        if j['type'] in ('revolute','continuous'):
            a=np.eye(4);a[:3,:3]=axis_rotation(j['axis'],positions[j['name']]);t=t@a
        return self.transform(j['parent'],positions)@t

    def feet_body(self,sdk_q):
        q=np.asarray(sdk_q,float)
        if q.shape!=(12,) or not np.isfinite(q).all():raise ValueError('invalid joint snapshot')
        positions=dict(zip(SDK_NAMES,q))
        return np.array([self.transform(link,positions)[:3,3] for link in self.foot_links])

    def leg_probes_body(self,sdk_q):
        """Thigh and calf capsules; 17 centers cover hip-to-knee-to-toe."""
        q=np.asarray(sdk_q,float)
        if q.shape!=(12,) or not np.isfinite(q).all():raise ValueError('invalid joint snapshot')
        positions=dict(zip(SDK_NAMES,q));feet=self.feet_body(q)
        knees=np.array([self.transform(leg+'_calf',positions)[:3,3] for leg in ('FR','FL','RR','RL')])
        hips=np.array([self.transform(leg+'_thigh',positions)[:3,3] for leg in ('FR','FL','RR','RL')])
        weights=np.linspace(0.,1.,9)[None,:,None]
        return np.concatenate((hips[:,None,:]*(1-weights)+knees[:,None,:]*weights,
                               knees[:,None,:]*(1-weights[:,1:])+feet[:,None,:]*weights[:,1:]),axis=1)

    def leg_motion_body(self,sdk_q,sdk_dq):
        q=np.asarray(sdk_q,float);dq=np.asarray(sdk_dq,float)
        if dq.shape!=(12,) or not np.isfinite(dq).all():raise ValueError('invalid joint velocity')
        epsilon=.0001
        return (self.leg_probes_body(q+epsilon*dq)-self.leg_probes_body(q-epsilon*dq))/(2*epsilon)

    def support_extent(self,sdk_q,direction_body):
        q=np.asarray(sdk_q,float);direction=np.asarray(direction_body,float)
        if q.shape!=(12,) or direction.shape!=(3,) or not np.isfinite(q).all() or not np.isfinite(direction).all():
            raise ValueError('invalid collision support query')
        positions=dict(zip(SDK_NAMES,q));maximum=-float('inf')
        for link,origin,kind,size in self.collisions:
            T=self.transform(link,positions)@origin;local=T[:3,:3].T@direction
            if kind=='box':extent=abs(local)@size
            elif kind=='sphere':extent=size[0]*np.linalg.norm(local)
            else:extent=size[0]*np.linalg.norm(local[:2])+size[1]*abs(local[2])
            maximum=max(maximum,float(T[:3,3]@direction+extent))
        return maximum

    def cloud_heading(self,points,wxyz,frame='utlidar_lidar'):
        a=np.asarray(points,float)
        if a.ndim!=2 or a.shape[1]!=3:raise ValueError('invalid XYZ cloud')
        if frame=='utlidar_lidar':a=a@self.lidar[:3,:3].T+self.lidar[:3,3]
        elif frame!='base_link':raise ValueError('unexpected cloud frame')
        return a@heading_rotation(wxyz).T

    def feet_heading(self,sdk_q,wxyz):return self.feet_body(sdk_q)@heading_rotation(wxyz).T

def parse_cloud(message):
    if message.header.frame_id not in ('utlidar_lidar','base_link'):raise ValueError('unsupported point frame')
    if message.width*message.height>250000 or message.point_step<12:raise ValueError('cloud size limit')
    fields={f.name:f for f in message.fields}
    if any(n not in fields or fields[n].datatype!=7 or fields[n].count!=1 for n in ('x','y','z')):
        raise ValueError('XYZ float32 required')
    if message.row_step<message.width*message.point_step or len(message.data)!=message.height*message.row_step:
        raise ValueError('malformed cloud layout')
    dtype=np.dtype({'names':['x','y','z'],'formats':[('>' if message.is_bigendian else '<')+'f4']*3,
                    'offsets':[fields[n].offset for n in ('x','y','z')],'itemsize':message.point_step})
    array=np.ndarray((message.height,message.width),dtype=dtype,buffer=memoryview(message.data),
                     strides=(message.row_step,message.point_step))
    return np.stack([array[n].reshape(-1) for n in ('x','y','z')],axis=1).astype(float)
