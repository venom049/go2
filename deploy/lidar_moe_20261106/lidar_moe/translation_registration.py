"""Attitude-constrained point-to-plane registration, with explicit degeneracy.

Attitude is measured by the body IMU. Translation is estimated only from
observed surfaces; a missing direction is rejected, never filled by a prior.
"""
import numpy as np
from scipy.spatial import cKDTree


def voxel(points,size=.05):
    points=np.asarray(points,float);_,ids=np.unique(np.floor(points/size).astype(np.int64),axis=0,return_index=True)
    return points[np.sort(ids)]


class SurfaceMap:
    def __init__(self,points):
        self.points=voxel(points);self.tree=cKDTree(self.points)
        if len(self.points)<150:raise ValueError('registration_map_sparse')
        distances,indices=self.tree.query(self.points,k=12)
        neighbors=self.points[indices];center=neighbors.mean(1);x=neighbors-center[:,None,:]
        cov=np.einsum('nki,nkj->nij',x,x)/12
        eigen,vectors=np.linalg.eigh(cov);self.normals=vectors[:,:,0]
        self.valid=(distances[:,-1]<.3)&(eigen[:,1]>.0002)&(eigen[:,0]<.15*eigen[:,1])&(eigen[:,0]<.025**2)
        if self.valid.sum()<100:raise ValueError('registration_map_no_planar_support')

    def register(self,predicted):
        p=voxel(predicted);correction=np.zeros(3)
        for _ in range(6):
            distances,indices=self.tree.query(p+correction)
            mask=(distances<.16)&self.valid[indices]
            if mask.sum()<100 or mask.mean()<.3:raise ValueError('registration_correspondence_support')
            n=self.normals[indices[mask]];matched=self.points[indices[mask]]
            residual=np.sum((p[mask]+correction-matched)*n,axis=1)
            weight=np.minimum(1.,.025/np.maximum(abs(residual),1e-9))
            h=(n.T*weight)@n;information=h/max(1,len(n));eigen=np.linalg.eigvalsh(information)
            ratio=float(eigen[0]/max(eigen[-1],1e-12))
            if ratio<.01:raise ValueError('registration_translation_degenerate')
            delta=-np.linalg.solve(h,n.T@(weight*residual))
            if not np.isfinite(delta).all() or np.linalg.norm(delta)>.12:raise ValueError('registration_correction_excess')
            correction+=delta
            if np.linalg.norm(delta)<.0005:break
        distances,indices=self.tree.query(p+correction);mask=(distances<.10)&self.valid[indices]
        if mask.sum()<100 or mask.mean()<.3:raise ValueError('registration_final_support')
        n=self.normals[indices[mask]];residual=np.sum((p[mask]+correction-self.points[indices[mask]])*n,axis=1)
        mean=float(np.mean(abs(residual)))
        if mean>.025:raise ValueError('registration_final_residual')
        covariance=np.linalg.inv(n.T@n)*max(.002**2,float(np.mean(residual**2)))+np.eye(3)*.005**2
        return correction,{'effective_points':int(mask.sum()),'mean_residual_m':mean,'pose_information_ratio':ratio,'covariance':covariance}


def interpolate_quaternions(times,quaternions,query_times):
    times=np.asarray(times);q=np.asarray(quaternions,float);query=np.asarray(query_times)
    right=np.searchsorted(times,query,side='right');right=np.clip(right,1,len(times)-1);left=right-1
    if np.min(query)<times[0] or np.max(query)>times[-1] or np.any(times[right]-times[left]>.05):raise ValueError('deskew_imu_coverage')
    a=q[left].copy();b=q[right].copy();b[np.sum(a*b,axis=1)<0]*=-1
    fraction=((query-times[left])/(times[right]-times[left]))[:,None]
    result=a*(1-fraction)+b*fraction
    return result/np.linalg.norm(result,axis=1)[:,None]


def rotate_points(q,points):
    xyz=q[:,1:];cross=np.cross(xyz,points)
    return points+2*(q[:,0,None]*cross+np.cross(xyz,cross))
