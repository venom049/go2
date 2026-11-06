"""Offline exploratory surface fits, not a deployable stair detector."""
from pathlib import Path
import json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]/'deploy'))
from go2_stairs.lidar_watch import yaw_local_points

def fit_horizontal(points,rng):
    best=None
    for _ in range(600):
        v=points[rng.choice(len(points),3,replace=False)]
        a=np.c_[v[:,:2],np.ones(3)]
        if abs(np.linalg.det(a))<1e-5:continue
        c=np.linalg.solve(a,v[:,2])
        if np.linalg.norm(c[:2])>.15:continue
        mask=abs(np.c_[points[:,:2],np.ones(len(points))]@c-points[:,2])<.012
        if best is None or mask.sum()>best[0]:best=(int(mask.sum()),mask,c)
    if best is None or best[0]<20:return None
    inliers=points[best[1]]
    c=np.linalg.lstsq(np.c_[inliers[:,:2],np.ones(len(inliers))],inliers[:,2],rcond=None)[0]
    return best[1],{'points':len(inliers),'plane_z_ax_by_c':c.tolist(),'xyz_min_m':inliers.min(0).tolist(),
                    'xyz_max_m':inliers.max(0).tolist(),'median_z_m':float(np.median(inliers[:,2]))}

def main():
    archive=np.load(ROOT/'onboard/go2-lidar-audit-20261106/detail_frames.npz')
    rng=np.random.default_rng(20261106);frames=[];rows=[]
    for key in sorted(k for k in archive.files if k.startswith('points_')):
        meta=json.loads(str(archive[key.replace('points_','meta_')]))
        pose=meta['nearest_pose'];q=np.array(pose[2]);x,y,z,w=q
        yaw=float(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
        a=yaw_local_points(archive[key],np.array(pose[1]),yaw)
        frames.append(a)
        roi=a[(a[:,0]>.25)&(a[:,0]<1.3)&(abs(a[:,1])<.48)]
        remaining=roi.copy();planes=[]
        for _ in range(3):
            if len(remaining)<20:break
            result=fit_horizontal(remaining,rng)
            if result is None:break
            mask,report=result;planes.append(report);remaining=remaining[~mask]
        rows.append({'frame':key,'source_stamp_s':meta['stamp_s'],
                     'odom_match_difference_s':abs(pose[0]-meta['stamp_s']),
                     'roi_points':len(roi),'exploratory_horizontal_planes':planes})
    cloud=np.concatenate(frames)
    roi=cloud[(cloud[:,0]>.15)&(cloud[:,0]<2.5)&(abs(cloud[:,1])<.55)]
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    front=roi[abs(roi[:,1])<.45]
    axes[0].scatter(front[:,0],front[:,2],s=2,c=front[:,2],cmap='viridis',alpha=.3,rasterized=True)
    axes[0].set(xlabel='Forward x from body origin (m)',ylabel='Gravity-up z relative to body (m)',
                title='Side view: floor, near surface, and far returns',xlim=(.15,2.1),ylim=(-.42,.5))
    axes[1].scatter(roi[:,0],roi[:,1],s=2,c=roi[:,2],cmap='viridis',alpha=.3,rasterized=True)
    axes[1].set(xlabel='Forward x from body origin (m)',ylabel='Left y from body origin (m)',
                title='Top view: visible footprint is incomplete',xlim=(.15,2.1),ylim=(-.55,.55))
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Native Go2 lidar: 12 pose-matched frames; exploratory inspection only',fontsize=12)
    fig.savefig(ROOT/'pointcloud_geometry.png',dpi=170)
    report={'scope':'Exploratory plane support only; not stair recognition or calibrated step dimensions',
            'frame_count':len(frames),'merged_points':len(cloud),'roi_points':len(roi),
            'gravity_alignment_calibrated':False,'distance_reference':'body origin, not front feet or lidar origin',
            'rows':rows,'plot':'pointcloud_geometry.png'}
    (ROOT/'offline_geometry.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'frames':len(frames),'roi_points':len(roi),
                      'horizontal_plane_counts':[len(r['exploratory_horizontal_planes']) for r in rows]}))

if __name__=='__main__':main()
