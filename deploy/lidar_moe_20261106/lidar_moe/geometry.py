"""Two-riser detector. Separates vertical faces, horizontal tread support and wall."""
import math
import numpy as np

def dot3(a,n):return a[:,0]*n[0]+a[:,1]*n[1]+a[:,2]*n[2]

def _plane(points,kind,rng,iterations=100,tolerance=.018):
    if len(points)<12:return None
    best=None
    if kind=='vertical':
        # Search gravity-constrained planes directly. Three-point RANSAC is
        # unreliable when a sparse wall is outnumbered by horizontal tread points.
        for theta in np.linspace(-.50,.50,25):
            n=np.array([math.cos(theta),math.sin(theta),0.]);projection=dot3(points,n)
            bins=np.floor(projection/.025).astype(int);ids,counts=np.unique(bins,return_counts=True)
            for index in ids[np.argsort(counts)[-12:]]:
                d=(index+.5)*.025;mask=abs(projection-d)<.021;sample=points[mask]
                if len(sample)<12 or np.ptp(sample[:,1])<.25:continue
                if best is not None and len(sample)<=best[0]:continue
                low,high=np.quantile(sample[:,2],[.05,.95]);span=high-low
                if span<.07:continue
                levels=np.unique(np.floor((sample[:,2]-low)/.025).astype(int))
                levels=levels[(levels>=0)&(levels<=int(span/.025))]
                if len(levels)<max(3,int(span/.025)*.65) or (len(levels)>1 and np.diff(levels).max()>2):continue
                if best is None or len(sample)>best[0]:best=(len(sample),mask,n,float(np.median(projection[mask])))
    for _ in range(iterations):
        if kind=='vertical':break
        a=points[rng.choice(len(points),3,replace=False)]
        n=np.cross(a[1]-a[0],a[2]-a[0]);size=np.linalg.norm(n)
        if size<1e-5:continue
        n/=size
        if kind=='horizontal':
            if abs(n[2])<.985:continue
            if n[2]<0:n=-n
        else:
            if abs(n[2])>.20 or abs(n[0])<.80:continue
            n[2]=0;n/=np.linalg.norm(n)
            if n[0]<0:n=-n
        d=float(np.median(dot3(a,n)));mask=abs(dot3(points,n)-d)<tolerance
        sample=points[mask]
        if len(sample)<12:continue
        if kind=='vertical' and (np.ptp(sample[:,2])<.07 or np.ptp(sample[:,1])<.20):continue
        score=len(sample)
        if best is None or score>best[0]:best=(score,mask,n,d)
    if best is None:return None
    _,mask,n,d=best;sample=points[mask]
    if kind=='horizontal':
        c=np.linalg.lstsq(np.c_[sample[:,:2],np.ones(len(sample))],sample[:,2],rcond=None)[0]
        n=np.array([-c[0],-c[1],1.]);n/=np.linalg.norm(n);d=float(np.median(dot3(sample,n)))
    else:
        # Least-squares vertical plane, with the gravity axis fixed.
        xy=sample[:,:2];_,_,v=np.linalg.svd(xy-xy.mean(0),full_matrices=False)
        n=np.r_[v[-1],0.]
        if n[0]<0:n=-n
        d=float(np.median(dot3(sample,n)))
    mask=abs(dot3(points,n)-d)<tolerance;sample=points[mask]
    return n,d,mask,sample

def detect(points,seed=1):
    """Return body-heading distances; unknown space is never filled as floor."""
    a=np.asarray(points,float)
    result={'valid':False,'reason':'insufficient_points','control_authorized':False}
    if a.ndim!=2 or a.shape[1]!=3 or len(a)>250000:raise ValueError('invalid cloud')
    a=a[np.isfinite(a).all(1)]
    a=a[(a[:,0]>.12)&(a[:,0]<3.)&(abs(a[:,1])<.65)&(a[:,2]>-.85)&(a[:,2]<1.2)]
    a=np.unique(np.round(a/.008).astype(np.int64),axis=0)*.008
    if len(a)<100:return result
    # A bounded deterministic subsample preserves reproducible replay.
    if len(a)>6000:a=a[::int(math.ceil(len(a)/6000))]
    rng=np.random.default_rng(seed)
    low=a[a[:,2]<np.quantile(a[:,2],.35)+.025]
    floor=_plane(low,'horizontal',rng,80,.014)
    if floor is None:return dict(result,reason='floor_not_observed')
    fn,fd,_,support=floor
    if len(support)<25 or np.ptp(support[:,0])<.15 or np.ptp(support[:,1])<.25:
        return dict(result,reason='floor_support_too_small')
    h=(dot3(a,fn)-fd)/fn[2]
    vertical=a[(h>.025)&(h<1.2)]
    faces=[];remaining=vertical.copy()
    for _ in range(5):
        fit=_plane(remaining,'vertical',rng,110)
        if fit is None:break
        n,d,mask,pts=fit;hh=(dot3(pts,fn)-fd)/fn[2]
        faces.append({'normal':n,'distance':d,'points':pts,'low':float(np.quantile(hh,.05)),
                      'high':float(np.quantile(hh,.95)),'span':float(np.ptp(hh)),
                      'width':float(np.quantile(pts[:,1],.95)-np.quantile(pts[:,1],.05))})
        remaining=remaining[~mask]
    faces.sort(key=lambda p:p['distance']/p['normal'][0])
    diagnostic=[{'distance_body_m':float(f['distance']/f['normal'][0]),'height_interval_m':[f['low'],f['high']],
                 'support_points':len(f['points']),'lateral_span_m':f['width']} for f in faces]
    result.update(reason='two_consistent_risers_not_observed',faces=diagnostic,
                  floor={'normal':fn.tolist(),'distance':fd,'points':len(support)})
    for first in faces:
        if not -.02<first['low']<.075 or not .07<first['high']<.26 or first['width']<.25:continue
        n=first['normal'];s1=first['distance'];along=dot3(a,n)
        for second in faces:
            s2=second['distance']
            if not .18<s2-s1<.60 or np.dot(n,second['normal'])<math.cos(.12) or second['width']<.25:continue
            result['reason']='middle_tread_not_observed'
            middle=(along>s1+.035)&(along<s2-.035)&(h>.06)&(h<.28)
            tread=a[middle];th=h[middle]
            if len(tread)<16 or np.ptp(tread[:,1])<.25 or np.ptp(along[middle])<.10:continue
            result['reason']='middle_tread_not_planar'
            height1=float(np.median(th))
            tread_mask=abs(th-height1)<.025
            if tread_mask.sum()<12 or np.mean(tread_mask)<.65:continue
            if np.ptp(tread[tread_mask,1])<.25 or np.ptp(along[middle][tread_mask])<.10:continue
            if abs(second['low']-height1)>.07:continue
            result['reason']='upper_tread_not_observed'
            landing=(along>s2+.045)&(along<s2+.75)&(h>height1+.06)&(h<height1+.27)
            upper=a[landing];uh=h[landing]
            if len(upper)<8 or np.ptp(upper[:,1])<.20 or np.ptp(along[landing])<.10:continue
            height2=float(np.quantile(uh,.30));supported=abs(uh-height2)<.025
            if supported.sum()<6:continue
            patch=upper[supported]
            if np.ptp(patch[:,1])<.20 or np.ptp(along[landing][supported])<.10:continue
            plane=np.linalg.lstsq(np.c_[patch[:,:2],np.ones(len(patch))],patch[:,2],rcond=None)[0]
            if np.linalg.norm(plane[:2])>.15 or np.median(abs(patch[:,2]-np.c_[patch[:,:2],np.ones(len(patch))]@plane))>.012:continue
            result['reason']='inconsistent_step_heights'
            rise2=height2-height1
            if not .07<rise2<.24 or abs(rise2-height1)>.05:continue
            walls=[f for f in faces if f['distance']>s2+.60 and f['span']>.20 and
                   f['high']>height2+.15 and np.dot(n,f['normal'])>math.cos(.2)]
            if not walls:return dict(result,reason='rear_wall_not_observed')
            wall=min(walls,key=lambda f:f['distance'])
            if any(s2+.15<f['distance']<wall['distance']-.15 and f['high']>height2+.07 and
                   abs(f['low']-height2)<.08 for f in faces):
                return dict(result,reason='additional_riser_before_wall')
            # Infer the common lateral corridor from both measured risers.
            yl=max(float(np.quantile(f['points'][:,1],.05)) for f in (first,second))
            yr=min(float(np.quantile(f['points'][:,1],.95)) for f in (first,second))
            if yr-yl<.28:continue
            return {'valid':True,'reason':'two_risers_tread_landing_and_wall_supported','control_authorized':False,
                'normal_xy':n[:2].tolist(),'heading_error_rad':math.atan2(n[1],n[0]),
                'first_edge_m':float(s1),'second_edge_m':float(s2),'wall_distance_m':float(wall['distance']),
                'step_heights_m':[height1,rise2],'tread_depth_m':float(s2-s1),
                'corridor_y_m':[yl,yr],'lateral_center_m':(yl+yr)/2,
                'floor':result['floor'],'support':{'first_riser':len(first['points']),'second_riser':len(second['points']),
                    'middle_tread':int(tread_mask.sum()),'upper_tread':int(supported.sum()),'wall':len(wall['points'])},
                'faces':diagnostic}
    return result
