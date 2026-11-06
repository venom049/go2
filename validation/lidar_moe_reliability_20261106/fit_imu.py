import json,numpy as np,sys
from pathlib import Path
p=Path(__file__).resolve().parent;d=np.load(p/'manual_motion.npz');b=d['lowstate'];l=d['lidar_imu'];bt=b[:,0];lt=l[:,1]
# Align gyroscope vectors over candidate transport/time offsets, then fit a
# proper rigid rotation. Differentiate constant biases from rotation.
def fit(offset,train=None):
 valid=(lt+offset>bt[0])&(lt+offset<bt[-1]);t=lt[valid]+offset;x=l[valid,2:5];y=np.stack([np.interp(t,bt,b[:,7+i]) for i in range(3)],1)
 moving=(np.linalg.norm(x-x.mean(0),axis=1)>.015)|(np.linalg.norm(y-y.mean(0),axis=1)>.015)
 if train is not None:moving &= train(np.arange(len(moving)))
 x=x[moving];y=y[moving];xm=x.mean(0);ym=y.mean(0);u,s,v=np.linalg.svd((x-xm).T@(y-ym));R=v.T@u.T
 if np.linalg.det(R)<0:v[-1]*=-1;R=v.T@u.T
 bias=ym-R@xm;err=np.linalg.norm(x@R.T+bias-y,axis=1);return np.mean(err**2),R,bias,s,err,len(x)
fits=[(fit(float(offset)),float(offset)) for offset in np.arange(-.15,.151,.001)];best,offset=min(fits,key=lambda x:x[0][0]);mse,R,bias,s,e,n=best
# Accelerometer fit permits a constant bias, but uses the gyro's rotation; test
# both conventions rather than silently guessing a sign.
t=lt+offset;valid=(t>bt[0])&(t<bt[-1]);t=t[valid];la=l[valid,5:8];ba=np.stack([np.interp(t,bt,b[:,10+i]) for i in range(3)],1);acc=[]
for sign in [1,-1]:
 diff=ba-sign*la@R.T;ab=diff.mean(0);res=np.linalg.norm(diff-ab,axis=1);acc.append({'sign':sign,'bias_body_m_s2':ab.tolist(),'residual_quantiles_m_s2':np.quantile(res,[.5,.95,.99]).tolist()})
out={'samples':{'lowstate':len(b),'lidar_imu':len(l),'moving':n},'gyro_axis_std_rad_s':{'body':b[:,7:10].std(0).tolist(),'lidar':l[:,2:5].std(0).tolist()},'gyro_fit':{'lidar_source_to_body_receipt_offset_s':offset,'R_body_imu':R.tolist(),'bias_body_rad_s':bias.tolist(),'cross_singular_values':s.tolist(),'residual_quantiles_rad_s':np.quantile(e,[.5,.95,.99]).tolist(),'rms_rad_s':float(np.sqrt(mse))},'accelerometer_sign_candidates':acc,'candidate_only_not_motion_validated':True}
(p/'imu_alignment_fit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
