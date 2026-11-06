"""Export original 225/12 interface with certified inward action bounds.

Post-training packaging only: does not modify the frozen experiment source.
Both float32 and float64 deployment affine decoders are checked at the extrema.
Positive affine scales are monotone, so all finite in-interval actions remain
within the same SDK target limits. History uses the returned bounded actions.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
import onnx
import onnxruntime as ort
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'source/scripts'),str(ROOT/'source')]
from train_moe_stairs import load_policy,target_action_contract


def certify_bounds(contract):
    scale=np.array(contract['scale']);offset=np.array(contract['offset'])
    lower=np.array(contract['lower_rad']);upper=np.array(contract['upper_rad'])
    lo=np.array(contract['raw_action_lower'],np.float32)
    hi=np.array(contract['raw_action_upper'],np.float32)
    # Stay at least one representable action value inside the double-decoder
    # boundary. Check both deployed arithmetic variants instead of assuming
    # algebraically equivalent floating point expressions have identical output.
    lo=np.nextafter(lo,np.float32(np.inf));hi=np.nextafter(hi,np.float32(-np.inf))
    def decoded(a):
        return (a*scale+offset,a*scale.astype(np.float32)+offset.astype(np.float32),
                (a*scale+offset).astype(np.float32))
    for _ in range(16):
        badlo=np.logical_or.reduce([v<lower for v in decoded(lo)])
        badhi=np.logical_or.reduce([v>upper for v in decoded(hi)])
        if not badlo.any() and not badhi.any():break
        lo=np.where(badlo,np.nextafter(lo,np.float32(np.inf)),lo)
        hi=np.where(badhi,np.nextafter(hi,np.float32(-np.inf)),hi)
    else:raise ValueError('Could not certify affine bounds')
    if not np.all(lo<hi):raise ValueError('Empty action interval')
    for endpoint in (lo,hi):
        for target in decoded(endpoint):
            assert np.all(target>=lower) and np.all(target<=upper)
    return lo,hi


class SafeActionPolicy(torch.nn.Module):
    def __init__(self,actor,lower,upper):
        super().__init__();self.actor=actor
        self.register_buffer('lower',torch.tensor(lower))
        self.register_buffer('upper',torch.tensor(upper))
    def forward(self,observations):
        return torch.clamp(self.actor(observations),min=self.lower,max=self.upper)


def export(checkpoint,out):
    torch.set_num_threads(1)
    checkpoint=Path(checkpoint);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    data=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if data.get('target_mode')!='bounded':raise ValueError('Wrong checkpoint target mode')
    c=target_action_contract();lo,hi=certify_bounds(c)
    model=SafeActionPolicy(load_policy(checkpoint).eval(),lo,hi).eval()
    path=out/'policy.onnx'
    torch.onnx.export(model,torch.zeros(1,225),str(path),opset_version=17,dynamo=False,
        input_names=['obs'],output_names=['actions'])
    onnx.checker.check_model(onnx.load(str(path)),full_check=True)
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    rng=np.random.default_rng(42)
    probes=np.concatenate([np.zeros((1,225)),rng.normal(0,1,(128,225)),
        rng.normal(0,10,(128,225)),np.full((1,225),100),np.full((1,225),-100)]).astype(np.float32)
    max_error=0.
    for obs in probes:
        with torch.no_grad():expected=model(torch.from_numpy(obs[None])).numpy()
        action=session.run(None,{'obs':obs[None]})[0]
        np.testing.assert_allclose(action,expected,atol=2e-4,rtol=0)
        assert np.isfinite(action).all() and np.all(action>=lo) and np.all(action<=hi)
        max_error=max(max_error,float(np.max(np.abs(action-expected))))
    scale=np.array(c['scale']);offset=np.array(c['offset'])
    lower=np.array(c['lower_rad']);upper=np.array(c['upper_rad'])
    actions=np.vstack([lo,hi,rng.uniform(lo,hi,(100000,12)).astype(np.float32)])
    for q in (actions*scale+offset,actions*scale.astype(np.float32)+offset.astype(np.float32),
              (actions*scale+offset).astype(np.float32)):
        assert np.all(q>=lower) and np.all(q<=upper)
    (out/'contract.json').write_text(json.dumps(dict(input='obs float32 [1,225]',output='actions float32 [1,12]',
        history='Append these bounded actions in original policy order; five-frame term-major observations unchanged.',
        joint_order=c['joint_names'],sdk_map=[3,4,5,0,1,2,9,10,11,6,7,8],
        scale=c['scale'],offset=c['offset'],lower_rad=c['lower_rad'],upper_rad=c['upper_rad'],
        action_lower=lo.tolist(),action_upper=hi.tolist(),kp=20.,kd=.5,control_dt=.02,
        finite_check='Reject nonfinite observations/actions/targets before publication.',
        scope='Only target-position bounds. Keep existing state, velocity, tracking, torque and handoff guards.',
        rounding='Inward float32 action endpoints certified for float32, float64, and float64-to-float32 affine decoding.',
        real_robot_deployed=False),indent=2)+'\n')
    report=dict(passed=True,onnx_cases=len(probes),max_action_absolute_error=max_error,
        action_error_tolerance=2e-4,affine_decoder_cases=len(actions)*3,
        target_excess_count=0,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        onnx_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        formal_training_update=data['update'])
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(export(a.checkpoint,a.out)))
