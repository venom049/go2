"""Export the complete bounded command policy, with SDK-order target output."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import hashlib
import json
from pathlib import Path
import sys
import shutil
import numpy as np
import torch
import onnx
import onnxruntime as ort
from train_moe_stairs import load_policy, target_action_contract
from go2_him.target_clip import transform_target_action


class BoundedCommandPolicy(torch.nn.Module):
    def __init__(self, actor):
        super().__init__();self.actor=actor
        c=target_action_contract()
        for key, source in (('scale','scale'),('offset','offset'),('lower','lower_rad'),('upper','upper_rad')):
            self.register_buffer(key,torch.tensor(c[source],dtype=torch.float64))
        self.register_buffer('sdk_map',torch.tensor([3,4,5,0,1,2,9,10,11,6,7,8]))

    def forward(self, observation):
        raw=self.actor(observation)
        target=torch.clamp(raw.double()*self.scale+self.offset,min=self.lower,max=self.upper)
        executed=((target-self.offset)/self.scale).float()
        sdk=target[:,self.sdk_map].float()
        return target,executed,sdk


def export(checkpoint,out):
    torch.set_num_threads(1)
    data=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if data.get('target_mode')!='bounded':raise ValueError('Not a bounded training checkpoint')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    actor=load_policy(checkpoint).eval();model=BoundedCommandPolicy(actor).eval()
    sample=torch.zeros(1,225)
    path=out/'command_policy.onnx'
    torch.onnx.export(model,sample,str(path),opset_version=17,dynamo=False,
        input_names=['observations'],output_names=['joint_targets_policy_order','executed_actions','joint_targets_sdk_order'],
        dynamic_axes={'observations':{0:'batch'},'joint_targets_policy_order':{0:'batch'},
                      'executed_actions':{0:'batch'},'joint_targets_sdk_order':{0:'batch'}})
    onnx.checker.check_model(onnx.load(str(path)),full_check=True)
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    rng=np.random.default_rng(20261104)
    inputs=np.concatenate([np.zeros((1,225)),rng.normal(0,1,(128,225)),
        rng.normal(0,10,(128,225)),np.full((1,225),100),np.full((1,225),-100)]).astype(np.float32)
    contract=target_action_contract()
    with torch.no_grad():
        expected=[t.numpy() for t in model(torch.from_numpy(inputs))]
        raw=actor(torch.from_numpy(inputs)).numpy()
    numpy_expected=transform_target_action(raw,np.array(contract['scale']),np.array(contract['offset']),
        np.array(contract['lower_rad']),np.array(contract['upper_rad']))
    np.testing.assert_array_equal(expected[0],numpy_expected.sent_target)
    np.testing.assert_array_equal(expected[1],numpy_expected.sent_action)
    actual=session.run(None,{'observations':inputs})
    errors=[]
    # Large, clipped observation probes amplify float32 GEMM order differences.
    # 50 microradians for target outputs, with the matching /0.25 action budget.
    # SDK limit assertions below remain exact: they use no tolerance.
    tolerances=[5e-5,2e-4,5e-5]
    for lhs,rhs,tolerance in zip(actual,expected,tolerances):
        np.testing.assert_allclose(lhs,rhs,atol=tolerance,rtol=0)
        if not np.isfinite(lhs).all():raise ValueError('Nonfinite export output')
        errors.append(float(np.max(np.abs(lhs-rhs))))
    lower=np.array(contract['lower_rad']);upper=np.array(contract['upper_rad'])
    mapping=[3,4,5,0,1,2,9,10,11,6,7,8]
    assert np.all(actual[0]>=lower) and np.all(actual[0]<=upper)
    assert np.all(actual[2]>=lower[mapping]) and np.all(actual[2]<=upper[mapping])
    contract.update(reference_anchor='unchanged source raw mean; no projection',raw_target_penalty_added=False,
        interface='Use joint_targets_sdk_order directly as SDK q; do not apply action scale/offset again.',
        history='Append executed_actions to last_action history in policy FL/FR/RL/RR order.',
        outputs=['joint_targets_policy_order float64','executed_actions float32','joint_targets_sdk_order float32'],
        input='float32 [batch,225], same term-major five-frame history',
        finite_validation='Caller must reject nonfinite observations and outputs before publication.',
        kp=20.,kd=.5,control_dt=.02,real_robot_deployed=False)
    report=dict(passed=True,cases=len(inputs),max_absolute_errors=errors,absolute_error_limits=tolerances,
        numpy_decoder_exact=True,onnx_sdk_target_excess_count=0,
        checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        onnx_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (out/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    (out/'parity.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(out/'parity_inputs.npz',observations=inputs)
    shutil.copy2(Path(__file__).resolve().parents[1]/'go2_him/sdk_target_runtime.py',out/'sdk_target_runtime.py')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(export(a.checkpoint,a.out)))
