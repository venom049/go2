"""Evaluate final ONNX itself using original legacy action decoding/history."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
import onnxruntime as ort
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'source/scripts'),str(ROOT/'source')]
from evaluate_moe_finetune import samples,run_episode,aggregate_guards,gait_summary
from go2_him.moe_stair_env import MoEStairEnv
from go2_him.target_clip import aggregate_target_clip_summaries


def evaluate(model,out,episodes_per_seed=64):
    torch.set_num_threads(1)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
    def actor(obs):
        inputs=obs.detach().cpu().numpy()
        if not np.isfinite(inputs).all():raise ValueError('Nonfinite observations')
        actions=session.run(None,{'obs':inputs})[0]
        if not np.isfinite(actions).all():raise ValueError('Nonfinite actions')
        return torch.from_numpy(actions)
    env=MoEStairEnv(seed=0,evaluation=True,guard_mode='audit',stair_steps=2,
                    position_constraint='native',target_mode='legacy')
    env.ramp=1.
    reports=[]
    try:
        for i,sample in enumerate(samples(episodes_per_seed,[42,43])):
            report=run_episode(env,actor,sample,torch.device('cpu'))
            commands=report['target_commands']
            assert commands['raw']['excess_count']==0 and commands['sent']['excess_count']==0
            (out/f'episode_{i:03}.json').write_text(json.dumps(report,indent=2)+'\n')
            reports.append(report)
            print(json.dumps({'episode':i,'status':report['status'],'target_excess':0}),flush=True)
    finally:env.exp.close()
    guard=aggregate_guards([r['guard'] for r in reports])
    summary=dict(model=str(model),model_sha256=hashlib.sha256(Path(model).read_bytes()).hexdigest(),
        episodes=len(reports),successes=sum(r['success'] for r in reports),
        statuses=dict(Counter(r['status'] for r in reports)),
        runtime='ONNX Runtime CPU; original affine decoder and raw-action history; NO additional per-joint target clamp in environment',
        target_commands=aggregate_target_clip_summaries([r['target_commands'] for r in reports]),
        gait_success_only=gait_summary([r['gait_integrals'] for r in reports if r['success']]),
        guard=guard,real_robot_driven=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return {k:summary[k] for k in ('episodes','successes','statuses','model_sha256')}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--episodes-per-seed',type=int,default=64);a=p.parse_args()
    print(json.dumps(evaluate(a.model,a.out,a.episodes_per_seed)))
