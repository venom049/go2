"""Matched official-model nominal/randomized ONNX evaluation; offline only."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,argparse,hashlib,multiprocessing as mp
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'source'),str(ROOT/'source/scripts')]
import numpy as np
from evaluate_moe_finetune import samples,run_episode,aggregate_guards,gait_summary
from go2_him.target_clip import aggregate_target_clip_summaries
SESSIONS={}
def episode(task):
    import torch,onnxruntime as ort
    from go2_him.moe_stair_env import MoEStairEnv
    torch.set_num_threads(1)
    model,mode,level,idx,sample,out,guard=task
    if model not in SESSIONS:
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
        SESSIONS[model]=ort.InferenceSession(model,sess_options=options,providers=['CPUExecutionProvider'])
    def actor(obs):return torch.from_numpy(SESSIONS[model].run(None,{'obs':obs.cpu().numpy()})[0])
    env=MoEStairEnv(seed=100000+idx,evaluation=True,guard_mode=guard,position_constraint='native',
        target_mode='legacy',domain_randomization=level is not None)
    env.dr.fixed_level=level;env.dr.rng=np.random.default_rng(200000+idx)
    try:
        row=run_episode(env,actor,sample,torch.device('cpu'))
        if 'domain_randomization' not in row:row['domain_randomization']=env.dr.summary()
        row.update(mode=mode,index=idx,robot_model='Unitree official Go2 (EDU hardware target)',dr_test_level=level)
        Path(out,f'episode_{idx:03d}.json').write_text(json.dumps(row,indent=2)+'\n')
        return row
    finally:env.exp.close()

def evaluate(model,out,level=None,count=64,workers=8,guard='audit'):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    mode='nominal' if level is None else f'dr{level}'
    tasks=[(str(Path(model).resolve()),mode,level,i,s,str(out),guard) for i,s in enumerate(samples(count,[442,443]))]
    rows=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        for row in pool.map(episode,tasks):
            rows.append(row)
            print(json.dumps({'mode':mode,'episode':row['index'],'status':row['status']}),flush=True)
    summary=dict(model=str(model),model_sha256=hashlib.sha256(Path(model).read_bytes()).hexdigest(),
        episodes=len(rows),successes=sum(r['success'] for r in rows),statuses=dict(Counter(r['status'] for r in rows)),
        mode=mode,dr_level=level,guard_mode=guard,
        target_commands=aggregate_target_clip_summaries([r['target_commands'] for r in rows]),
        guard=aggregate_guards([r['guard'] for r in rows]),
        gait_success_only=gait_summary([r['gait_integrals'] for r in rows if r['success']]),
        randomized_episodes=sum(r['domain_randomization']['active'] for r in rows),
        delayed_episodes=sum(r['domain_randomization']['delay_steps']>0 for r in rows),
        push_events=sum(r['domain_randomization']['realized']['push_events'] for r in rows),
        real_robot_driven=False,seeds=[442,443],sampling='first episodes; same parameters/seeds for both policies; no cherry-picking',
        model_source='Unitree official ae6a8403e272733e9996ef59990880330496177f; no extra payload',
        command_history='bounded known policy command; actuator lag/noise hidden from actor',
        limitations='Simulation ground truth triggers platform stop; hard limits only protect commanded target angles')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return {k:summary[k] for k in ('episodes','successes','statuses','mode','delayed_episodes','push_events')}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--out',required=True)
    p.add_argument('--level',type=float);p.add_argument('--count',type=int,default=64);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--guard',choices=['audit','strict'],default='audit');a=p.parse_args()
    print(json.dumps(evaluate(a.model,a.out,a.level,a.count,a.workers,a.guard)))
