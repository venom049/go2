"""Reproduce MoE1000 with SDK target projection as the sole training change."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import torch
from train_moe_stairs import atomic_json

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'validation/moe_cts_actor_conversion_20261104/actor.pt'
REFERENCE=ROOT/'validation/moe_finetune_20261104/formal_run/training/model_1000.pt'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(out,updates=1000,num_envs=128,workers=32):
    torch.set_num_threads(1)
    if sha(SOURCE)!='4b7c51ee0771a375001b1176f9e7e78cc3ad4c1603253ce1e65aeedd2af103a3':
        raise ValueError('Changed original actor')
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,str(ROOT/'scripts/train_moe_stairs.py'),
        '--actor',str(SOURCE),'--out',str(out/'training'),'--updates',str(updates),
        '--num-envs',str(num_envs),'--workers',str(workers),'--steps','24',
        '--epochs','3','--minibatches','4','--critic-warmup','30','--actor-lr','3e-6',
        '--std','.03','--target-kl','.005','--anchor-coef','.2','--seed','20261104',
        '--device','cuda:0','--stair-steps','2','--position-constraint','native',
        '--guard-objective','balanced','--guard-ramp-start','.2','--guard-ramp-updates','300',
        '--target-mode','bounded']
    files={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*')
        if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.yaml','.xml','.onnx','.pt','.urdf')}
    job=dict(status='preparing',pid=os.getpid(),started_unix=time.time(),max_updates=updates,
        training_command=command,source_sha256=sha(SOURCE),source_files=files,
        changes=['SDK per-joint submitted-target projection','executed-action observation history'],
        unchanged=['original actor initialization','balanced rewards and weights','native MuJoCo physics',
                   'two stair risers','PPO hyperparameters','raw Gaussian log probabilities','raw reference anchor'],
        new_target_penalty=False,real_robot_driven=False,completion_verified=False)
    def save():atomic_json(out/'job.json',job)
    def execute(cmd,name):
        with (out/name).open('x') as f:
            subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    save()
    try:
        job['status']='training';save();execute(command,'training.log')
        records=[json.loads(l) for l in (out/'training/progress.jsonl').read_text().splitlines()]
        assert [r['update'] for r in records]==list(range(1,updates+1))
        assert all(r['all_losses_finite'] and r['target_commands_update']['sent']['excess_count']==0
                   and r['mean_raw_target_cost_per_second']==0 for r in records)
        c=torch.load(out/f'training/model_{updates}.pt',map_location='cpu',weights_only=True)
        original=torch.load(SOURCE,map_location='cpu',weights_only=True)
        assert c['target_mode']=='bounded' and c['guard_objective']=='balanced'
        assert c['initialization']['source_cumulative_update']==0
        assert all(torch.equal(v,c['actor_state_dict'][k]) for k,v in original['actor_state_dict'].items()
                   if k.startswith('student_moe_encoder.'))
        assert all(torch.isfinite(v).all() for v in c['actor_state_dict'].values())
        job.update(status='exporting',training_completed=True,training_target_excess_count=0,
                   final_checkpoint=str(out/f'training/model_{updates}.pt'));save()
        execute([sys.executable,str(ROOT/'scripts/export_sdk_target_policy.py'),
            '--checkpoint',job['final_checkpoint'],'--out',str(out/'exported')],'export.log')
        job['status']='evaluating';save()
        actors=[f'original_moe1000={REFERENCE}']
        if updates>=500 and updates!=500:actors.append(f'sdk_target500={out / "training/model_500.pt"}')
        actors.append(f'sdk_target{updates}={out / f"training/model_{updates}.pt"}')
        for mode in ('audit','strict'):
            job['evaluation_phase']=mode;save()
            execute([sys.executable,str(ROOT/'scripts/evaluate_moe_finetune.py'),'--actor',*actors,
                '--out',str(out/f'evaluation_{mode}'),'--episodes-per-seed','64' if updates==1000 else '0',
                '--seeds','42','43','--device','cpu','--stair-steps','2','--position-constraint','native',
                '--guard-mode',mode,'--target-mode','auto'],f'evaluation_{mode}.log')
            rows=json.loads((out/f'evaluation_{mode}/comparison.json').read_text())['rows']
            assert all(r['episodes']==(129 if updates==1000 else 1) for r in rows)
            assert all(r['target_command_acceptance_passed'] for r in rows if r['label'].startswith('sdk_target'))
        assert all(sha(ROOT/p)==h for p,h in files.items())
        job.update(status='completed',completion_verified=True,completed_unix=time.time(),
                   model_sha256=sha(job['final_checkpoint']),export=str(out/'exported/command_policy.onnx'))
        save()
    except BaseException as e:
        job.update(status='failed',error=repr(e),traceback=traceback.format_exc());save();raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--updates',type=int,default=1000);p.add_argument('--num-envs',type=int,default=128)
    p.add_argument('--workers',type=int,default=32);a=p.parse_args()
    run(a.out,a.updates,a.num_envs,a.workers)
