#!/usr/bin/env python3
"""One bounded offline training run followed by its predeclared comparisons."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]

def write(path,data):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    actor=ROOT/'validation/moe_cts_actor_conversion_20261104/actor.pt'
    train=out/'training';evaluation=out/'evaluation'
    job={'status':'preparing','pid':os.getpid(),'started_unix':time.time(),
         'training_dir':str(train),'training_job':str(train/'job.json'),
         'eval_summary_path':str(evaluation/'comparison.json'),'source_actor':str(actor),
         'source_actor_sha256':hashlib.sha256(actor.read_bytes()).hexdigest(),
         'max_updates':1000,'num_envs':128,'workers':32,
         'final_checkpoint':str(train/'model_1000.pt'),'real_robot_driven':False,
         'reference_method':'https://github.com/saifahmadgit/go2-sim2real-locomotion-rl/tree/0830f3e3582acd5c9b3a5c09186d4f028f54d9dd'}
    source_files=['scripts/train_moe_stairs.py','scripts/evaluate_moe_finetune.py','scripts/run_moe_finetune.py',
                  'scripts/test_robogauge_stairs.py','scripts/test_robogauge_two_steps_wall.py',
                  'go2_him/moe_stair_env.py','go2_him/moe_stair_rewards.py','go2_him/official_guard.py',
                  'go2_him/moe_cts_actor.py','validation/official_joint_limits_20261104/limit_profile.json']
    for item in source_files:
        dest=out/'source'/item;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/item,dest)
    write(out/'job.json',job)
    try:
        job['status']='training';write(out/'job.json',job)
        with (out/'training.log').open('x') as stream:
            subprocess.run([sys.executable,str(ROOT/'scripts/train_moe_stairs.py'),
                            '--actor',str(actor),'--out',str(train),'--updates','1000',
                            '--num-envs','128','--workers','32'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        assert (train/'model_500.pt').is_file() and (train/'model_1000.pt').is_file()
        job.update(status='evaluating',evaluation_started_unix=time.time());write(out/'job.json',job)
        with (out/'evaluation.log').open('x') as stream:
            subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_moe_finetune.py'),
                            '--actor',f'reference_moe={actor}',f'moe500={train / "model_500.pt"}',f'moe1000={train / "model_1000.pt"}',
                            '--out',str(evaluation),'--episodes-per-seed','64','--device','cuda:0'],
                           cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        summary=json.loads((evaluation/'comparison.json').read_text())
        assert len(summary['rows'])==3 and all(r['episodes']==129 for r in summary['rows'])
        job.update(status='completed',completed_unix=time.time(),completion_verified=True,
                   final_checkpoint_exists=True,eval_summary_exists=True)
        write(out/'job.json',job)
    except BaseException as exc:
        job.update(status='failed',error=repr(exc),traceback=traceback.format_exc(),failed_unix=time.time())
        write(out/'job.json',job);raise

if __name__=='__main__':main()
