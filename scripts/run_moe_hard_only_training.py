#!/usr/bin/env python3
"""Resume the original MoE1000 with its balanced PPO and hard actual joint stops.

No new target clamp, target penalty, history remapping, or anchor projection.
Six stairs were previously requested. Worker count/device are execution choices.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from run_moe_target_clip_training import (
    ROOT, SOURCE_CHECKPOINT, SOURCE_SHA256, PROFILE_PATH, PROFILE_SHA256,
    SOURCE_FILES, digest, write, environment_contract, validate_resources,
)

LABELS = ('reference_moe1000_hard_only', 'hard_only1500', 'hard_only2000')


def training_command(source, out, num_envs=128, workers=8, updates=1000):
    validate_resources(num_envs, workers)
    return [sys.executable, str(ROOT/'scripts/train_moe_stairs.py'),
            '--actor', str(source), '--out', str(out), '--resume-training',
            '--updates', str(updates), '--num-envs', str(num_envs), '--workers', str(workers),
            '--steps', '24', '--epochs', '3', '--minibatches', '4',
            '--stair-steps', '6', '--position-constraint', 'hard',
            '--guard-objective', 'balanced', '--critic-warmup', '30',
            '--guard-ramp-start', '.2', '--guard-ramp-updates', '300',
            '--anchor-coef', '.2', '--actor-lr', '3e-6', '--std', '.03',
            '--target-kl', '.005', '--seed', '20261104', '--device', 'cpu']


def validate_checkpoint(path, source, update):
    import torch
    c = torch.load(path, map_location='cpu', weights_only=True)
    original = torch.load(source, map_location='cpu', weights_only=True)
    if not (c['update'] == update and c['cumulative_update'] == 1000+update
            and c['guard_objective'] == 'balanced'
            and c['action_contract'] == {'kind':'legacy_raw_action_history'}
            and c['environment_contract'] == environment_contract(6)
            and c['std'] == original['std'] and c['frozen_moe_encoder'] is True):
        raise ValueError('Hard-only checkpoint contract mismatch')
    init = c['initialization']
    if init['source_cumulative_update'] != 1000 or not all(init[k] is True for k in (
            'full_training_state_resumed','actor_restored','critic_restored','optimizer_restored')):
        raise ValueError('Source training state was not fully restored')
    def finite(x):
        if torch.is_tensor(x) and not torch.isfinite(x).all():
            raise ValueError('Nonfinite checkpoint tensor')
        if isinstance(x,dict):
            for v in x.values(): finite(v)
        elif isinstance(x,(tuple,list)):
            for v in x: finite(v)
    finite(c)
    for key, value in original['actor_state_dict'].items():
        if key.startswith('student_moe_encoder.') and not torch.equal(value,c['actor_state_dict'][key]):
            raise ValueError('Frozen encoder changed')
    return dict(update=update,cumulative_update=1000+update,sha256=digest(path),
                all_tensors_finite=True,encoder_unchanged=True,full_training_state_resumed=True)


def run(out):
    out = Path(out).resolve()
    if digest(SOURCE_CHECKPOINT) != SOURCE_SHA256 or digest(PROFILE_PATH) != PROFILE_SHA256:
        raise ValueError('Source checkpoint or joint profile changed')
    out.mkdir(parents=True,exist_ok=False)
    train = out/'training'
    source = out/'source_checkpoint/model_1000.pt'
    source.parent.mkdir(); shutil.copy2(SOURCE_CHECKPOINT, source)
    files = (*SOURCE_FILES, 'scripts/run_moe_finetune.py','scripts/run_moe_hard_only_training.py')
    hashes = {}
    for name in files:
        dest = out/'source'/name; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name, dest); hashes[name] = digest(dest)
    write(out/'source_manifest.json',dict(files=hashes,source_checkpoint_sha256=SOURCE_SHA256))
    command = training_command(source,train)
    job = dict(status='preparing',pid=os.getpid(),started_unix=time.time(),
               training_dir=str(train),training_job=str(train/'job.json'),
               source_actor=str(source),source_actor_sha256=SOURCE_SHA256,
               source_update=1000,max_updates=1000,num_envs=128,workers=8,device='cpu',
               stair_steps=6,environment_contract=environment_contract(6),
               training_command=command,source_hashes=hashes,
               guard_objective='balanced',action_contract={'kind':'legacy_raw_action_history'},
               algorithm_changes=False,added_physics='hard actual-position joint stops',
               new_raw_target_penalty=False,per_joint_target_clamp=False,
               original_training_protocol=dict(critic_warmup=30,guard_ramp_start=.2,
                   guard_ramp_updates=300,anchor_coef=.2,epochs=3,minibatches=4,steps=24),
               resume_actor_critic_optimizer=True,environment_and_rng_restored=False,
               reference_anchor='original moe1000 raw mean, no projection',
               final_checkpoint=str(train/'model_1000.pt'),
               eval_summary_path=str(out/'evaluation/comparison.json'),
               strict_eval_summary_path=str(out/'strict_evaluation/comparison.json'),
               real_robot_driven=False,completion_verified=False)
    write(out/'job.json', job)
    def verify_inputs():
        if digest(source)!=SOURCE_SHA256 or digest(SOURCE_CHECKPOINT)!=SOURCE_SHA256:
            raise ValueError('Original model changed')
        if any(digest(ROOT/n)!=h for n,h in hashes.items()):
            raise ValueError('Snapshot source changed')
    try:
        verify_inputs()
        job['status']='training'; write(out/'job.json',job)
        with (out/'training.log').open('x') as stream:
            subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        checks={str(train/f'model_{n}.pt'):validate_checkpoint(train/f'model_{n}.pt',source,n)
                for n in (500,1000)}
        records=[json.loads(line) for line in (train/'progress.jsonl').read_text().splitlines()]
        if [r['update'] for r in records] != list(range(1,1001)):
            raise ValueError('Incomplete update sequence')
        if any(not r['all_losses_finite'] or r['actual_position_update']['excess_count'] != 0
               or r['mean_raw_target_cost_per_second'] != 0 for r in records):
            raise ValueError('Training invariant failed')
        job.update(status='evaluating',checkpoint_validation=checks,
                   evaluation_started_unix=time.time(),training_completed=True)
        summaries={}
        for mode,dirname in (('audit','evaluation'),('strict','strict_evaluation')):
            verify_inputs(); job['evaluation_phase']=mode; write(out/'job.json',job)
            cmd=[sys.executable,str(ROOT/'scripts/evaluate_moe_finetune.py'),'--actor',
                 f'{LABELS[0]}={source}',f'{LABELS[1]}={train / "model_500.pt"}',
                 f'{LABELS[2]}={train / "model_1000.pt"}','--out',str(out/dirname),
                 '--episodes-per-seed','64','--seeds','42','43','--device','cpu',
                 '--stair-steps','6','--position-constraint','hard',
                 '--target-mode','legacy','--guard-mode',mode]
            with (out/f'{dirname}.log').open('x') as stream:
                subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
            summary=json.loads((out/dirname/'comparison.json').read_text())
            if summary['environment_contract'] != environment_contract(6) or summary['guard_mode']!=mode:
                raise ValueError('Evaluation environment mismatch')
            rows=summary['rows']
            if [r['label'] for r in rows] != list(LABELS) or any(r['episodes']!=129 for r in rows):
                raise ValueError('Incomplete comparison')
            for row in rows:
                if row['target_mode']!='legacy' or row['guard']['metrics']['position']['excess_count']!=0:
                    raise ValueError('Evaluation hard-only invariant failed')
            summaries[mode]=rows
        verify_inputs()
        if any(digest(p)!=v['sha256'] for p,v in checks.items()):
            raise ValueError('Checkpoint changed during evaluation')
        report=dict(experiment_complete=True,training_updates=1000,cumulative_update=2000,
                    checkpoint_validation=checks,actual_position_bounds_passed=True,
                    source_unchanged=True,new_raw_target_penalty=False,per_joint_target_clamp=False,
                    evaluations=summaries,real_robot_driven=False)
        write(out/'validation_report.json',report)
        job.update(status='completed',completed_unix=time.time(),completion_verified=True,
                   validation_report=str(out/'validation_report.json'))
        write(out/'job.json',job)
    except BaseException as exc:
        job.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
        write(out/'job.json',job); raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args().out)
