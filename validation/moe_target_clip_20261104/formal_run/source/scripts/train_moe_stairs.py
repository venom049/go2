#!/usr/bin/env python3
"""PPO fine-tuning of a verified MoE deployment actor. No robot I/O."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse
from collections import deque
import copy
import hashlib
import json
import multiprocessing as mp
import time
import traceback
import numpy as np
import torch
import yaml
from torch import nn
from go2_him.moe_cts_actor import MoECTSActor, load_actor_checkpoint
from go2_him.moe_stair_env import env_worker, original
from go2_him.moe_stair_rewards import DEFAULT_CONFIG
from go2_him.official_guard import load_official_profile
from go2_him.joint_recovery import CONFIG as JOINT_RECOVERY_CONFIG
from go2_him.target_clip import TARGET_CLIP_CONFIG, TargetClipAccumulator

FORMAT='moe_cts_ppo_finetune_v1'

def atomic_json(path,data):
    tmp=Path(str(path)+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)

def load_policy(path,device='cpu'):
    data=torch.load(path,map_location='cpu',weights_only=True)
    if data.get('format')==FORMAT:
        model=MoECTSActor()
        model.load_state_dict(data['actor_state_dict'],strict=True)
        return model.to(device).eval()
    return load_actor_checkpoint(path,device=device).eval()

def target_action_contract():
    """Record the same affine decoder and named bounds used by the simulator."""
    config_path=original.POLICY_DIR/'params/deploy.yaml'
    cfg=yaml.safe_load(config_path.read_text())['actions']['JointPositionAction']
    profile=load_official_profile(joint_names=original.JOINTS)
    scale=np.asarray(cfg['scale'],dtype=np.float64)
    offset=np.asarray(cfg['offset'],dtype=np.float64)
    if scale.shape!=(12,) or offset.shape!=(12,) or not np.isfinite(scale).all() or np.any(scale<=0):
        raise ValueError('Expected 12 finite positive action scales')
    if not np.isfinite(offset).all() or np.any(offset<profile.lower) or np.any(offset>profile.upper):
        raise ValueError('Default action offset must lie inside joint target limits')
    lower_exact=(profile.lower-offset)/scale
    upper_exact=(profile.upper-offset)/scale
    lower=lower_exact.astype(np.float32)
    upper=upper_exact.astype(np.float32)
    lower=np.where(lower.astype(np.float64)<lower_exact,np.nextafter(lower,np.float32(np.inf)),lower)
    upper=np.where(upper.astype(np.float64)>upper_exact,np.nextafter(upper,np.float32(-np.inf)),upper)
    return {'kind':'bounded_joint_target_v1','joint_names':list(profile.names),
            'scale':scale.tolist(),'offset':offset.tolist(),
            'lower_rad':profile.lower.tolist(),'upper_rad':profile.upper.tolist(),
            'raw_action_lower':lower.tolist(),'raw_action_upper':upper.tolist(),
            'reference_anchor_bound_rounding':'inward float32 to avoid affine boundary overshoot',
            'history':'inverse-mapped post-clipping submitted target; float32 action',
            'initial_history':'zero action at settled simulation pose; no hardware handoff simulated',
            'ppo_distribution':'unbounded Gaussian; original sampled action and log probability retained',
            'target_slew_limiter':False,'actual_state_clipping':False,
            'reference_anchor':'source mean projected to per-joint target-action bounds',
            'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
            'profile_sha256':profile.source_sha256}

class Vec:
    def __init__(self,n,workers,seed):
        assert n%workers==0
        self.count=n//workers
        self.pipes=[];self.processes=[]
        # Fork before any CUDA context; workers only use NumPy/MuJoCo/ORT.
        ctx=mp.get_context('fork')
        for i in range(workers):
            parent,child=ctx.Pipe()
            proc=ctx.Process(target=env_worker,args=(child,seed+i*self.count,self.count),daemon=True)
            proc.start();child.close();self.pipes.append(parent);self.processes.append(proc)
        self.obs=np.concatenate([self.recv(p) for p in self.pipes])
    def recv(self,p):
        if not p.poll(180):raise RuntimeError('MuJoCo worker timed out')
        return p.recv()
    def settings(self,frontier,ramp):
        for p in self.pipes:p.send(('settings',{'frontier':frontier,'ramp':ramp,
                                              'guard_objective':getattr(self,'guard_objective','balanced')}))
        for p in self.pipes:self.recv(p)
    def step(self,actions):
        for i,p in enumerate(self.pipes):p.send(('step',actions[i*self.count:(i+1)*self.count]))
        out=[self.recv(p) for p in self.pipes]
        return np.concatenate([x[0] for x in out]),np.concatenate([x[1] for x in out]),np.concatenate([x[2] for x in out]),sum([x[3] for x in out],[])
    def close(self):
        for p in self.pipes:
            try:p.send(('close',None))
            except (EOFError,BrokenPipeError):pass
        for proc in self.processes:
            proc.join(10)
            if proc.is_alive():proc.terminate();proc.join(5)

def gae(rewards,values,dones,last_value,gamma=.99,lam=.95):
    adv=torch.zeros_like(rewards);running=torch.zeros_like(last_value)
    for t in reversed(range(len(rewards))):
        nxt=last_value if t==len(rewards)-1 else values[t+1]
        mask=1.-dones[t]
        running=rewards[t]+gamma*nxt*mask-values[t]+gamma*lam*mask*running
        adv[t]=running
    return adv,adv+values

def initialize_training_state(args,device):
    """Explicit full resume, or the original actor-only warm start.

    The new PPO anchor is the supplied actor. A full resume retains critic,
    Adam moments/LRs and exploration std; its changed reward objective is
    recorded separately. Environment and RNG state are intentionally fresh.
    """
    actor=load_policy(args.actor,device=device)
    for p in actor.student_moe_encoder.parameters():p.requires_grad_(False)
    critic=nn.Sequential(nn.Linear(225,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,1)).to(device)
    optimizer=torch.optim.Adam([{'params':actor.actor.parameters(),'lr':args.actor_lr},
                                {'params':critic.parameters(),'lr':3e-4}])
    metadata={'full_training_state_resumed':False,'source_cumulative_update':0,
              'actor_restored':True,'critic_restored':False,'optimizer_restored':False,
              'environment_and_rng_restored':False,'reference_anchor':'supplied checkpoint actor'}
    if getattr(args,'resume_training',False):
        data=torch.load(args.actor,map_location='cpu',weights_only=True)
        if data.get('format')!=FORMAT or data.get('frozen_moe_encoder') is not True:
            raise ValueError('Full resume requires an audited frozen-encoder MoE training checkpoint')
        source_update=data.get('cumulative_update',data.get('update'))
        if type(source_update) is not int or source_update<=0:
            raise ValueError('Resume checkpoint lacks a valid completed update count')
        if data.get('profile',{}).get('source_sha256')!=load_official_profile().source_sha256:
            raise ValueError('Resume checkpoint guard profile differs from the installed profile')
        if data.get('std')!=args.std:
            raise ValueError('Full resume must preserve the checkpoint exploration std')
        critic.load_state_dict(data['critic_state_dict'],strict=True)
        optimizer.load_state_dict(data['optimizer_state_dict'])
        if len(optimizer.param_groups)!=2:
            raise ValueError('Unexpected optimizer parameter groups')
        metadata.update(full_training_state_resumed=True,source_cumulative_update=source_update,
                        critic_restored=True,optimizer_restored=True)
    tensors=list(actor.state_dict().values())+list(critic.state_dict().values())
    tensors += [v for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v)]
    if any(not torch.isfinite(value).all() for value in tensors):
        raise FloatingPointError('Nonfinite checkpoint weights or optimizer state')
    reference=copy.deepcopy(actor).eval()
    for p in reference.parameters():p.requires_grad_(False)
    return actor,critic,optimizer,reference,metadata

def train(args):
    objective=getattr(args,'guard_objective','balanced')
    ramp_start=getattr(args,'guard_ramp_start',.2)
    ramp_updates=getattr(args,'guard_ramp_updates',300)
    if objective not in ('balanced','calf_velocity','joint_recovery','target_clipped'):
        raise ValueError('Unknown guard objective')
    if not 0<ramp_start<=1 or ramp_updates<0 or args.updates<=0:
        raise ValueError('Invalid reward ramp or update count')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    if args.out.exists():raise FileExistsError('Refusing to overwrite a training run')
    args.out.mkdir(parents=True)
    start=time.time()
    job={'status':'initializing','pid':os.getpid(),'started_unix':start,'updates':0,
         'max_updates':args.updates,'num_envs':args.num_envs,'workers':args.workers,
         'warm_start':str(args.actor),'source_sha256':hashlib.sha256(args.actor.read_bytes()).hexdigest(),
         'profile':load_official_profile().metadata(),'task_rewards':DEFAULT_CONFIG,
         'training_method':('PPO actor-head continuation; frozen MoE; restore actor/critic/optimizer'
                            if getattr(args,'resume_training',False) else
                            'PPO actor-head fine-tuning; frozen pretrained MoE; new critic and optimizer'),
         'guard_objective':objective,
         'guard_cost_config':({'kind':'calf_velocity','joint_selector':'four named *_calf joints',
                              'soft_ratio':.90,'budget_reward_units_per_second':4.,
                              'raw_cost':'.5*margin^2 + 4*hard + 4*hard^2',
                              'other_joint_guard_reward_weights':0,
                              'reduction':'worst calf and pre/post physics boundary',
                              'physical_velocity_clipping':False}
                             if objective=='calf_velocity' else TARGET_CLIP_CONFIG if objective=='target_clipped'
                             else JOINT_RECOVERY_CONFIG if objective=='joint_recovery'
                             else {'kind':'balanced','budget_reward_units_per_second':3.}),
         'external_platform_stop':True,'real_robot_driven':False,
         'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}}
    sources=['scripts/train_moe_stairs.py','go2_him/moe_stair_env.py','go2_him/moe_stair_rewards.py','go2_him/official_guard.py','go2_him/calf_speed.py','go2_him/moe_cts_actor.py','scripts/test_robogauge_stairs.py','scripts/test_robogauge_two_steps_wall.py']
    sources.extend(['go2_him/joint_recovery.py','go2_him/target_clip.py'])
    job['action_contract']=target_action_contract() if objective=='target_clipped' else {'kind':'legacy_raw_action_history'}
    job['source_hashes']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
    atomic_json(args.out/'job.json',job)
    vec=None
    try:
        vec=Vec(args.num_envs,args.workers,args.seed)
        vec.guard_objective=objective
        device=torch.device(args.device)
        actor,critic,optimizer,reference,resume_metadata=initialize_training_state(args,device)
        if objective=='target_clipped':
            action_lower=torch.tensor(job['action_contract']['raw_action_lower'],device=device,dtype=torch.float32)
            action_upper=torch.tensor(job['action_contract']['raw_action_upper'],device=device,dtype=torch.float32)
        profile=load_official_profile(joint_names=original.JOINTS)
        job['initialization']=resume_metadata
        source_update=resume_metadata['source_cumulative_update']
        std=torch.full((12,),args.std,device=device)
        obs=torch.as_tensor(vec.obs,device=device)
        recent=deque(maxlen=256);frontier=.15;last_adjust=0
        job['status']='training';atomic_json(args.out/'job.json',job)
        for update in range(1,args.updates+1):
            t0=time.time()
            ramp=ramp_start+(1-ramp_start)*min((update-1)/ramp_updates,1.) if ramp_updates else 1.
            vec.settings(frontier,ramp)
            buf={k:[] for k in ('obs','actions','logp','values','rewards','dones','mean')}
            episodes=[];costs=[];calf_samples=[];joint_samples=[];cost_components=[]
            target_samples=[];raw_target_costs=[]
            for _ in range(args.steps):
                with torch.no_grad():
                    mean=actor(obs);distribution=torch.distributions.Normal(mean,std)
                    action=distribution.sample();logp=distribution.log_prob(action).sum(-1)
                    value=critic(obs).squeeze(-1)
                next_obs,reward,done,infos=vec.step(action.cpu().numpy())
                for info in infos:
                    costs.append(info['guard_cost'])
                    if 'calf_speed' in info:calf_samples.append(info['calf_speed'])
                    if 'joint_guard_update' in info:joint_samples.append(info['joint_guard_update'])
                    if 'guard_cost_components' in info:cost_components.append(info['guard_cost_components'])
                    if 'target_step' in info:target_samples.append(info['target_step'])
                    if 'raw_target_cost_rate' in info:raw_target_costs.append(info['raw_target_cost_rate'])
                    if 'episode' in info:episodes.append(info['episode']);recent.append(info['episode'])
                for key,val in (('obs',obs),('actions',action),('logp',logp),('values',value),('rewards',torch.as_tensor(reward,device=device)),('dones',torch.as_tensor(done,dtype=torch.float32,device=device)),('mean',mean)):
                    buf[key].append(val)
                obs=torch.as_tensor(next_obs,device=device)
            buf={k:torch.stack(v) for k,v in buf.items()}
            with torch.no_grad():
                adv,returns=gae(buf['rewards'],buf['values'],buf['dones'],critic(obs).squeeze(-1))
                adv=(adv-adv.mean())/(adv.std()+1e-8)
                flat_obs=buf['obs'].flatten(0,1);anchor=reference(flat_obs)
                if objective=='target_clipped':
                    # An illegal teacher request must not pull the student back
                    # outside its command bounds. PPO samples/logp remain raw.
                    anchor=torch.clamp(anchor,min=action_lower,max=action_upper)
            flat={k:v.flatten(0,1) for k,v in buf.items()}
            advantage=adv.flatten();target_value=returns.flatten()
            enabled=update>args.critic_warmup
            losses=[];kl_max=0.;early=False;rejected_updates=0;accepted_post_kl_max=0.
            for epoch in range(args.epochs):
                for idx in torch.randperm(len(flat_obs),device=device).chunk(args.minibatches):
                    mean=actor(flat_obs[idx]);dist=torch.distributions.Normal(mean,std)
                    ratio=(dist.log_prob(flat['actions'][idx]).sum(-1)-flat['logp'][idx]).exp()
                    pi_loss=-torch.minimum(ratio*advantage[idx],ratio.clamp(.9,1.1)*advantage[idx]).mean()
                    anchor_loss=(mean-anchor[idx]).square().mean()
                    value_loss=(critic(flat_obs[idx]).squeeze(-1)-target_value[idx]).square().mean()
                    kl=((mean-flat['mean'][idx]).square()/(2*std.square())).sum(-1).mean()
                    kl_max=max(kl_max,float(kl.detach()))
                    if enabled and kl>args.target_kl*2:early=True;break
                    loss=.5*value_loss+((pi_loss+args.anchor_coef*anchor_loss) if enabled else 0.)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite PPO loss')
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group['params']],.5,error_if_nonfinite=True)
                    head=list(actor.actor.parameters())
                    saved_params=[p.detach().clone() for p in head] if enabled else []
                    saved_adam=[copy.deepcopy(optimizer.state[p]) for p in head] if enabled else []
                    optimizer.step()
                    if enabled:
                        with torch.no_grad():
                            post_kl=((actor(flat_obs[idx])-flat['mean'][idx]).square()/(2*std.square())).sum(-1).mean()
                        kl_max=max(kl_max,float(post_kl))
                        if not torch.isfinite(post_kl) or post_kl>2*args.target_kl:
                            # Reject the actor update and its Adam moments;
                            # keep the independent critic update. No parameter
                            # or action clipping disguises the rejected step.
                            with torch.no_grad():
                                for p,value,state in zip(head,saved_params,saved_adam):
                                    p.copy_(value);optimizer.state[p]=state
                            optimizer.param_groups[0]['lr']=max(1e-7,optimizer.param_groups[0]['lr']*.5)
                            rejected_updates+=1
                        else:accepted_post_kl_max=max(accepted_post_kl_max,float(post_kl))
                    losses.append((float(value_loss.detach()),float(pi_loss.detach()),float(anchor_loss.detach())))
                if early:break
            success=float(np.mean([e['success'] for e in recent])) if recent else None
            if len(recent)>=64 and update-last_adjust>=25:
                if success<.6:frontier=max(.10,frontier-.015)
                elif success>.85:frontier=min(.15,frontier+.005)
                last_adjust=update
            record={'update':update,'cumulative_update':source_update+update,
                    'transitions':update*args.num_envs*args.steps,'elapsed_s':time.time()-start,
                    'update_seconds':time.time()-t0,'eta_s':(time.time()-start)/update*(args.updates-update),
                    'sample_success_rate_recent256':success,'completed_episodes_this_update':len(episodes),
                    'mean_reward_per_step':float(buf['rewards'].mean()),'frontier_height_m':frontier,'guard_ramp':ramp,
                    'mean_guard_cost_per_second':float(np.mean(costs)),'max_guard_cost_per_second':float(np.max(costs)),
                    'guard_objective':objective,'mean_effective_guard_cost_per_second':ramp*float(np.mean(costs)),
                    'critic_only_warmup':not enabled,'all_losses_finite':True,'kl_max':kl_max,
                    'accepted_minibatch_post_kl_max':accepted_post_kl_max,'rejected_actor_updates':rejected_updates,
                    'actor_learning_rate':optimizer.param_groups[0]['lr'],
                    'loss_mean_value_policy_anchor':np.mean(losses,axis=0).tolist() if losses else None,
                    'mean_forward_m_recent':float(np.mean([e['forward_m'] for e in recent])) if recent else None,
                    'timeout_fraction_recent':float(np.mean([e['status']=='timeout' for e in recent])) if recent else None}
            if target_samples:
                target_audit=TargetClipAccumulator(profile.names)
                target_audit.record(np.stack([s['raw_target_rad'] for s in target_samples]),
                                    np.stack([s['sent_target_rad'] for s in target_samples]),
                                    profile.lower,profile.upper)
                record['target_commands_update']=target_audit.summary()
            if raw_target_costs:
                record['mean_raw_target_cost_per_second']=float(np.mean(raw_target_costs))
            if calf_samples:
                count=sum(s['excess_count'] for s in calf_samples);n=sum(s['sample_count'] for s in calf_samples)
                record['calf_speed_update']={'scope':'all rollout physical pre/post boundaries of this update, including partial and terminal episodes',
                    'excess_count':count,'sample_count':n,'excess_fraction':count/n if n else None,
                    'peak_absolute_value':max(s['peak_absolute_value'] for s in calf_samples),
                    'max_ratio':max(s['max_ratio'] for s in calf_samples),
                    'per_joint_peak_rad_s':{name:max(s['per_joint_peak_rad_s'][name] for s in calf_samples)
                                           for name in calf_samples[0]['per_joint_peak_rad_s']}}
            if joint_samples:
                metrics={}
                for key in joint_samples[0]:
                    rows=[item[key] for item in joint_samples]
                    count=sum(r['excess_count'] for r in rows); n=sum(r['sample_count'] for r in rows)
                    per_joint={}
                    for name in rows[0]['per_joint']:
                        jr=[r['per_joint'][name] for r in rows]
                        jc=sum(r['excess_count'] for r in jr); jn=sum(r['sample_count'] for r in jr)
                        per_joint[name]={'excess_count':jc,'sample_count':jn,'excess_fraction':jc/jn,
                            'max_excess':max(r['max_excess'] for r in jr),
                            'min_value':min(r['min_value'] for r in jr),'max_value':max(r['max_value'] for r in jr)}
                    metrics[key]={'excess_count':count,'sample_count':n,'excess_fraction':count/n,
                        'max_excess':max(r['max_excess'] for r in rows),
                        'peak_absolute_value':max(r['peak_absolute_value'] for r in rows),'per_joint':per_joint}
                record['joint_guard_update']={'scope':'All partial and completed rollout episodes in this update; every pre/post physics boundary',
                                              'metrics':metrics}
            if cost_components:
                record['mean_guard_cost_components_per_second']={key:float(np.mean([r[key] for r in cost_components]))
                                                               for key in cost_components[0]}
            if update in {1,50,250,500,args.updates}:
                checkpoint=args.out/f'model_{update}.pt'
                torch.save({'format':FORMAT,'actor_state_dict':actor.state_dict(),'critic_state_dict':critic.state_dict(),
                            'optimizer_state_dict':optimizer.state_dict(),'update':update,'frozen_moe_encoder':True,
                            'cumulative_update':source_update+update,'initialization':resume_metadata,
                            'guard_objective':objective,'guard_cost_config':job['guard_cost_config'],
                            'action_contract':job['action_contract'],
                            'std':args.std,'source_conversion_sha256':job['source_sha256'],'profile':job['profile'],
                            'training_record':record},checkpoint)
                record['checkpoint']=str(checkpoint)
            with (args.out/'progress.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
            if episodes:
                with (args.out/'episodes.jsonl').open('a') as f:
                    for e in episodes:f.write(json.dumps({'update':update,**e},allow_nan=False)+'\n')
            job.update(updates=update,last_record=record,last_update_unix=time.time())
            atomic_json(args.out/'job.json',job)
            print(json.dumps(record),flush=True)
        job.update(status='trained',completed_unix=time.time(),final_checkpoint=str(args.out/f'model_{args.updates}.pt'))
        atomic_json(args.out/'job.json',job)
    except BaseException as exc:
        job.update(status='failed',error=repr(exc),traceback=traceback.format_exc(),last_update_unix=time.time())
        atomic_json(args.out/'job.json',job);raise
    finally:
        if vec is not None:vec.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--actor',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--updates',type=int,default=1000);p.add_argument('--num-envs',type=int,default=128)
    p.add_argument('--workers',type=int,default=32);p.add_argument('--steps',type=int,default=24)
    p.add_argument('--epochs',type=int,default=3);p.add_argument('--minibatches',type=int,default=4)
    p.add_argument('--critic-warmup',type=int,default=30);p.add_argument('--actor-lr',type=float,default=3e-6)
    p.add_argument('--std',type=float,default=.03);p.add_argument('--target-kl',type=float,default=.005)
    p.add_argument('--anchor-coef',type=float,default=.2);p.add_argument('--seed',type=int,default=20261104)
    p.add_argument('--resume-training',action='store_true',help='Restore actor, critic, Adam and std from a training checkpoint')
    p.add_argument('--guard-objective',choices=['balanced','calf_velocity','joint_recovery','target_clipped'],default='balanced')
    p.add_argument('--guard-ramp-start',type=float,default=.2)
    p.add_argument('--guard-ramp-updates',type=int,default=300)
    p.add_argument('--device',default='cuda:0');train(p.parse_args())

if __name__=='__main__':main()
