"""Validate completed run and build a self-contained delivery directory."""
from pathlib import Path
import sys,json,hashlib,shutil,tarfile,gzip,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'source'),str(ROOT/'source/scripts')]
from train_moe_stairs import load_policy

def read(p):return json.loads(p.read_text())
def write(p,data):p.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    run=ROOT/'formal_run';j=read(run/'job.json');assert j['status']=='completed'
    train=read(run/'training/job.json');assert train['updates']==1000
    records=[json.loads(s) for s in (run/'training/progress.jsonl').read_text().splitlines()]
    episodes=[json.loads(s) for s in (run/'training/episodes.jsonl').read_text().splitlines()]
    assert len(records)==1000 and all(v['all_losses_finite'] for v in records)
    ckpt=run/'training/model_1000.pt';data=torch.load(ckpt,map_location='cpu',weights_only=True)
    source=load_policy(ROOT/'source/validation/moe_cts_actor_conversion_20261104/actor.pt').state_dict()
    actual=data['actor_state_dict'];enc=[k for k in source if k.startswith('student_moe_encoder.')]
    assert all(torch.equal(source[k],actual[k]) for k in enc)
    changed=[k for k in source if not torch.equal(source[k],actual[k])]
    assert changed and all(k.startswith('actor.') for k in changed)
    tensors=list(actual.values())+list(data['critic_state_dict'].values())
    tensors += [v for state in data['optimizer_state_dict']['state'].values() for v in state.values() if torch.is_tensor(v)]
    assert all(torch.isfinite(t).all() for t in tensors)
    assert data['critic_state_dict']['0.weight'].shape==(512,370)
    out=ROOT/'delivery';out.mkdir(exist_ok=False);ev=out/'evidence';ev.mkdir()
    shutil.copytree(run/'exported',out/'exported');shutil.copy2(ckpt,out/'model_1000.pt')
    (out/'params').mkdir()
    cfg=ROOT/'source/upstream/robogauge/unitree_cpp_deploy/logs/rsl_rl/go2_moe_cts_expert_goal_137000_0.6745/params/deploy.yaml'
    shutil.copy2(cfg,out/'params/deploy.yaml')
    for name in ('job.json','comparison.json'):shutil.copy2(run/name,ev/name)
    shutil.copy2(run/'training/job.json',ev/'training_job.json')
    for name in ('progress.jsonl','episodes.jsonl'):
        with (run/'training'/name).open('rb') as src,gzip.open(ev/(name+'.gz'),'wb') as dst:shutil.copyfileobj(src,dst)
    for name in ('server_tests.log','launch.json','run_config.json','local_validation.json','actor_tests.log','target_tests.log','reward_tests.log'):
        if (ROOT/name).exists():shutil.copy2(ROOT/name,ev/name)
    results=[];strict=read(run/'evaluations/new_strict/summary.json')
    pairs=[]
    for mode in ('nominal','light','moderate','full'):
        for i in range(129):
            previous=read(run/f'evaluations/previous_{mode}/episode_{i:03d}.json')
            new=read(run/f'evaluations/new_{mode}/episode_{i:03d}.json')
            a={k:v for k,v in previous['domain_randomization'].items() if k!='realized'}
            b={k:v for k,v in new['domain_randomization'].items() if k!='realized'}
            assert a==b,(mode,i,'unmatched randomization')
            np.testing.assert_allclose(previous['initial_state']['base_position_m'],new['initial_state']['base_position_m'],atol=0,rtol=0)
        pairs.append({'mode':mode,'paired_episodes':129,'identical_initial_parameters':True})
    for summary in sorted((run/'evaluations').glob('*/summary.json')):
        v=read(summary);shutil.copy2(summary,ev/(summary.parent.name+'.json'))
        results.append({'test':summary.parent.name,'successes':v['successes'],'episodes':v['episodes'],
            'success_rate':v['successes']/v['episodes'],'statuses':v['statuses'],
            'target_excess_count':v['target_commands']['sent']['excess_count'],
            'target_samples':v['target_commands']['sent']['sample_count'],
            'climb_horizontal_speed_m_s':v['gait_success_only']['climb_horizontal_speed_m_s'],
            'guard_acceptance_passed':v['guard']['joint_and_torque_acceptance_passed'],
            'guard_excess_counts':{k:x['excess_count'] for k,x in v['guard']['metrics'].items()},
            'delayed_episodes':v['delayed_episodes'],'push_events':v['push_events']})
    dr=[e['domain_randomization'] for e in episodes]
    realized={'completed_episodes':len(dr),'randomized_episodes':sum(d['active'] for d in dr),
        'nominal_episodes':sum(not d['active'] for d in dr),
        'delayed_episodes':sum(d['delay_steps']>0 for d in dr),
        'push_events':sum(d['realized']['push_events'] for d in dr),
        'final_curriculum':records[-1]['dr_curriculum'],
        'max_episode_level':max(d['level'] for d in dr),
        'observed_ranges':{k:{'min':float(np.min([d[k] for d in dr])), 'max':float(np.max([d[k] for d in dr]))}
            for k in ('friction','kp_factors','kd_factors','motor_strength','base_mass_shift_kg','base_com_shift_m','hip_mass_shift_kg','gravity_offset_m_s2','total_mass_kg')},
        'scope':'Completed episodes only; update command counts below include all rollout steps'}
    n=sum(v['target_commands_update']['sent']['sample_count'] for v in records)
    bad=sum(v['target_commands_update']['sent']['excess_count'] for v in records);assert n==36864000 and bad==0
    report={'completed':True,'updates':1000,'transitions':3072000,'robot':'Go2 EDU without extra payload',
        'official_model_commit':'ae6a8403e272733e9996ef59990880330496177f',
        'official_model_xml_sha256':digest(ROOT/'source/assets/unitree_go2/go2.xml'),
        'official_generic_go2_not_individually_calibrated_edu':True,
        'training_started_unix':train['started_unix'],'training_completed_unix':train['completed_unix'],
        'training_seconds':train['completed_unix']-train['started_unix'],
        'target_samples':n,'target_excess_count':bad,'frozen_encoder_tensors':len(enc),'changed_actor_tensors':len(changed),
        'finite_checkpoint_tensors':len(tensors),'critic_input_size':370,'actor_input_size':225,'actor_output_size':12,
        'training_last_record':{k:records[-1][k] for k in ('sample_success_rate_recent256','actor_learning_rate','frontier_height_m','all_losses_finite')},
        'realized_randomization':realized,'paired_evaluation_validation':pairs,'evaluations':results,
        'onnx_validation':read(run/'exported/validation.json'),'real_robot_driven':False,
        'limits':'Targets bounded; actual position uses native official soft limits; velocity and tracking guards not guaranteed.',
        'runtime':'Deployment actor needs only original 225 inputs; privileged critic is training-only; deployment uses fixed 20/.5 PD. Noise/delay/pushes are training/evaluation uncertainty, not deployment features.'}
    write(out/'evaluation_summary.json',report)
    write(out/'manifest.json',{'model_source':train['robot_model'],'domain_randomization':train['domain_randomization'],
        'warm_start':train['warm_start'],'warm_start_sha256':train['source_sha256'],
        'checkpoint_sha256':digest(ckpt),'onnx_sha256':digest(out/'exported/policy.onnx'),
        'source_hashes':train['source_hashes'],'runtime_contract':'exported/contract.json'})
    with tarfile.open(out/'source_and_reference.tar.gz','w:gz') as archive:
        for name in ('source','reference','run_pipeline.py','evaluate_transfer.py','export_safe_action_policy.py','package_results.py','finish_run.py'):
            archive.add(ROOT/name,arcname=name,filter=lambda info:None if '__pycache__' in info.name or info.name.endswith('.pyc') else info)
    with tarfile.open(ev/'full_evaluation.tar.gz','w:gz') as archive:archive.add(run/'evaluations',arcname='evaluations')
    (out/'USAGE.txt').write_text('Go2 EDU 官方模型 + Sim2Real 随机化 MoE1000\n主模型 exported/policy.onnx，输入 obs float32[1,225]、输出 actions float32[1,12]。\n沿用 params/deploy.yaml 的五帧分组历史、动作缩放0.25、默认关节角、固定PD20/0.5和20ms周期。历史记录模型返回的有界动作。\n训练使用官方通用Go2 MJCF，无额外载荷，未对特定EDU实机标定。价值网络370维只用于训练，不用于部署。\n策略初始化仍来自RoboGauge MoE-CTS；使用官方机器人模型不等于使用官方策略。\n模型内含目标角度限幅；单独提取PT原始actor会失去此保证。保留控制器原有实际位置、速度、力矩、跟踪及接管保护。\n实际随机化范围、扰动测试和失败结果见 evaluation_summary.json；不得把配置最大范围当成训练实际达到的范围。\n本次未连接或驱动真实机器人。\n')
    files=sorted(p for p in out.rglob('*') if p.is_file())
    (out/'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.relative_to(out)}\n' for p in files))
    archive=ROOT/'moe1000_go2edu_sim2real_20261105.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:tar.add(out,arcname='moe1000_go2edu_sim2real_20261105')
    print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':digest(archive),'summary':report},ensure_ascii=False))
if __name__=='__main__':main()
