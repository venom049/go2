import json,pathlib,hashlib
r=pathlib.Path('/root/gpufree-data/moe1000-sim2real-20261105')
out={}
for name in ['server_smoke','formal_run/training']:
    p=r/name/'job.json'
    if not p.exists():continue
    j=json.loads(p.read_text());v=j.get('last_record',{})
    out[name]={k:j.get(k) for k in ('status','updates','error','critic_input_size')}
    out[name].update({k:v.get(k) for k in ('sample_success_rate_recent256','dr_curriculum','eta_s','frontier_height_m','all_losses_finite')})
    logs=r/name/'episodes.jsonl'
    if logs.exists():
        rows=[json.loads(s) for s in logs.read_text().splitlines()]
        out[name]['completed_episodes']=len(rows)
        out[name]['delayed_episodes']=sum(x['domain_randomization']['delay_steps']>0 for x in rows)
        out[name]['push_events']=sum(x['domain_randomization']['realized']['push_events'] for x in rows)
        out[name]['randomized_episodes']=sum(x['domain_randomization']['active'] for x in rows)
p=r/'formal_run/job.json'
if p.exists():out['pipeline']=json.loads(p.read_text())
p=r/'formal_run/comparison.json'
if p.exists():out['evaluation']=[{k:v[k] for k in ('model','mode','episodes','successes','statuses','push_events','delayed_episodes')} for v in json.loads(p.read_text())]
p=r/'delivery_job.json'
if p.exists():out['delivery']=json.loads(p.read_text())
out['evaluation_episode_counts']={p.name:len(list(p.glob('episode_*.json'))) for p in (r/'formal_run/evaluations').glob('*') if p.is_dir()}
p=r/'formal_run/exported/validation.json'
if p.exists():out['export_validation']=json.loads(p.read_text())
print(json.dumps(out))
