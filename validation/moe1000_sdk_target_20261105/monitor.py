"""Read-only remote training monitor; no credentials stored."""
import argparse
import json
from pathlib import Path
import subprocess
import time

REMOTE=r'''
import json,pathlib,time,os
r=pathlib.Path('/root/gpufree-data/moe1000-sdk-target-20261105/formal_run')
j=json.loads((r/'job.json').read_text())
t=json.loads((r/'training/job.json').read_text())
v=t.get('last_record',{})
lines=(r/'training/progress.jsonl').read_text().splitlines()
records=[]
for line in lines:
    try:records.append(json.loads(line))
    except json.JSONDecodeError:break
out=dict(status=j['status'],pipeline_alive=pathlib.Path('/proc/'+str(j['pid'])).exists(),
    update=t['updates'],max_updates=t['max_updates'],eta_minutes=round(v.get('eta_s',0)/60,1),
    sample_success_rate=v.get('sample_success_rate_recent256'),
    sent_target_samples=sum(x['target_commands_update']['sent']['sample_count'] for x in records),
    sent_target_excess=sum(x['target_commands_update']['sent']['excess_count'] for x in records),
    losses_finite=all(x['all_losses_finite'] for x in records),
    evaluation_phase=j.get('evaluation_phase'),error=j.get('error'),
    completion_verified=j.get('completion_verified',False),checked_unix=time.time())
if j['status']=='evaluating':
    p=r/('evaluation_'+j['evaluation_phase']+'.log')
    if p.exists():out['evaluation_last_line']=p.read_text().splitlines()[-1:]
print(json.dumps(out))
'''

def check():
    cmd=['ssh','-p','30916','-S','/tmp/go2-training-30916.sock','-o','BatchMode=yes',
         '-o','ConnectTimeout=15','-o','UserKnownHostsFile=/tmp/go2-training-known-hosts',
         'root@183.147.142.40','/root/go2-himloco/.venv-moe-finetune/bin/python -']
    p=subprocess.run(cmd,input=REMOTE,text=True,capture_output=True,timeout=30,check=True)
    result=json.loads(p.stdout)
    path=Path(__file__).with_name('latest_snapshot.json')
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(path)
    print(json.dumps(result),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');args=p.parse_args()
    while True:
        state=check()
        if not args.watch or state['status'] in ('completed','failed') or not state['pipeline_alive']:break
        time.sleep(45)
