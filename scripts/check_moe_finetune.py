#!/usr/bin/env python3
"""Read-only SSH progress inspection; acknowledgements are local only."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
REMOTE=r'''
import json,os,time
from pathlib import Path
root=Path(RUN_PATH)
job=json.loads((root/'job.json').read_text())
train=Path(job['training_dir'])
training=json.loads((train/'job.json').read_text()) if (train/'job.json').exists() else {}
def jsonl(path):
    if not path.exists():return []
    raw=path.read_text();lines=raw.splitlines()
    # A concurrent append can expose an incomplete last line, not corruption.
    if raw and not raw.endswith('\n'):lines=lines[:-1]
    return [json.loads(line) for line in lines]
records=jsonl(train/'progress.jsonl')
episodes=jsonl(train/'episodes.jsonl')
def window_guard(node):
    subset=[e for e in episodes if e['update']<=node][-256:]
    metrics={}
    for key in ('velocity','position','target_position','tracking','requested_torque','applied_torque'):
        rows=[e['guard']['metrics'][key] for e in subset]
        count=sum(r['excess_count'] for r in rows);n=sum(r['sample_count'] for r in rows)
        metrics[key]={'excess_count':count,'sample_count':n,'excess_fraction':count/n if n else None,
                      'peak_absolute_value':max((r['peak_absolute_value'] for r in rows),default=None),
                      'max_ratio':max((r['max_ratio'] for r in rows),default=None),
                      'max_excess':max((r['max_excess'] for r in rows),default=None)}
        if key=='velocity' and rows:
            metrics[key]['per_joint_peak_rad_s']={name:max(r['per_joint'][name]['peak_absolute_value'] for r in rows) for name in rows[0]['per_joint']}
    return {'scope':'latest up to 256 completed training episodes through this update; pre/post boundaries include failed terminal states; not all partial episodes of this update','episodes':len(subset),'metrics':metrics}
milestones=[]
for node in (500,1000):
    found=next((r for r in records if r['update']==node),None)
    if found is not None:
        milestones.append({'node':node,'record':found,'guard_window':window_guard(node),
                           'checkpoint_exists':(train/f'model_{node}.pt').is_file()})
pid=job.get('pid');alive=bool(pid and Path(f'/proc/{pid}/cmdline').exists() and b'run_moe_finetune.py' in Path(f'/proc/{pid}/cmdline').read_bytes())
status=job['status']
if status in ('training','preparing','evaluating') and not alive:status='stopped'
if status=='training' and time.time()-training.get('last_update_unix',job['started_unix'])>900:status='stalled'
summary=Path(job['eval_summary_path'])
final=Path(job['final_checkpoint'])
verified=False
if status=='completed' and summary.is_file() and final.is_file():
    rows=json.loads(summary.read_text()).get('rows',[])
    verified=job.get('completion_verified') is True and len(rows)==3 and all(r.get('episodes')==129 for r in rows)
print(json.dumps({'status':status,'pipeline_alive':alive,'pipeline':job,'training':training,
                  'milestones':milestones,'completion_verified':verified,
                  'final_checkpoint_exists':final.is_file(),'eval_summary_exists':summary.is_file(),
                  'eval_summary_path':str(summary)},allow_nan=False))
'''

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ack',type=int,choices=[500,1000])
    parser.add_argument('--ack-event',choices=['evaluating','completed','failed','stopped','stalled','unreachable'])
    args=parser.parse_args()
    cfg=json.loads((ROOT/'.monitor/moe_finetune_job.json').read_text())
    state_path=ROOT/'.monitor/moe_finetune_state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {'ack':0,'events':[]}
    if args.ack is not None:state['ack']=max(state['ack'],args.ack)
    if args.ack_event is not None:state['events']=sorted(set(state['events'])|{args.ack_event})
    if args.ack is not None or args.ack_event is not None:
        state_path.write_text(json.dumps(state,indent=2)+'\n');print(json.dumps({'acknowledged':state}));return
    code=REMOTE.replace('RUN_PATH',repr(cfg['remote_run']))
    cmd=['ssh','-S',cfg['ssh_control'],'-o','BatchMode=yes','-o','ConnectTimeout=10',
         '-o','UserKnownHostsFile='+cfg['known_hosts'],'-p',str(cfg['port']),cfg['host'],
         'python3 -']
    try:
        result=subprocess.run(cmd,input=code,text=True,capture_output=True,timeout=60,check=True)
        data=json.loads(result.stdout)
        data['pending_milestones']=[m for m in data.pop('milestones') if m['node']>state['ack']]
        event=data['status'] if data['status'] in ('evaluating','completed','failed','stopped','stalled') else None
    except (subprocess.SubprocessError,OSError,ValueError) as exc:
        data={'status':'unreachable','error':str(exc),'details':getattr(exc,'stderr',None),'pending_milestones':[]}
        event='unreachable'
    data['event_to_report']=event if event and event not in state['events'] else None
    data['checked_unix']=time.time()
    print(json.dumps(data,indent=2,allow_nan=False))

if __name__=='__main__':main()
