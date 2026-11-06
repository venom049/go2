#!/usr/bin/env python3
"""Read-only remote inspection; acknowledgements affect local state only."""
import argparse
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'validation/moe_remote_20261105/monitor_state.json'
SNAPSHOT = STATE.with_name('monitor_snapshot.json')
NODES = (200, 400, 600, 800, 1000)
REMOTE = r'''
import json, time
from pathlib import Path
root = Path('/root/gpufree-data/moe_hard_20261105/formal_run')
job = json.loads((root/'job.json').read_text())
training = json.loads((root/'training/job.json').read_text())
raw = (root/'training/progress.jsonl').read_text()
lines = raw.splitlines() if raw.endswith('\n') else raw.splitlines()[:-1]
records = [json.loads(line) for line in lines if line.strip()]
status = job['status']
try:
    alive = b'run_moe_target_clip_training.py' in Path('/proc/%s/cmdline' % job['pid']).read_bytes()
except OSError:
    alive = False
if status in ('training', 'preparing', 'evaluating') and not alive:
    status = 'stopped'
elif training.get('status') == 'failed':
    status = 'failed'
elif status == 'training' and time.time()-training.get('last_update_unix',job['started_unix'])>900:
    status = 'stalled'
result = dict(status=status, pipeline_alive=alive, pipeline=job,
              latest=records[-1] if records else {}, error=training.get('error', job.get('error')),
              milestones=[r for r in records if r['update'] in (200,400,600,800,1000)])
report = root/'validation_report.json'
if status == 'completed' and report.exists():
    result['validation_report'] = json.loads(report.read_text())
print(json.dumps(result,allow_nan=False))
'''


def save(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ack', type=int, choices=NODES)
    parser.add_argument('--ack-event')
    args = parser.parse_args()
    state = json.loads(STATE.read_text()) if STATE.exists() else {'reported_nodes':[], 'reported_events':[]}
    if args.ack or args.ack_event:
        snapshot = json.loads(SNAPSHOT.read_text())
        if args.ack:
            observed = {r['update'] for r in snapshot.get('milestones',[])}
            if args.ack not in observed:
                raise ValueError('Cannot acknowledge an unobserved milestone')
            state['reported_nodes'] = sorted(set(state['reported_nodes']) | {args.ack})
        if args.ack_event:
            if args.ack_event != snapshot['status']:
                raise ValueError('Event differs from latest observed status')
            state['reported_events'] = sorted(set(state['reported_events']) | {args.ack_event})
        save(STATE,state)
        print(json.dumps(state)); return
    command = ['ssh','-p','30916','-o','ControlPath=/tmp/moe-train-30916.sock',
               '-o','UserKnownHostsFile=/tmp/moe-train-30916-known-hosts',
               '-o','StrictHostKeyChecking=yes','-o','BatchMode=yes','-o','ConnectTimeout=15',
               'root@183.147.142.40',
               '/root/gpufree-data/moe_hard_20261105/.venv/bin/python -']
    try:
        process = subprocess.run(command,input=REMOTE,text=True,capture_output=True,timeout=45,check=True)
        result = json.loads(process.stdout)
    except (subprocess.SubprocessError, ValueError, OSError) as exc:
        result = {'status':'unreachable','error':str(exc),
                  'detail':getattr(exc,'stderr',None),'milestones':[]}
    result['checked_unix'] = time.time()
    result['pending_milestones'] = [r for r in result.get('milestones',[]) if r['update'] not in state['reported_nodes']]
    events = ('evaluating','completed','failed','stopped','stalled','unreachable')
    result['pending_event'] = result['status'] if result['status'] in events and result['status'] not in state['reported_events'] else None
    save(SNAPSHOT,result)
    save(STATE,state)
    # Keep normal polls compact; complete evidence remains in the snapshot.
    output = {k:result.get(k) for k in ('status','checked_unix','pending_milestones','pending_event','error')}
    output['latest'] = {k:result.get('latest',{}).get(k) for k in (
        'update','cumulative_update','eta_s','all_losses_finite','actual_position_update')}
    print(json.dumps(output,ensure_ascii=False,allow_nan=False))


if __name__ == '__main__':
    main()
