#!/usr/bin/env python3
"""Fetch status over the temporary SSH connection; track reported milestones locally."""
import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / '.monitor/report_state.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ack', type=int, help='Record the highest milestone reported to the user; no network access')
    parser.add_argument('--ack-event', choices=['completed','stopped','stalled','unreachable'], help='Record an event already reported')
    args = parser.parse_args()
    STATE.parent.mkdir(exist_ok=True)
    state = json.loads(STATE.read_text()) if STATE.exists() else {'last_reported_milestone':0, 'last_reported_event':None}
    if args.ack is not None or args.ack_event:
        if args.ack is not None:
            if args.ack not in range(500,3001,500): parser.error('Milestone must be 500,1000,...,3000')
            state['last_reported_milestone'] = max(state['last_reported_milestone'],args.ack)
        if args.ack_event: state['last_reported_event'] = args.ack_event
        STATE.write_text(json.dumps(state,indent=2))
        print(json.dumps(state)); return
    command = ['ssh','-S',str(ROOT/'.monitor/control.sock'),'-o','BatchMode=yes',
               '-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=/tmp/go2_server_known_hosts',
               '-o','ConnectTimeout=10','-p','31910','root@183.147.142.40',
               '/opt/conda/envs/isaaclab/bin/python -']
    try:
        result = subprocess.run(command,input=(ROOT/'scripts/monitor_remote.py').read_text(),
                                capture_output=True,text=True,timeout=40)
        if result.returncode: raise RuntimeError(result.stderr.strip())
        data = json.loads(result.stdout)
        data['pending_milestones'] = [x for x in data.pop('milestones') if x['completed_updates'] > state['last_reported_milestone']]
        data['event_to_report'] = data['status'] if data['status'] != 'running' and data['status'] != state['last_reported_event'] else None
        if data['status'] == 'running' and state['last_reported_event'] in ('unreachable','stalled'):
            data['connection_recovered'] = True
            state['last_reported_event'] = None
    except (subprocess.TimeoutExpired,RuntimeError,ValueError) as exc:
        data = {'status':'unreachable','detail':str(exc),'pending_milestones':[],
                'event_to_report':'unreachable' if state['last_reported_event'] != 'unreachable' else None}
    data['checked_at_utc'] = datetime.now(timezone.utc).isoformat()
    state['last_check'] = data['checked_at_utc']
    STATE.write_text(json.dumps(state,indent=2))
    (ROOT/'.monitor/latest_snapshot.json').write_text(json.dumps(data,indent=2,ensure_ascii=False))
    print(json.dumps(data,indent=2,ensure_ascii=False))


if __name__ == '__main__': main()
