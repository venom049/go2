"""Read-only snapshot of the current server run; executed via SSH stdin."""
import json
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path('/root/go2-himloco')
RUN = ROOT / 'logs/20261102_154245_stairs_baseline'
CONSOLE = ROOT / 'logs/jobs/20261102_154240_stairs_baseline.console.log'
PID = 3474


def parse_iterations(text):
    text = re.sub(r'\x1b\[[0-9;]*m', '', text)
    headings = list(re.finditer(r'Learning iteration (\d+)/(\d+)', text))
    records = []
    for i, match in enumerate(headings):
        block = text[match.end():headings[i + 1].start() if i + 1 < len(headings) else len(text)]
        if 'Total timesteps:' not in block or 'ETA:' not in block:
            continue
        values = dict(re.findall(r'^\s*([^:\n]+):\s*([^\n]+)', block, re.M))
        records.append({'completed_updates':int(match[1]) + 1, 'target_updates':int(match[2]),
                        'metrics':{key.strip():value.strip() for key,value in values.items()}})
    return records


def snapshot():
    text = CONSOLE.read_text(errors='replace')
    records = parse_iterations(text)
    latest = records[-1] if records else {'completed_updates':0, 'target_updates':3000, 'metrics':{}}
    try:
        command = Path(f'/proc/{PID}/cmdline').read_bytes().replace(b'\0', b' ').decode()
        state = Path(f'/proc/{PID}/stat').read_text().split(') ',1)[1].split()[0]
        running = 'scripts/train_lab.py' in command and 'stairs_baseline' in command and state != 'Z'
    except FileNotFoundError:
        running = False
    checkpoints = sorted(RUN.glob('model_*.pt'), key=lambda p:int(p.stem.split('_')[-1]))
    completed_file = RUN / 'completed.json'
    completion = json.loads(completed_file.read_text()) if completed_file.exists() else None
    status = 'completed' if completion and completion.get('iterations',0) >= 3000 else ('running' if running else 'stopped')
    age = time.time() - CONSOLE.stat().st_mtime
    if status == 'running' and age > 300: status = 'stalled'
    errors = [line for line in text.splitlines() if re.search(r'Traceback|NaN/Inf detected|CUDA out of memory|FloatingPointError', line)]
    gpu = subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],
                         capture_output=True,text=True,timeout=15)
    milestones = [record for record in records if record['completed_updates'] % 500 == 0]
    for record in milestones:
        checkpoint = RUN / f"model_{record['completed_updates']}.pt"
        record['checkpoint'] = str(checkpoint)
        record['checkpoint_exists'] = checkpoint.is_file()
    return {'status':status, 'pid':PID, 'run_directory':str(RUN), 'console_log':str(CONSOLE),
        'latest':latest, 'milestones':milestones, 'completion':completion,
        'latest_checkpoint':str(checkpoints[-1]) if checkpoints else None,
        'seconds_since_log_update':round(age,1), 'gpu_util_memory_mib':gpu.stdout.strip(), 'errors':errors[-5:]}


if __name__ == '__main__':
    print(json.dumps(snapshot(),ensure_ascii=False,indent=2))
