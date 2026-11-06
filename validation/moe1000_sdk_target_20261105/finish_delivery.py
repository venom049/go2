"""Finish original-interface export and ONNX evaluation after the fixed run."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'formal_run'
state=dict(status='waiting_for_training',pid=os.getpid(),started_unix=time.time())

def save():
    tmp=ROOT/'delivery_job.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(ROOT/'delivery_job.json')

def execute(args,log):
    with (ROOT/log).open('x') as stream:
        subprocess.run([sys.executable,*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)

save()
try:
    while True:
        p=json.loads((RUN/'job.json').read_text())
        t=json.loads((RUN/'training/job.json').read_text())
        if p['status']=='failed' or t['status']=='failed':raise RuntimeError('Fixed training pipeline failed')
        if t['status']=='trained':break
        time.sleep(10)
    state['status']='exporting_compatible_policy';save()
    execute(['export_safe_action_policy.py','--checkpoint',str(RUN/'training/model_1000.pt'),
        '--out',str(RUN/'compatible_export')],'compatible_export.log')
    state['status']='evaluating_final_onnx';save()
    execute(['evaluate_safe_action_policy.py','--model',str(RUN/'compatible_export/policy.onnx'),
        '--out',str(RUN/'onnx_evaluation'),'--episodes-per-seed','64'],'onnx_evaluation.log')
    summary=json.loads((RUN/'onnx_evaluation/summary.json').read_text())
    assert summary['episodes']==129
    assert summary['target_commands']['raw']['excess_count']==0
    assert summary['target_commands']['sent']['excess_count']==0
    state.update(status='completed',completed_unix=time.time(),episodes=129,
        successes=summary['successes'],target_excess_count=0,
        model_sha256=summary['model_sha256'],final_onnx=str(RUN/'compatible_export/policy.onnx'))
    save()
except BaseException as e:
    state.update(status='failed',error=repr(e),traceback=traceback.format_exc());save();raise
