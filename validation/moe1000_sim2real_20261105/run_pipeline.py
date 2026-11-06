"""Detached, fail-fast training/export/matched-evaluation pipeline."""
from pathlib import Path
import os,sys,json,subprocess,time,hashlib,traceback
ROOT=Path(__file__).resolve().parent
RUN=ROOT/'formal_run';PY=sys.executable
PREVIOUS=Path('/root/gpufree-data/moe1000-sdk-target-20261105/formal_run/compatible_export/policy.onnx')
def atomic(path,value):
    t=path.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(path)
def run(cmd,log):
    with log.open('x') as f:subprocess.run([PY,*map(str,cmd)],stdout=f,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
def main():
    RUN.mkdir(exist_ok=False)
    job=dict(status='training',pid=os.getpid(),started_unix=time.time(),max_updates=1000,
        robot='Go2 EDU without payload; official generic Go2 MJCF ae6a8403',real_robot_driven=False)
    atomic(RUN/'job.json',job)
    try:
        run(['source/scripts/train_moe_stairs.py','--actor',ROOT/'source/validation/moe_cts_actor_conversion_20261104/actor.pt',
            '--out',RUN/'training','--updates','1000','--num-envs','128','--workers','32',
            '--position-constraint','native','--target-mode','bounded','--domain-randomization',
            '--seed','20261104','--device','cuda:0'],RUN/'training.log')
        job.update(status='exporting');atomic(RUN/'job.json',job)
        run(['export_safe_action_policy.py','--checkpoint',RUN/'training/model_1000.pt','--out',RUN/'exported'],RUN/'export.log')
        job.update(status='evaluating');atomic(RUN/'job.json',job)
        results=[]
        for label,model in [('previous',PREVIOUS),('new',RUN/'exported/policy.onnx')]:
            for name,level in [('nominal',None),('moderate',.5),('full',1.)]:
                phase=f'{label}_{name}';job['phase']=phase;atomic(RUN/'job.json',job)
                cmd=['evaluate_transfer.py','--model',model,'--out',RUN/'evaluations'/phase,'--count','64','--workers','8']
                if level is not None:cmd+=['--level',str(level)]
                run(cmd,RUN/f'{phase}.log')
                result=json.loads((RUN/'evaluations'/phase/'summary.json').read_text());results.append(result)
                atomic(RUN/'comparison.json',results)
        job['phase']='new_strict';atomic(RUN/'job.json',job)
        run(['evaluate_transfer.py','--model',RUN/'exported/policy.onnx','--out',RUN/'evaluations/new_strict',
            '--count','64','--workers','8','--guard','strict'],RUN/'new_strict.log')
        records=[json.loads(s) for s in (RUN/'training/progress.jsonl').read_text().splitlines()]
        assert len(records)==1000 and records[-1]['update']==1000
        assert all(r['all_losses_finite'] for r in records)
        assert sum(r['target_commands_update']['sent']['excess_count'] for r in records)==0
        job.update(status='completed',completed_unix=time.time(),completion_verified=True)
        atomic(RUN/'job.json',job)
    except BaseException as exc:
        job.update(status='failed',error=repr(exc),traceback=traceback.format_exc(),ended_unix=time.time())
        atomic(RUN/'job.json',job);raise
if __name__=='__main__':main()
