from pathlib import Path
import time,json,subprocess,sys,os,traceback
ROOT=Path(__file__).resolve().parent
RUN=ROOT/'formal_run'
def write(j):
    p=ROOT/'delivery_job.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(j,indent=2)+'\n');q.replace(p)
def run(args,log):
    with (ROOT/log).open('x') as f:subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
def main():
    j={'pid':os.getpid(),'status':'waiting_for_training_and_main_evaluation','started_unix':time.time()};write(j)
    try:
        while True:
            p=RUN/'job.json'
            state=json.loads(p.read_text()) if p.exists() else {}
            if state.get('status')=='completed':break
            if state.get('status')=='failed':raise RuntimeError(state.get('error','main pipeline failed'))
            time.sleep(10)
        j['status']='evaluating_light';write(j)
        for label,model in [('previous',Path('/root/gpufree-data/moe1000-sdk-target-20261105/formal_run/compatible_export/policy.onnx')),
                            ('new',RUN/'exported/policy.onnx')]:
            j['phase']=label;write(j)
            run(['evaluate_transfer.py','--model',model,'--out',RUN/f'evaluations/{label}_light',
                 '--level','.15','--count','64','--workers','8'],f'{label}_light.log')
        j['status']='packaging';write(j)
        run(['package_results.py'],'packaging.log')
        j.update(status='completed',completed_unix=time.time());write(j)
    except BaseException as exc:
        j.update(status='failed',error=repr(exc),traceback=traceback.format_exc());write(j);raise
if __name__=='__main__':main()
