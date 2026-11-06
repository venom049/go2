from pathlib import Path
import torch,json,subprocess,hashlib
r=Path('/root/gpufree-data/moe1000-sim2real-20261105')
j=json.loads((r/'server_smoke/job.json').read_text())
assert j['status']=='trained' and j['updates']==40
initial=torch.load(r/'source/validation/moe_cts_actor_conversion_20261104/actor.pt',map_location='cpu',weights_only=True)
final=torch.load(r/'server_smoke/model_40.pt',map_location='cpu',weights_only=True)
warm=torch.load(r/'server_smoke/model_1.pt',map_location='cpu',weights_only=True)
assert final['critic_state_dict']['0.weight'].shape==(512,370)
enc=[k for k in final['actor_state_dict'] if k.startswith('student_moe_encoder.')]
assert all(torch.equal(final['actor_state_dict'][k],warm['actor_state_dict'][k]) for k in enc)
assert any(not torch.equal(final['actor_state_dict'][k],warm['actor_state_dict'][k]) for k in final['actor_state_dict'] if k.startswith('actor.'))
records=[json.loads(s) for s in (r/'server_smoke/progress.jsonl').read_text().splitlines()]
assert all(s['all_losses_finite'] for s in records)
assert sum(s['target_commands_update']['sent']['excess_count'] for s in records)==0
assert not (r/'formal_run').exists()
assert Path('/root/gpufree-data/moe1000-sdk-target-20261105/formal_run/compatible_export/policy.onnx').exists()
with (r/'pipeline.log').open('x') as log:
    p=subprocess.Popen(['/root/go2-himloco/.venv-moe-finetune/bin/python',str(r/'run_pipeline.py')],cwd=r,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(r/'launch.json').write_text(json.dumps({'pid':p.pid,'server_smoke_updates':40,'passed':True,'frozen_encoder_tensors':len(enc)},indent=2)+'\n')
print(json.dumps({'launched':True,'pid':p.pid,'server_smoke':'passed'}))
