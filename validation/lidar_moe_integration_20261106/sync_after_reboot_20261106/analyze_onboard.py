"""Read-only synchronization acceptance collector; never initializes DDS."""
import hashlib,json,re,sys,time
from pathlib import Path
import numpy as np
stage=Path(sys.argv[1]);root=Path('/home/unitree/go2-lidar-moe-20261106')
launch=(stage/'readonly_launch.log').read_text()
match=re.search(r'^Logs: (.+)$',launch,re.M)
run=Path(match.group(1)) if match else None
result=json.loads((stage/'install_result.json').read_text())
result.update({'analysis_epoch_s':time.time(),'run_directory':str(run) if run else None,'motor_publisher_created':False,'release_mode_called':False})
result['manifest_sha256']=hashlib.sha256((root/'SHA256SUMS').read_bytes()).hexdigest()
result['binary_sha256']=hashlib.sha256((root/'controller/robots/go2/build/go2_ctrl').read_bytes()).hexdigest()
result['binary_manifest']=(root/'BINARY_SHA256SUMS').read_text().strip()
failures=[]
for line in (root/'SHA256SUMS').read_text().splitlines():
 digest,name=line.split('  ',1)
 if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:failures.append(name)
result['source_hash_failures']=failures
if run:
 controller=(run/'controller.log').read_text();sensor=(run/'sensor.log').read_text()
 summary=re.search(r'LIDAR_READONLY (.+)',controller)
 result['cpp_summary']=dict(re.findall(r'(\w+)=([^\s]+)',summary.group(1))) if summary else None
 result['cpp_error_lines']=[line for line in controller.splitlines() if any(s in line for s in ('[error]','[critical]'))]
 result['sensor_error_lines']=[line for line in sensor.splitlines() if any(s in line for s in ('Traceback','Error:'))]
 path=run/'supervisor.jsonl';rows=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
 commands=[r for r in rows if r['event']=='command'];geometries=[r for r in rows if r['event']=='geometry']
 from collections import Counter
 result['sensor']={'commands':len(commands),'duration_s':commands[-1]['mono_s']-commands[0]['mono_s'] if len(commands)>1 else 0,'mode_counts':dict(Counter(r['mode'] for r in commands)),'reason_counts':dict(Counter(r['reason'] for r in commands)),'geometry_count':len(geometries),'valid_geometry_count':sum(r['geometry']['valid'] for r in geometries),'geometry_reason_counts':dict(Counter(r['geometry']['reason'] for r in geometries)),'geometry_compute_quantiles_s':np.quantile([r['duration_s'] for r in geometries],[.5,.95,1]).tolist() if geometries else [],'source_age_quantiles_s':np.quantile([r['source_age_s'] for r in commands if r['fresh']],[.5,.95,1]).tolist() if any(r['fresh'] for r in commands) else [],'nonzero_commands':sum(any(abs(v)>0 for v in r['command']) for r in commands),'last_command':commands[-1] if commands else None}
active=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  args=(proc/'cmdline').read_bytes().split(b'\0')
  if args and (args[0].endswith(b'go2_ctrl') or b'lidar_moe.ros_bridge' in args):active.append({'pid':int(proc.name),'argv':[v.decode(errors='replace') for v in args if v]})
 except (PermissionError,FileNotFoundError):pass
result['remaining_task_processes']=active
result['remaining_sockets']=[p.name for p in (root/'run').glob('*.sock')]
result['actuation_launcher_disabled']='运动入口暂时禁用' in (root/'start.sh').read_text()
(stage/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
