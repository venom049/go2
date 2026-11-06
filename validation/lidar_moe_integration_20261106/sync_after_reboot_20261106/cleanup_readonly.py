"""Stop only this deployment's hung sensor-only process tree; no motor process."""
from pathlib import Path
import json,os,signal,sys
root=b'/home/unitree/go2-lidar-moe-20261106';rows={};roots=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  args=(p/'cmdline').read_bytes().split(b'\0');status=(p/'status').read_text();ppid=int(next(s.split(':',1)[1] for s in status.splitlines() if s.startswith('PPid:')))
  rows[int(p.name)]={'ppid':ppid,'args':[a.decode(errors='replace') for a in args if a],'wchan':(p/'wchan').read_text().strip()}
  if b'lidar_moe.ros_bridge' in args and root in args:roots.append(int(p.name))
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
known=set(roots)
while True:
 children={pid for pid,r in rows.items() if r['ppid'] in known};new=known|children
 if new==known:break
 known=new
report={'reason':'sensor-only ProcessPoolExecutor Python 3.8 shutdown(wait=False) handle-closed hang','roots':roots,'processes':{str(pid):rows[pid] for pid in known}}
for pid in known:
 if rows[pid]['args'] and rows[pid]['args'][0].endswith('go2_ctrl'):raise RuntimeError('Unexpected motor-controller process in cleanup tree')
Path(sys.argv[1]).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
for pid in known:
 try:os.kill(pid,signal.SIGKILL)
 except ProcessLookupError:pass
