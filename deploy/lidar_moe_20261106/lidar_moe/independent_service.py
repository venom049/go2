"""Own only sensor processes; any child exit stops the complete sensor group."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--lio-workspace',type=Path);ap.add_argument('--seconds',type=float,default=60.)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--config',type=Path)
    ap.add_argument('--send-local',action='store_true');ap.add_argument('--no-supervisor',action='store_true');a=ap.parse_args()
    workspace=a.lio_workspace or a.root/'independent_lio';bin_dir=workspace/'install/moe_fast_lio/lib/moe_fast_lio'
    cfg=json.loads((a.root/'config/independent_lio.json').read_text())
    mapping=[sys.executable,'-m','lidar_moe.attitude_lio','--root',str(a.root)] if cfg.get('odometry_algorithm')=='body_attitude_translation_registration' else [str(bin_dir/'moe_fastlio_mapping'),'--ros-args','--params-file',str(a.root/'config/fastlio_l1.yaml'),
                          '-r','/cloud_registered:=/moe_lio/internal/cloud_registered','-r','/Odometry:=/moe_lio/internal/imu_odom',
                          '-r','/tf:=/moe_lio/internal/tf']
    commands=[('imu',[str(bin_dir/'moe_body_imu_bridge')]),
              ('mapping',mapping),
              ('adapter',[sys.executable,'-m','lidar_moe.lio_adapter','--root',str(a.root)])]
    if not a.no_supervisor:
        cmd=[sys.executable,'-m','lidar_moe.ros_bridge','--root',str(a.root),'--seconds',str(a.seconds),'--output',str(a.output)]
        if a.send_local:cmd+=['--send-local']
        if a.config:cmd+=['--config',str(a.config)]
        commands.append(('supervisor',cmd))
    children=[];files=[];running=True;code=0;a.output.parent.mkdir(parents=True,exist_ok=True)
    def stop(*_):
        nonlocal running
        running=False
    for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(s,stop)
    try:
        for name,command in commands:
            stream=a.output.with_name(a.output.stem+'.'+name+'.log').open('x');files.append(stream)
            child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);children.append((name,child))
            print(json.dumps({'event':'sensor_child_started','name':name,'pid':child.pid,'motor_publisher':False}),flush=True)
        end=time.monotonic()+a.seconds
        while running and time.monotonic()<end:
            failed=[(name,p.returncode) for name,p in children if p.poll() is not None]
            if failed:
                print(json.dumps({'event':'sensor_child_exited','children':failed}),flush=True);code=1;break
            time.sleep(.05)
    finally:
        for _,p in reversed(children):
            if p.poll() is None:os.killpg(p.pid,signal.SIGINT)
        deadline=time.monotonic()+3
        for _,p in reversed(children):
            try:p.wait(timeout=max(.01,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL);p.wait()
        for f in files:f.close()
        print(json.dumps({'event':'sensor_group_stopped','children':[(n,p.returncode) for n,p in children]}),flush=True)
    return code


if __name__=='__main__':raise SystemExit(main())
