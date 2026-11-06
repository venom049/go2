"""Bounded sensor-process recovery; never owns a DDS motor publisher.

After a failure, wait for the command watchdog, then query the controller's
actual state. Only PRE_HANDOFF / POST_HANDOFF_HOLD permits a restart. Missing,
active-motion or stopped replies prohibit restart. No controller is restarted.
"""
import argparse,json,os,signal,socket,stat,subprocess,sys,time
from pathlib import Path
from .transport import QUERY,feedback

def query_controller(root,timeout=.25):
    path=root/'run'/('health-'+str(os.getpid())+'.sock');sock=socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM)
    try:
        sock.bind(str(path));os.chmod(path,0o600);sock.settimeout(timeout)
        sock.sendto(QUERY.pack(b'G2Q2',2),str(root/'run/perception.sock'))
        packet=sock.recv(128);return feedback(packet,time.monotonic())
    except (OSError,ValueError):return None
    finally:
        sock.close()
        try:path.unlink()
        except FileNotFoundError:pass

def may_restart(reply):return reply is not None and reply[0] in (0,3)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--seconds',type=float,default=3600)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path);a=p.parse_args()
    running=True;child=None
    def stop(*_):
        nonlocal running
        running=False
        if child and child.poll() is None:child.terminate()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGHUP,stop)
    audit=a.output.with_name('sensor_service.jsonl');audit.parent.mkdir(parents=True,exist_ok=True)
    def record(event,**values):
        with audit.open('a') as f:f.write(json.dumps({'event':event,'mono_s':time.monotonic(),**values})+'\n')
    deadline=time.monotonic()+a.seconds;restart=0
    try:
        while running and time.monotonic()<deadline:
            output=a.output if restart==0 else a.output.with_name(a.output.stem+'.restart'+str(restart)+a.output.suffix)
            if output.exists():raise RuntimeError('Refusing to overwrite sensor evidence')
            cfg=json.loads((a.config or a.root/'config/supervisor.json').read_text())
            module='lidar_moe.independent_service' if cfg.get('backend') in ('raw_lidar_body_imu','raw_lidar_body_imu_fastlio') else 'lidar_moe.ros_bridge'
            cmd=[sys.executable,'-m',module,'--root',str(a.root),'--send-local','--seconds',str(max(1.,deadline-time.monotonic())),'--output',str(output)]
            if a.config:cmd+=['--config',str(a.config)]
            child=subprocess.Popen(cmd,start_new_session=True);record('sensor_started',pid=child.pid,restart=restart,output=str(output))
            while running and child.poll() is None:time.sleep(.05)
            if not running:break
            rc=child.wait();record('sensor_exited',code=rc)
            if rc==0 or restart>=3:break
            # In motion, native expiry must occur before any recovery query.
            until=time.monotonic()+.35
            while running and time.monotonic()<until:time.sleep(.02)
            if not running:break
            reply=query_controller(a.root);record('restart_decision',reply=reply,permitted=may_restart(reply))
            if not may_restart(reply):break
            client=a.root/'run/perception_client.sock'
            if client.exists():
                info=client.lstat()
                if not stat.S_ISSOCK(info.st_mode) or info.st_uid!=os.getuid():raise RuntimeError('Foreign client path; recovery refused')
                client.unlink() # child has exited; exclusive launcher owns path.
            restart+=1
        return 0 if not running or (child and child.returncode==0) else 1
    finally:
        if child and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait()
        record('service_stopped',restarts=restart)

if __name__=='__main__':raise SystemExit(main())
