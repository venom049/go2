"""One-shot MotionSwitcher bridge. No LowCmd publisher; default is check-only.

The explicit --allow-release mode permits exactly one release RPC, with no
automatic retry, stand command, mode restoration or service modification.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--allow-release',action='store_true')
    args=p.parse_args()
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
    ChannelFactoryInitialize(0,'eth0')
    client=MotionSwitcherClient();client.SetTimeout(1.0);client.Init()
    code,data=client.CheckMode()
    print(json.dumps({'event':'bridge_check','code':code,'data':data,'read_only':True}),flush=True)
    if code!=0: return 2
    if not args.allow_release: return 0
    path=Path('/home/unitree/go2-stairs-handoff-20261103/validation/motion_rpc.sock')
    # An existing socket is never silently replaced: avoid duplicate bridges.
    if path.exists(): raise RuntimeError('RPC socket already exists')
    used=False
    server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    os.umask(0o077);server.bind(str(path));server.listen(1);server.settimeout(1.)
    print(json.dumps({'event':'bridge_ready','socket':str(path),'max_releases':1}),flush=True)
    deadline=time.monotonic()+180
    try:
        while time.monotonic()<deadline:
            try: conn,_=server.accept()
            except socket.timeout: continue
            with conn:
                conn.settimeout(1.)
                op=conn.recv(64).decode().strip()
                result={'ok':False,'operation':op,'release_attempted':False}
                if op=='check':
                    rc,mode=client.CheckMode()
                    result.update(code=rc,mode=mode,ok=rc==0 and mode.get('name')=='mcf')
                elif op=='release' and not used:
                    # Recheck before changing control. Never retry an uncertain release.
                    rc,mode=client.CheckMode()
                    if rc==0 and mode.get('name')=='mcf':
                        used=True
                        started=time.monotonic()
                        release_code,_=client.ReleaseMode()
                        check_code,after=client.CheckMode()
                        result.update(release_attempted=True,release_code=release_code,
                                      check_code=check_code,mode=after,
                                      elapsed_ms=1000*(time.monotonic()-started),
                                      ok=release_code==0 and check_code==0 and after.get('name')=='')
                    else: result.update(code=rc,mode=mode)
                print(json.dumps(result),flush=True)
                conn.sendall((json.dumps(result)+'\n').encode())
    finally:
        server.close();path.unlink(missing_ok=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
