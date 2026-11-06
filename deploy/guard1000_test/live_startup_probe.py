"""Read-only counterfactual startup check; never sends LowCmd or changes mode."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-mode', required=True)
    parser.add_argument('--stairs-speed', type=float, choices=[0.45, 0.6], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Report exists; refusing overwrite')
    sys.path.append('/home/unitree/go2-stairs-handoff-20261103/.venv/lib/python3.9/site-packages')
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, LowCmd_
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
    model = Path('/home/unitree/go2-stairs-guard1000-20261104/legacy_candidate/v0/exported/policy.onnx')
    model_hash = hashlib.sha256(model.read_bytes()).hexdigest()
    if model_hash != '8eb200bf9ae5e72428d5d0b9d0d7977bdae569b39c32b57bc7a5ad9c15942dc3':
        raise RuntimeError('Unexpected model hash')
    config = model.parent.parent / 'params/deploy.yaml'
    config_hash = hashlib.sha256(config.read_bytes()).hexdigest()
    if config_hash != 'c3817799231b7a0b9e0148e4a72a943ce111c248ecd9c31388fbdc89e44a73bf':
        raise RuntimeError('Unexpected deployment configuration hash')
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model), options, providers=['CPUExecutionProvider'])
    offset = np.array([0,.8,-1.5,0,.8,-1.5,0,1.,-1.5,0,1.,-1.5], np.float32)
    limits = np.array([23.7,23.7,45.43]*4, np.float32)
    lower = np.array([-1.0472,-1.5708,-2.7227]*2+[-1.0472,-.5236,-2.7227]*2)
    upper = np.array([1.0472,3.4907,-.83776]*2+[1.0472,4.5379,-.83776]*2)
    names = [leg+'_'+joint for leg in ['FR','FL','RR','RL'] for joint in ['hip','thigh','calf']]
    state, command, lock = {}, {}, threading.Lock()
    def receive_state(m):
        with lock:
            state.update(tick=m.tick, q=[x.q for x in m.motor_state[:12]],
                         dq=[x.dq for x in m.motor_state[:12]], quat=list(m.imu_state.quaternion),
                         rpy=list(m.imu_state.rpy), gyro=list(m.imu_state.gyroscope), received=time.monotonic())
    def receive_command(m):
        with lock:
            command.update(q=[x.q for x in m.motor_cmd[:12]], kp=[x.kp for x in m.motor_cmd[:12]],
                           kd=[x.kd for x in m.motor_cmd[:12]], received=time.monotonic())
    ChannelFactoryInitialize(0, 'eth1')
    subscriber = ChannelSubscriber('rt/lowstate', LowState_); subscriber.Init(receive_state, 1)
    cmd_subscriber = ChannelSubscriber('rt/lowcmd', LowCmd_); cmd_subscriber.Init(receive_command, 1)
    client = MotionSwitcherClient(); client.SetTimeout(1.0); client.Init()
    before = client.CheckMode()
    if before[0] != 0 or before[1].get('name') != args.expected_mode:
        raise RuntimeError('Unexpected mode before read-only probe: '+str(before))
    time.sleep(.3)
    rows, history, previous_tick = [], None, None
    started = time.monotonic()
    try:
        for step in range(50):
            with lock:
                s, c = dict(state), dict(command)
            now = time.monotonic()
            if not s or not c or max(now-s['received'], now-c['received']) > .1:
                raise RuntimeError('Missing/stale telemetry or observed LowCmd')
            if previous_tick is not None and not 0 < (s['tick']-previous_tick) % 2**32 < 2**31:
                raise RuntimeError('State tick did not advance')
            previous_tick = s['tick']
            q, dq, quat = [np.asarray(s[k], np.float32) for k in ('q','dq','quat')]
            gyro, previous_target = np.asarray(s['gyro'], np.float32), np.asarray(c['q'], np.float32)
            all_values = np.concatenate([q,dq,quat,gyro,previous_target,c['kp'],c['kd']])
            if not np.isfinite(all_values).all() or abs(float(np.linalg.norm(quat))-1.) > .01:
                raise RuntimeError('Invalid telemetry')
            if np.any(q<lower) or np.any(q>upper) or np.any(previous_target<lower) or np.any(previous_target>upper):
                raise RuntimeError('Measured pose or observed target outside guard bounds')
            if np.any(np.asarray(c['kp'])<=0):
                raise RuntimeError('Observed controller is not holding a position')
            down = np.array([0,0,-1],np.float32)
            cross = np.cross(quat[1:],down)
            gravity = down+2*(-quat[0]*cross+np.cross(quat[1:],cross))
            applied = np.concatenate([(previous_target-offset)/.25,np.zeros(4,np.float32)])
            frame = np.concatenate([np.array([args.stairs_speed,0,0],np.float32),gyro*.25,gravity,q-offset,dq*.05,applied])
            history = np.tile(frame,(6,1)) if history is None else np.concatenate([history[1:],frame[None]],axis=0)
            output = session.run(None,{'obs':history.reshape(1,294)})[0][0]
            if output.shape!=(16,) or not np.isfinite(output).all() or np.any(output[12:]!=0):
                raise RuntimeError('Invalid model output')
            target = offset+.25*np.clip(output[:12],-4.5,4.5)
            effort = 40*(target-q)-dq
            ratios = np.abs(effort)/limits
            outside = bool(np.any(target<lower) or np.any(target>upper))
            tracking_error = float(np.max(np.abs(target-q)))
            tilt = float(np.arccos(np.clip(gravity[2]*-1.,-1,1)))
            settled = bool(tilt<=.25 and np.max(np.abs(gyro))<=.35 and np.max(np.abs(dq))<=.8 and np.max(np.abs(q-previous_target))<=.18)
            rows.append({'step':step,'tick':s['tick'],'q':q.tolist(),'dq':dq.tolist(),'rpy':s['rpy'],
                         'observed_target':previous_target.tolist(),'proposed_target':target.tolist(),
                         'estimated_effort_nm':effort.tolist(),'max_effort_ratio':float(ratios.max()),
                         'peak_joint':names[int(ratios.argmax())],'target_outside_bounds':outside,
                         'settled_pose':settled,'overlimit':bool(np.any(ratios>1)),
                         'maximum_proposed_tracking_error_rad':tracking_error,
                         'tracking_limit_exceeded':tracking_error>0.9})
            time.sleep(max(0,started+(step+1)*.02-time.monotonic()))
        after = client.CheckMode()
        passed = (after[0]==0 and after[1].get('name')==args.expected_mode and
                  all(not r['overlimit'] and not r['target_outside_bounds'] and
                      not r['tracking_limit_exceeded'] and r['settled_pose'] for r in rows))
        result = {'read_only':True,'motor_commands_sent':False,'model_sha256':model_hash,'config_sha256':config_hash,
                  'command_speed_mps':args.stairs_speed,
                  'mode_before':before,'mode_after':after,'frames':len(rows),'passed':passed,
                  'overlimit_frames':sum(r['overlimit'] for r in rows),
                  'maximum_requested_ratio':max(r['max_effort_ratio'] for r in rows),
                  'scope':'Live held-pose counterfactual proposals only, not closed-loop execution or measured torque.',
                  'history':'Observed LowCmd position targets; first frame repeated six times. Never feed proposals back.',
                  'rows':rows}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as stream: json.dump(result,stream,indent=2)
        print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
        return 0 if passed else 3
    finally:
        subscriber.Close();cmd_subscriber.Close()


if __name__=='__main__':
    raise SystemExit(main())
