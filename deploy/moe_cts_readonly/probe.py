"""Bounded ONNX parity and counterfactual first-frame checks; no motor writer.

The live phase only subscribes to LowState and calls CheckMode. Each measured
snapshot is treated as a fresh policy initialization: repeated current history,
zero prior action. Predicted actions are never applied to the robot. This is
neither a closed-loop test nor a motion controller.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import numpy as np
import onnxruntime as ort

TERMS = ('base_ang_vel', 'projected_gravity', 'velocity_commands',
         'joint_pos_rel', 'joint_vel_rel', 'last_action')
MAP = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])


def initial_observation(sample, cfg, vx):
    q = np.array([m['q_rad'] for m in sample['motors']])[MAP]
    dq = np.array([m['dq_rad_s'] for m in sample['motors']])[MAP]
    quat = np.array(sample['imu']['quaternion_wxyz'])
    gyro = np.array(sample['imu']['gyroscope_rad_s'])
    values = np.concatenate((q, dq, quat, gyro))
    if not np.isfinite(values).all() or abs(np.linalg.norm(quat)-1.) > .01:
        raise ValueError('Invalid telemetry')
    down = np.array([0., 0., -1.])
    cross = np.cross(quat[1:], down)
    gravity = down + 2*(-quat[0]*cross + np.cross(quat[1:], cross))
    terms = (gyro, gravity, np.array([vx, 0., 0.]),
             q-np.array(cfg['default_joint_pos']), dq, np.zeros(12))
    chunks = []
    for name, value in zip(TERMS, terms):
        c = cfg['observations'][name]
        if c['history_length'] != 5:
            raise ValueError('Expected five frames per term')
        chunks.append(np.tile(np.clip(value, *c['clip'])*np.array(c['scale']), 5))
    obs = np.concatenate(chunks).astype(np.float32)[None, :]
    if obs.shape != (1, 225) or not np.isfinite(obs).all():
        raise ValueError('Invalid observation')
    return obs, q, dq


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--interface', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite a probe report')
    bundle = args.bundle.resolve()
    manifest = json.loads((bundle/'manifest.json').read_text())
    for filename, expected in manifest['files'].items():
        if hashlib.sha256((bundle/filename).read_bytes()).hexdigest() != expected:
            raise ValueError('Bundle hash mismatch: '+filename)
    cfg = json.loads((bundle/'params/deploy.json').read_text())
    if cfg['joint_ids_map'] != MAP.tolist() or tuple(cfg['observations']) != TERMS:
        raise ValueError('Unexpected policy contract')
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(bundle/'exported/policy.onnx'), options,
                                  providers=['CPUExecutionProvider'])
    if [(v.name, v.shape) for v in session.get_inputs()] != [('obs', [1, 225])]:
        raise ValueError('Unexpected ONNX input')
    fixture = np.load(bundle/'parity_fixtures.npz', allow_pickle=False)
    parity_errors = {}
    for name, obs, expected in zip(fixture['names'], fixture['obs'], fixture['actions']):
        actual = session.run(['actions'], {'obs': obs[None, :]})[0][0]
        family = 'boundary' if str(name).startswith(('clip_', 'boundary_')) else 'ordinary'
        tolerance = 2e-4+2e-4*np.abs(expected)
        if family == 'boundary':
            tolerance += 1e-5*np.abs(expected).max()
        error = np.abs(actual.astype(float)-expected.astype(float))
        if not np.isfinite(actual).all() or np.any(error > tolerance):
            raise AssertionError('ARM ONNX parity failed: '+str(name))
        parity_errors[family] = max(parity_errors.get(family, 0.), float(error.max()))
    print(json.dumps({'event': 'arm_parity_passed', 'cases': len(fixture['obs']),
                      'max_absolute_action_error': parity_errors}), flush=True)
    sys.path.insert(0, str(bundle))
    from read_state import StateSummary, open_subscription
    from official_guard import load_official_profile, evaluate_guard
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
    state = StateSummary()
    subscriber = open_subscription(args.interface, state.receive)
    mode = MotionSwitcherClient(); mode.SetTimeout(1.); mode.Init()
    before = mode.CheckMode()
    if before[0] != 0:
        raise RuntimeError('Unable to read current mode')
    profile = load_official_profile(bundle/'limit_profile.json')
    deadline = time.monotonic()+5.
    while state.count < 20 and time.monotonic() < deadline:
        time.sleep(.02)
    rows = []
    action_cfg = cfg['actions']['JointPositionAction']
    last_tick = None
    for index in range(100):
        snap = state.report(interface=args.interface, max_age=.1)
        sample = snap['latest']
        if not snap['fresh_receipt'] or sample is None or not sample['checks']['policy_input_fields_finite']:
            raise RuntimeError('Missing, stale or invalid telemetry')
        tick = sample['tick_raw']
        if last_tick is not None and not 0 < (tick-last_tick) % 2**32 < 2**31:
            raise RuntimeError('Telemetry tick did not advance')
        last_tick = tick
        for vx in (0., .6):
            obs, q, dq = initial_observation(sample, cfg, vx)
            t0 = time.monotonic()
            action = session.run(['actions'], {'obs': obs})[0][0]
            latency = 1000*(time.monotonic()-t0)
            if action.shape != (12,) or not np.isfinite(action).all():
                raise FloatingPointError('Invalid action')
            clip = np.array(action_cfg['clip'])
            target = np.clip(action*np.array(action_cfg['scale'])+np.array(action_cfg['offset']),
                             clip[:, 0], clip[:, 1])
            audit = evaluate_guard(profile, q, dq, target, cfg['stiffness'], cfg['damping'])
            metrics = {k: {'excess_count': int(audit.exceeded[k].sum()),
                           'peak_abs': float(np.abs(audit.values[k]).max()),
                           'max_ratio': float(audit.ratios[k].max()),
                           'max_excess': float(audit.excess[k].max())}
                       for k in ('velocity', 'position', 'target_position', 'tracking', 'requested_torque')}
            rows.append({'sample': index, 'tick': tick, 'counterfactual_vx_m_s': vx,
                         'inference_ms': latency, 'metrics': metrics,
                         'any_observed_violation': bool(audit.hard_violation),
                         'measured_q': q.tolist(), 'measured_dq': dq.tolist(),
                         'proposed_target': target.tolist()})
        time.sleep(.02)
    after = mode.CheckMode()
    report = {'read_only': True, 'motor_commands_sent': False, 'mode_release_called': False,
              'policy': manifest['model'], 'onnx_sha256': manifest['onnx_sha256'],
              'arm_parity_passed': True, 'parity_cases': len(fixture['obs']),
              'parity_max_errors': parity_errors, 'mode_before': before, 'mode_after': after,
              'mode_unchanged': before == after, 'telemetry': state.report(interface=args.interface, max_age=.1),
              'rows': rows, 'scope': __doc__,
              'not_closed_loop_validation': True, 'actuation_ready': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({'event': 'readonly_probe_finished', 'rows': len(rows),
                      'rows_with_guard_violation': sum(r['any_observed_violation'] for r in rows),
                      'max_inference_ms': max(r['inference_ms'] for r in rows),
                      'mode_before': before, 'mode_after': after,
                      'motor_commands_sent': False}), flush=True)


if __name__ == '__main__':
    main()
