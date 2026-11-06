#!/usr/bin/env python3
"""Offline held-stance torque probes; no networking or motor commands.

These synthetic static poses are regression cases, not reconstructed real
robot telemetry. A pass does not authorize deployment or replace simulation.
"""
import argparse
import json
from pathlib import Path
import _bootstrap
import torch
from deploy.go2_stairs.policy import (load_checkpoint, build_observation,
                                    DEFAULT_JOINT_POSITION, JOINT_NAMES)
from go2_him.torque import joint_effort_limits
from go2_him.settings import load_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-pass', action='store_true')
    args = parser.parse_args()
    settings = load_settings(args.checkpoint.parent / 'settings.yaml')
    control = settings['control']
    net, metadata = load_checkpoint(args.checkpoint)
    default = torch.tensor(DEFAULT_JOINT_POSITION).reshape(1,12)
    legacy = default.clone(); legacy[:,::3] = 0
    limits = joint_effort_limits(JOINT_NAMES)
    cases = []
    with torch.inference_mode():
        for label, pose in [('native_default', default), ('legacy_stand_hips_zero', legacy)]:
            for speed in (0., .3, .6):
                # A held position target is fed back as the previous action.
                previous = (pose-default)/control['action_scale']
                obs = build_observation(torch.tensor([[speed,0.,0.]]), torch.zeros(1,3),
                    torch.tensor([[1.,0.,0.,0.]]), pose, torch.zeros(1,12), previous)
                raw = net(obs.repeat(1,6))
                target = default + raw.clamp(-control['action_clip'],control['action_clip'])*control['action_scale']
                requested = control['stiffness']*(target-pose)
                ratio = requested.abs()/limits
                cases.append({'pose':label,'command_m_s':speed,'requested_peak_ratio':float(ratio.max()),
                    'passed':bool((ratio<=1).all()),
                    'requested_nm':dict(zip(JOINT_NAMES,requested[0].tolist()))})
    report = {'checkpoint':str(args.checkpoint), 'checkpoint_metadata':metadata,
              'scope':'Synthetic static level stance, zero joint/gyro velocity, repeated 6-frame history, one proposed action; not recorded physical telemetry or a dynamics rollout',
              'all_cases_passed':all(c['passed'] for c in cases),'cases':cases}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'all_cases_passed':report['all_cases_passed'],
                      'maximum_requested_ratio':max(c['requested_peak_ratio'] for c in cases),
                      'output':str(args.output)}))
    if args.require_pass and not report['all_cases_passed']: raise SystemExit(2)


if __name__ == '__main__': main()
