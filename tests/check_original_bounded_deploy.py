"""Offline SDK projection, guard and handoff regression for this deployment."""
from pathlib import Path
import json
import subprocess
import tempfile
import sys
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'deploy/original_bounded_20261106'
INCLUDE=BUNDLE/'controller/include'
sys.path.insert(0,str(ROOT/'validation/moe1000_sim2real_20261105/source'))
from go2_him.official_guard import load_official_profile
from go2_him.target_clip import transform_target_action

with tempfile.TemporaryDirectory() as temp:
    temp=Path(temp)
    regression={}
    for name in ('test_first_test_guard','test_handoff_hold','test_moe_stand_readiness'):
        binary=temp/name
        subprocess.run(['g++','-std=c++17','-O2','-I'+str(INCLUDE),str(ROOT/'tests'/f'{name}.cpp'),'-o',str(binary)],check=True)
        regression[name]=subprocess.check_output([str(binary)],text=True).strip()
    oracle=temp/'projection.cpp'
    oracle.write_text('''#include "MoePolicy.h"
#include <iostream>
int main() {
 go2_moe::Action a;
 while(std::cin.read(reinterpret_cast<char*>(a.data()),sizeof(a))) {
  try {
   auto c=go2_moe::decode(a); auto previous=go2_moe::inverse_submitted(c);
   for(size_t i=0;i<12;++i) {
    auto j=go2_moe::kMap[i];
    if(c.q[j]<go2_joint_limits::kPositionLower[j] || c.q[j]>go2_joint_limits::kPositionUpper[j]) return 2;
    if(c.kp[j]!=20. || c.kd[j]!=.5 || c.dq[j]!=0. || c.tau[j]!=0.) return 3;
    double v=c.q[j]; std::cout.write(reinterpret_cast<char*>(&v),sizeof(v));
   }
   for(float v:previous) {double d=v;std::cout.write(reinterpret_cast<char*>(&d),sizeof(d));}
  } catch(...) {return 4;}
 }
}
''')
    binary=temp/'projection'
    subprocess.run(['g++','-std=c++17','-O2','-I'+str(INCLUDE),str(oracle),'-o',str(binary)],check=True)
    cfg=yaml.safe_load((BUNDLE/'model/params/deploy.yaml').read_text())
    names=[f'{leg}_{joint}' for leg in ('FL','FR','RL','RR') for joint in ('hip','thigh','calf')]
    profile=load_official_profile(joint_names=names)
    scale=np.array(cfg['actions']['JointPositionAction']['scale'])
    offset=np.array(cfg['actions']['JointPositionAction']['offset'])
    rng=np.random.default_rng(20261106)
    boundary=np.stack([(profile.lower-offset)/scale,(profile.upper-offset)/scale]).astype(np.float32)
    cases=np.concatenate([rng.normal(0,20,(20000,12)).astype(np.float32),boundary,
        np.nextafter(boundary,np.float32(np.inf)),np.nextafter(boundary,np.float32(-np.inf)),
        np.full((1,12),np.finfo(np.float32).max,np.float32),np.full((1,12),-np.finfo(np.float32).max,np.float32)])
    actual=np.frombuffer(subprocess.check_output([str(binary)],input=cases.tobytes()),np.float64).reshape(-1,24)
    reference=transform_target_action(cases,scale,offset,profile.lower,profile.upper)
    wire=reference.sent_target.astype(np.float32).astype(np.float64)
    previous=((wire-offset)/scale).astype(np.float32)
    np.testing.assert_array_equal(actual[:,:12],wire)
    np.testing.assert_array_equal(actual[:,12:],previous)
    for invalid in (np.nan,np.inf,-np.inf):
        row=np.zeros(12,np.float32);row[0]=invalid
        assert subprocess.run([str(binary)],input=row.tobytes(),stdout=subprocess.DEVNULL).returncode==4
    # Reuse the existing independent 200-frame YAML observation oracle with this include tree.
    existing=(ROOT/'tests/test_moe_observation_parity.py').read_text()
    existing=existing.replace("ROOT = Path(__file__).resolve().parents[1]",f"ROOT = Path({str(ROOT)!r})")
    existing=existing.replace("deploy/moe1000_controller/overlay/include","deploy/original_bounded_20261106/controller/include")
    script=temp/'observations.py';script.write_text(existing)
    observation=json.loads(subprocess.check_output([sys.executable,str(script)],text=True))
    report=dict(passed=True,projection_cases=len(cases),joint_targets=len(cases)*12,
        sdk_float32_target_max_error=0.,submitted_history_max_error=0.,
        max_wire_rounding_from_float64_sim_rad=float(np.abs(wire-reference.sent_target).max()),
        nonfinite_rejections=3,observation=observation,regression=regression,
        robot_commands_sent=False)
    (BUNDLE/'validation/local_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
