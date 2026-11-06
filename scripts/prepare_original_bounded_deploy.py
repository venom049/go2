"""Prepare an isolated original-MoE deployment; no network or motor operations."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'deploy/original_bounded_20261106'
REMOTE = '/home/unitree/go2-original-bounded-20261106'
ORIGINAL = ROOT / 'validation/moe1000_sim2real_20261105/source/upstream/robogauge/unitree_cpp_deploy/logs/rsl_rl/go2_moe_cts_expert_goal_137000_0.6745'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    if OUT.exists():
        raise RuntimeError('Destination already exists; do not overwrite deployment evidence')
    OUT.mkdir(parents=True)
    shutil.copytree(ROOT/'deploy/moe1000_controller/base', OUT/'controller')
    shutil.copytree(ROOT/'deploy/moe1000_controller/overlay', OUT/'controller', dirs_exist_ok=True)
    for sub in ('exported/policy.onnx', 'params/deploy.yaml'):
        target=OUT/'model'/sub
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ORIGINAL/sub,target)
    assert sha(OUT/'model/exported/policy.onnx') == 'bc08253fd68f661bf6be5ce5ca70cec555c3fde2c397d23fbe9814a998e056f5'
    shutil.copyfile(ROOT/'validation/official_joint_limits_20261104/limit_profile.json',OUT/'model/limit_profile.json')
    shutil.copyfile(ROOT/'deploy/handoff_overlay/include/Go2JointLimits.h',OUT/'controller/include/Go2JointLimits.h')
    policy=OUT/'controller/include/MoePolicy.h'
    text=policy.read_text().replace('std::array<float,12> kOffset','std::array<double,12> kOffset')
    text=text.replace('.1f,.8f,-1.5f,-.1f,.8f,-1.5f,.1f,1.f,-1.5f,-.1f,1.f,-1.5f',
                      '.1,.8,-1.5,-.1,.8,-1.5,.1,1.,-1.5,-.1,1.,-1.5')
    text=text.replace('(static_cast<float>(cmd.q[kMap[i]])-kOffset[i])/.25f',
                      'static_cast<float>((cmd.q[kMap[i]]-kOffset[i])/.25)')
    old='''        // Training clips the scaled target to [-100,100], not the raw action.
        // Physical target/tracking/effort rejection remains at publication.
        const float target=action[i]*.25f+kOffset[i];finite(target);
        const auto sdk=kMap[i];result.q[sdk]=std::clamp(target,-100.f,100.f);'''
    new='''        // Keep the original ONNX unchanged. Project the affine target into
        // SDK float32 position bounds BEFORE the independent publication guard.
        // Round exactly as LowCmd serialization does, then retain that submitted
        // target in action history. Other guard failures still latch damping.
        const double target=static_cast<double>(action[i])*.25+kOffset[i];finite(target);
        const auto sdk=kMap[i];
        result.q[sdk]=static_cast<float>(std::clamp(target,
            go2_joint_limits::kPositionLower[sdk],go2_joint_limits::kPositionUpper[sdk]));'''
    assert old in text
    policy.write_text(text.replace(old,new))
    cfg=OUT/'controller/robots/go2/config/config.yaml'
    cfg.write_text(cfg.read_text().replace('unitree_go2_2ded279_hip48_calf30_user20261104','unitree_go2_2ded279_hip48'))
    identity=OUT/'controller/include/CandidateIdentity.h'
    text=identity.read_text()
    replacements={
      '/home/unitree/go2-moe1000-20261104':REMOTE+'/model',
      'f0bfb5ac6758b72fbeae96f1b82462950127440b5210807016d5fa5466529d56':sha(OUT/'model/exported/policy.onnx'),
      '7d04bfd00a60cf858572597667d9f33e78b94f71dca2b101104162caef0aca00':sha(OUT/'model/params/deploy.yaml'),
      'ee56644806f2bb38c5fd5061be56823dc68d746076e6b2bf1d9c01e534aa8cff':sha(cfg),
      'b62566e6bc95385b15a0423a449b3d033c5e76f77a7ce650ba0a2c112e9585f0':sha(OUT/'model/limit_profile.json')}
    for old,new in replacements.items():
        assert old in text,old
        text=text.replace(old,new)
    identity.write_text(text)
    main_cpp=OUT/'controller/robots/go2/main.cpp'
    main_cpp.write_text(main_cpp.read_text().replace('/home/unitree/go2-moe1000-controller-20261104',REMOTE))
    bridge=(ROOT/'deploy/moe1000_controller/motion_rpc_bridge.py').read_text()
    bridge=bridge.replace("p.add_argument('--allow-release',action='store_true')", "p.add_argument('--allow-release',action='store_true')\n    p.add_argument('--network', default='eth0')")
    bridge=bridge.replace("ChannelFactoryInitialize(0,'eth1')",'ChannelFactoryInitialize(0,args.network)')
    bridge=bridge.replace('/home/unitree/go2-moe1000-controller-20261104',REMOTE)
    (OUT/'motion_rpc_bridge.py').write_text(bridge)
    shutil.copyfile(ROOT/'deploy/moe1000_controller/native_parity.cpp',OUT/'native_parity.cpp')
    (OUT/'validation').mkdir()
    (OUT/'manifest.json').write_text(json.dumps(dict(
        robot='Go2 EDU, no extra payload',remote_root=REMOTE,
        weights='Unmodified RoboGauge / MoE-CTS original ONNX',onnx_sha256=sha(OUT/'model/exported/policy.onnx'),
        local_wifi_interface='wlo1',remote_wifi_ip='192.168.1.180',remote_dds_interface='eth0',
        target_clip='SDK float32 joint position bounds; sent target inverse-encoded into history',
        profile='unitree_go2_2ded279_hip48',calf_speed_limit_rad_s=15.7,
        stand_gains=[40,1],policy_gains=[20,.5],policy_period_s=.02,
        handoff_source='Existing protected handoff, original main.cpp SHA ea3d036a3797395e3824cf5e48209e754006cc38e8c4c89ed34fb442135cc86b',
        policy_run_limit_s=2,autostart=False,
        caveats=['Fixed damping does not maintain standing pose.',
                 'Existing ReleaseMode RPC and post-release checks do not guarantee an uninterrupted hold.',
                 'Simulation success is not a real robot test; actual speed/position/tracking checks may stop the policy.']),indent=2)+'\n')
    print(OUT)

if __name__=='__main__': main()
