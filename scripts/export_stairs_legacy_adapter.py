#!/usr/bin/env python3
"""Export an inactive ONNX candidate for the inspected existing controller.

No controller files are changed and no robot is contacted. Reference fixtures
come from the actual trained HIM actor/estimator, using independently specified
synthetic sensor states and previously applied targets, not model predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torch.nn import functional as F
import yaml

from export_stairs import DEFAULT_CHECKPOINT, validate_source, load_reference, load_source_record
from deploy.go2_stairs.policy import DEFAULT_JOINT_POSITION, JOINT_NAMES, load_checkpoint

OLD_DEFAULT = (0., .8, -1.5, 0., .8, -1.5, 0., 1., -1.5, 0., 1., -1.5)
SDK_NAMES = tuple(f"{leg}_{joint}_joint" for leg in ("FR", "FL", "RR", "RL")
                  for joint in ("hip", "thigh", "calf"))
PERMUTATION = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)
DEFAULT_OUTPUT = ROOT / "_exports/go2-stairs-addon-1100-20261103/legacy_candidate/v0"
SOURCE_ROOT = ROOT / "validation/deployment_1100/existing_controller"


class LegacyAdapter(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.encoder, self.actor = network.encoder, network.actor
        self.register_buffer("permutation", torch.tensor(PERMUTATION, dtype=torch.long))
        self.register_buffer("old_default", torch.tensor(OLD_DEFAULT, dtype=torch.float32))
        self.register_buffer("him_default", torch.tensor(DEFAULT_JOINT_POSITION, dtype=torch.float32))

    def him_history(self, obs):
        frames = torch.flip(obs.reshape(1, 6, 49), dims=[1])
        position = (frames[:, :, 9:21]+self.old_default).index_select(2, self.permutation)
        previous_target = (frames[:, :, 33:45]*.25+self.old_default).index_select(2, self.permutation)
        return torch.cat((frames[:, :, :9], position-self.him_default,
                          frames[:, :, 21:33].index_select(2, self.permutation),
                          (previous_target-self.him_default)/.25), dim=-1).reshape(1, 270)

    def forward(self, obs):
        history = self.him_history(obs)
        parts = self.encoder(history)
        latent = F.normalize(parts[:, 3:], dim=-1, p=2., eps=1e-12)
        raw = self.actor(torch.cat((history[:, :45], parts[:, :3], latent), dim=-1))
        target = (self.him_default+.25*raw.clamp(-4., 4.)).index_select(1, self.permutation)
        legacy_action = (target-self.old_default)/.25
        return torch.cat((legacy_action, torch.zeros_like(legacy_action[:, :4])), dim=-1)


def configuration():
    def term(width, scale=1., params=None):
        return {"params": {} if params is None else params, "clip": None,
                "scale": [scale]*width, "history_length": 6}
    # First entry must be a real term: the C++ manager checks its 'params' key
    # to decide whether this is a single observation group. Put the flag last.
    observations = {
        "keyboard_stairs_velocity_commands": term(3, params={
            "min_forward_speed": .6, "max_forward_speed": .6,
            "speed_step": .1, "initial_forward_speed": .6}),
        "base_ang_vel": term(3, .25), "projected_gravity": term(3),
        "joint_pos_rel": term(12), "joint_vel_rel": term(12, .05),
        "applied_action": term(16), "use_gym_history": True,
    }
    return {
        "joint_ids_map": list(range(12)), "step_dt": .02,
        "stiffness": [40.]*12, "damping": [1.]*12,
        "default_joint_pos": list(OLD_DEFAULT),
        "commands": {"base_velocity": {"ranges": {"lin_vel_x": [.6, .6],
                     "lin_vel_y": [0., 0.], "ang_vel_z": [0., 0.], "heading": None}}},
        "actions": {"AdaptiveStiffnessJointPositionAction": {
            "position_scale": .25, "action_clip": 4.5, "offset": list(OLD_DEFAULT),
            "kp_range": [40., 40.], "kp_default": 40., "kp_action_scale": 0.,
            "kd_sqrt_scale": 1/math.sqrt(40.)}},
        "observations": observations,
    }


def legacy_process(output, cfg):
    """Offline reproduction of the inspected C++ 16-to-36 action formula."""
    action = cfg["actions"]["AdaptiveStiffnessJointPositionAction"]
    raw = np.clip(output, -action["action_clip"], action["action_clip"])
    target = np.array(action["offset"], np.float32)+np.float32(action["position_scale"])*raw[:, :12]
    kp = np.clip(np.float32(action["kp_default"])+np.float32(action["kp_action_scale"])*raw[:, 12:],
                 *action["kp_range"]).repeat(3, axis=1)
    kd = np.float32(action["kd_sqrt_scale"])*np.sqrt(kp)
    return target, kp, kd


@torch.inference_mode()
def make_fixtures(reference):
    """Canonical HIM-order states -> two independently assembled histories."""
    rows, targets, actions, histories, trajectory_ids, step_ids = [], [], [], [], [], []
    him_default = torch.tensor(DEFAULT_JOINT_POSITION, dtype=torch.float32)
    old_default = torch.tensor(OLD_DEFAULT, dtype=torch.float32)
    sdk_indices = [JOINT_NAMES.index(name) for name in SDK_NAMES]
    him_indices = [SDK_NAMES.index(name) for name in JOINT_NAMES]
    saturated = 0
    for trajectory in range(4):
        old_history, him_history = [], []
        for step in range(40):
            phase = torch.arange(12, dtype=torch.float32)*.37+step*.12+trajectory*.7
            # These measurements and targets are predetermined, independent of
            # either policy output. This is not a simulated or actual robot.
            q_him = him_default + (.08+.04*trajectory)*torch.sin(phase)
            dq_him = (.5+.2*trajectory)*torch.cos(phase*.8)
            previous_sdk = (old_default if step == 0 else
                            old_default+.12*torch.sin(phase[sdk_indices]+.4))
            previous_him = previous_sdk[him_indices]
            roll, pitch = .10*math.sin(step*.12+trajectory), .16*math.cos(step*.07+trajectory)
            gravity = torch.tensor([math.sin(pitch), -math.cos(pitch)*math.sin(roll),
                                    -math.cos(pitch)*math.cos(roll)], dtype=torch.float32)
            command = torch.tensor([.6, 0., 0.], dtype=torch.float32)
            gyro = torch.tensor([.2*math.cos(step*.12), -.1*math.sin(step*.1), .03*trajectory])
            q_sdk, dq_sdk = q_him[sdk_indices], dq_him[sdk_indices]
            gain_history = torch.tensor([math.sin(step+i+trajectory) for i in range(4)])
            # Existing controller frame: cmd, gyro, gravity, SDK qrel, SDK dq,
            # last 16 clipped actions. Four gain channels must be discarded.
            old_frame = torch.cat((command, gyro*.25, gravity, q_sdk-old_default,
                                   dq_sdk*.05, (previous_sdk-old_default)/.25, gain_history))
            # Oracle starts directly from canonical HIM physical states; it
            # never invokes LegacyAdapter's frame or action conversion.
            him_frame = torch.cat((command, gyro*.25, gravity, q_him-him_default,
                                   dq_him*.05, (previous_him-him_default)/.25))
            if step == 0:
                old_history = [old_frame.clone() for _ in range(6)]
                him_history = [him_frame.clone() for _ in range(6)]
            else:
                old_history = old_history[1:]+[old_frame]
                him_history = [him_frame]+him_history[:5]
            old_input = torch.cat(old_history)[None, :]
            reference_history = torch.cat(him_history)[None, :]
            raw = reference.act_inference(reference_history)
            saturated += int((raw.abs() > 4.).sum())
            target_him = him_default+.25*raw.clamp(-4., 4.)
            expected_target = target_him[:, sdk_indices]
            expected_output = torch.cat(((expected_target-old_default)/.25, torch.zeros(1, 4)), -1)
            rows.append(old_input.numpy()[0]); histories.append(reference_history.numpy()[0])
            targets.append(expected_target.numpy()[0]); actions.append(expected_output.numpy()[0])
            trajectory_ids.append(trajectory); step_ids.append(step)
    return {"obs": np.stack(rows), "him_history": np.stack(histories),
            "target_sdk": np.stack(targets), "output_legacy": np.stack(actions),
            "kp": np.full((len(rows), 12), 40., dtype=np.float32),
            "kd": np.ones((len(rows), 12), dtype=np.float32),
            "trajectory": np.array(trajectory_ids, np.int32), "step": np.array(step_ids, np.int32)}, saturated


VERIFY_SOURCE = r'''#!/usr/bin/env python3
"""Standalone synthetic-fixture ONNX verification; no SDK, network or control."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import onnxruntime as ort

def verify(directory, iterations=100, warmup=20):
    directory = Path(directory).resolve()
    if iterations < 1 or warmup < 0:
        raise ValueError("iterations must be positive and warmup nonnegative")
    checked = []
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        path = (directory / relative).resolve()
        if directory not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Bundle checksum failed: " + relative)
        checked.append(relative)
    required = {"exported/policy.onnx", "params/deploy.yaml", "reference.npz", "provenance.json", "README.md", "verify.py"}
    if set(checked) != required or len(checked) != len(required):
        raise ValueError("Unexpected SHA256SUMS file set")
    config = json.loads((directory / "provenance.json").read_text())
    fixtures = np.load(directory / "reference.npz", allow_pickle=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(directory / "exported/policy.onnx"), options, providers=["CPUExecutionProvider"])
    if len(session.get_inputs()) != 1 or session.get_inputs()[0].name != "obs" or session.get_inputs()[0].shape != [1,294]:
        raise ValueError("Expected the existing controller's static obs[1,294] input")
    if len(session.get_outputs()) != 1 or session.get_outputs()[0].shape != [1,16]:
        raise ValueError("Expected static [1,16] output")
    x = fixtures["obs"]
    if x.dtype != np.float32 or x.ndim != 2 or x.shape[1] != 294 or not np.isfinite(x).all():
        raise ValueError("Invalid synthetic fixture")
    outputs = np.concatenate([session.run(None, {"obs": row[None,:]})[0] for row in x])
    if not np.isfinite(outputs).all():
        raise ValueError("Non-finite ONNX output")
    action = config["legacy_action_contract"]
    clipped = np.clip(outputs, -action["action_clip"], action["action_clip"])
    targets = np.asarray(action["offset"], np.float32)+np.float32(action["position_scale"])*clipped[:,:12]
    kp = np.clip(np.float32(action["kp_default"])+np.float32(action["kp_action_scale"])*clipped[:,12:], *action["kp_range"]).repeat(3,axis=1)
    kd = np.float32(action["kd_sqrt_scale"])*np.sqrt(kp)
    errors = {}
    for name, actual, expected in (("output_legacy",outputs,fixtures["output_legacy"]),
                                    ("target_sdk",targets,fixtures["target_sdk"]),
                                    ("kp",kp,fixtures["kp"]),("kd",kd,fixtures["kd"])):
        np.testing.assert_allclose(actual, expected, atol=5e-6, rtol=2e-5)
        errors[name+"_max_abs"] = float(np.max(np.abs(actual-expected)))
    if not np.array_equal(outputs[:,12:],np.zeros_like(outputs[:,12:])):
        raise ValueError("Adaptive gain outputs must be exactly zero")
    if np.max(np.abs(outputs)) > 4.40001:
        raise ValueError("Output exceeds the proven target encoding range")
    sample = x[-1:]
    for _ in range(warmup): session.run(None,{"obs":sample})
    timings = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        session.run(None,{"obs":sample})
        timings.append((time.perf_counter_ns()-start)/1e6)
    return {"timestamp_utc":datetime.now(timezone.utc).isoformat(),
            "status":"passed", "scope":"synthetic reference parity and CPU ONNX timing; no real-robot or controller execution",
            "platform":platform.platform(), "machine":platform.machine(),
            "numpy_version":np.__version__, "onnxruntime_version":ort.__version__,
            "provider":"CPUExecutionProvider", "checked_files":checked,
            "fixture_samples":int(len(x)), "trajectories":int(len(np.unique(fixtures["trajectory"]))),
            "numerical_errors":errors, "atol":5e-6,"rtol":2e-5,
            "benchmark":{"warmup_iterations":warmup,"timed_iterations":iterations,
                "p50_ms":float(np.percentile(timings,50)),"p95_ms":float(np.percentile(timings,95)),
                "maximum_ms":max(timings),"budget_ms":20.,
                "p95_within_budget":bool(np.percentile(timings,95)<=20.),
                "samples_over_budget":sum(t>20. for t in timings)},
            "robot_commands_sent":False,"robot_validation":False}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,default=Path(__file__).resolve().parent)
    p.add_argument("--iterations",type=int,default=100)
    p.add_argument("--warmup",type=int,default=20)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    try:
        if a.output is not None and a.output.exists(): raise ValueError("Output report already exists")
        result=verify(a.directory,a.iterations,a.warmup)
        text=json.dumps(result,indent=2,ensure_ascii=False)+"\n"
        if a.output is not None:
            with a.output.open("x",encoding="utf-8") as f: f.write(text)
    except (OSError,ValueError,RuntimeError,AssertionError,KeyError) as e: p.error(str(e))
    print(text,end="")
if __name__=="__main__": main()
'''


@torch.inference_mode()
def validate(adapter, session, fixtures, cfg):
    eager, mapped = [], []
    for row in fixtures["obs"]:
        inp = torch.from_numpy(row[None, :])
        eager.append(adapter(inp).numpy()[0]); mapped.append(adapter.him_history(inp).numpy()[0])
    eager, mapped = np.stack(eager), np.stack(mapped)
    outputs = np.concatenate([session.run(None, {"obs": row[None, :]})[0] for row in fixtures["obs"]])
    errors = {}
    for name, actual, expected in (("converted_history", mapped, fixtures["him_history"]),
                                   ("eager_output", eager, fixtures["output_legacy"]),
                                   ("onnx_output", outputs, fixtures["output_legacy"])):
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=5e-6)
        errors[name+"_max_abs"] = float(np.abs(actual-expected).max())
    for backend, output in (("eager", eager), ("onnx", outputs)):
        target, kp, kd = legacy_process(output, cfg)
        for name, actual in (("target_sdk", target), ("kp", kp), ("kd", kd)):
            np.testing.assert_allclose(actual, fixtures[name], rtol=2e-5, atol=5e-6)
            errors[backend+"_"+name+"_max_abs"] = float(np.abs(actual-fixtures[name]).max())
    if not np.all(outputs[:, 12:] == 0.):
        raise ValueError("Legacy gain action channels must be zero")
    # Endpoint stress of the separate existing action postprocessor: the offset
    # conversion needs +/-4.4 legacy units, so clipping it at 4 would be wrong.
    him_default = np.asarray(DEFAULT_JOINT_POSITION, np.float32)
    extremes = np.stack((np.full(12, -4., np.float32), np.full(12, 4., np.float32)))
    extreme_target = (him_default+.25*extremes)[:, PERMUTATION]
    encoded = np.concatenate(((extreme_target-np.asarray(OLD_DEFAULT, np.float32))/.25,
                               np.zeros((2, 4), np.float32)), -1)
    np.testing.assert_allclose(legacy_process(encoded, cfg)[0], extreme_target, atol=2e-7, rtol=0)
    return {"samples": len(outputs), "trajectories": 4, "frames_per_trajectory": 40,
            "errors": errors, "atol": 5e-6, "rtol": 2e-5,
            "clipped_action_extreme_targets_checked": True,
            "previous_targets": "predetermined synthetic physical targets shared by both paths; never inferred policy outputs"}


def export(output: Path, checkpoint: Path = DEFAULT_CHECKPOINT,
           source_record: str | Path | dict | None = None):
    if output.is_symlink() or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
        raise ValueError("Candidate output must be new or empty; refusing overwrite")
    torch.set_num_threads(1)
    checkpoint = Path(checkpoint).resolve()
    identity = load_source_record(source_record)
    _, agent = validate_source(checkpoint, identity)
    reference = load_reference(checkpoint, agent, identity["checkpoint_iteration"])
    network, checkpoint_metadata = load_checkpoint(checkpoint)
    adapter = LegacyAdapter(network).cpu().eval()
    fixtures, saturation_count = make_fixtures(reference)
    cfg = configuration()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".legacy-candidate-", dir=output.parent) as temporary:
        stage = Path(temporary)
        (stage / "exported").mkdir(); (stage / "params").mkdir()
        model = stage / "exported/policy.onnx"
        torch.onnx.export(adapter, torch.zeros(1, 294), str(model), input_names=["obs"],
                          output_names=["actions"], opset_version=17, dynamo=False,
                          export_params=True, do_constant_folding=True)
        graph = onnx.load(str(model)); onnx.checker.check_model(graph, full_check=True)
        options = ort.SessionOptions(); options.intra_op_num_threads = 1; options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
        verification = validate(adapter, session, fixtures, cfg)
        (stage / "params/deploy.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        # Verify YAML roundtrip does not reorder terms or alter control values.
        restored = yaml.safe_load((stage / "params/deploy.yaml").read_text())
        if restored != cfg or list(restored["observations"]) != list(cfg["observations"]):
            raise ValueError("Deployment YAML contract/order changed during serialization")
        np.savez_compressed(stage / "reference.npz", **fixtures)
        source_files = ["include/isaaclab/manager/manager_term_cfg.h",
                        "include/isaaclab/manager/observation_manager.h",
                        "include/isaaclab/algorithms/algorithms.h",
                        "include/isaaclab/envs/manager_based_rl_env.h",
                        "robots/go2/src/State_RLBase.cpp", "robots/go2/include/StairsCommandBuffer.h"]
        provenance = {
            "format": "go2-him-legacy-candidate-v1", "active": False,
            "checkpoint_filename": checkpoint.name,
            "checkpoint_iteration": checkpoint_metadata["iteration"],
            "iteration_semantics": checkpoint_metadata["iteration_semantics"],
            "checkpoint_sha256": identity["checkpoint_sha256"],
            "source_training_files_sha256": identity["source_hashes"],
            "onnx_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            "dependencies": {"torch": str(torch.__version__), "numpy": np.__version__,
                             "onnx": onnx.__version__, "onnxruntime": ort.__version__},
            "opset": 17, "input": {"name": "obs", "shape": [1, 294], "dtype": "float32",
                                    "history": "six 49D frames, oldest first"},
            "output": {"name": "actions", "shape": [1, 16],
                       "semantics": "SDK-ordered position target encoded about legacy default; four zero gain channels"},
            "observation_terms": list(cfg["observations"]),
            "sdk_joint_order": list(SDK_NAMES), "him_joint_order": list(JOINT_NAMES),
            "sdk_to_him_indices": list(PERMUTATION),
            "legacy_action_contract": cfg["actions"]["AdaptiveStiffnessJointPositionAction"],
            "reference": "actual vendored HIMActorCritic.act_inference; trained HIMEstimator; canonical HIM-order synthetic physical states",
            "inspected_controller_sources_sha256": {p: hashlib.sha256((SOURCE_ROOT/p).read_bytes()).hexdigest() for p in source_files},
            "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "offline_verification": {**verification, "reference_raw_actions_clipped": saturation_count},
            "scope": "candidate artifacts only; no original controller edits, robot commands, or activation",
            "limitations": [
                "No hybrid terrain sensing or flat/stair transition equivalence; this model is only a stair actor.",
                "First reset uses legacy zero-hip default target, so HIM previous hip action is nonzero, unlike a training reset.",
                "FixStand, target holding, asynchronous threads, safety stops and recovery are outside this numerical mapping proof.",
                "applied_action is existing ActionManager raw_actions, not measured actuator execution; a held/overridden target would break exact previous-target equivalence.",
                "Keyboard stop/auto-zero commands and arbitrary FSM speed overrides are not verified as trained standing or recovery behavior.",
                "No extra joint hard limits or torque bounds are inserted; existing low-level protection remains unchanged.",
                "Synthetic fixture and CPU checks do not establish safe real-robot stair performance.",
            ],
        }
        (stage / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False)+"\n")
        (stage / "verify.py").write_text(VERIFY_SOURCE)
        (stage / "README.md").write_text(f"""# 本轮 {identity['checkpoint_iteration']} 次更新楼梯模型：原控制器接口候选包

本目录只新增 ONNX、deploy.yaml 和离线验证文件，尚未激活，不修改原控制器、平地策略或服务。
`exported/policy.onnx` 接受固定 `obs[1,294]`，6 帧旧到新、每帧49维；内部转换为训练时的新到旧270维。
命令必须位于首项且不缩放，顺序和历史开关以本目录 `params/deploy.yaml` 为准，不能沿用原49维单帧配置。
输出是SDK顺序16维，前12维表达相同HIM关节目标，后4维为0。现有AdaptiveStiffness动作处理产生12目标+12Kp+12Kd。
HIM先裁剪±4，目标换算后旧接口可能达到±4.4，故旧action_clip=4.5；Kp=40、Kd=1。

独立离线验证只需 NumPy 1.26.4 和 ONNX Runtime 1.19.2，不需要训练项目或 Unitree SDK：

```bash
sha256sum -c SHA256SUMS
../../.venv/bin/python verify.py --iterations 100 --warmup 20
```

也可显式用已安装依赖的Python运行 verify.py。脚本仅加载本目录模型和真实HIM参考fixture，
不读取传感器、不联网、不启动原控制器、不发布电机命令。
40步×4条合成轨迹的上一目标是共同预设物理目标；不是把推理结果假冒成已执行动作。
报告比较观测转换、关节目标、Kp/Kd及ONNX数值误差，另给CPU p50/p95与20ms预算；耗时不是实时保证。

首次重置时，旧默认髋0与训练±0.1不同，转换后的HIM上一动作对应真实旧目标而非强制归零。
原控制器的applied_action实为动作管理器raw记录；持有/覆盖目标、FixStand、异步调度、安全停机与恢复不在等价性证明范围内。
原平地策略未纳入模型，仿真hybrid所需地形raycast也未接入实机；不能宣称本包提供原仿真的自动切换或实机安全。
键盘归零/自动归零与FSM速度覆盖亦不等于经过训练验证的站立行为。
完整来源、控制合同和限制见provenance.json。不要把本次离线通过当作实机爬楼通过。
""", encoding="utf-8")
        files = sorted(p for p in stage.rglob("*") if p.is_file())
        (stage / "SHA256SUMS").write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(stage).as_posix()}\n" for p in files))
        validate_source(checkpoint, identity)
        if output.is_symlink() or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
            raise ValueError("Output became nonempty")
        os.rename(stage, output)
    return {"output": str(output), "onnx_sha256": provenance["onnx_sha256"],
            "verification": provenance["offline_verification"], "active": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                   help="Checkpoint beside pinned training sidecars; defaults to original 1100")
    p.add_argument("--source-record", type=Path,
                   help="Explicit JSON identity for another checkpoint; omission keeps strict original hashes")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    a = p.parse_args()
    try:
        result = export(a.output.absolute(), a.checkpoint, a.source_record)
    except (OSError, ValueError, RuntimeError, AssertionError) as e:
        p.error(str(e))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
