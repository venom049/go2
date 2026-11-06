#!/usr/bin/env python3
"""Build an allowlisted, standalone stair inference/receive-only bundle.

Only the three deployment modules, an empty package initializer, the verified
TorchScript model/manifest, generated README and checksums enter this bundle.
No robot connection, dependency installation, telemetry capture or control is
performed. Existing nonempty output directories are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "_exports/go2-stairs-addon-1100-20261103"
CHECKPOINT_SHA256 = "877ff67ce7619516fa0e9350eccdd6537e7170640e5ef0a8b095a3624b2c27f5"
MODULES = ("policy.py", "read_state.py", "offline_check.py")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def readme(metadata: dict) -> str:
    return f"""# Go2 楼梯附加策略：1100 次更新

这是独立的模型及检查工具包。选中的 1100 次模型来自本项目的 HIMLoco 训练，
不替换机器人原有平地策略、固件、控制服务或启动配置。

本包只包含 CPU 推理、合成输入的离线检查，以及 `rt/lowstate` 只读订阅。
**没有电机命令发布、控制权切换、开机自启或实际爬楼入口。文件安装完成不表示机器人已经能爬楼。**
仿真中的自动平地/楼梯切换使用 raycast 地形数据，尚未接入实机感知。

## 文件与模型身份

- `go2_stairs/policy.py`：45 维观测 × 6 帧历史的推理适配器，只返回关节目标建议。
- `go2_stairs/offline_check.py`：模型校验与 CPU 合成输入耗时检查，不联网。
- `go2_stairs/read_state.py`：有限时长读取 `rt/lowstate`；需要 Unitree SDK2 Python/CycloneDDS。
- `model/stairs_policy.pt`：CPU TorchScript，输入 float32 `[batch,270]`，输出原始 `[batch,12]` 动作。
- `model/manifest.json`：完整观测、历史、关节顺序、控制合同与导出一致性结果。
- `SHA256SUMS`：本包文件逐项校验。

训练 checkpoint SHA256：`{metadata['checkpoint_sha256']}`

导出 TorchScript SHA256：`{metadata['export_sha256']}`

导出前与训练时实际 HIMActorCritic/Estimator 比较了 176 份输入历史（含 40 步动态观测），
本次 CPU eager 与重载 TorchScript 最大绝对误差均为 0。这只验证数值一致性。

## 独立环境与完整性检查

当前目标机为 Orin Nano、Ubuntu 20.04、aarch64、Python 3.9.5。
使用此目录自己的 `.venv`，依赖为 PyTorch 2.7.0 CPU 和 NumPy 1.26.4；
这些依赖不随本包复制，不应向系统 Python 或现有机器人控制环境安装或覆盖。

在机载电脑的独立目录中运行：

```bash
cd /home/unitree/go2-stairs-addon-1100-20261103
sha256sum -c SHA256SUMS
.venv/bin/python -c 'import torch; print(torch.__version__)'
```

## 离线 CPU 检查

```bash
mkdir -p reports
.venv/bin/python -m go2_stairs.offline_check --model model --warmup 20 --iterations 100 --output reports/offline_check.json
```

输入是训练默认关节姿态、水平静止姿态与固定 `0.6 m/s` 命令。
不会读取实机传感器，也不会向电机发送输出。报告给出模型哈希、输出形状/有限性、
100 次有效样本的 p50/p95/最大耗时，以及是否超过 50 Hz 对应的 20 ms 预算；
20 次预热不计入统计。`0.6 m/s` 是合成命令，不是机器人实测速度。

在开发电脑得到的耗时不能替代机载电脑耗时；低于 20 ms 也不保证实时调度或通信性能。
`--output` 拒绝覆盖已有报告，再次运行时请使用新的文件名，或省略此参数查看标准输出。

## 只读状态检查

先确认实际有线网卡；下面的 `eth0` 只是示例，必须替换成确认过的网卡名。
只读订阅还要求当前 Python 能导入 `unitree_sdk2py` 及 CycloneDDS。
若 `.venv` 未安装 SDK，应使用已确认具有 SDK 的 Python 解释器替换下列 `.venv/bin/python`，
无需更改机器人原有控制代码、服务或模式。

```bash
ip -brief link
mkdir -p reports
timeout 15s .venv/bin/python -m go2_stairs.read_state --interface eth0 --duration 5 --output reports/lowstate.json
```

订阅会产生必要的 DDS 发现通信，但不发布控制主题、不调用运动服务或释放原有控制器。
报告只保留必要状态字段，不包含设备序列号或遥控器原始字节；报告是本地运行时生成的，
不属于本包。若无数据或 SDK/网卡不匹配，工具会报告失败，不会尝试发出运动命令。
收到状态只说明数据链路可读，不表示故障已排除、控制权已获得或适合运动。

## 接口边界

策略关节顺序是 `FL、FR、RL、RR`，每腿 `hip、thigh、calf`；SDK 顺序是
`FR、FL、RR、RL`。二者不能直接混用，完整映射见 `model/manifest.json`。
50 Hz 推理，默认 Kp=40、Kd=1，动作先限幅 ±4，再乘 0.25 并加训练默认姿态。
观测最后 12 维必须反映上一步实际执行的最终目标，不能用未执行的网络建议替代。

该策略与原仓库 49 维观测/16 维输出的平地策略不兼容，不能只替换原策略文件路径。
本包也不实现原控制器的关节硬限位、扭矩限幅、超时保护或控制权仲裁。
后续接入运动应通过单独适配层保留现有保护和原平地功能；当前包不执行这一接入步骤。
"""


def build_bundle(output: str | Path, model_directory: str | Path) -> dict:
    output, model_directory = Path(output).absolute(), Path(model_directory).resolve()
    if output.is_symlink() or (output.exists() and
                              (not output.is_dir() or any(output.iterdir()))):
        raise ValueError("Output must be a new or empty directory; refusing to overwrite files")
    metadata = json.loads((model_directory / "manifest.json").read_text())
    if (metadata.get("format") != "go2-him-stairs-cpu-v1"
            or metadata.get("checkpoint_iteration") != 1100
            or metadata.get("checkpoint_sha256") != CHECKPOINT_SHA256):
        raise ValueError("Model manifest is not the selected 1100-update stair export")
    if sha256(model_directory / "stairs_policy.pt") != metadata.get("export_sha256"):
        raise ValueError("TorchScript SHA256 does not match the model manifest")
    for filename in MODULES:
        if not (ROOT / "deploy/go2_stairs" / filename).is_file():
            raise ValueError(f"Required deployment module missing: {filename}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as staging:
        stage = Path(staging)
        package = stage / "go2_stairs"
        package.mkdir()
        (package / "__init__.py").write_text('"""Go2 stair inference and bounded receive-only checks; no actuation."""\n')
        for filename in MODULES:
            shutil.copy2(ROOT / "deploy/go2_stairs" / filename, package / filename)
        destination = stage / "model"
        destination.mkdir()
        for filename in ("stairs_policy.pt", "manifest.json"):
            shutil.copy2(model_directory / filename, destination / filename)
        (stage / "README.md").write_text(readme(metadata), encoding="utf-8")
        files = sorted(path for path in stage.rglob("*") if path.is_file())
        checksums = "".join(f"{sha256(path)}  {path.relative_to(stage).as_posix()}\n" for path in files)
        (stage / "SHA256SUMS").write_text(checksums)
        # This exact allowlist prevents accidental inclusion of environment,
        # credentials, monitor configuration, captured state, or local caches.
        expected = {"go2_stairs/__init__.py", *(f"go2_stairs/{name}" for name in MODULES),
                    "model/stairs_policy.pt", "model/manifest.json", "README.md", "SHA256SUMS"}
        if {path.relative_to(stage).as_posix() for path in stage.rglob("*") if path.is_file()} != expected:
            raise RuntimeError("Unexpected file in the deployment bundle")
        if output.is_symlink() or (output.exists() and
                                  (not output.is_dir() or any(output.iterdir()))):
            raise ValueError("Output became nonempty; refusing to overwrite files")
        os.rename(stage, output)
    return {"directory": str(output), "files": sorted(expected),
            "checkpoint_iteration": 1100, "export_sha256": metadata["export_sha256"],
            "sha256sums_sha256": sha256(output / "SHA256SUMS"), "actuation_included": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="New or empty bundle directory; use a new versioned path when rebuilding")
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts/deploy_stairs_1100")
    args = parser.parse_args()
    try:
        report = build_bundle(args.output, args.model)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
