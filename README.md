# Go2 HIMLoco：稳定爬楼与平地策略切换归档

本仓库当前推荐 **2026-11-02 续训中选定的 `model_1100.pt`**，并完整保留上一轮600次PPO更新的代码、模型、训练记录与独立仿真评估。目标平台为 **Isaac Sim 5.1 / Isaac Lab 2.3.2 / Python 3.11**。楼梯由本项目训练的HIMLoco策略控制，平地使用原仓库未修改的 `walk.pt`；通过独立地形扫描器提前1.1 m识别台阶，并以0.3 s过渡切换控制器。

这是独立的私有归档仓库：[`venom049/go2-himloco-stable-fast`](https://github.com/venom049/go2-himloco-stable-fast)。归档中的模型可以回放或作为后续训练起点；没有加载上游楼梯策略，也没有进行实机部署。

## 当前选定模型：累计1100次更新

推荐模型为 [`artifacts/20261102_195635_stairs_stable_fast_continue/model_1100.pt`](artifacts/20261102_195635_stairs_stable_fast_continue/model_1100.pt)。它从原600检查点恢复训练，在新增500次PPO更新时保存，累计编号为1100。整个续训任务新增1000次更新并运行到1600，评估后选择1100作为当前模型；它不是本次任务最后保存的检查点。“累计”指当前稳定爬楼训练链的更新编号，不包含更早warm-start之前的历史训练。

相同组合控制和评估条件下，固定18 cm阶高、32 cm踏面、8级楼梯，指令0.6 m/s，种子42/43各64个首回合，关闭观测噪声与域随机化、保留初始姿态采样，回合限时25 s：

| 指标 | 原600 | 选定1100 | 最终1600 |
| --- | ---: | ---: | ---: |
| 成功数 | 128/128 | **128/128** | 127/128 |
| 实测水平上楼速度 | 0.429 m/s | **0.464 m/s** | 0.491 m/s |
| 楼梯段左右侧倾RMS | 3.445° | **2.287°** | 3.999° |
| 楼梯段平均髋外展正分量 | 10.482° | **8.589°** | 8.491° |
| 支撑足滑移RMS | 0.133 m/s | **0.129 m/s** | 0.127 m/s |
| 足端内侧越界采样比例 | 2.55% | **11.60%** | 2.75% |

1100在保持这组测试128/128通过的同时，改善了相对600的速度、侧倾与平均外展。1600更快，但出现一次失败且侧倾更大，因此未选为当前模型。**1100的内侧足位仍有不足**：足端位于每侧0.12 m内侧边界以内的采样比例升至11.6%，主要来自左后足（该足45.93%）；足端跨过身体中线的采样比例为0。这不是11.6%的回合失败，也不能据此宣称交叉步或所有步态指标都已改善。

**0.6 m/s仍是命令值，1100实测水平上楼速度约0.464 m/s。** 质量指标统计成功首回合，通过率单独报告；这些是开发场景检查，不是未见地形或实机验证。模型选择、更新计数、文件校验与限制见 [`model_selection.json`](validation/stable_fast_continue/model_selection.json)，完整对照见 [`comparison.json`](validation/stable_fast_continue/comparison.json)。原600的长进场、纯平地及114项代码测试记录保留在下方历史章节，不作为1100重新完成这些检查的证据。

## 归档内容

| 路径 | 内容 |
| --- | --- |
| `go2_him/`、`configs/`、`scripts/`、`tests/` | 环境、奖励、模型接口、切换控制器、训练/回放/评估入口与测试 |
| `upstream/himloco_lab/source/` | 实际运行依赖的HIMLoco Lab源码及本地修补；不能直接用未修补的上游版本替换 |
| `upstream/SOURCES.json` | 上游固定提交与补丁清单 |
| `assets/go2/` | Go2 URDF、原始MJCF、16个OBJ、转换记录与许可证 |
| `assets/policies/upstream_walk/` | 原行走策略、接口来源记录、校验值与许可证 |
| `artifacts/20261102_195635_stairs_stable_fast_continue/` | 当前选定1100检查点及其实际配置、来源记录 |
| `artifacts/20261102_183927_stairs_stable_fast/` | 原600轮的6个检查点、实际配置、训练日志/TensorBoard事件、环境/算法/运行检查记录与训练时源码快照 |
| `artifacts/20261102_175509_stairs_lateral_refinement/` | 原600轮warm-start起点模型及配置、来源记录 |
| `validation/stable_fast_continue/` | 600/1100/1600对照、当前模型选择记录与仿真截图 |
| `validation/stable_fast/` | 原600轮评估、汇总、测试结果和仿真截图 |
| `validation/flat_policy/adapter_validation.json` | 原行走策略适配一致性检查 |
| `validation/server_environment.json` | 实际验证环境的软件版本与GPU信息 |
| `docs/PROJECT_GUIDE.md` | 原工作区完整说明与历史调参记录 |
| `ARCHIVE_SHA256SUMS` | 归档文件内容校验清单 |

原说明与历史JSON保留了原服务器绝对路径，它们是溯源记录；下列命令使用本仓库相对路径。历史参考工程、其他旧训练、服务器访问配置、仿真缓存和完整Python环境未全部打包。`docs/PROJECT_GUIDE.md`中的历史目录因此不一定存在于本仓库。

## 环境前提与完整性检查

下面命令在仓库根目录执行，前提是已有可用的NVIDIA GPU与兼容驱动，以及安装好的Isaac Sim/Isaac Lab环境。本仓库不打包NVIDIA运行时，也不提供未经验证的全新安装脚本；在另一台机器安装时遵循对应版本的官方安装说明和许可要求。已验证环境为：

| 组件 | 版本 |
| --- | --- |
| Python | 3.11.15 |
| Isaac Sim | 5.1.0.0 |
| Isaac Lab | 2.3.2.post1 |
| PyTorch / CUDA运行时 | 2.7.0+cu128 / 12.8 |
| NumPy / trimesh | 1.26.0 / 4.5.1 |
| Gymnasium / TensorBoard / PyYAML | 1.2.0 / 2.20.0 / 6.0.2 |

激活上述环境后，明确指定它的Python；所有后续命令沿用此环境变量。`run_lab.sh`和`_bootstrap.py`负责加载Isaac Lab及本仓库的HIMLoco源码路径，无需另装另一个版本的HIMLoco。

```bash
export GO2_PYTHON="$(command -v python)"

# 检查文件内容；应在修改文件或运行会生成文件的命令之前执行。
sha256sum -c ARCHIVE_SHA256SUMS

# 检查实际解释器、依赖、CUDA和项目文件。
bash scripts/run_lab.sh scripts/doctor_lab.py

# 配置检查不启动Isaac Sim。
"$GO2_PYTHON" scripts/train_lab.py --dry-run --config configs/stairs_stable_fast.yaml

# 单元测试不启动仿真窗口。
bash scripts/run_lab.sh -m unittest discover -s tests -v
```

第一次仿真启动会从随附URDF和网格生成本地USD缓存。模型文件、`settings.yaml`和`manifest.json`应保留在同一目录，回放和恢复入口需要读取它们。

## 回放最新组合控制

在已有图形会话的机器运行以下命令。服务器需由使用者设置其实际的 `DISPLAY` / X授权，仓库不携带桌面访问凭据。

```bash
bash scripts/run_lab.sh scripts/play_lab.py \
  --checkpoint artifacts/20261102_195635_stairs_stable_fast_continue/model_1100.pt \
  --controller hybrid --gate-lookahead 1.1 \
  --walk-checkpoint assets/policies/upstream_walk/walk.pt \
  --num-envs 1 --interactive --view front \
  --set terrain.rows=1 --set terrain.cols=1 \
  --set terrain.initial_level=0 \
  --set terrain.height_min=0.18 --set terrain.height_max=0.18 \
  --set commands.forward_min=0.6 --set commands.forward_max=0.6
```

`--controller hybrid`是显式选择：平地使用原行走策略，台阶附近由HIM策略接管，到上方平地后恢复行走策略。命令行默认控制器仍是 `stairs`；不加 `hybrid` 会全程使用楼梯策略。切换用独立2.4×1.0 m扫描器，原HIM观测接口保持不变。该扫描器是仿真高度射线，不是实机感知实现。

## 无窗口独立评估

输出写入新目录，避免覆盖归档评估。脚本会拒绝覆盖同名结果，重跑时更换 `--label` 或 `--output-dir`。

```bash
"$GO2_PYTHON" scripts/evaluate_stability.py \
  --checkpoint artifacts/20261102_195635_stairs_stable_fast_continue/model_1100.pt \
  --label archive_recheck --controller hybrid \
  --gate-lookahead 1.1 --speeds 0.6 --seeds 42 43 \
  --height 0.18 --num-envs 64 \
  --output-dir validation/recheck

"$GO2_PYTHON" scripts/summarize_stability.py \
  --input-dir validation/recheck --print-table
```

若要复核长距离进场，在评估命令中改为 `--seeds 44 --spawn-x-offset -0.3` 并换标签；纯平地检查改为 `--height 0 --seeds 44` 并换标签。独立通过率与训练日志里的课程采样成功率含义不同。

## 重跑原600次更新实验

该命令从归档的早期lateral模型重新微调，使用原600轮实际保存的完整配置；新结果写入 `logs/<新时间戳>_stairs_stable_fast_reproduce/`。它重跑的是原600轮实验，不是从1100继续训练。GPU仿真与优化不保证逐位重现原模型。

```bash
bash scripts/run_lab.sh scripts/train_lab.py \
  --headless \
  --config artifacts/20261102_183927_stairs_stable_fast/settings.yaml \
  --warm-start artifacts/20261102_175509_stairs_lateral_refinement/model_400.pt \
  --num-envs 1024 --max-iterations 600 \
  --run-name stairs_stable_fast_reproduce
```

`--warm-start`加载网络权重，并重置优化器、更新计数和配置中的探索噪声；速度课程从头开始。修改奖励可编辑 `configs/stairs_stable_fast.yaml` 后改用该配置，或通过重复 `--set key=value` 覆盖。

## 从最新模型继续训练

下面例子从当前选定的1100检查点恢复网络、优化器和已完成更新计数，再追加100次更新；保存编号预计继续为1200。示例关闭升速课程，以保持保存配置中的最终0.45–0.65 m/s命令采样范围。

```bash
bash scripts/run_lab.sh scripts/train_lab.py \
  --headless \
  --config artifacts/20261102_195635_stairs_stable_fast_continue/settings.yaml \
  --resume artifacts/20261102_195635_stairs_stable_fast_continue/model_1100.pt \
  --set commands.speed_ramp_updates=0 \
  --num-envs 1024 --max-iterations 100 \
  --run-name stairs_stable_fast_continue
```

`--resume`与`--warm-start`互斥。`--max-iterations`在恢复时表示追加更新数。checkpoint不保存完整仿真场景、随机数状态和地形课程现场；恢复会创建新环境。若不关闭速度课程，它会按新环境的步数重新渐进，因此不能把恢复理解为逐步无缝继续原仿真。

第三方来源及许可证边界见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
