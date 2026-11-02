# Go2 爬楼梯：Isaac Lab + HIMLoco 从零训练

本项目使用指定的 [go2-stairs-sim2real](https://github.com/wxz-521/go2-stairs-sim2real) 仓库所引用的 Go2 机器人模型，建立可调奖励、可调楼梯课程的仿真训练工程。当前主运行环境为 **Isaac Sim 5.1、Isaac Lab 2.3.2、Python 3.11**。楼梯训练使用 HIMLoco 的历史观测估计器、Actor-Critic 和 PPO，默认随机初始化网络，也可显式从本项目自己的 checkpoint 微调；没有加载上游楼梯策略。按照后续需求，仿真回放另接入原仓库的 `walk.pt` 专门控制平地行走，它与楼梯 HIM 策略独立，不用于楼梯训练初始化。

最初的 Isaac Gym 版 HIMLoco 保留在本地工作区的 `upstream/HIMLoco/`，供算法对照；服务器只部署 Lab 运行所需的源码与模型。实际训练入口使用 `upstream/himloco_lab/` 中的 Isaac Lab 移植版及本项目的 Go2 楼梯环境。原版 Gym 环境与当前 Lab 环境的启动方式、观测组织和物理参数不能直接混用。

HIMLoco 参考源码固定为 `ef289acaa62795009363b7b819c9186690630441`；[IsaacZH/himloco_lab 移植版](https://github.com/IsaacZH/himloco_lab/tree/c3fbd0173f2a53a86e472c099f8e2e9744cf7639) 固定为 `c3fbd0173f2a53a86e472c099f8e2e9744cf7639`。本地补丁修正了回合重置时的历史观测、估计器观测切片、超时价值引导、小批次余数和单样本归一化、checkpoint 恢复计数与设备映射；同时适配 Lab 2.3.2 API、延迟加载无关任务，并在缩放关节动作前进行裁剪。完整版本和补丁清单见 [`upstream/SOURCES.json`](upstream/SOURCES.json)。

## 文件位置

服务器项目目录为 `/root/go2-himloco`。项目内的 `logs/` 指向数据盘目录 `/root/gpufree-data/go2-himloco/logs`，训练日志和自己的 checkpoint 保存到这里。

| 文件/目录 | 用途 |
| --- | --- |
| `configs/stairs.yaml` | 环境规模、奖励权重、楼梯、控制器、随机化和 PPO 参数 |
| `configs/stairs_stable_fast.yaml` | 稳定性、髋外展与逐步升速的可选微调配置 |
| `go2_him/` | 实验配置、楼梯几何及 Isaac Lab 环境代码 |
| `scripts/run_lab.sh` | 使用服务器的 Isaac Lab Python 环境启动脚本 |
| `scripts/train_lab.py` | 从零训练或显式继续自己的 checkpoint |
| `scripts/play_lab.py` | 楼梯 HIM、原仓库平地策略或两者自动切换的仿真回放 |
| `scripts/evaluate_stability.py` | 固定速度、楼梯高度和随机种子的首回合对照评估 |
| `go2_him/flat_policy.py`、`policy_switch.py` | 原平地策略接口适配及高度扫描驱动的切换 |
| `assets/policies/upstream_walk/` | 固定版本 `walk.pt`、来源校验记录和 BSD-3-Clause 许可证 |
| `scripts/prepare_go2_asset.py` | 离线重新生成 URDF，或获取固定版本的源模型 |
| `assets/go2/urdf/go2.urdf` | 实际使用的 Go2 模型 |
| `assets/go2/mjcf/` | 原始 MJCF 和 16 个 OBJ 网格 |
| `assets/go2/PROVENANCE.json` | 模型来源、版本、校验值及转换差异 |
| `tests/` | 无需启动仿真器的模型、地形和算法检查 |
| `upstream/` | 已获取的上游源码和算法参考 |

## 先验证运行链路

以下命令在已配置 Isaac Sim/Isaac Lab 的服务器执行。`run_lab.sh` 封装 Isaac Lab 的源码路径，默认使用当前环境的 `python`；如果安装在其他解释器下，先设置 `GO2_PYTHON=/绝对路径/python`。运行前应确认该解释器为安装了 Isaac Sim 5.1 和 Isaac Lab 2.3.2 的 Python 3.11。

用户已于 2026-11-02 明确接受本次服务器安装所需的 NVIDIA EULA；接受记录 `.nvidia_eula_accepted` 仅保留在服务器项目目录。启动脚本读取此标记后设置 `OMNI_KIT_ACCEPT_EULA=YES`。工程中不保存服务器口令。

```bash
cd /root/go2-himloco

# 少量环境、两次 PPO 更新：检查模型导入、步进、梯度更新和保存链路。
bash scripts/run_lab.sh scripts/train_lab.py \
  --headless \
  --num-envs 64 \
  --max-iterations 2 \
  --run-name smoke
```

首次启动可能需要生成 USD 资产及初始化仿真缓存。两次更新只用于检查训练链路，不能证明已经学会爬楼梯。无桌面的服务器应保留 `--headless`；有可用图形会话时，可移除它观察场景。

模型本身还可以在普通 Python 环境中检查：

```bash
python3 scripts/prepare_go2_asset.py
python3 -m unittest discover -s tests -p test_assets.py -v

# 仅校验训练参数；不导入或启动 Isaac Sim。
python3 scripts/train_lab.py --dry-run --set rewards.tracking_lin_vel=2.0
```

## 开始训练与继续训练

默认实验使用 1024 个并行环境和 3000 次更新；该迭代数是训练预算，不是收敛保证。显存不足时先减小 `--num-envs`。

```bash
cd /root/go2-himloco

# 从随机网络开始训练，不加载任何现成策略。
bash scripts/run_lab.sh scripts/train_lab.py \
  --headless \
  --config configs/stairs.yaml \
  --num-envs 1024 \
  --max-iterations 3000 \
  --run-name stairs_baseline
```

仅当明确传入 `--resume` 时继续一个已有训练。将下例路径替换为本项目在 `logs/` 下生成的实际 checkpoint：

```bash
bash scripts/run_lab.sh scripts/train_lab.py \
  --headless \
  --config "/root/go2-himloco/logs/<训练目录>/settings.yaml" \
  --num-envs 1024 \
  --max-iterations 1000 \
  --run-name stairs_continue \
  --resume "/root/go2-himloco/logs/<训练目录>/model_<编号>.pt"
```

继续训练时保持观测维度、动作定义及网络结构一致。对奖励或楼梯难度做修改时，使用新的 `--run-name` 保存实验，记录变更，便于比较。原部署仓库的策略具有不同接口，不能作为这里的恢复点。

## 调整奖励和楼梯课程

直接修改 `configs/stairs.yaml`，或用重复的 `--set key=value` 为单次实验覆盖参数。未知配置键会报错，避免拼写错误被静默忽略。

```bash
bash scripts/run_lab.sh scripts/train_lab.py \
  --headless \
  --num-envs 1024 \
  --max-iterations 3000 \
  --run-name stairs_tracking2 \
  --set rewards.tracking_lin_vel=2.0 \
  --set rewards.action_rate=-0.02 \
  --set terrain.height_max=0.12
```

配置使用米、秒、弧度。正权重奖励期望行为，负权重惩罚代价；改变一个权重后，需要同时观察对应原始指标与训练稳定性。

| 配置键 | 默认值 | 调整作用 |
| --- | ---: | --- |
| `rewards.tracking_lin_vel` | `1.5` | 跟踪前进速度命令 |
| `rewards.tracking_ang_vel` | `0.5` | 跟踪角速度命令 |
| `rewards.forward_progress` | `0.3` | 鼓励沿楼梯方向推进 |
| `rewards.success` | `5.0` | 到达楼梯目标的奖励 |
| `rewards.failure` | `-2.0` | 失败惩罚 |
| `rewards.orientation` | `-0.15` | 限制机身倾斜；过强可能抑制爬阶所需姿态 |
| `rewards.action_rate` | `-0.01` | 限制连续动作变化，减少抖动 |
| `rewards.joint_power` | `-0.00002` | 限制关节功率代价 |
| `rewards.undesired_contacts` | `-0.5` | 惩罚非足端的异常接触 |
| `rewards.foot_slip` | `-0.1` | 减少支撑期足端滑动 |
| `rewards.foot_clearance` | `-0.1` | 约束摆动足的抬脚高度误差 |
| `targets.foot_clearance` | `0.08` | 摆动足抬脚目标，单位米 |
| `targets.base_height` | `0.30` | 机身相对地形高度目标，单位米 |

默认楼梯为 8 级、踏面深度 0.32 m，台阶高度在课程范围 0–0.18 m 内变化。默认 `terrain.rows=10`、`terrain.cols=4`、`terrain.initial_level=1`，启用 `terrain.curriculum=true`。机器人从楼梯下方进入，经过台阶到达顶部平台；楼梯几何和任务进度采用同一组参数。

`initial_level` 是初始地形行的上限，环境在第 0 行至该行之间随机分配；各行内的难度也会采样。高度范围代表课程覆盖区间，不保证实际某一块地形恰好为 0 cm 或 18 cm，也不表示训练已经到达最高难度。需要验证指定高度时，将 `height_min` 和 `height_max` 同时设为该高度。

建议先保留基线配置跑通训练，再一次改变少量参数。需要降低起步难度时，可降低 `terrain.height_max`；需要单独验证平地步行时，可同时设置 `terrain.height_min=0.0` 和 `terrain.height_max=0.0`。调高楼梯前先确认当前难度下的通过率和跌倒情况，单纯提高总奖励不等于学会爬梯。

其他常用参数包括：

- `commands.forward_min/forward_max`：前进速度范围，默认 0.3–0.6 m/s。
- `control.stiffness/damping`：关节 PD 控制参数，默认 40.0/1.0。
- `control.action_scale`：动作到关节位置偏移的缩放，默认 0.25 rad。
- `randomization.enabled`：域随机化开关；摩擦与负载范围也在该配置块中。
- `training.learning_rate`、`entropy_coef`、`num_steps_per_env`：PPO 学习率、探索强度与采样长度。

## 仿真回放与日志

### 调整步态自然度

`configs/stairs_natural.yaml` 是可选的步态改进配置，原始 `stairs.yaml` 以及旧模型保存的配置保持原有行为。新配置保留 45×6 维 HIM 历史观测、12 维关节目标动作和原来的执行限幅，加入以下约束：

- `rewards.action_smoothness`：动作二阶差分，减少目标角度来回突变；回合重置后的前两步不计入。
- `roll_orientation`、`hip_deviation`、`joint_deviation`、`joint_vel`：约束侧倾、外展和夸张腿部动作，同时允许爬楼所需的俯仰。
- `feet_air_time`、`feet_stumble`、`foot_slip`：使用正确的落地前离地时间，约束撞阶沿和支撑脚滑移，不强制固定步态相位。
- `training.min_noise_std` / `max_noise_std`：限制训练探索噪声；`action_mean_penalty_coef` 对超过执行限幅的策略均值施加惩罚。日志中的 `Raw action clip fraction` 是随机训练动作的截断比例，与确定性评估指标区分。

这些权重是需通过仿真对照验证的起始值，不能保证步态效果；改变奖励不会改变已经保存的模型。使用本项目自己的 checkpoint 微调：

```bash
GO2_PYTHON=/opt/conda/envs/isaaclab/bin/python bash scripts/run_lab.sh scripts/train_lab.py \
  --headless --config configs/stairs_natural.yaml \
  --warm-start /root/go2-himloco/logs/<训练目录>/model_<编号>.pt \
  --num-envs 1024 --max-iterations 500 --run-name stairs_natural_refinement
```

`--warm-start` 只加载本项目模型权重，重置优化器、更新计数和探索噪声；`--resume` 恢复模型与优化器继续原训练，两者不能同时使用。每次输出到新目录，不覆盖原模型。未指定这两个选项时仍从随机权重训练。

用相同场景分别评估原模型和候选模型；每个环境只计第一个完整回合，防止失败后快速重试改变分母。评估时接触力历史统一覆盖完整控制周期（当前为 4 个物理步），使基线和候选模型使用相同的机身碰撞判定窗口；旧的 3 步历史评估结果需要重新测量后再比较：

```bash
GO2_PYTHON=/opt/conda/envs/isaaclab/bin/python bash scripts/run_lab.sh scripts/play_lab.py \
  --headless --checkpoint /root/go2-himloco/logs/<训练目录>/model_<编号>.pt \
  --num-envs 64 --steps 1500 --episodes-per-env 1 --gait-metrics \
  --set seed=42 --set terrain.rows=1 --set terrain.cols=4 \
  --set terrain.initial_level=0 --set terrain.height_min=0.18 --set terrain.height_max=0.18 \
  --set commands.forward_min=0.4 --set commands.forward_max=0.4 \
  --output validation/natural_gait/<模型名称>_18cm_seed42.json
```

诊断包含独立通过率、未完成回合数、动作截断比例、一阶/二阶动作差分、机身姿态与角速度、足端滑移和机械功率。动作差分排除跨回合跳变；同时检查成功数，避免把提前摔倒造成的低运动量误认为改善。机械功率不是电功耗，训练采样成功率也不是此独立通过率。

参考依据：[原版 HIMLoco 的二阶动作平滑](https://github.com/InternRobotics/HIMLoco/blob/ef289acaa62795009363b7b819c9186690630441/legged_gym/legged_gym/envs/base/legged_robot.py#L1163)、[Go2 楼梯开源配置](https://github.com/saifahmadgit/go2-sim2real-locomotion-rl/blob/0830f3e3582acd5c9b3a5c09186d4f028f54d9dd/examples/locomotion/final/go2_train_stair.py#L310)、[实机/仿真参考视频](https://www.youtube.com/watch?v=m1GyvmJ9dBw)。后者使用 Genesis、15 cm 楼梯和带刚度控制的 16 维动作；本项目借鉴其姿态/接触质量思路，不加载它的策略或照搬动作接口。视频作为定性参考，未从视频提取关节轨迹或训练 AMP。

### 针对腿部过度外展的惩罚

`configs/stairs_lateral.yaml` 在前一版自然步态配置上增加两个软约束。它们读取实际关节角和足端位置，不改变推理限幅、默认关节角或 HIM 接口。设腿顺序为 FL、FR、RL、RR，左右符号为 `s=(+1,-1,+1,-1)`，`q_i` 是实际髋角，`u_i=s_i*y_i` 是足端在完整机身坐标系中的外向距离，`[x]+ = max(x,0)`：

- `hip_abduction = Σ [s_i*q_i - 0.30]+²`，权重 `-2.0`。0.30 rad（约17.2°）以内不额外惩罚，仅针对过度外展。
- `feet_lateral = Σ ([u_i - 0.22]+² + [0.10 - u_i]+²)`，权重 `-30.0`。足端在每侧10–22 cm走廊内可自由变化，外侧约束限制劈腿，内侧约束避免穿过身体中线。

位置通过根刚体的完整 wxyz 四元数逆旋转取得，不能用世界 Y 坐标或两足三维距离替代：转向、机身俯仰和前后足处于不同台阶，不应被当成侧向伸展。支撑与摆动阶段均约束，允许足端沿前后方向摆动和抬高。上述阈值是本项目的可调设计值，不是从视频中测得的数值。

原仓库的[策略动作定义](https://github.com/wxz-521/go2-stairs-sim2real/blob/530a10fee4232317aef4343f2b85b6bad9cf7905/mujoco_sim/go2_policy_mujoco.py#L21)使用四髋默认0 rad、16维动作（含4维刚度）；其[MuJoCo微调奖励](https://github.com/wxz-521/go2-stairs-sim2real/blob/530a10fee4232317aef4343f2b85b6bad9cf7905/mujoco_sim/train_stairs_curriculum.py#L783)没有专门的髋外展或足宽项。这里参考其姿态定义和连续动作控制，新增适配本工程的惩罚；保留现有默认髋角±0.1 rad，避免改变旧模型的观测与动作偏置。[作者视频约0:46–0:56](https://www.youtube.com/watch?v=nrwN8KrsD2c&t=46s)中的前后足分阶、躯干上俯作为定性参考，不做视频轨迹模仿或AMP训练。

常用调参入口：

| 配置 | 初始值 | 作用 |
| --- | ---: | --- |
| `rewards.hip_abduction` | -2.0 | 超额髋外展的惩罚权重 |
| `targets.hip_abduction_max` | 0.30 rad | 允许的外展角死区 |
| `rewards.feet_lateral` | -30.0 | 足端横向越界惩罚权重 |
| `targets.feet_lateral_min` | 0.10 m | 每侧足端内侧边界 |
| `targets.feet_lateral_max` | 0.22 m | 每侧足端外侧边界 |

独立评估额外输出每条腿的实际髋外展均值、RMS、超限比例，前后足宽、足端外越和内越比例；同时给出成功首回合的对应统计。应结合通过率判断，不能把跌倒后脚不动当成步态改善。旧配置中新奖励默认为0，重新回放旧模型不会自动改变其步态。

将路径替换为训练脚本输出的实际 checkpoint：

```bash
cd /root/go2-himloco
bash scripts/run_lab.sh scripts/play_lab.py \
  --checkpoint "/root/go2-himloco/logs/<训练目录>/model_<编号>.pt" \
  --headless \
  --steps 1000
```

无窗口回放用于检查策略能否加载并连续运行；有图形会话时移除 `--headless` 查看动作。回放测试应覆盖多个初始状态和楼梯难度，避免只观察一次成功轨迹。

在图形桌面中增加 `--interactive --num-envs 1` 可持续查看单只机器狗，镜头跟随机器人，回放按实时速度限速；关闭仿真窗口后退出。此模式忽略 `--steps`，也不写评估报告。服务器当前图形会话为 `DISPLAY=:20`，X 授权文件为 `/home/ubuntu/.Xauthority`。

例如，固定在 16 cm 楼梯上评估自己的策略：

```bash
bash scripts/run_lab.sh scripts/play_lab.py \
  --checkpoint "/root/go2-himloco/logs/<训练目录>/model_<编号>.pt" \
  --headless --steps 3000 \
  --set terrain.height_min=0.16 \
  --set terrain.height_max=0.16 \
  --set terrain.initial_level=0
```

每次训练创建 `logs/<北京时间戳>_<run-name>/`。其中保存实际使用的 `settings.yaml`、环境配置 `env.yaml`、算法配置 `agent.json`、来源和初始化记录 `manifest.json`、运行时检查 `runtime_check.json`、TensorBoard 事件文件及模型 checkpoint；正常完成后另写入 `completed.json`。训练入口会打印 `GO2_RUN_DIR`，便于找到本次输出。

回放从 checkpoint 所在目录读取保存的训练配置，也支持重复的 `--set` 覆盖参数。评估结果写入同目录的 `model_<编号>_evaluation.json`，包含回合数、成功、失败和超时统计。没有结束任何回合时，通过率为 `null`，不能解释为评估成功。

若运行 TensorBoard，可在安装了它的 Python 环境中执行：

```bash
tensorboard --logdir /root/go2-himloco/logs --host 127.0.0.1 --port 6006
```

重点查看速度跟踪、通过率/失败情况、各奖励分项、策略损失、价值损失和估计器误差。最终是否学会爬楼梯，以独立回放和重复评估为准。

## 稳定性、速度和独立平地控制

`configs/stairs_stable_fast.yaml` 是从上一轮防外展模型继续微调的独立实验配置，预算为 **1024 个环境、600 次 PPO 更新**。前300次更新将速度采样范围从0.35–0.50 m/s线性提升至0.45–0.65 m/s，后300次保持最终范围，覆盖0.6 m/s目标。课程按本次仿真的控制步数计算（24步/更新），不使用旧 checkpoint 的轮次，也不会由回合长度或成功数提前推进；命令在正常重采样时更新。回放与独立评估关闭速度课程，使用明确指定的固定命令。命令速度是目标值，实际爬楼速度必须另行测量。

新配置保留原动作尺度、默认关节角和HIM观测定义，增加以下约束；权重与阈值是待验证的设计值，不是效果保证：

| 配置 | 新值 | 意图 |
| --- | ---: | --- |
| `rewards.roll_orientation` | −3.0 | 抑制左右倾斜，允许爬阶俯仰 |
| `rewards.roll_rate` | −0.25 | 惩罚机身x轴角速度，减少横滚摆动 |
| `rewards.lateral_velocity` | −0.4 | 惩罚机身y轴平移速度，减少侧向滑动 |
| `rewards.foot_slip` | −0.6 | 减少触地足端滑移 |
| `rewards.tracking_lin_vel`、`forward_progress` | 2.0、0.8 | 加强速度跟踪与实际向前推进 |
| `rewards.hip_abduction`、`targets.hip_abduction_max` | −3.0、0.22 rad | 继续压缩过大的外展角 |
| `rewards.feet_lateral` | −40.0 | 约束足端横向伸展 |
| `targets.feet_lateral_min/max` | 0.12/0.205 m | 每侧保留12–20.5 cm软走廊 |
| `targets.feet_lateral_inner_weight` | 2.0 | 加强内侧越界惩罚，防止减小外展时后脚过度内收 |

足端惩罚为 `Σ([u_i−0.205]+² + 2*[0.12−u_i]+²)`，仍使用完整机身坐标系中的足端位置。没有强制四腿角度一致，也没有锁死髋关节。全姿态和俯仰角速度权重维持前一轮较弱约束，新增稳定性项主要针对横滚与侧滑；前后方向摆腿、抬脚与必要俯仰保留。默认 `stairs.yaml` 中新增奖励权重和速度课程均关闭，旧配置行为保持兼容。

可复现的训练命令如下。`--warm-start` 重置优化器与更新计数，只加载自己的楼梯模型权重；输出新实验目录，不覆盖来源模型：

```bash
cd /root/go2-himloco
GO2_PYTHON=/opt/conda/envs/isaaclab/bin/python bash scripts/run_lab.sh scripts/train_lab.py \
  --headless --config configs/stairs_stable_fast.yaml \
  --warm-start logs/20261102_175509_stairs_lateral_refinement/model_400.pt \
  --num-envs 1024 --max-iterations 600 --run-name stairs_stable_fast
```

### 原仓库平地策略与切换

`scripts/play_lab.py --controller` 提供三个控制模式，默认仍为 `stairs`，便于复现旧实验：

- `stairs`：仅使用 `--checkpoint` 指定的本项目HIM楼梯策略。
- `walk`：仅执行 `--walk-checkpoint` 指定的原仓库平地策略，适合纯平地检查。
- `hybrid`：平地执行原 `walk.pt`，发现楼梯后渐变切换为HIM楼梯策略，离开楼梯后切回平地策略。

原策略来自 [wxz-521/go2-stairs-sim2real 的固定版本](https://github.com/wxz-521/go2-stairs-sim2real/tree/530a10fee4232317aef4343f2b85b6bad9cf7905)，文件为 `example/go2/low_level/final/walk.pt`。本地保存在 `assets/policies/upstream_walk/walk.pt`，权重未修改，SHA-256为 `65367faebe875faddd8513e12fc3211b9185857e59b3e5cabbef454d46976dfc`。来源、接口及校验值保存在同目录 `manifest.json`，原 BSD-3-Clause 许可证保存在 `LICENSE`。未加载上游 `stairs.pt`。

适配器按原接口构造49维观测并保留全部16维动作：12维关节目标和4维逐腿刚度。关节顺序按名称从原FR、FL、RR、RL映射到当前模型；保留50 Hz控制频率、0.25 rad动作缩放、每步0.1 rad目标变化限制、20–60的动态刚度、`Kd=0.2*sqrt(Kp)`阻尼公式和原关节力矩限制。没有将这份策略当成12维HIM模型加载，也没有丢弃刚度输出。两种策略分别维护所需的历史状态；切换时渐变混合实际关节目标及PD参数。

`hybrid` 使用独立的 `transition_scanner` 仿真高度扫描，默认网格范围为2.4×1.0 m，检查机身偏航坐标系前后 `x=[−0.45,1.10] m`、左右 `|y|≤0.30 m` 区域。原 `height_scanner` 和HIM Critic的187条高度射线保持不变；扩大切换感知范围不会改变已有模型的观测维度。`--gate-lookahead` 可在0.4–2.0 m内调整前视距离，默认1.10 m，独立扫描器范围随之配置。

高度跨度大于5 cm进入楼梯模式，小于2.5 cm并持续0.4 s后退出，进入与退出均保留0.3 s渐变混合。有效射线需覆盖前、后、左、右区域；扫描无效时保持之前模式。进入与退出阈值不同，避免边界反复切换。它没有用楼梯编号、预设位置或回合成败决定模式；**仿真高度扫描不等于真实机器人已有楼梯感知能力**，实机接入仍需要传感器、标定、延迟和异常处理验证。

在可用的图形桌面打开自动切换回放；将 `--checkpoint` 替换为独立评估后选定的本项目模型。此例仍使用上一轮防外展模型，不表示新模型已验收：

```bash
cd /root/go2-himloco
DISPLAY=:20 XAUTHORITY=/home/ubuntu/.Xauthority \
GO2_PYTHON=/opt/conda/envs/isaaclab/bin/python bash scripts/run_lab.sh scripts/play_lab.py \
  --checkpoint logs/20261102_175509_stairs_lateral_refinement/model_400.pt \
  --controller hybrid --walk-checkpoint assets/policies/upstream_walk/walk.pt \
  --gate-lookahead 1.1 \
  --interactive --view front --num-envs 1 \
  --set terrain.rows=1 --set terrain.cols=1 --set terrain.initial_level=0 \
  --set terrain.height_min=0.18 --set terrain.height_max=0.18 \
  --set commands.forward_min=0.6 --set commands.forward_max=0.6
```

纯平地检查将上例改为 `--controller walk`，同时把 `terrain.height_min/max` 均设为 `0.0`。此模式不使用楼梯策略输出控制机器人；当前脚本仍要求 `--checkpoint`，用于读取本工程的模型/环境配置并初始化公共回放链路。`--walk-checkpoint` 默认即为上述固定版本文件，不传该参数也会使用它。

### 用统一条件验证速度和稳定性

`scripts/evaluate_stability.py` 为每种条件启动独立仿真，只统计每个环境的首个完整回合，统一为64环境、25 s回合上限、关闭域随机化与观测噪声。旧模型与候选模型的诊断阈值统一为髋0.22 rad、足端12–20.5 cm，因此结果不能直接与上一轮使用0.30 rad/10–22 cm阈值的超限率比较。已有结果文件拒绝覆盖；重跑请更换 `--label` 或 `--output-dir`。

```bash
cd /root/go2-himloco
# 旧模型与新候选均测试0.4、0.6 m/s命令，固定18 cm楼梯、两个种子。
python3 scripts/evaluate_stability.py \
  --checkpoint logs/20261102_175509_stairs_lateral_refinement/model_400.pt \
  --label lateral400 --controller stairs --speeds 0.4 0.6 --seeds 42 43 \
  --height 0.18 --num-envs 64 --output-dir validation/stable_fast

python3 scripts/evaluate_stability.py \
  --checkpoint "logs/<新训练目录>/model_<候选编号>.pt" \
  --label stable_candidate --controller stairs --speeds 0.4 0.6 --seeds 42 43 \
  --height 0.18 --num-envs 64 --output-dir validation/stable_fast

# 平地控制独立检查，之后用hybrid检查完整进场、爬楼和退出切换。
python3 scripts/evaluate_stability.py \
  --checkpoint logs/20261102_175509_stairs_lateral_refinement/model_400.pt \
  --label upstream_walk --controller walk --speeds 0.6 --seeds 42 43 \
  --height 0.0 --num-envs 64 --output-dir validation/stable_fast

python3 scripts/evaluate_stability.py \
  --checkpoint "logs/<新训练目录>/model_<候选编号>.pt" \
  --label stable_hybrid --controller hybrid --speeds 0.6 --seeds 42 43 \
  --gate-lookahead 1.1 \
  --height 0.18 --num-envs 64 --output-dir validation/stable_fast

# 增加0.3 m平地进场距离，检查不同接管步相，保持同一楼梯与速度。
python3 scripts/evaluate_stability.py \
  --checkpoint "logs/<新训练目录>/model_<候选编号>.pt" \
  --label stable_hybrid_long_approach --controller hybrid --speeds 0.6 --seeds 44 \
  --gate-lookahead 1.1 --spawn-x-offset -0.3 \
  --height 0.18 --num-envs 64 --output-dir validation/stable_fast
```

`--spawn-x-offset` 同时适用于 `play_lab.py` 和 `evaluate_stability.py`，允许范围为−0.3至0.3 m，默认0。它平移原来的 `[-0.1,0.1] m` 初始x位置采样范围；设为−0.3后为 `[-0.4,−0.2] m`，使机器人在接管前多走一段平地，用于检查进场步相敏感性。此参数不改变楼梯几何、成功位置或策略的观测定义。不同前视距离/进场位置的实验应使用不同 `--label`，输出JSON会记录实际设置。

同时查看通过率、失败和超时数、楼梯区间实际速度/耗时、横滚姿态与角速度、侧向速度、足端滑移、髋外展与逐腿内收。只报告成功回合的运动质量时，应同时给出成功回合数量，防止更慢或更不稳定的回合被筛掉。`hybrid` 报告还包含平地/混合/楼梯控制采样数、模式转换次数和每回合是否完成“平地→楼梯→平地”；完成切换序列本身不等于成功爬完楼梯。

## 模型来源及物理差异

指定仓库版本 `530a10fee4232317aef4343f2b85b6bad9cf7905` 的 `mujoco_sim/setup.sh` 锁定了 Unitree 官方模型版本 `ae6a8403e272733e9996ef59990880330496177f`。本项目从该模型生成 URDF，保留原始 MJCF 和所有网格的 SHA-256，附带 Unitree 许可证。没有把其他 Go2 模型替代成指定仓库的模型。

模型总质量为 15.206408 kg。为独立读取足端接触，将原 calf 刚体上的足部几何移入已有固定足端，并重新分配 calf/foot 质量与惯量；组合质量、质心和惯量保持不变。导入时保留固定足端刚体。原模型没有速度限值，URDF 增加的 30 rad/s 是明确记录的配置假设。

MuJoCo 与 PhysX 的接触、摩擦和求解器并不完全一致。模型转换保留几何和惯性属性，接触参数由 Isaac Lab 环境单独配置。详细说明见 [`assets/go2/README.md`](assets/go2/README.md) 和 [`assets/go2/PROVENANCE.json`](assets/go2/PROVENANCE.json)。本工程当前面向仿真训练，不包含真实机器人的部署步骤。

## 验证记录

2026-11-02 已在服务器验证以下环境：Ubuntu 24.04、RTX 4090（24564 MiB）、NVIDIA 驱动 580.126.09、Python 3.11.15、Isaac Sim 5.1.0.0、Isaac Lab 2.3.2.post1、PyTorch 2.7/CUDA 12.8。

已完成 32 个环境 × 3 次 PPO 更新，以及 1024 个环境 × 10 次 PPO 更新；后者采集 245760 个环境转移样本。运行时检查通过 16 个零动作仿真步，确认 Actor 观测 270 维、Critic 观测 235 维、动作 12 维，根刚体 z 坐标均值约 0.290 m。这些结果验证模型导入、仿真步进和训练链路，未达到或宣称爬楼梯策略已经收敛。

模型转换的 7 项 CPU 测试已通过，包括关节轴/限位、总质量、惯量正定性、组合刚体惯量守恒、足端碰撞分离、网格校验和离线重建一致性。

首次环境验收时服务器 **47 项测试通过**。自己的 `model_10.pt` 成功恢复并完成到第 12 次更新，运行时额外核对了 12 个动作关节与观测的顺序。使用 16 个环境回放 1500 步，完成 16 个回合：0 次成功、0 次摔倒/越界终止、16 次超时。这些是早期训练链路短测记录，不代表后续训练模型的表现。

初始验收记录位于 [`validation/summary.json`](validation/summary.json)、[`validation/server_environment.json`](validation/server_environment.json)、[`validation/tests.txt`](validation/tests.txt) 和同目录的训练/回放日志。服务器短测 checkpoint 位于 `/root/go2-himloco/logs/20261102_153359_scale_check/model_10.pt`。

后续已完成 3000 次更新的基线训练，目录为 `/root/go2-himloco/logs/20261102_154245_stairs_baseline`。步态改进新增了奖励边界、探索噪声、PPO 动作一致性及回合统计测试，完整 **63 项测试通过**，见 [`validation/natural_gait/tests.txt`](validation/natural_gait/tests.txt)。固定 18 cm、0.4 m/s、seed=42、64 个首回合、统一接触判定窗口的独立评估中，基线 `model_1000.pt` 通过 55/64，`model_3000.pt` 通过 42/64；更新次数更多不保证更优。对应原始 JSON 与实现来源记录保存在 [`validation/natural_gait/`](validation/natural_gait/)。

2026-11-02 的步态微调从自己的基线 `model_1000.pt` 开始，使用 `configs/stairs_natural.yaml` 完成 1024 环境 × 500 次更新，全部模型参数为有限数值。根据 seed=42 的对照选择 **第 400 次更新**的检查点，再用 seed=43 复核；最终第 500 次更新的模型在 seed=42 中只有 47/64 次通过，因此没有选用。

| 模型 | seed=42 通过数 | seed=43 通过数 | 合计通过率 | 动作二阶差分 RMS / rad | 支撑足滑移 RMS / m/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原基线 3000 | 42/64 | 40/64 | 64.1% | 0.2983 | 0.2688 |
| 微调起点：原基线 1000 | 55/64 | 54/64 | 85.2% | 0.1273 | 0.1890 |
| 选定微调 400 | 60/64 | 62/64 | 95.3% | 0.0709 | 0.1561 |

运动质量指标仅统计成功首回合，按有效元素数汇总；动作二阶差分未除以控制周期平方，不是角加速度。两组使用固定 18 cm 台阶、32 cm 踏面、8 级楼梯、0.4 m/s 命令、25 s 回合上限，均关闭观测噪声和域随机化，保留相同的初始姿态采样。seed=42 用于模型选择，seed=43 是选择后的复核；这不是实机结果，也不证明不同楼梯或扰动下的泛化。原始数据和汇总见 [`validation/natural_gait/comparison.json`](validation/natural_gait/comparison.json)。

在上述模型基础上，随后使用 `configs/stairs_lateral.yaml` 完成400次防外展微调，全部模型参数为有限值，新增坐标、左右符号、惩罚梯度、逐腿与楼梯区间统计后完整73项测试通过。第100次检查点固定18cm仅5/64通过，没有采用；第300次为63/64，但前足仍较宽。第400次在seed42通过59/64、seed43通过63/64，综合采用第400次。

| 成功回合的爬楼区间姿态指标 | 前一版自然步态模型 | 防外展模型 |
| --- | ---: | ---: |
| 前足平均横向间距 | 56.7 cm | 44.5 cm |
| 后足平均横向间距 | 39.5 cm | 31.6 cm |
| 髋外展超过0.30 rad的采样比例 | 78.5% | 38.7% |
| 足端外向距离超过22 cm的采样比例 | 61.6% | 27.9% |
| 足端内侧越界（距中线不足10 cm）采样比例 | 0.009% | 3.41% |
| 两组固定18cm测试合计通过率 | 122/128（95.3%） | 122/128（95.3%） |

这里的爬楼区间按基座中心相对环境原点的x坐标处于 `[1.2,3.76)` m筛选，排除了进场与顶部平台；足宽是完整机身坐标系中的横向投影。超限比例分母是有效时刻×四腿，不是失败回合比例。内侧越界是低于10cm，不等于足端穿过身体中线。模型仍允许短时软阈值越界，没有把腿锁死。

这轮主要改善侧展，并非所有运动质量指标同时提升：成功回合全程的侧倾RMS由0.109增至0.135 rad（约6.2°→7.7°），支撑足滑移RMS由0.156增至0.163 m/s；动作二阶差分RMS由0.0709降至0.0609 rad。通过率在这两组仿真中相同，不代表所有地形与扰动下等价。原始数据、逐腿统计与汇总见 [`validation/lateral_gait/comparison.json`](validation/lateral_gait/comparison.json)，实现与示教来源见 [`validation/lateral_gait/implementation_provenance.json`](validation/lateral_gait/implementation_provenance.json)。

上一轮选定模型为 `/root/go2-himloco/logs/20261102_175509_stairs_lateral_refinement/model_400.pt`，同目录保存训练配置、来源记录及最终完成标记；本地副本位于 `artifacts/20261102_175509_stairs_lateral_refinement/`。查看该轮实际动作（正面视角便于检查左右伸展）：

```bash
cd /root/go2-himloco
DISPLAY=:20 XAUTHORITY=/home/ubuntu/.Xauthority \
GO2_PYTHON=/opt/conda/envs/isaaclab/bin/python bash scripts/run_lab.sh scripts/play_lab.py \
  --checkpoint logs/20261102_175509_stairs_lateral_refinement/model_400.pt \
  --interactive --view front --num-envs 1 \
  --set terrain.rows=1 --set terrain.cols=1 --set terrain.initial_level=0 \
  --set terrain.height_min=0.18 --set terrain.height_max=0.18 \
  --set commands.forward_min=0.4 --set commands.forward_max=0.4
```

### 2026-11-02 稳定性、提速与平地策略切换验收

当前选定楼梯模型为 `logs/20261102_183927_stairs_stable_fast/model_600.pt`。从上一轮自己的防外展 `model_400.pt` 热启动，使用 `configs/stairs_stable_fast.yaml` 完成 **600 次 PPO 更新、1024 个环境、14,745,600 个环境转移样本**，训练计时约15分44秒。完成标记确认所有模型张量为有限值，本地备份 SHA-256 与服务器一致。这里的600是本次微调轮数。

增加横滚角速度与侧向速度惩罚，加强侧倾、滑移和髋外展约束，保留上楼所需的俯仰自由度；前300次更新逐渐把速度采样区间提高到0.45–0.65 m/s。平地使用原仓库未经修改的 `walk.pt`，接近楼梯时提前切换至自己的 HIM 策略，上楼后返回 walk。选定运行命令必须显式包含 `--controller hybrid --gate-lookahead 1.1`；默认 `stairs` 模式用于保留单策略对照。直接构造控制器的代码也应显式传入 `gate_lookahead_m=1.1`。

下表为固定18 cm台阶、32 cm踏面、8级楼梯、相同 **0.6 m/s指令**、seed42/43各64个首回合的对照。观测噪声和域随机化关闭，保留初始姿态采样。速度与姿态仅统计成功首回合，汇总按实际有效样本数加权。

| 指标 | 上一轮防外展模型（仅楼梯策略） | 当前600模型＋原仓库walk（组合控制） |
| --- | ---: | ---: |
| 通过数 | 121/128（94.5%） | 128/128（100%） |
| 实测水平上楼速度 | 0.316 m/s | 0.429 m/s |
| 完整楼梯区间平均耗时 | 8.151 s | 5.974 s |
| 楼梯区间横滚 RMS | 6.72° | 3.44° |
| 四腿髋外展正分量均值 | 13.08° | 10.48° |
| 支撑足滑移 RMS | 0.141 m/s | 0.133 m/s |
| 足端距身体中线不足12 cm的比例 | 9.56% | 2.55% |

相同指令下，实测上楼速度提高约36%，左右侧倾减少约49%，平均髋外展减少约20%。**0.6 m/s是指令目标，实测水平上楼速度约0.43 m/s，尚未达到实测0.6 m/s。** 髋外展惩罚是软约束，前腿仍可超过0.22 rad，不能解释成硬限幅或步态已经完全自然。

额外复核：seed44向后平移出生位置0.3 m、增加平地进场距离后，组合控制通过64/64，全部完成“walk→stairs→walk”；默认进场的128个回合全部在终点恢复walk，其中124个记录完整三阶段序列，另外4个因出生已接近检测区而没有初始纯walk采样。纯平地seed44通过64/64，52,742个有效控制采样全部使用walk，没有楼梯或混合控制采样；原仓库walk在此仿真中的实测区间速度约0.281 m/s。当前模型仅使用楼梯控制时，0.6 m/s两组测试也通过128/128，速度约0.434 m/s。

切换前视距离0.7 m的初始组合测试仅116/128，0.8 m在seed42为61/64，主要问题是在第一级台阶前超时。保留这些数据，采用1.1 m前视提前接管后复核上述结果。seed42参与模型/接管参数选择，seed43和44用于开发复核；这些结果不等于训练采样成功率，也不代表不同地形、扰动或实机的独立泛化验收。

完整114项测试通过。原始首回合报告与加权汇总在 [`validation/stable_fast/comparison.json`](validation/stable_fast/comparison.json)，来源、参数、模型校验和与限制在 [`validation/stable_fast/implementation_provenance.json`](validation/stable_fast/implementation_provenance.json)，模型本地副本位于 `artifacts/20261102_183927_stairs_stable_fast/model_600.pt`。

```bash
cd /root/go2-himloco
DISPLAY=:20 XAUTHORITY=/home/ubuntu/.Xauthority \
GO2_PYTHON=/opt/conda/envs/isaaclab/bin/python bash scripts/run_lab.sh scripts/play_lab.py \
  --checkpoint logs/20261102_183927_stairs_stable_fast/model_600.pt \
  --controller hybrid --gate-lookahead 1.1 \
  --interactive --view front --num-envs 1 \
  --set terrain.rows=1 --set terrain.cols=1 --set terrain.initial_level=0 \
  --set terrain.height_min=0.18 --set terrain.height_max=0.18 \
  --set commands.forward_min=0.6 --set commands.forward_max=0.6
```
