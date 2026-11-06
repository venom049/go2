# Go2 楼梯策略键盘控制

默认 `Stairs` 状态仍使用固定 15 cm 连续上下楼策略：

- 模型：`config/policy/stairs/v0/exported/policy.onnx`
- 输入：49 维本体状态、速度命令和上一帧动作
- 输出：12 个关节位置目标和 4 个逐腿刚度动作
- 控制周期：20 ms

## 控制方式

键盘状态切换和速度预选：

1. `F`：`Passive -> FixStand`
2. 在 `FixStand` 中选择一次速度
3. `G`：以预选速度直接进入 `Stairs`
4. 第一次 `X`：`FixStand/Stairs -> LieDown`
5. 趴下完成后第二次 `X`：`LieDown -> Passive`

遥控器 `L2 + Up` 可进入固定站立，`L2 + B` 与第一次 `X` 一样只启动
受控趴下。为避免绕过速度预选，`Stairs` 只能由键盘 `G` 启动；进入
`Passive` 必须等待趴下完成后再按键盘 `X`。

在 `FixStand` 中预选前进速度：

| 按键 | 动作 |
|---|---|
| `W` / `Up` | 将预选速度增加 `0.1 m/s` |
| `S` / `Down` | 将预选速度减少 `0.1 m/s`，最低为 `0.5 m/s` |
| `1` 到 `4` | 直接预选 `0.5` 到 `0.8 m/s` |
| `G` | 固定站立完成且已预选速度后，启动楼梯策略 |
| `Space` / `0` | 进入 `Stairs` 后将本次速度命令置为零 |
| 第一次 `X` | 开始约 `2.5 s` 的受控趴下轨迹 |
| 第二次 `X` | 趴下完成后进入 `Passive` 阻尼模式 |

速度在进入 `Stairs` 前锁存，运行中不再调速。要更换速度，先完成两次
`X` 的趴下和阻尼流程，再按 `F` 进入固定站立并预选，然后按 `G`。
策略没有训练后退、横移或转向，部署端不会向它提供这些命令。

每次进入 `FixStand` 后都必须重新按一次速度键，未选择速度时 `G` 无效。
如果固定站立的两秒插值尚未完成，`G` 同样无效，终端会提示等待后重试。

默认模型从零速稳态切换到 `0.3-0.4 m/s` 时可能只前倾而不迈步；离线测试
显示 `0.5 m/s` 能可靠触发步态。因此部署端不再提供低于 `0.5 m/s` 的档位，
并在策略首次决策前完成速度预选。

## 模型适用边界

该模型训练使用固定 `15 cm` 阶高、物理尺寸 `40 cm` 踏深、每段六级的重复
上下楼地形。训练重置时机身中心距第一阶立面 `1.00 m`，初始前脚距第一阶
约 `0.822 m`，并不是贴阶起步；actor 输入也不含前方高度扫描，只能依靠
本体状态和上一帧动作维持楼梯步态。

使用当前部署入口直接给定 `0.5 m/s` 的离线贴阶对照中，机身中心距第一阶
`0.28 m`、前脚距第一阶约 `0.102 m`。32 个环境中 29 个在六级踏面上都
检测到连续稳定落脚，各级摆动脚相对目标踏面的典型余量约 `5-10 cm`。因此
当前权重在匹配尺寸的仿真中会持续抬腿，而不是只在第一步抬腿。实机首测时
建议前脚与第一阶保留约 `10 cm`，不要让脚或机身贴住立面；楼梯尺寸不匹配
时该结论不成立。

逐足验收结果保存在
`results/go2_saif_gait/model450_speed050_adjacent028.json`，可用
`scripts/evaluate_go2_saif_gait.py` 对后续 checkpoint 重复检查。

默认部署的 `model_450` 仍是上述 `15 cm` 权重。针对实测约 `15.5 cm` 的楼梯，
训练/回放环境已调整为固定 `16 cm`。新训练的机身中心到第一阶距离随机为
`0.25-0.35 m`，约等于前脚保留 `7-17 cm`；回放固定使用约 `10 cm` 前脚
间隙。

16 cm 候选保存在 `config/policy/stairs_16cm_kp20/v0`，不会覆盖默认模型。
它使用 16 cm 续训的 `model_500`，并把逐腿 Kp 下限从 `10` 提高到 `20`。
固定种子、`0.5 m/s`、前脚约 `10 cm` 的 32 环境测试为：六级完整通过
`32/32`、每一级至少两脚稳定落地 `29/32`、提前终止 `0/32`。机身距首级
`0.25 m` 和 `0.35 m` 两端测试均为 `16/16` 完成。Kp 下限 `30` 曾造成
`12/32` 提前终止，因此不得在实机侧继续随意提高增益。

候选验收结果保存在：

- `results/go2_saif_gait/model500_train16_eval16_speed050_adjacent028_seed42_kp20.json`
- `results/go2_saif_gait/model500_train16_eval16_speed050_distance025_seed42_kp20.json`
- `results/go2_saif_gait/model500_train16_eval16_speed050_distance035_seed42_kp20.json`

## 有线连接

推荐在通过网线连接机器狗的电脑上直接运行。当前电脑的有线网卡为
`enp7s0`（`192.168.123.100/24`）：

```bash
cd /home/fishros/unitree_rl_mjlab/deploy/robots/go2
./build/go2_ctrl --network=enp7s0 --check-lowcmd
./build/go2_ctrl --network=enp7s0
```

正式控制前先运行只读策略检查。该模式订阅真实 `LowState`，构造 49 维
observation 并执行 ONNX 推理，但不会创建 `LowCmd` 发布者，也不会发送
电机命令：

```bash
./build/go2_ctrl --network=enp7s0 --robot-print \
  --robot-print-steps=100 --stairs-speed=0.5
```

终端应显示 `No LowCmd publisher was created`，并打印重力方向、陀螺仪、
动作幅度和 Kp 范围。只有这些值正常，才继续启动正式控制器。

只有在现场确认要测试 16 cm 候选时，才使用下面的显式命令。它同时以 50 Hz
记录策略目标、实测关节角/速度以及 Kp/Kd，便于区分“策略没有继续抬腿”和
“策略给了目标但实机没有跟上”：

```bash
./build/go2_ctrl --network=enp7s0 \
  --stairs-policy=config/policy/stairs_16cm_kp20 \
  --stairs-trace=log/stairs_16cm_kp20.csv
```

不带 `--stairs-policy` 时仍加载默认 15 cm 模型；不带 `--stairs-trace` 时不会
生成关节跟踪 CSV。

首次或靠近障碍物实测必须加入自动趴下硬限时，避免 SSH/键盘延迟导致机器狗
持续前进：

```bash
LD_LIBRARY_PATH=/tmp/sdk2new_lib ./bin/go2_ctrl --network=eth0 \
  --stairs-speed=0.5 \
  --stairs-auto-lie-seconds=2.0 \
  --stairs-trace=log/realtest_short.csv
```

`--stairs-auto-lie-seconds` 只在进入 `Stairs` 后计时，到时自动切换到
`LieDown`，不依赖人工按 `Space` 或 `X`。如果未显式指定，默认不会自动趴下。
`--stairs-speed` 会同时固定真实运行和 `--robot-print` 使用的前进速度。

如果要做短距离上楼后站稳测试，可以不用人工抢按 `Space`，而是让控制器在
进入 `Stairs` 后自动把前进速度命令置零：

```bash
LD_LIBRARY_PATH=/tmp/sdk2new_lib ./bin/go2_ctrl --network=eth0 \
  --stairs-speed=0.5 \
  --stairs-auto-zero-seconds=4.5 \
  --stairs-auto-lie-seconds=8.0 \
  --stairs-trace=log/three_step_auto_zero.csv
```

`--stairs-auto-zero-seconds` 只把策略速度命令改成 0，并继续保持 `Stairs`
状态；`--stairs-auto-lie-seconds` 是额外保险，到时切换到受控趴下。三阶任务
应先在吊装或空旷环境用保守秒数试跑，再根据 CSV 和现场位置调整自动置零时间。

先执行只读检查。`LOWCMD_STATUS=ACTIVE` 表示原生运动控制或其他低层控制器
仍在发布，此时不能启动正式控制器；`LOWCMD_STATUS=IDLE` 才能继续。检查模式
不会创建发布者或发送电机命令。

机载运行作为备用方案。通过 SSH 启动时必须分配交互式终端：

```bash
ssh -tt unitree@192.168.123.18
cd /home/unitree/unitree_rl_mjlab_stairs_20260814/deploy/robots/go2
./bin/go2_ctrl --network=eth0
```

启动控制器之前，机器狗必须吊装并处于 `zero-torque` 状态，再通过遥控器
按 `L2 + R2` 进入 Debug Mode。

`Space` 只是零速度命令，不等价于硬件急停。第一次 `X` 或遥控器
`L2 + B` 只启动受控趴下；低姿态完成后第二次 `X` 才切换到仅有阻尼、
没有位置支撑的 `Passive` 状态。它们都不能替代实体急停和现场保护。
首次实机运行必须吊装，并先完成仿真部署验证。
