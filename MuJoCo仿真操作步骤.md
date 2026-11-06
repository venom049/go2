# 在 MuJoCo 中打开我们的 Go2 机器狗进行仿真

适用环境：本机 Linux 桌面，项目目录 `/home/venom/文档/ChatGPT/机器狗`。核对日期：2026-11-04。

下面复现项目里已经使用过的 **moe1000 策略 + 两级台阶 + 尽头墙体** 仿真。直接按第 1～3 步操作即可打开窗口，机器狗会自动开始运动。本流程在电脑上运行，不需要连接实机，也不需要启动 Isaac Lab 或 ROS。

## 1. 打开本机桌面的终端，进入项目

```bash
cd "/home/venom/文档/ChatGPT/机器狗"
```

后面的命令都在这个目录执行。窗口需要可用的桌面图形会话；如果使用没有桌面的 SSH 终端，先看第 5 节的无窗口仿真。

## 2. 启用项目已有 Python 环境

```bash
source .venv-deploy/bin/activate

python -c 'import sys, mujoco, onnxruntime, numpy, yaml; print("Python:", sys.executable); print("MuJoCo:", mujoco.__version__); print("ONNX Runtime:", onnxruntime.__version__)'
```

正常应看到 Python 路径指向本项目的 `.venv-deploy/bin/python`，MuJoCo 版本为 `3.3.7`，ONNX Runtime 为 `1.19.2`。本机已经装好这些依赖，检查通过后直接下一步。

如果找不到环境或提示缺少模块，使用第 7 节创建独立环境，然后继续第 3 步。

## 3. 启动我们的策略仿真窗口

完整复制下面这一段。每次运行都会生成新的结果目录，不要提前创建该目录。

```bash
GO2_SIM_OUT="validation/mujoco_manual_$(date +%Y%m%d_%H%M%S_%N)"

MUJOCO_GL=glfw python scripts/test_robogauge_two_steps_wall.py \
  --policy "_exports/go2-moe1000-20261104/exported/policy.onnx" \
  --config "_exports/go2-moe1000-20261104/params/deploy.yaml" \
  --robot-xml "upstream/robogauge/RoboGauge/resources/robots/go2/go2.xml" \
  --viewer \
  --hold-final \
  --episodes 0 \
  --vx 0.6 \
  --duration 25 \
  --command-mode stop_on_platform \
  --torque-limit-mode physical \
  --outdir "$GO2_SIM_OUT"
```

打开后，场景里有一只 Go2、两级各高 15 cm 的台阶，以及平台尽头的墙。楼梯宽 90 cm，第一踏面深 30 cm，第二层踏面连同平台总深 90 cm。初始化后，前脚碰撞球前缘距第一阶立面 50 cm。

机器狗自动执行前进速度命令，上平台后由脚本切换到零速度命令。回合结束后，窗口保留最后画面，此时物理仿真已经停止；这由 `--hold-final` 控制，不是程序卡死。**关闭仿真窗口即可退出，再次运行整段命令即可重播。**

### 这些参数分别控制什么

| 参数 | 作用 |
| --- | --- |
| `--policy`、`--config` | 配套加载我们导出的 moe1000 ONNX 策略与部署参数；省略时会加载脚本默认的上游原版策略 |
| `--robot-xml` | 加载与这套 MoE 策略仿真对应的 RoboGauge Go2 模型 |
| `--viewer` | 打开 MuJoCo 图形窗口，默认按实时速度运行 |
| `--hold-final` | 回合完成、保存结果后保留最终画面 |
| `--episodes 0` | 不增加随机扰动回合，仍运行一个固定初始状态的基准回合 |
| `--vx 0.6` | 前进速度命令为 0.6 m/s；实际运动速度由仿真结果决定 |
| `--duration 25` | 单回合最长 25 秒仿真时间；成功或失败可以提前结束；此脚本要求至少为 25 |
| `--command-mode stop_on_platform` | 使用仿真中的位置、接触信息判断登台并给零速命令；不是策略自主识墙 |
| `--torque-limit-mode physical` | 对 PD 请求力矩限幅：髋/大腿 23.7 N·m，小腿 45.43 N·m |
| `--outdir` | 保存本次场景、参数和结果，必须使用尚不存在的目录 |

这个入口没有实现 W/A/S/D 键盘驾驶，也没有把空格绑定为 Python 物理循环的暂停键。调整速度时，退出后修改 `--vx` 再运行，例如 `--vx 0.4`；改变速度后的效果需要重新观察。

双台阶脚本不支持 `--loop`，也不接收可调台阶尺寸参数。重播时重新执行上面的完整命令即可。

## 4. 查看仿真结果

关闭窗口、回到同一个终端后执行：

```bash
printf '本次结果目录：%s\n' "$GO2_SIM_OUT"
python -m json.tool "$GO2_SIM_OUT/summary.json"
```

主要文件如下：

| 文件 | 内容 |
| --- | --- |
| `summary.json` | 成功回合数、终止原因、墙体接触等汇总 |
| `episode_000.json` | 本次基准回合的详细记录 |
| `manifest.json` | 实际加载的模型路径、哈希、参数和场景说明 |
| `scene.xml` | 本次运行生成的完整 MuJoCo 场景 |

`status_counts` 中的 `success` 表示完成了仿真登台停稳任务。关节保护审计是另一个指标，任务成功不代表所有保护检查都通过。

## 5. 没有桌面时，先跑无窗口仿真

在第 1、2 步完成后执行。此命令不打开窗口，也不录制视频；`--no-realtime` 取消按真实时间等待，用于快速验证模型和策略能否运行。

```bash
GO2_HEADLESS_OUT="validation/mujoco_headless_$(date +%Y%m%d_%H%M%S_%N)"

MUJOCO_GL=egl python scripts/test_robogauge_two_steps_wall.py \
  --policy "_exports/go2-moe1000-20261104/exported/policy.onnx" \
  --config "_exports/go2-moe1000-20261104/params/deploy.yaml" \
  --robot-xml "upstream/robogauge/RoboGauge/resources/robots/go2/go2.xml" \
  --headless \
  --no-realtime \
  --episodes 0 \
  --vx 0.6 \
  --duration 25 \
  --command-mode stop_on_platform \
  --torque-limit-mode physical \
  --outdir "$GO2_HEADLESS_OUT"

python -m json.tool "$GO2_HEADLESS_OUT/summary.json"
```

终端出现 `ROBOGAUGE_TWO_STEP_WALL_COMPLETE` 表示评估运行结束；任务是否成功还要看后面的 `successes` 和 `status_counts`。

## 6. 如果只想打开模型查看

下面使用 MuJoCo 自带查看器打开项目保存的原始 Go2 MJCF。它只加载机器人模型，不加载 moe1000 策略，也不自动创建上述楼梯场景。

```bash
MUJOCO_GL=glfw python -m mujoco.viewer \
  --mjcf="/home/venom/文档/ChatGPT/机器狗/assets/go2/mjcf/go2.xml"
```

该原始模型没有地面和行走控制器，推进物理后不能据此判断站立或爬梯效果。要看机器狗按策略运动，使用第 3 节。

也可以在完成第 3 节后，在同一个终端打开生成的完整场景：

```bash
MUJOCO_GL=glfw python -m mujoco.viewer --mjcf="$GO2_SIM_OUT/scene.xml"
```

这会显示场景定义，不会重放刚才的动作轨迹或恢复结束时的状态；`scene.xml` 不包含 ONNX 控制循环。

## 7. 仅在已有环境不可用时：创建独立环境

当前机器不需要重新安装。若以后重新部署项目，可在 Linux 的 Python 3.10 环境下执行以下命令。新环境使用 `.venv-mujoco`，保留原来的 `.venv-deploy`。

```bash
cd "/home/venom/文档/ChatGPT/机器狗"

python3.10 -m venv .venv-mujoco
source .venv-mujoco/bin/activate

python -m pip install \
  "mujoco==3.3.7" \
  "numpy==1.26.4" \
  "onnxruntime==1.19.2" \
  "PyYAML==6.0.2" \
  "glfw==2.11.0"

python -c 'import mujoco, onnxruntime, numpy, yaml, glfw; print("MuJoCo 仿真依赖检查通过")'
```

安装需要网络。安装完成后，在这个已激活的环境里执行第 3 节或第 5 节。上述 ONNX 仿真入口不需要安装 Isaac Sim、Isaac Lab 或 PyTorch。

## 8. 常见问题

### 提示 `No module named mujoco` 或 `onnxruntime`

先检查是否用了项目环境：

```bash
python -c 'import sys; print(sys.executable)'
```

应指向 `.venv-deploy/bin/python`，或第 7 节新建的 `.venv-mujoco/bin/python`。如果指向系统 Python，重新激活对应环境。

### 提示 `X11: Failed to open display`、`DISPLAY` 缺失或 GLFW 初始化失败

查看当前终端的显示变量：

```bash
printf 'DISPLAY=%s\nXAUTHORITY=%s\n' "${DISPLAY:-未设置}" "${XAUTHORITY:-未设置}"
```

请在已登录 Linux 桌面的终端运行第 3 节，沿用该会话的显示变量。不要从旧命令中照抄 `DISPLAY=:0` 或 `:20`；这些编号和权限属于具体会话。只有 SSH、没有可用图形会话时，使用第 5 节。

### 提示 `Output directory already exists`

重新执行第 3 节的整段命令，让第一行重新生成 `GO2_SIM_OUT`。不要复用旧目录，也不要提前 `mkdir "$GO2_SIM_OUT"`。

### 窗口停在最终姿态

这是 `--hold-final` 的预期行为。终端出现 `ROBOGAUGE_VIEWER_HOLD_FINAL` 时，结果已经保存，物理已停止。关闭窗口后重新运行即可。

### 加载了不同策略，或把 `.pt` 文件传给了 `--policy`

此入口要求配套的 ONNX 和 YAML。保留第 3 节中明确写出的 `--policy` 与 `--config`；训练用的 `.pt` 检查点不能直接替换这里的 `.onnx`。

## 本次核对记录

- 已核对本机依赖版本、ONNX/YAML 配对、模型网格加载，以及上述脚本参数。
- 使用第 5 节对应的参数完成一次无窗口回归：1 个基准回合成功，约 5.702 秒仿真时间完成登台停稳，无墙体接触。关节保护审计未全部通过，因此这里只报告仿真任务成功。
- 本次执行环境的图形探测返回 `X11: Failed to open display :0`，未验证实际窗口显示；图形启动命令依据现有脚本和项目此前的窗口回放记录整理。
- 本指南使用的是已导出的 `moe1000`，不把正在训练的检查点称为新的已完成策略。

MuJoCo 自带查看器的命令格式见 [MuJoCo 官方 Python 文档](https://mujoco.readthedocs.io/en/stable/python.html#standalone-app)；策略启动参数以项目当前的 `scripts/test_robogauge_two_steps_wall.py` 为准。
