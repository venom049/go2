# Go2 累计 1100 模型：独立首次测试保护副本

这是供审查、编译与只读预检的增量 overlay。未通过电机响应测试，不替换原控制器或厂家服务。当前机器人若仍有 `rt/lowcmd` 发布者，或处于折叠姿态并超出本训练 URDF 的位置范围，必须拒绝位置控制；不裁剪实测姿态，不通过该程序强行站立。

## 安装方式

在机器人上另建隔离目录，完整复制现有 `deploy` 树及其依赖，然后将本目录的 `include/`、`robots/` 同路径文件覆盖到**新副本**。保留原有 `include`、`thirdparty`、`robots/go2/include`、配置与 CMake 文件，按原 `robots/go2/CMakeLists.txt` 编译。只覆盖这里列出的增量文件，不复制覆盖原安装目录：

- `include/FirstTestGuard.h`
- `include/Go2JointLimits.h`
- `include/FSM/FSMState.h`
- `include/FSM/CtrlFSM.h`
- `include/FSM/State_RLBase.h`
- `robots/go2/main.cpp`
- `robots/go2/src/State_RLBase.cpp`
- `robots/go2/include/State_StairsReady.h`

最终二进制须保留在新副本的 `robots/go2/bin` 或 `robots/go2/build` 下，原参数解析器据此寻找新副本 `robots/go2/config/config.yaml`。安装前后校验原目录清单和 SHA-256；本 overlay 不提供服务注册、开机启动、自动接管或恢复厂家模式。

2026-11-04 保护参数更新：新副本配置的 `FSM.Stairs` 必须删除旧 `safety_dq_limit`，并设置 `safety_joint_limit_profile: unitree_go2_2ded279_hip48`。策略检查与发送边界共用 `Go2JointLimits.h`：髋/大腿 30.1 rad/s、小腿 15.7 rad/s，髋位置取官方网页较窄的 ±48°。这些是官方公开参数构成的应用层检查，不是固件阈值读回。0.9 rad 跟踪误差、姿态、站稳交接和锁存停止等保护保留。缺失或不匹配的 profile 会拒绝启动。

模型路径必须显式为 `/home/unitree/go2-stairs-addon-1100-20261103/legacy_candidate/v0`。模型和配置内容的 SHA-256 由外部部署记录/检查器再次验证；本程序只检查明确的绝对路径和文件存在性，不能替代内容校验。

## 默认及只读入口

不带自定义激活开关时，普通启动只打印 dry-run，不初始化 DDS、不创建 publisher。可使用以下入口进行只读检查，`EXE` 应替换为**新副本**的可执行文件路径，`IFACE` 替换为核实的机器人有线网卡：

```sh
EXE --first-test-preflight --network IFACE --stairs-policy /home/unitree/go2-stairs-addon-1100-20261103/legacy_candidate/v0
EXE --check-lowcmd --network IFACE
EXE --robot-print --robot-print-steps 100 --network IFACE --stairs-policy /home/unitree/go2-stairs-addon-1100-20261103/legacy_candidate/v0
```

`--first-test-preflight` 总是只读：检查候选路径、观察 LowCmd、进行有限时姿态/递增 tick 检查，在创建 publisher 之前返回；输出 `control_authorized=false`。返回 0 仅表示这组软件前置条件成立，不代表允许上楼或获得了控制权。`--check-lowcmd` 检测 ACTIVE 返回 2，IDLE 返回 0。

`--robot-print` 是形状/推理运行诊断。没有已提交电机目标时，历史中的 applied action 不可知，使用明示的零占位；它不能验证闭环表现或真实的动作反馈。`--service-list` 在此副本禁用，因为原 C++ 调用在该设备上曾异常退出；使用独立已验证的 Python 只读查询。`--release-motion` 和所有 `--stairs-safety-*`、`--stairs-auto-*` 均禁用。

## 激活门控与停止语义

只有单独显式给出 `--first-test-actuate`、正确候选路径、交互 TTY，并通过无竞争 LowCmd/有限姿态预检后，才可能创建 publisher。这里只说明开关，**不提供可复制粘贴的实机激活命令**；真实交接、姿态、支撑、急停和模型哈希必须由现场受控流程确认。

新进程仅在内存中将 FixStand 改成从当前范围内姿态直接用两秒插值到候选默认站姿，去掉原深蹲中间点，原配置文件不变。

从 Passive 开始；F 进入 FixStand；等待其两秒定位结束后，1 选择唯一的 0.6 m/s 指令，G 才进入 Stairs。2–9、W/S 和方向速度键不参与选择，并清除已选状态；只有 1 能重新确认 0.6 m/s。进入 Stairs 后软件限时 2 秒，锁存固定 `kp=0, kd=3, q=0, dq=0, tau=0` 阻尼。X、Space、0 和原遥控停止状态转换同样锁存阻尼，永不自动 LieDown、重新进入策略或重新解锁。

每个发布周期先获得 SDK `trylock`，所有状态 `run/enter` 写入均在锁内，最后检查并替换完整 12 关节输出再 `unlockAndPublish`。多余电机槽清零并 mode=0。模型的全部 16 个原始输出先检查有限性再 clip，防止 Inf 被夹成有限值。检测非有限输入/输出、四元数范数异常、tick 倒退或超过 100 ms 不前进、逐关节位置/速度/gain/前馈力矩超限，以及快照估算的 PD+前馈力矩超 URDF effort，均锁存阻尼，绝不裁剪后继续策略。FixStand 等非楼梯位置控制也受此边界约束；被动阻尼不使用实测 q 做保持。

LowState 在其互斥锁内复制；本 SDK 没有无竞争的接收时间戳或同步的绝对源时间，所以两种绝对年龄保持 unavailable。新鲜度依赖本地单调时间内源 tick 的前进，不把轮询时刻当作接收时刻。

模型 env、状态、历史、过渡检查在同一 FSM 线程串行执行，推理目标周期 20 ms。反馈项来自上一次通过保护且交给 SDK 发布的目标 q 的逆编码；该值不是 DDS 送达/电机执行确认。首帧从实际已提交 FixStand 目标建立历史，不使用尚未发布的模型 raw。保护拒绝一次输出后直接停止，不再把被拒绝动作反馈进策略。

SIGINT/SIGTERM 的正常退出先停止 FSM，再请求约 250 ms 的有限固定阻尼序列并留 20 ms 发送时间，最后沿用设备已存在的 `_Exit` 析构规避方式。日志统计的是发布请求，不能确认网络送达或电机执行；退出后不恢复厂家模式。

## 尚未验证的边界

2 秒限时、100 ms tick 检查及 20 ms 推理超时都依赖该线程继续运行。推理超过 20 ms 时在返回后锁存，**没有独立硬实时看门狗**；SDK/ORT 永久阻塞、进程被 SIGKILL、断电或网络中断时，本进程无法保证阻尼发送或限时退出。LowCmd 两次空闲检查也不是控制权租约，另一个发布者可能随后启动。必须保留现场硬件停止与支撑。

这些 URDF 软件限位和快照 PD 力矩估算并不建立真实电机标定、实际总力矩限制、接触/支撑稳定性或楼梯成功率。尚未进行实机受保护电机测试；15 cm/30 cm 楼梯与仿真评估也不等价。

## 离线验证

纯保护器测试不包含 SDK、DDS 或网络：

```sh
g++ -std=c++17 -Wall -Wextra -Werror -pedantic -Ideploy/first_test_overlay/include tests/test_first_test_guard.cpp -o /tmp/go2_first_test_guard
/tmp/go2_first_test_guard
```

完整副本须在目标 ARM64 环境编译；可以运行默认 dry-run 和只读 preflight。编译成功或只读数据健康均不构成实机爬梯验证。
