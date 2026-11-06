# Go2 官方公开参数保护更新（2026-11-04）

本次只更新独立爬楼控制程序的关节保护，不修改原厂底层程序、模型权重或训练参数，也不启动实机运动。机载安装状态及备份目录以本目录 `installation.json` 为准。

## 采用的限制

| 关节 | SDK q 范围（rad） | 速度绝对值（rad/s） | 请求力矩检查（Nm） |
|---|---|---|---|
| 四个髋关节 | ±0.837758041（±48°） | 30.1 | 23.7 |
| 前腿大腿 | -1.5708 ～ 3.4907 | 30.1 | 23.7 |
| 后腿大腿 | -0.5236 ～ 4.5379 | 30.1 | 23.7 |
| 四个小腿 | -2.7227 ～ -0.83776 | 15.7 | 45.43 |

来源为[宇树 Go2 开发文档](https://support.unitree.com/home/zh/developer/about_Go2)及[官方固定版本 URDF](https://github.com/unitreerobotics/unitree_ros/blob/2ded279cfb3207d410b1ade88303e1a8971b0ac3/robots/go2_description/urdf/go2_description.urdf)。原始公开文件及 SHA256 保存在 `sources/manifest.json`，逐关节来源与解释保存在 `limit_profile.json`。

文档髋关节为 ±48°，URDF 为 ±60°，这里取较窄范围。大腿文档叙述的正方向与 URDF 区间符号相反，代码保持 SDK/URDF 的 q 坐标：官方 SDK2 站立示例也使用正大腿角、负小腿角；此符号解释是来源对照推断，不是假定厂家已发布更正。速度采用关节输出端 rad/s，不换算为 RPM，也不再乘减速比。

上述值是公开模型/文档参数，**不是从电机固件读回的保护阈值**。力矩检查计算 PD＋前馈请求的估算值，不能保证实测力矩或所有温度、电压下的持续输出能力。

## 代码行为

- `Go2JointLimits.h` 是两层保护共用的单一参数表，按 SDK 的 FR、FL、RR、RL 顺序，每腿依次 hip、thigh、calf。拒绝非 12 关节、非有限值和不匹配的 SDK 映射。
- 策略检查由统一 13 rad/s 改为逐关节上限；发送边界原来统一 30 rad/s 的检查同步更新。
- 按速度与对应上限的比值寻找违规关节，避免合法的 20 rad/s 髋关节掩盖 16 rad/s 小腿超限。
- 比较边界采用 SDK float32 的表示；原始十进制来源另存，无可调的超限容差，也不裁剪测得的速度。
- 配置必须明确选择 `safety_joint_limit_profile: unitree_go2_2ded279_hip48`；旧 `safety_dq_limit` 配置会拒绝启动，避免旧标量与新表混用。
- 保留自定义 0.9 rad 跟踪误差、姿态、增益、原始动作、站稳交接、100 ms 状态时效、最长两秒测试和首次故障锁存阻尼。

## 更新对象与验证

机载目标：`/home/unitree/go2-stairs-torque1000-test-20261103/controller`。这个目录名表示独立测试控制器的来源；本次不改变其选择模型的逻辑。

保护来源文件位于本地 `deploy/handoff_overlay`，同步到 `deploy/first_test_overlay` 和准备中的 `deploy/guard1000_test/overlay`。历史 `validation/onboard_joint_limits_20261104/protection_sources` 审计快照保持原样。

安装包只替换清单中的六个源码/配置文件及重新编译的 `robots/go2/build/go2_ctrl`。安装前检查无控制器进程、源码和旧二进制校验值，备份旧文件；编译或测试失败即回滚。运行编译后的无参数检查前校验已审阅的 main、参数处理、状态机和构建配置。该入口在 DDS 初始化前返回，不发布电机命令。

离线测试涵盖所有 12 关节正负速度/位置边界、float32 边界、逐关节力矩、隐藏小腿超限、NaN/Inf、错误映射、超时锁存与禁止重启。另检查原交接站姿测试和候选短时测试。测试通过不等于实机爬楼通过；旧仿真结果也不能因阈值调整自动视为通过。
