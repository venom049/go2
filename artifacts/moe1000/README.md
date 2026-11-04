# MoE-CTS moe1000 模型归档

这是用户指定的 **moe1000**（有时写作 meo1000），不是 HIMLoco，也不是后续小腿速度续训的 calf1500。
基于 RoboGauge / MoE-CTS 的部署 actor 恢复可训练网络并完成 ONNX 数值校验后，冻结 MoE 编码器、初始化 critic 和优化器，在 128 个 MuJoCo 环境、32 workers 中进行 1000 次 PPO 策略头微调。

## 文件

- `exported/policy.onnx`：微调后推理模型。
- `model_1000.pt`：完整训练检查点，含 actor、critic、Adam 状态及更新计数。恢复训练需要配套 MoE-CTS 训练环境，不能使用本仓库历史 HIMLoco 的加载入口。
- `moe_cts_actor.py`：本次导出对应的 actor 实现快照。
- `params/deploy.yaml`、`params/deploy.json`：观测、动作、关节映射和 PD 参数。
- `manifest.json`：来源、接口与模型哈希。
- `export_parity.json`、`parity_fixtures.npz`：199 组输入的 PyTorch / ONNX 导出一致性检查及测试数据，覆盖 81 个计算节点输出。
- `evaluation.json`：独立仿真对照中 moe1000 的结果，保留完整关节及力矩统计。
- `UPSTREAM_LICENSE`：原上游部署目录随附 MIT 许可证，保留原文声明。
- `SHA256SUMS`：本目录文件校验值。

## 接口

输入 `[1,225]`，12 维关节动作；每个观测项按时间排列 5 帧（旧到新），再按观测项拼接。初始化历史时重复首帧实测观测。
观测项顺序：角速度、投影重力、速度命令、相对关节位置、关节速度、上一动作。具体缩放见部署参数。
策略关节顺序为 FL、FR、RL、RR，各腿 hip、thigh、calf；SDK 映射为 `[3,4,5,0,1,2,9,10,11,6,7,8]`。
控制周期 0.02 s，Kp=20、Kd=0.5，动作缩放 0.25。此归档不提供电机接管或自动启动脚本。

## 已完成的仿真检查与限制

场景：宽 0.9 m、两级各高 0.15 m、首踏深 0.30 m、顶部平台深 0.90 m 后接墙；前脚距首级 0.50 m。命令速度 0.6 m/s；nominal 1 回合加 seed42/43 各 64 个首回合，共 129 回合。

| 指标 | moe1000 |
| --- | ---: |
| 任务成功数 | 129/129 |
| 实测水平上楼速度 | 0.377117 m/s |
| 横滚 RMS | 2.47229° |
| 平均髋外展正分量 | 1.35107° |
| 足滑 RMS | 0.165756 m/s |
| 四小腿速度加权超限比例 | 30766 / 2793376 = 1.101391% |
| 小腿最大实测速度 | 26.340046 rad/s |
| 小腿保护上限 | 15.7 rad/s（float32 值 15.699999809265137） |
| calf_speed_acceptance_passed | false |
| joint_and_torque_acceptance_passed | false |

步态指标仅统计成功首回合；关节审计包含失败末子步，并覆盖每个物理步 pre/post 两边界。小腿比例使用四轴采样数，不以全部 12 轴稀释。其他实际位置、目标位置与跟踪误差也有超限，详见 evaluation.json。
**任务成功不等于保护验收通过；本模型仍有速度等超限。** 平台停车由仿真几何触发外部零命令，不是自主识墙。0.6 m/s 是命令值，不能当作实测速度；仿真结果不代表实机运行许可。

## 校验

在本目录执行 `sha256sum -c SHA256SUMS`。

- ONNX SHA-256：`f0bfb5ac6758b72fbeae96f1b82462950127440b5210807016d5fa5466529d56`
- 检查点 SHA-256：`3bad0efcc2e751e834bea1e85a5914c410a2a1ff1a1b80193426805f69e4881f`

## 上游来源

来源：[robogauge/code](https://github.com/robogauge/code)，固定提交 `f116b00a934262348f41cbfd0d07cc4cb6fbf8d8`。
原始策略位于 `unitree_cpp_deploy/logs/rsl_rl/go2_moe_cts_expert_goal_137000_0.6745/exported/policy.onnx`。
本归档保留原有许可证与来源，不为仓库其他材料重新授权。
