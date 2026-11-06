# 第三方来源与许可记录

本文件记录随本次私有训练归档保存的第三方材料及其已有许可声明，不替代各文件的版权头部或原许可证，也不将不同来源统一重新授权。本仓库的可见性设置不改变第三方材料原有的权利和义务。

## Go2机器人模型与网格

- 实际材料：`assets/go2/mjcf/go2.xml`、`assets/go2/mjcf/assets/*.obj`，以及从它们转换的 `assets/go2/urdf/go2.urdf`。
- 来源：[unitreerobotics/unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco/tree/ae6a8403e272733e9996ef59990880330496177f/unitree_robots/go2)，固定提交 `ae6a8403e272733e9996ef59990880330496177f`。
- 该模型依赖由用户指定的 [wxz-521/go2-stairs-sim2real](https://github.com/wxz-521/go2-stairs-sim2real/tree/530a10fee4232317aef4343f2b85b6bad9cf7905) 的 `mujoco_sim/setup.sh` 声明。
- 随附许可证：[BSD-3-Clause](assets/go2/LICENSE)，版权声明为 `Copyright (c) 2016-2024 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics")`。
- 原MJCF和16个OBJ按原始字节保存；URDF转换差异、源文件校验值与物理参数假设见 [`assets/go2/PROVENANCE.json`](assets/go2/PROVENANCE.json)。

## 原仓库平地行走策略

- 实际材料：`assets/policies/upstream_walk/walk.pt`。
- 来源：[wxz-521/go2-stairs-sim2real 的 `example/go2/low_level/final/walk.pt`](https://github.com/wxz-521/go2-stairs-sim2real/blob/530a10fee4232317aef4343f2b85b6bad9cf7905/example/go2/low_level/final/walk.pt)，固定提交 `530a10fee4232317aef4343f2b85b6bad9cf7905`。
- 保留该来源随附的 [BSD-3-Clause许可证](assets/policies/upstream_walk/LICENSE)，原文版权声明为Unitree Robotics。
- 文件SHA-256：`65367faebe875faddd8513e12fc3211b9185857e59b3e5cabbef454d46976dfc`。
- 本项目没有修改该权重；适配代码按原49维观测、16维动作、关节顺序和动态PD接口读取它。来源与接口记录见 [`manifest.json`](assets/policies/upstream_walk/manifest.json)。
- 该策略只用于平地回放/组合控制，不作为楼梯训练的初始化模型。未使用原仓库的楼梯策略权重。

## HIMLoco Lab移植源码及本地修补

- 实际材料：`upstream/himloco_lab/source/`。
- 来源：[IsaacZH/himloco_lab](https://github.com/IsaacZH/himloco_lab/tree/c3fbd0173f2a53a86e472c099f8e2e9744cf7639)，固定提交 `c3fbd0173f2a53a86e472c099f8e2e9744cf7639`。
- 本地修补及原归档校验值记录于 [`upstream/SOURCES.json`](upstream/SOURCES.json)。修补包括历史观测重置、估计器切片、超时价值引导、小批次处理、恢复计数与设备映射、Lab 2.3.2兼容、探索噪声与动作诊断等。
- RSL算法文件保留原有BSD-3-Clause完整许可文本，相关版权声明包括NVIDIA Corporation & Affiliates、ETH Zurich和Nikita Rudin；部分Isaac Lab派生文件保留 `SPDX-License-Identifier: BSD-3-Clause` 及Isaac Lab Project Developers版权头部。
- **上游许可元数据存在差异**：`source/himloco_lab/setup.py`自身头部标注BSD-3-Clause，但其 `setup(..., license="Apache-2.0")` 元数据填写Apache-2.0。该固定快照未附顶层LICENSE文件，部分源码也没有独立许可头部。本归档保留已有声明及此差异，不把包元数据解释为对所有文件统一授权或覆盖已有BSD通知。
- 原Isaac Gym算法参考：[InternRobotics/HIMLoco](https://github.com/InternRobotics/HIMLoco/tree/ef289acaa62795009363b7b819c9186690630441)，固定提交 `ef289acaa62795009363b7b819c9186690630441`；完整Gym工程未打包，也不是本归档Lab训练入口的运行依赖。

## 外部运行时

Isaac Sim、Isaac Lab、PyTorch、CUDA相关运行库和其他Python依赖由运行环境提供，没有作为完整运行时分发到本仓库。具体已验证版本见 [`validation/server_environment.json`](validation/server_environment.json)。下载或使用这些组件时适用各自的许可与安装要求；本归档不授予它们的新许可，也不携带服务器上的NVIDIA许可接受标记。

本项目训练生成的checkpoint及新增环境/适配代码，与上述第三方文件分目录保留；本文件没有为新增内容设定统一的开源许可证。后续改变仓库可见性或再分发时，应继续保留这些来源、原文件声明和附带许可证。

## RoboGauge / MoE-CTS moe1000

新增材料、固定上游提交、微调来源及原许可证见 [moe1000 说明](artifacts/moe1000/README.md) 和 [上游许可证](artifacts/moe1000/UPSTREAM_LICENSE)。该模型与历史 HIMLoco 训练链相互独立。


## 2026-11-06 新增的原始 MoE、SDK 与独立感知

- `upstream/robogauge/`、原始部署 ONNX 及 MoE 适配来自 [robogauge/code](https://github.com/robogauge/code)，固定来源提交 `f116b00a934262348f41cbfd0d07cc4cb6fbf8d8`；保留 [RoboGauge MIT](upstream/robogauge/RoboGauge/LICENSE) 和 [C++ 部署 MIT](upstream/robogauge/unitree_cpp_deploy/LICENSE) 原文。新增训练检查点与原始作者权重分别保存，不混用模型身份。
- `upstream/unitree_sdk2_python/` 保留其 [Unitree BSD-3-Clause](upstream/unitree_sdk2_python/LICENSE)。机载 SDK/ROS 运行时没有从机器狗系统目录打包进来，构建脚本记录既有外部依赖路径。
- 官方 Go2 MJCF 的固定来源与许可证同上。新训练使用的原字节副本位于 `validation/moe1000_sim2real_20261105/source/assets/unitree_go2/`；机载 URDF 单独记录来源与哈希，不代表已完成该机个体物理标定。
- `deploy/lidar_moe_20261106/independent_lio/src/moe_fast_lio/` 是已适配但未选为默认估计的 FAST-LIO ROS 2 候选，保留 [GPL-2.0 许可证](deploy/lidar_moe_20261106/independent_lio/src/moe_fast_lio/LICENSE)。来源为 [Ericsii/FAST_LIO_ROS2](https://github.com/Ericsii/FAST_LIO_ROS2) 提交 `2fffc570a25d0df172720bac034fbdb6a13d2162`，算法来自 [hku-mars/FAST_LIO](https://github.com/hku-mars/FAST_LIO)，ikd-Tree 提交及适配记录见 [SOURCE.json](deploy/lidar_moe_20261106/independent_lio/SOURCE.json)。当前默认独立估计实现和算法选择见部署 README；没有把此 GPL 候选源码统一改为其他许可。
- 参考的适应性训练仓库、RealSense 排查文章和 NVIDIA 原始资料在本次结果总览及对应报告中链接；未打包外部文章、论文或报告图片。
