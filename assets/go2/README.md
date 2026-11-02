# Go2 模型来源与转换

训练入口使用 `urdf/go2.urdf`。该文件由 `scripts/prepare_go2_asset.py` 生成；原始模型与全部 16 个 OBJ 网格保存在 `mjcf/`，可离线重新生成。

用户指定的 [go2-stairs-sim2real](https://github.com/wxz-521/go2-stairs-sim2real/tree/530a10fee4232317aef4343f2b85b6bad9cf7905) 没有直接包含机器人模型文件。其 `mujoco_sim/setup.sh` 引用了 [Unitree 官方模型](https://github.com/unitreerobotics/unitree_mujoco/tree/ae6a8403e272733e9996ef59990880330496177f/unitree_robots/go2)，版本固定为 `ae6a8403e272733e9996ef59990880330496177f`。本目录使用同一版本，不使用任何预训练策略。

模型与网格遵循随附的 `LICENSE`；完整来源、内容校验值和转换差异见 `PROVENANCE.json`。

## 物理属性

- 总质量：15.206408 kg；12 个旋转关节，4 个固定足端关节。
- 根刚体：`base_link`；足端：`FL_foot`、`FR_foot`、`RL_foot`、`RR_foot`。
- 关节名：`{FL,FR,RL,RR}_{hip,thigh,calf}_joint`；不要依赖导入器产生的数组顺序。
- 保留原模型质量、组合质心/惯量、关节坐标系/限位、碰撞几何和可视网格。
- hip/thigh 力矩上限 23.7 N·m，calf 力矩上限 45.43 N·m。
- 原始 MJCF 没有关节速度上限。URDF 中的 30 rad/s 是明确增加的训练配置值，可在执行器配置中调整。

原始模型的足端球体属于 calf 刚体，`*_foot` 是空的固定子刚体。为支持单独读取足端接触力，转换器把足端几何移入对应的固定足端刚体，并从 calf 划出 0.01 kg 给足端。转换器使用平行轴定理调整 calf 的质心及完整惯量张量，因此 calf + foot 的组合质量、质心与惯量保持不变。CPU 测试对这些不变量进行数值验证。

## Isaac Lab 导入要点

使用浮动基座，设置 `fix_base=False`、`merge_fixed_joints=False`。保留原始圆柱碰撞体；不要通过胶囊体替换来声称保持完全相同几何。足端需要保留为独立刚体，才能按上述名称建立接触观测。

MJCF 的 `armature=0.01` 无法由标准 URDF 表达，应在 Isaac Lab 的执行器配置中设置。URDF 保留了关节 `damping=0.1`、`friction=0.2`，但导入器及执行器配置可以覆盖这些值。MuJoCo 的 `frictionloss` 与 PhysX 的关节摩擦并非严格相同的动力学模型。

MuJoCo 的接触摩擦、接触维数、求解器参数和 margin 无法通过 URDF 精确迁移，需要在 PhysX 材料和仿真配置中单独指定。原始世界坐标初始位置、传感器和 home keyframe 未转为 URDF：浮动根状态、观测和初始关节姿态由训练环境负责。

## 重新生成与验证

在项目根目录执行：

```bash
python3 scripts/prepare_go2_asset.py
python3 -m unittest discover -s tests -p test_assets.py -v
```

如果原始模型文件尚未下载，执行 `python3 scripts/prepare_go2_asset.py --fetch`。此操作仅获取固定版本的官方机器人模型。转换器会校验原始 MJCF 和网格的整体 SHA-256，拒绝悄悄转换其他版本。

这些测试验证模型结构、文件来源、关节参数和刚体惯量，不代替 Isaac Sim/PhysX 中的实际导入、接触和训练验证。
