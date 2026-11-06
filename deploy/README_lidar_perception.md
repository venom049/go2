# Go2 原生雷达楼梯识别：只读原型

本模块独立添加在机载电脑 `/home/unitree/go2-stairs-perception-20261103`，不修改原厂代码、既有控制器、ROS 安装、服务或模型。它不发布运动命令，不切换控制模式，不接管电机。ROS 自身仍有 DDS 发现及基础参数事件通信。

## 本轮目的

用户确认机器狗当前面对“平地或室内杂物”。这类数据用于检查误报，不能用于证明真实楼梯检出率。本轮保留累计 1100 次训练的策略及已有部署目录，识别模块与策略权重相互独立。

原生雷达的 `/utlidar/cloud_deskewed` 提供 `odom` 坐标系点云；`/utlidar/robot_odom` 提供 `odom → base_link` 位姿。读取时按最近源时间戳配准（最多 50 ms），减去机身位置并只消除航向角，保留世界竖直方向。`odom` 已重力对齐仍是一项待完整标定的假设；初次采样中其横滚、俯仰与 `/lowstate` 接近，但这不等于完成外参标定。

## 运行

从机载新增目录运行一次，默认 10 秒，最长 60 秒，自动退出：

```bash
cd /home/unitree/go2-stairs-perception-20261103
bash run_lidar_readonly.sh --duration 10
```

若已知源时钟与机载时钟不同，可显式采集几何预览；输出文件必须是新文件：

```bash
bash run_lidar_readonly.sh --duration 10 --allow-unsynced-preview \
  --output validation/scene_new.json --capture validation/scene_new.npz
```

不需要安装全局依赖。机载系统 Python 3.8、NumPy 1.17.4 和现有 ROS 2 Foxy 可运行；独立的模型 Python 3.9 环境不用于这个节点。脚本只为本进程指定现有 `eth0` DDS 通道，不修改已有 `setup.sh`。

## 输出和识别规则

- `state`：`stairs_up`、`stairs_down`、`flat`、`obstacle` 或 `unknown`。这只是观察结果，不是通行许可。
- `geometry`：候选第一条可见台阶边缘距离、阶高、踏深、可见高度变化数量及横向覆盖。距离不保证对应物理上的第一阶。
- `score`：几何支持程度的启发式分数，不是概率或可靠性认证。
- `geometry_preview`：时钟未对齐时仅供检查的几何结果；总 `state` 仍为 `unknown`，`source_age_s` 为 `null`，不自动补偿偏差。
- `geometry_state_counts`：本次处理帧的候选分类数量，可能包含时钟未对齐的预览，不能解释为准确率。
- `control_authorized` 始终为 `false`。

几何算法使用前方 0.15–2.5 m、横向 ±0.55 m 的区域，要求左右及中间三个条带都有连续踏面，并有至少三次一致的阶高变化。默认候选阶高 7–24 cm、踏深 18–60 cm；这是检测范围，不是现有策略可通过的范围。单个箱子、单级台阶、坡面或墙面不能仅凭高度差被当作楼梯。缺测不填成平地。该保守覆盖条件容易漏检稀疏、遮挡或斜对楼梯，尚未使用真实楼梯数据调参。

`perception.py` 另提供 `TemporalConfirmation`，检查同一时钟下的连续独立新帧；当前短时 watcher 只统计单帧几何候选，没有连接自动切换逻辑。现有仿真中的高度跨度策略门控不能直接当作实机楼梯识别器使用。

## 当前边界与下一步

初次实采的点云与里程计时间相差约 0.6 ms，但源时间与机载墙钟相差约 499 s。当前只能确认点云持续到达、源时间递增、点云与里程计可配准，不能宣称源数据真实年龄已知。接入控制前需完成时钟同步和坐标标定，并在真实楼梯正面进行只读验证。

下一步应采集已知阶高、踏深的楼梯以及单阶、箱子、坡道等对照场景，检查检出、误报与距离误差。随后才开发带人工启用、状态新鲜度检查及退出条件的独立切换层。现有模型尚未完成实机爬楼梯验证，也不支持以本模块的 `stairs_down` 结果自动下楼。

几何思路参考作者开源的 [staircase_autonomy 独立点云检测文档](https://github.com/Prassi07/staircase_autonomy/blob/main/docs/ros2_nodes/staircase_standalone_detection_node.md)：使用重力对齐的点云、连续阶梯结构及几何约束。当前为独立的 NumPy 实现，没有安装或移植其 Humble 软件栈。
