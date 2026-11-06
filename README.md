# Go2 EDU：MoE 训练、实机测试与感知接入

当前结果更新于 **2026-11-06**。机器人已由用户关机，实机工作暂停。**完整自主爬阶验收未通过。** 当前集成基线是原始 RoboGauge / MoE-CTS 权重加目标硬限幅；雷达运动入口保持禁用。

完整结果与模型区别见 **[当前结果总览](docs/RESULTS_20261106.md)**。

| 内容 | 入口 |
|---|---|
| 官方模型、适应性方法、1000 更新的最终权重及证据 | [Go2 EDU sim2real 产物](artifacts/moe1000_go2edu_sim2real_20261105/) |
| SDK 目标硬限幅、1000 更新的最终权重及证据 | [SDK target 产物](artifacts/moe1000_sdk_target_20261105/) |
| 独立 L1＋机身 IMU、楼梯识别与腿部风险停车 | [部署代码](deploy/lidar_moe_20261106/)、[验证结果](validation/lidar_moe_reliability_20261106/resume/RESULTS.md) |
| 原始权重＋目标限幅的实机测试 | [15.7 rad/s 配置](validation/original_bounded_deployment_20261106/retest_20261106_1449/RESULTS.md)、[用户选择的 30 rad/s 配置](validation/original_bounded_deployment_20261106/retest_20261106_1507/RESULTS.md) |
| 最新一次爬阶启动检查 | [17:14–17:15 记录](validation/lidar_moe_reliability_20261106/retry_20261106_1714/RESULTS.md) |
| RealSense 检测、Type-C 初始化异常与暂停点 | [相机排查](validation/camera_detection_20261106/RESULTS.md) |
| 历史 HIMLoco、MoE-CTS 归档 | [截至 11 月 4 日的说明](README_HISTORY_20261104.md) |
| 工程、训练和历史调参说明 | [工程指南](PROJECT_GUIDE_20261106.md) |
| 上传范围、大型日志压缩记录 | [上传清单](docs/UPLOAD_MANIFEST_20261106.json) |

目标角硬限幅不能保证实测关节角、速度或接触都满足要求。当前点云地面与 URDF 脚底存在厘米级不一致，楼梯后墙/上平台观测也不连续；新版只读检查保持 WAIT，没有将其记为实机爬阶成功。RealSense 的 Type-C 修复工具仅已准备，尚未执行。

模型与证据按各自 SHA256 清单保存；当前快照校验为 `ARCHIVE_SHA256SUMS`，第三方来源见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。历史 600/1100 和 moe1000 产物均保留，不与新训练或实机结论混用。
