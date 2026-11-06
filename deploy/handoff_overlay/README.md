# 独立爬楼交接控制器 overlay

本目录提供独立爬楼控制器的增量源码。2026-11-04 更新了两层逐关节保护，公共参数在 `include/Go2JointLimits.h`。

合并到已核验的独立副本后，必须在该副本 `robots/go2/config/config.yaml` 的 `FSM.Stairs` 下移除 `safety_dq_limit`，添加：

```yaml
safety_joint_limit_profile: unitree_go2_2ded279_hip48
```

未设置正确 profile 的控制器会拒绝启动，避免新源码继续使用旧的统一速度阈值。髋/大腿速度30.1 rad/s、小腿15.7 rad/s来自固定版本官方 URDF；髋位置采用官方网页较窄的±48°。其余关节位置使用SDK/URDF坐标，力矩请求检查仍为髋/大腿23.7 Nm、小腿45.43 Nm。模型权重、厂家服务与固件不属于此 overlay。

本目录源码已同步到机载哪个版本、是否编译安装，以 `validation/official_joint_limits_20261104/installation.json` 为准。编译与离线测试通过不表示已启动或通过实机爬楼测试。
