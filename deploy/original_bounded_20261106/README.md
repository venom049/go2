# 原始 RoboGauge 权重＋目标硬限幅

机载目录：`/home/unitree/go2-original-bounded-20261106`。原始 ONNX 不改动、不重新训练，SHA-256 为 `bc08253fd68f661bf6be5ce5ca70cec555c3fde2c397d23fbe9814a998e056f5`。

通过 Wi-Fi SSH 连接 `unitree@192.168.1.180`；控制器在机载运行，用内部 `eth0` 通信。不要把 PC 的 `wlo1` 或机载的 `wlan0` 填入电机控制接口。

策略输出先经原 scale/offset 转为角度，再按 SDK float32 边界硬限幅。进入动作历史的是实际提交目标的逆编码。保护层仍独立检查实际角度、速度、0.9 rad 跟踪误差、请求力矩、45°倾角及状态超时。小腿速度采用 15.7 rad/s；旧控制器的用户覆盖 30 rad/s 未沿用。

只读检查（无 LowCmd 发布、无模式释放）：

```bash
cd /home/unitree/go2-original-bounded-20261106
./start.sh --dry-run
./start.sh --mode-check
./start.sh --infer
./start.sh --handoff-probe
```

现场需要实际接管时，在交互 SSH 终端运行 `./start.sh --start`。沿用既有保护流程：先预加载和预热策略、检查原厂站立指令；H 才请求一次控制权释放并进入 40/1 保持与两秒站姿插值；达到 STAND_LOCKED 后，1 选择速度、G 才启动 20/0.5 策略。策略限时两秒，X/空格/0 锁存固定阻尼，Ctrl-C 请求退出前阻尼。

交接沿用既有 HandoffHold，不自动恢复原厂模式、重试 ReleaseMode 或操作厂家服务。原有代码仍在释放后等待 RPC 结果、检查竞争指令并重新核验状态；该流程不能保证释放至开始保持之间绝无控制空窗。固定阻尼也不是站姿保持，不能承诺故障或断连时不趴下。部署完成及只读推理通过不代表闭环或实机爬阶通过。

`SHA256SUMS` 校验模型、源码和入口，`BINARY_SHA256SUMS` 校验机载生成的二进制。旧部署和厂家服务不修改，不安装自启动。
