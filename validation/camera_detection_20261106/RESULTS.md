# RealSense 检测及 Type-C 初始化核查

2026-11-06，机载电脑 `192.168.1.180`（内网 `192.168.123.10`）只读检查结果：**尚未检测到可用的 RealSense 深度设备，未读取到深度帧，具体相机型号尚不能确认。** 用户确认所接设备是 RealSense。

- USB 缓存编号只有 Wi-Fi `0bda:c811`、集线器 `214b:7260`、网卡 `0bda:8152` 和根集线器；未见 RealSense 设备。
- 没有 `/dev/video*` 视频节点，没有运行中的 RealSense 驱动进程。ROS 2 未发现深度图、相机内参话题；ROS 1 当前无可连接 master。已有 `/frontvideostream` 不能据此认定为 RealSense 深度数据。
- 系统安装有 `rs-enumerate-devices`、RealSense udev 规则及 ROS 相机包，保存有启用深度、点云的 `camera.yaml`。配置和软件存在不证明相机硬件在线。
- `lsusb` 阻塞在 `read_descriptors`，读取 USB product 字符串也阻塞；本次创建的两项阻塞查询已发送 TERM 并确认退出。限时 RealSense 枚举返回超时。内核还持续报告 eth0 注销等待，并有 USB 集线器断连记录；不能据此断定相机本体损坏。

用户提供文章：[解决宇树 Go2 EDU 无法显示连接 D435i 相机的问题](https://blog.csdn.net/onelate/article/details/151190474)。按其线索检查发现，**该机现有 `/etc/rc.local` 本来就包含相同地址及 GPIO 的初始化命令**，无需为了排查而先修改此文件。

实机证据：

1. 系统报告 `NVIDIA Orin Nano Developer Kit`，L4T R35.3.1。
2. 开机脚本执行 `busybox devmem 0x02430030 w 0x004` 时报告 `busybox: not found`，随后脚本仍以 0 退出，因此 rc-local 显示成功并不代表寄存器初始化成功。
3. `PP.06` 已导出，方向为 out、值为 1。
4. 使用现有账户的 sudo 进行一次只读 MMIO 核查，`0x02430030` 实际值为 `0x20`，与该机启动脚本目标 `0x04` 不同。这是一个具体的初始化异常，是否为相机缺失的全部原因尚需恢复后验证。

对应技术核对使用 NVIDIA [Orin NX/Nano 开发指南](https://docs.nvidia.com/jetson/archives/r35.4.1/DeveloperGuide/text/HR/JetsonModuleAdaptationAndBringUp/JetsonOrinNxNanoSeries.html)及 NVIDIA 著作权声明的[上游 Tegra234 pinctrl 源码](https://github.com/torvalds/linux/blob/master/drivers/pinctrl/tegra/pinctrl-tegra234.c)。后者将 PP6 对应的寄存器偏移列为 `0x0030`；载板连接关系仍以该机已有启动配置为依据，不能将博客指令推广到任意硬件。

已在本地准备 `restore_existing_typec_pinmux.py`：限制板型、地址、旧值、目标值和现有 GPIO 状态，写前保存回退记录，只进行一次对齐的 32 位写入并回读；不复位 USB、不改 GPIO、不改启动文件、不发送机器人运动指令。

**修复尚未执行。** 在 scp 同步该工具时 SSH 报 `No route to host`，文件上传失败；随后连接尝试也未到达远端。未输入写入命令的 sudo 密码，没有成功执行该脚本或硬件写入。未安装软件包，未更改 `/etc/rc.local` 或官方 SDK/ROS。

本机 `wlo1=192.168.1.182`，到 `192.168.1.180` 的路由仍存在；等待用户确认机器狗电源与 Wi-Fi。恢复连接后应先重新核对硬件状态和原厂控制模式，再决定是否仍需恢复该项既有引脚初始化，并检查 USB 枚举、深度帧及 CameraInfo。当前不能把本次排查记为“已检测到相机”或“已修好深度数据”。
