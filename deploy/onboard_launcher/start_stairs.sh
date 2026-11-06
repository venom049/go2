#!/usr/bin/env bash
# Go2 HIMLoco cumulative-1100 launcher; install only, never auto-start.
set -euo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CONTROLLER_DIR=/home/unitree/go2-stairs-first-test-20261103/controller/robots/go2
CONTROLLER="$CONTROLLER_DIR/build/go2_ctrl"
POLICY_DIR=/home/unitree/go2-stairs-addon-1100-20261103/legacy_candidate/v0
NETWORK=eth0
MODE=start
MODE_SET=0

usage() {
    cat <<'EOF'
Go2 HIMLoco 1100：独立保护控制器启动入口

用法：./start_stairs.sh [--check | --infer | --dry-run] [--network 网卡]
  无模式参数    在交互终端启动受保护控制器；需要独立的控制权交接条件
  --check       只读检查控制占用和关节姿态，不发布电机指令
  --infer       使用实机数据做 100 步只读推理，不发布电机指令
  --dry-run     校验文件并打印启动命令，不连接 DDS、不运行控制器
  --network     机载网卡，默认 eth0
  -h, --help    显示本说明

固定模型：累计 1100 次更新，速度命令 0.6 m/s。
本脚本不释放原厂控制、不切模式、不关闭服务、不自动按键。
当前站立保持/控制交接问题尚未修复；此入口不是已通过实机验证的一键爬楼。
EOF
}

fail() { printf '错误：%s\n' "$1" >&2; exit "${2:-1}"; }
while (($#)); do
    case "$1" in
        --check|--infer|--dry-run)
            ((MODE_SET == 0)) || fail '模式参数只能选择一个。' 64
            MODE="${1#--}"; MODE_SET=1; shift ;;
        --network)
            (($# >= 2)) || fail '--network 缺少网卡名称。' 64
            NETWORK="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) fail "不支持的参数：$1（使用 --help 查看用法）" 64 ;;
    esac
done
[[ "$NETWORK" =~ ^[a-zA-Z0-9_.:-]+$ ]] || fail '网卡名称不合法。' 64

verify_file() {
    local expected="$1" path="$2" actual
    [[ -f "$path" && -r "$path" ]] || fail "缺少文件或不可读：$path" 66
    actual="$(sha256sum -- "$path")"
    [[ "${actual%% *}" == "$expected" ]] || fail "文件校验不匹配，拒绝启动：$path" 65
}
command -v sha256sum >/dev/null || fail '缺少 sha256sum。' 69
verify_file 2155cb53443d96f1b294009fcf25058e701220543a41acf414a9feade83f0539 "$CONTROLLER"
verify_file ba85d67ef27b32874d4ac180c5439d72ff0c9f13452495a0b5f7881a5a907432 "$POLICY_DIR/exported/policy.onnx"
verify_file c3817799231b7a0b9e0148e4a72a943ce111c248ecd9c31388fbdc89e44a73bf "$POLICY_DIR/params/deploy.yaml"
[[ -x "$CONTROLLER" ]] || fail "控制器不可执行：$CONTROLLER" 66

# This executable has a verified RPATH for /usr/local/lib and its ARM64 ORT.
# Do not copy the old controller's /tmp/sdk2new_lib override: the DDS builds differ.
BASE_ARGS=(--network "$NETWORK" --stairs-policy "$POLICY_DIR" --stairs-speed 0.6)
if [[ "$MODE" == dry-run ]]; then
    printf '文件校验通过。以下仅为命令预览，未启动控制器：\n'
    printf '%q ' "$CONTROLLER" --first-test-actuate "${BASE_ARGS[@]}" \
        --stairs-trace "$SCRIPT_DIR/logs/<本次运行目录>/stairs_trace.csv"
    printf '\n已知限制：本脚本没有修复控制交接时的站立保持问题。\n'
    exit 0
fi
[[ -d "/sys/class/net/$NETWORK" ]] || fail "机载网卡不存在：$NETWORK" 69
if [[ "$MODE" == start ]]; then
    [[ -t 0 && -t 1 ]] || fail '启动控制器需要交互终端；SSH 请使用 ssh -tt。' 64
    command -v flock >/dev/null || fail '缺少防止重复启动所需的 flock。' 69
    exec 9>>"$SCRIPT_DIR/.controller.lock"
    flock -n 9 || fail '此启动目录已有控制器进程，请勿重复启动。' 75
else
    command -v timeout >/dev/null || fail '只读检查需要 timeout。' 69
fi

mkdir -p -- "$SCRIPT_DIR/logs"
RUN_DIR="$(mktemp -d "$SCRIPT_DIR/logs/$(date +%Y%m%d_%H%M%S)_${MODE}.XXXXXX")"
printf 'model=HIMLoco cumulative 1100\nmode=%s\nnetwork=%s\ncontroller=%s\npolicy=%s\n' \
    "$MODE" "$NETWORK" "$CONTROLLER" "$POLICY_DIR" > "$RUN_DIR/session.txt"
printf '模型：HIMLoco 1100；命令速度：0.6 m/s；网卡：%s\n日志：%s\n' "$NETWORK" "$RUN_DIR"
cd -- "$CONTROLLER_DIR"
ulimit -c 0
case "$MODE" in
    check)
        printf '只读预检：ACTIVE 或姿态不满足条件时返回非零，不会自动释放控制。\n'
        timeout --signal=TERM --kill-after=2s 12s "$CONTROLLER" \
            --first-test-preflight "${BASE_ARGS[@]}" 2>&1 | tee "$RUN_DIR/check.log"
        ;;
    infer)
        timeout --signal=TERM --kill-after=2s 15s "$CONTROLLER" \
            --robot-print --robot-print-steps 100 "${BASE_ARGS[@]}" 2>&1 | tee "$RUN_DIR/inference.log"
        ;;
    start)
        cat <<'EOF'
启动的是首次验证保护副本，尚未完成实机爬梯验证。
原厂控制仍在发送指令或姿态不满足条件时，程序会拒绝接管。
已知交接问题尚未修复；不要为绕过拒绝而临时释放原厂控制或取消限位。
通过受控交接后，按键分别发送：F → 等待站姿稳定 → 1（0.6 m/s）→ G。
G 后软件限时 2 秒；X / 空格 / 0 锁存阻尼，Ctrl-C 请求退出。
初始及停止后的阻尼不保持站立；现场支撑和急停必须持续有效。
该程序不自动恢复原厂模式，软件限时也不是独立硬件看门狗。
EOF
        # exec preserves stdin/TTY and forwards terminal signals to the controller.
        # No pipe, nohup, sudo, injected key sequence, or automatic mode release.
        exec "$CONTROLLER" --first-test-actuate "${BASE_ARGS[@]}" \
            --stairs-trace "$RUN_DIR/stairs_trace.csv"
        ;;
esac
