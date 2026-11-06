#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/unitree/go2-original-bounded-20261106
cd "$ROOT"
MODE=${1:---dry-run}
[[ $# -le 1 ]] || { echo 'Only one mode is accepted.' >&2; exit 64; }
case "$MODE" in
  --dry-run|--infer|--handoff-probe|--mode-check|--start) ;;
  *) echo 'Modes: --dry-run, --infer, --handoff-probe, --mode-check, --start' >&2; exit 64 ;;
esac
sha256sum --check --quiet SHA256SUMS
[[ -f BINARY_SHA256SUMS ]] || { echo 'Controller build verification is missing.' >&2; exit 1; }
sha256sum --check --quiet BINARY_SHA256SUMS
BIN="$ROOT/controller/robots/go2/build/go2_ctrl"
ARGS=(--network eth0 --stairs-policy "$ROOT/model")
PYTHON=/home/unitree/go2-stairs-handoff-20261103/.venv/bin/python
export PYTHONPATH=/home/unitree/go2-stairs-handoff-20261103/python_sdk
export CYCLONEDDS_HOME=/home/unitree/go2-stairs-handoff-20261103/dds_runtime_build
export LD_LIBRARY_PATH="$CYCLONEDDS_HOME/lib"
if [[ "$MODE" == --dry-run ]]; then
  echo 'Original RoboGauge weights + SDK target clamp; DDS interface eth0; no motor commands.'
  "$BIN"
  exit
fi
if [[ "$MODE" == --mode-check ]]; then
  exec "$PYTHON" "$ROOT/motion_rpc_bridge.py" --network eth0
fi
if [[ "$MODE" == --infer ]]; then
  exec "$BIN" "${ARGS[@]}" --robot-print --robot-print-steps 100
fi
if [[ "$MODE" == --handoff-probe ]]; then
  exec "$BIN" "${ARGS[@]}" --moe-handoff-probe
fi
[[ -t 0 && -t 1 ]] || { echo 'Use an attended terminal: ssh -tt unitree@192.168.1.180' >&2; exit 64; }
exec 9>"$ROOT/.controller.lock"
flock -n 9 || { echo 'This deployment already has an active launcher.' >&2; exit 1; }
[[ ! -e "$ROOT/validation/motion_rpc.sock" ]] || { echo 'Existing RPC socket; inspect the previous session first.' >&2; exit 1; }
RUN="$ROOT/logs/$(date +%Y%m%dT%H%M%S)-$$"
mkdir -p "$RUN"
echo '受保护交接：H 请求交接，等待 STAND_LOCKED；1 选择 0.6 m/s，再按 G 启动策略。'
echo '策略最多运行 2 秒。X / 空格 / 0 锁存阻尼；固定阻尼不能保证站立。现场支撑须保持有效。'
echo "Logs: $RUN"
"$PYTHON" "$ROOT/motion_rpc_bridge.py" --network eth0 --allow-release >"$RUN/bridge.log" 2>&1 &
BRIDGE_PID=$!
cleanup() { kill "$BRIDGE_PID" 2>/dev/null || true; wait "$BRIDGE_PID" 2>/dev/null || true; }
trap cleanup EXIT
READY=0
for ((i=0;i<50;i++)); do
  kill -0 "$BRIDGE_PID" 2>/dev/null || { cat "$RUN/bridge.log"; exit 1; }
  if [[ -S "$ROOT/validation/motion_rpc.sock" ]]; then READY=1; break; fi
  sleep 0.1
done
[[ $READY == 1 ]] || { echo 'RPC bridge did not become ready; no takeover requested.' >&2; exit 1; }
"$BIN" "${ARGS[@]}" --protected-handoff --stairs-trace "$RUN/policy.csv"
