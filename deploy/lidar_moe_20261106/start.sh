#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/unitree/go2-lidar-moe-20261106
cd "$ROOT"
MODE=${1:---dry-run}
[[ $# -le 1 ]] || exit 64
case "$MODE" in --dry-run|--sense|--sense-factory|--readonly|--infer|--handoff-probe|--mode-check|--shutdown-selftest|--start|--commission) ;; *) echo 'Modes: --dry-run --sense --sense-factory --readonly --infer --handoff-probe --mode-check --shutdown-selftest --start --commission';exit 64;; esac
sha256sum --check --quiet SHA256SUMS
[[ -f BINARY_SHA256SUMS ]] || { echo 'Missing verified native build';exit 1; }
sha256sum --check --quiet BINARY_SHA256SUMS
if [[ $MODE == --start ]];then
 echo '运动入口暂时禁用：接管后感知连续性及完整爬阶尚待验收。可使用 --readonly。' >&2
 exit 78
fi
if [[ $MODE == --commission ]];then
 echo '新独立感知及腿部防撞尚未通过运动验收；本版接管入口禁用，先完成只读和仿真验证。' >&2
 exit 78
fi
BIN="$ROOT/controller/robots/go2/build/go2_ctrl"
ARGS=(--network eth1 --stairs-policy "$ROOT/model")
PYTHON=/home/unitree/go2-stairs-handoff-20261103/.venv/bin/python
export CYCLONEDDS_HOME=/home/unitree/go2-stairs-handoff-20261103/dds_runtime_build
export LD_LIBRARY_PATH="$CYCLONEDDS_HOME/lib:/usr/local/lib"
export GCOV_PREFIX="$ROOT/logs/gcov" GCOV_PREFIX_STRIP=3
export PYTHONPATH=/home/unitree/go2-stairs-handoff-20261103/python_sdk
if [[ $MODE == --dry-run ]]; then exec "$BIN";fi
if [[ $MODE == --shutdown-selftest ]];then exec "$BIN" --thread-shutdown-selftest;fi
if [[ $MODE == --infer ]]; then exec "$BIN" "${ARGS[@]}" --robot-print --robot-print-steps 100;fi
if [[ $MODE == --mode-check ]]; then exec "$PYTHON" "$ROOT/motion_rpc_bridge.py" --network eth1;fi
if [[ $MODE == --handoff-probe ]]; then exec "$BIN" "${ARGS[@]}" --moe-handoff-probe;fi
if [[ $MODE != --sense-factory ]];then
 [[ -f LIO_BINARY_SHA256SUMS ]] || { echo 'Missing verified independent LIO build';exit 1; }
 sha256sum --check --quiet LIO_BINARY_SHA256SUMS
fi
if [[ $MODE == --commission ]]; then [[ -t 0 && -t 1 ]] || { echo 'Attended TTY required';exit 64; };fi
exec 9>"$ROOT/.controller.lock"
flock -n 9 || { echo 'This deployment already has an active launcher';exit 1; }
for s in perception.sock perception_client.sock motion_rpc.sock; do
 [[ ! -e "$ROOT/run/$s" ]] || { echo "Existing socket $s; inspect previous session";exit 1; }
done
RUN="$ROOT/logs/$(date +%Y%m%dT%H%M%S)-$$"
mkdir -p "$RUN"
SENSOR_PID='';BRIDGE_PID=''
cleanup() {
 if [[ -n $SENSOR_PID ]];then kill "$SENSOR_PID" 2>/dev/null || true;wait "$SENSOR_PID" 2>/dev/null || true;fi
 if [[ -n $BRIDGE_PID ]];then kill "$BRIDGE_PID" 2>/dev/null || true;wait "$BRIDGE_PID" 2>/dev/null || true;fi
 # We hold the exclusive lock and verified absence before creating any child.
 rm -f "$ROOT/run/perception.sock" "$ROOT/run/perception_client.sock" "$ROOT/run/motion_rpc.sock"
}
trap cleanup EXIT
sensor() {
 set +u
 source /opt/ros/foxy/setup.bash
 source /home/unitree/go2_foxy/go2/install/setup.bash
 source "$ROOT/independent_lio/install/setup.bash"
 set -u
 export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=0
 export CYCLONEDDS_URI=/home/unitree/cyclonedds_ws/cyclonedds.xml
 export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"
 export PYTHONPATH="$ROOT:${PYTHONPATH:-}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
 exec /usr/bin/python3 -m lidar_moe.sensor_service --root "$ROOT" --config "$ROOT/config/supervisor_independent.json" --seconds 3600 --output "$RUN/supervisor.jsonl" "$@"
}
echo "Logs: $RUN"
if [[ $MODE == --sense || $MODE == --sense-factory ]];then
 set +u
 source /opt/ros/foxy/setup.bash
 source /home/unitree/go2_foxy/go2/install/setup.bash
 if [[ $MODE == --sense ]];then source "$ROOT/independent_lio/install/setup.bash";fi
 set -u
 export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=0 CYCLONEDDS_URI=/home/unitree/cyclonedds_ws/cyclonedds.xml
 export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}" PYTHONPATH="$ROOT:${PYTHONPATH:-}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
 if [[ $MODE == --sense ]];then
  /usr/bin/python3 -m lidar_moe.independent_service --root "$ROOT" --config "$ROOT/config/supervisor_independent.json" --seconds 3600 --output "$RUN/supervisor.jsonl"
 else
  /usr/bin/python3 -m lidar_moe.ros_bridge --root "$ROOT" --seconds 3600 --output "$RUN/supervisor.jsonl"
 fi
 exit
fi
sensor >"$RUN/sensor.log" 2>&1 &
SENSOR_PID=$!
if [[ $MODE == --readonly ]];then
 "$BIN" "${ARGS[@]}" --lidar-readonly >"$RUN/controller.log" 2>&1
 cat "$RUN/controller.log"
 exit
fi
"$PYTHON" "$ROOT/motion_rpc_bridge.py" --network eth1 --allow-release >"$RUN/bridge.log" 2>&1 &
BRIDGE_PID=$!
for ((i=0;i<50;i++));do
 kill -0 "$BRIDGE_PID" "$SENSOR_PID" 2>/dev/null || { cat "$RUN/sensor.log" "$RUN/bridge.log";exit 1; }
 [[ ! -S "$ROOT/run/motion_rpc.sock" ]] || break
 sleep .1
done
[[ -S "$ROOT/run/motion_rpc.sock" ]] || exit 1
echo '限时实机试验：现场承重支撑和急停须就绪；全部现有保护继续生效。'
echo 'H：稳定站姿和雷达确认后交接；等待 STAND_LOCKED、POST_HANDOFF_SENSORS_LOCKED 和新几何，1 武装、G 开始。'
echo '最多运动 20 秒；正常停车保持站姿，故障及 X/空格/0 请求锁存阻尼。'
"$BIN" "${ARGS[@]}" --protected-handoff --stairs-trace "$RUN/policy.csv"
