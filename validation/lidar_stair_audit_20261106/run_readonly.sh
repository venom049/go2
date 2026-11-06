#!/usr/bin/env bash
set -eo pipefail
ROOT=/home/unitree/go2-lidar-audit-20261106
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
source /opt/ros/foxy/setup.bash
source /home/unitree/unitree_ros2/cyclonedds_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="eth1"/></Interfaces></General></Domain></CycloneDDS>'
timeout 20 /usr/bin/python3 "$ROOT/ros_inventory.py" >"$ROOT/ros_inventory.json" 2>"$ROOT/ros_inventory_stderr.log"
cd /home/unitree/go2-stairs-perception-20261103
timeout 25 /usr/bin/python3 -m go2_stairs.lidar_watch --duration 12 --allow-unsynced-preview \
  --output "$ROOT/live_cloud_report.json" --capture "$ROOT/live_cloud.npz" \
  >"$ROOT/live_cloud_stdout.log" 2>"$ROOT/live_cloud_stderr.log"
