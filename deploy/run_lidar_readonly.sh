#!/usr/bin/env bash
# Run only the bounded sensor subscriber; never a motor controller.
set -eo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
source /opt/ros/foxy/setup.bash
source /home/unitree/unitree_ros2/cyclonedds_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="eth0"/></Interfaces></General></Domain></CycloneDDS>'
exec /usr/bin/python3 -m go2_stairs.lidar_watch "$@"
