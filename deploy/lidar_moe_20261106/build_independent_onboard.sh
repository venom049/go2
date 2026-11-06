#!/usr/bin/env bash
# Builds sensor-only programs in this independent deployment; no robot RPC.
set -eo pipefail
ROOT=/home/unitree/go2-lidar-moe-20261106
cd "$ROOT"
sha256sum --check --quiet SHA256SUMS
source /opt/ros/foxy/setup.bash
source /home/unitree/go2_foxy/go2/install/setup.bash
cd independent_lio
MAKEFLAGS=-j1 colcon build --packages-select moe_fast_lio --executor sequential
cd "$ROOT"
sha256sum independent_lio/install/moe_fast_lio/lib/moe_fast_lio/moe_body_imu_bridge independent_lio/install/moe_fast_lio/lib/moe_fast_lio/moe_fastlio_mapping > LIO_BINARY_SHA256SUMS
