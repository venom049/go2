#!/bin/bash
  source /opt/ros/foxy/setup.bash
  ros2 run tf2_ros static_transform_publisher 0.33 0.0 0.04 0 0 0 base_link camera_link
