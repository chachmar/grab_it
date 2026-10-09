#!/usr/bin/env bash
set -euo pipefail
docker exec -u admin -w /workspaces/isaac_ros-dev isaac_ros_dev-aarch64-container bash -lc '
set +u
source /opt/ros/humble/setup.bash
source /workspaces/isaac_ros-dev/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
unset CYCLONEDDS_URI
export FASTRTPS_DEFAULT_PROFILES_FILE=/workspaces/isaac_ros-dev/bridge_both.xml
echo nodes:
ros2 node list || true
echo topics_count:
ros2 topic list | wc -l
echo topics:
ros2 topic list || true
echo play_procs:
ps aux | grep play_policy | grep -v grep || true
'
