#!/usr/bin/env bash
set -eo pipefail
CMD="${1:-play}"
docker exec -u admin -w /workspaces/isaac_ros-dev isaac_ros_dev-aarch64-container bash -lc "
set +u
source /opt/ros/humble/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
unset CYCLONEDDS_URI
unset FASTRTPS_DEFAULT_PROFILES_FILE
python3 /workspaces/isaac_ros-dev/scripts/_il_pub_cmd.py $CMD
"
