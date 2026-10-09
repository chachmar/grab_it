#!/usr/bin/env bash
set -eo pipefail
pkill -f il_twin_cmd_relay.py 2>/dev/null || true
docker exec isaac_ros_dev-aarch64-container bash -lc 'pkill -f il_twin_cmd_relay.py 2>/dev/null || true' || true
sleep 1
nohup docker exec -u admin -w /workspaces/isaac_ros-dev isaac_ros_dev-aarch64-container bash -lc '
set +u
source /opt/ros/humble/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
unset CYCLONEDDS_URI
unset FASTRTPS_DEFAULT_PROFILES_FILE
export GRAB_IT_AGX_HUB=http://192.168.50.207:8081/isaac/cmd
exec python3 /workspaces/isaac_ros-dev/scripts/il_twin_cmd_relay.py
' >/tmp/il_twin_relay.log 2>&1 &
echo "relay started pid=$!"
sleep 3
tail -20 /tmp/il_twin_relay.log || true
