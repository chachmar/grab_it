#!/usr/bin/env bash
# Host-side twin relay (no isaac_ros container required).
set -eo pipefail
WS="${ISAAC_ROS_WS:-$HOME/dev-ws}"
pkill -f il_twin_cmd_relay.py 2>/dev/null || true
sleep 1
set +u
source /opt/ros/humble/setup.bash
set -u
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
unset CYCLONEDDS_URI || true
unset FASTRTPS_DEFAULT_PROFILES_FILE || true
export GRAB_IT_AGX_HUB="${GRAB_IT_AGX_HUB:-http://127.0.0.1:8081/isaac/cmd}"
export PYTHONPATH="${WS}/install/interbotix_xs_msgs/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages:${PYTHONPATH:-}"
export AMENT_PREFIX_PATH="${WS}/install/interbotix_xs_msgs:${AMENT_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="${WS}/install/interbotix_xs_msgs/lib:${LD_LIBRARY_PATH:-}"
nohup python3 "${WS}/scripts/il_twin_cmd_relay.py" >/tmp/il_twin_relay.log 2>&1 &
echo "relay started pid=$! log=/tmp/il_twin_relay.log"
sleep 2
tail -20 /tmp/il_twin_relay.log || true
