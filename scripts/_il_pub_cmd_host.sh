#!/usr/bin/env bash
# Publish IL play/stop on the AGX host (no isaac_ros container).
set -eo pipefail
CMD="${1:-play}"
WS="${ISAAC_ROS_WS:-$HOME/dev-ws}"
set +u
source /opt/ros/humble/setup.bash
set -u
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
unset CYCLONEDDS_URI || true
unset FASTRTPS_DEFAULT_PROFILES_FILE || true
export PYTHONPATH="${WS}/install/interbotix_xs_msgs/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages:${PYTHONPATH:-}"
export AMENT_PREFIX_PATH="${WS}/install/interbotix_xs_msgs:${AMENT_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="${WS}/install/interbotix_xs_msgs/lib:${LD_LIBRARY_PATH:-}"
python3 - <<PY
import json, rclpy
from rclpy.node import Node
from std_msgs.msg import String
rclpy.init()
n = Node("il_cmd_once")
pub = n.create_publisher(String, "/grab_it/il_command", 10)
# let discovery settle
import time; time.sleep(0.5)
msg = String(data=json.dumps({"cmd": "${CMD}"}))
for _ in range(3):
    pub.publish(msg)
    time.sleep(0.15)
print(f"published il_command cmd=${CMD}")
n.destroy_node()
rclpy.shutdown()
PY
