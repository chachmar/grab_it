#!/usr/bin/env bash
set -eo pipefail
WS=/home/ubuntu/dev-ws
pids=$(pgrep -f 'grab_it_il/play_policy.py' || true)
if [[ -n "$pids" ]]; then
  # shellcheck disable=SC2086
  kill $pids 2>/dev/null || true
  echo "ACT stopped"
fi
source /opt/ros/humble/setup.bash
GEN="${WS}/build/interbotix_xs_msgs/rosidl_generator_py/interbotix_xs_msgs"
ROOT="${WS}/build/interbotix_xs_msgs"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
unset CYCLONEDDS_URI || true
export FASTRTPS_DEFAULT_PROFILES_FILE="$WS/rtabmap_profile.xml"
export PYTHONPATH="${WS}/build/interbotix_xs_msgs/rosidl_generator_py:/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${GEN}:${ROOT}:${LD_LIBRARY_PATH:-}"
exec python3 "$WS/scripts/_il_to_demo_pose.py"
