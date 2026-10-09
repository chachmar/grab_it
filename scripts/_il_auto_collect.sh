#!/usr/bin/env bash
# Record N scripted grasps around the current click.
set -eo pipefail
WS="${GRAB_IT_WS:-$HOME/dev-ws}"
COUNT="${1:-5}"
source /opt/ros/humble/setup.bash
GEN="${WS}/build/interbotix_xs_msgs/rosidl_generator_py/interbotix_xs_msgs"
ROOT="${WS}/build/interbotix_xs_msgs"
MODE=real
if [ -f /tmp/grabit_mode ]; then
  MODE=$(tr -d '\r\n' </tmp/grabit_mode)
fi
if [ "$MODE" = "sim" ]; then
  PROFILE="$WS/bridge_both.xml"
else
  PROFILE="$WS/rtabmap_profile.xml"
fi
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
unset CYCLONEDDS_URI || true
export FASTRTPS_DEFAULT_PROFILES_FILE="$PROFILE"
export PYTHONPATH="${WS}/src/grab_it_il:${WS}/build/interbotix_xs_msgs/rosidl_generator_py:/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${GEN}:${ROOT}:${LD_LIBRARY_PATH:-}"
if ! pgrep -f 'grab_it_il/record_demos.py' >/dev/null; then
  bash "$WS/scripts/_il_real_record_host.sh" >>/tmp/il_record_host.log 2>&1 &
  sleep 2
fi
exec python3 "$WS/scripts/_il_auto_collect.py" "$COUNT"
