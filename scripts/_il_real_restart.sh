#!/usr/bin/env bash
# Bring back the live AGX stack: camera relay, GUI, click-to-grasp, idle ACT.
# Nano cameras and the arm driver must already be up.
set -eo pipefail
WS=/home/ubuntu/dev-ws
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
unset CYCLONEDDS_URI || true
export FASTRTPS_DEFAULT_PROFILES_FILE="$WS/rtabmap_profile.xml"
export GRAB_IT_IGNORE_ISAAC=1

if ! pgrep -f 'dev-ws/mjpeg_relay.py' >/dev/null; then
  nohup python3 "$WS/mjpeg_relay.py" >/tmp/mjpeg_relay.log 2>&1 &
  echo "mjpeg_pid=$!"
fi

if ! pgrep -f 'grab_it_gui/gui_node.py' >/dev/null; then
  GEN="${WS}/build/interbotix_xs_msgs/rosidl_generator_py/interbotix_xs_msgs"
  ROOT="${WS}/build/interbotix_xs_msgs"
  export PYTHONPATH="/home/ubuntu/cosmos3-venv/lib/python3.10/site-packages:${WS}/build/interbotix_xs_msgs/rosidl_generator_py:/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages:${WS}/src/grab_it_gui:${PYTHONPATH:-}"
  export LD_LIBRARY_PATH="${GEN}:${ROOT}:${LD_LIBRARY_PATH:-}"
  nohup python3 "$WS/src/grab_it_gui/grab_it_gui/gui_node.py" >/tmp/gui_node.log 2>&1 &
  echo "gui_pid=$!"
fi

bash "$WS/scripts/_il_real_ensure.sh"
echo "GUI http://192.168.50.207:8765"
