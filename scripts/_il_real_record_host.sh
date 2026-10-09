#!/usr/bin/env bash
# Host-side demo recorder. Idle until the GUI REC button sends start.
set -eo pipefail
WS="${GRAB_IT_WS:-$HOME/dev-ws}"
if pgrep -f 'grab_it_il/record_demos.py' >/dev/null; then
  echo "recorder already running"
  exit 0
fi
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
if [ -z "${GRAB_IT_IL_DATASET:-}" ] && [ -f /tmp/grabit_collection ]; then
  _col=$(tr -d '\r\n' </tmp/grabit_collection)
  case "$_col" in
    ""|.*|*/*|*..*) ;;
    *) GRAB_IT_IL_DATASET="$WS/datasets/$_col" ;;
  esac
fi
export GRAB_IT_IL_DATASET="${GRAB_IT_IL_DATASET:-$WS/datasets/tabletop_pick}"
export PYTHONPATH="${WS}/src/grab_it_il:${WS}/build/interbotix_xs_msgs/rosidl_generator_py:/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${GEN}:${ROOT}:${LD_LIBRARY_PATH:-}"
mkdir -p "$GRAB_IT_IL_DATASET"
exec python3 "$WS/src/grab_it_il/grab_it_il/record_demos.py" --ros-args \
  -p dataset_dir:="$GRAB_IT_IL_DATASET"
