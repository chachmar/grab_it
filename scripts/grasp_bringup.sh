#!/bin/bash
# grasp_bringup.sh — start click-to-grasp node (requires MoveIt + cameras already up).
#
# Usage (inside Isaac container):
#   bash /workspaces/isaac_ros-dev/scripts/grasp_bringup.sh
#   bash .../grasp_bringup.sh --foreground
WS=/workspaces/isaac_ros-dev
XPROF=${XPROF:-$WS/rtabmap_profile.xml}
LOGDIR=${LOGDIR:-/tmp/grabit_session}
mkdir -p "$LOGDIR"

FOREGROUND=false
for arg in "$@"; do
  case "$arg" in
    --foreground) FOREGROUND=true ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
  esac
done

source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash" 2>/dev/null || true
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
unset CYCLONEDDS_URI
export GRAB_IT_GRASP_PARAMS="${GRAB_IT_GRASP_PARAMS:-$WS/config/grasp_params.yaml}"
export FASTRTPS_DEFAULT_PROFILES_FILE="$XPROF"

if ! ros2 pkg prefix grab_it_grasp >/dev/null 2>&1; then
  echo "[grasp] building grab_it_grasp…"
  cd "$WS"
  colcon build --packages-select grab_it_grasp --symlink-install
  source "$WS/install/setup.bash"
fi

pkill -f 'grasp_from_click' 2>/dev/null || true
sleep 0.5

bash "$WS/scripts/calibration/grasp_cal_static_tf.sh" 2>/dev/null || true

echo "[grasp] starting grasp_from_click (log=$LOGDIR/grasp.log)"
if [ "$FOREGROUND" = "true" ]; then
  ros2 run grab_it_grasp grasp_from_click
else
  ros2 run grab_it_grasp grasp_from_click >"$LOGDIR/grasp.log" 2>&1 &
  echo "[grasp] pid=$!"
fi
