#!/bin/bash
# teach_cal_bringup.sh — start teach-tip calibration node.
#
# Usage (inside Isaac container):
#   bash /workspaces/isaac_ros-dev/scripts/calibration/teach_cal_bringup.sh
WS=${WS:-/workspaces/isaac_ros-dev}
BPROF=${BPROF:-$WS/bridge_both.xml}
LOGDIR=${LOGDIR:-/tmp/grabit_session}
mkdir -p "$LOGDIR" "$WS/config/calibration"

source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash" 2>/dev/null || true
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
unset CYCLONEDDS_URI
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-$BPROF}"
export GRAB_IT_GRASP_PARAMS="${GRAB_IT_GRASP_PARAMS:-$WS/config/grasp_params.yaml}"
export GRAB_IT_WS="$WS"

pkill -f 'teach_cal_node.py' 2>/dev/null || true
sleep 0.4

bash "$WS/scripts/calibration/grasp_cal_static_tf.sh" 2>/dev/null || true

echo "[teach_cal] starting (log=$LOGDIR/teach_cal.log)"
python3 "$WS/scripts/calibration/teach_cal_node.py" \
  >"$LOGDIR/teach_cal.log" 2>&1 &
echo "[teach_cal] pid=$!"
echo "[teach_cal] GUI: Free arm → touch tip → Hold → Save tip → Sleep→taught → Run pan cal"
