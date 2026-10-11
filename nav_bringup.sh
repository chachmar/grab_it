#!/bin/bash
# nav_bringup.sh — thin wrapper around grab_it_nav.
#
# Existing container command (unchanged):
#   bash /workspaces/isaac_ros-dev/nav_bringup.sh
#
# grabit_session.sh calls this file. The process list, parameters, DDS profiles,
# and gates live in:
#   ros2 launch grab_it_nav nav.launch.py
#
# One-time, inside the Isaac container, after this package lands:
#   cd /workspaces/isaac_ros-dev
#   colcon build --packages-select grab_it_nav --symlink-install
#   source install/setup.bash
#
# Paths below are the same files nav_bringup.sh used before the package existed.
# The originals are still in the repo; this wrapper points the launch at them.
set -m
WS=${GRABIT_WS:-/workspaces/isaac_ros-dev}

# Do not enable `set -u` before sourcing ROS — ament setup refs unbound vars.
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source "$WS/install/setup.bash" 2>/dev/null || true
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
unset CYCLONEDDS_URI

if ! ros2 pkg prefix grab_it_nav >/dev/null 2>&1; then
  echo "[nav] ERROR: grab_it_nav is not on the ROS package path."
  echo "       cd $WS && colcon build --packages-select grab_it_nav --symlink-install"
  echo "       source $WS/install/setup.bash"
  exit 1
fi

exec ros2 launch grab_it_nav nav.launch.py \
  "workspace:=$WS" \
  "database_path:=$WS/rtabmap.db" \
  "params_file:=$WS/config/nav2_params_create3.yaml" \
  "bridge_yaml:=$WS/create3_bridge.yaml" \
  "crossover_profile:=$WS/rtabmap_profile.xml" \
  "bridge_profile:=$WS/bridge_both.xml" \
  "camera_tf_env:=$WS/config/camera_tf.env"
