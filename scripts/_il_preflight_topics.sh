#!/usr/bin/env bash
set -eo pipefail
WS="${HOME}/dev-ws"
set +u
source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash" 2>/dev/null || true
set -u
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
unset CYCLONEDDS_URI || true
export FASTRTPS_DEFAULT_PROFILES_FILE="${WS}/bridge_both.xml"

echo "=== topic list (filtered) ==="
timeout 8 ros2 topic list 2>/dev/null | grep -E 'joint|d435i|wide_cam|il_|grab_it' | sort || true

echo "=== hz joint_states (8s) ==="
timeout 8 ros2 topic hz /px150_no_elbow/joint_states 2>&1 | tail -8 || true

echo "=== hz d435i raw (8s) ==="
timeout 8 ros2 topic hz /d435i/d435i/color/image_raw 2>&1 | tail -8 || true

echo "=== hz d435i compressed (8s) ==="
timeout 8 ros2 topic hz /d435i/d435i/color/image_raw/compressed 2>&1 | tail -8 || true

echo "=== hz joint_targets (3s) ==="
timeout 3 ros2 topic hz /px150_no_elbow/commands/joint_targets 2>&1 | tail -5 || true
