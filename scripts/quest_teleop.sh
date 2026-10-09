#!/usr/bin/env bash
# Quest 3 → PX150 teleop (domain 0). Args pass through to the launch file.
#   bash scripts/quest_teleop.sh
#   bash scripts/quest_teleop.sh target:=real
#   bash scripts/quest_teleop.sh target:=both allow_both:=true
set -eo pipefail
WS="${WS:-/workspaces/isaac_ros-dev}"
# Humble setup.bash reads unset AMENT_* vars; nounset must stay off.
set +u
source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash" 2>/dev/null || true
set -u
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
unset FASTRTPS_DEFAULT_PROFILES_FILE
unset CYCLONEDDS_URI
exec ros2 launch grab_it_quest quest_teleop.launch.py "$@"
