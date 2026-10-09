#!/bin/bash
# go_to_named.sh — navigate to a named pose (nav_bringup already running).
#
# Usage (Isaac container):
#   bash /workspaces/isaac_ros-dev/scripts/go_to_named.sh kitchen
set -e
WS=/workspaces/isaac_ros-dev
NAME=${1:?usage: $0 <location_name>}
YAML=${2:-$WS/config/named_locations.yaml}

source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash" 2>/dev/null
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
unset CYCLONEDDS_URI
# Prefer crossover profile when camera/RTAB live; fall back to both for wifi-only
if [ -f "$WS/rtabmap_profile.xml" ]; then
  export FASTRTPS_DEFAULT_PROFILES_FILE=${FASTRTPS_DEFAULT_PROFILES_FILE:-$WS/rtabmap_profile.xml}
fi

exec python3 "$WS/scripts/go_to_named.py" "$NAME" --yaml "$YAML"
