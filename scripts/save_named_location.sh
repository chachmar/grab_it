#!/bin/bash
# save_named_location.sh — write current map->base_link pose into YAML.
#
# Usage (Isaac container, nav up, Fixed Frame=map localized):
#   bash /workspaces/isaac_ros-dev/scripts/save_named_location.sh kitchen
set -e
WS=/workspaces/isaac_ros-dev
NAME=${1:?usage: $0 <location_name>}
YAML=${2:-$WS/config/named_locations.yaml}

source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
unset CYCLONEDDS_URI
export FASTRTPS_DEFAULT_PROFILES_FILE=${FASTRTPS_DEFAULT_PROFILES_FILE:-$WS/rtabmap_profile.xml}

exec python3 "$WS/scripts/save_named_location.py" "$NAME" --yaml "$YAML"
