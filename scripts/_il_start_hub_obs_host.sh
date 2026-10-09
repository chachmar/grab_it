#!/usr/bin/env bash
# Start hub→ROS obs feed for host twin play (cameras + joint_states).
set -eo pipefail
WS="${ISAAC_ROS_WS:-$HOME/dev-ws}"
pkill -f il_hub_obs_feed.py 2>/dev/null || true
sleep 1
set +u
source /opt/ros/humble/setup.bash
set -u
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
unset CYCLONEDDS_URI || true
unset FASTRTPS_DEFAULT_PROFILES_FILE || true
export GRAB_IT_AGX_HUB_BASE="${GRAB_IT_AGX_HUB_BASE:-http://127.0.0.1:8081}"
export PYTHONPATH="/opt/ros/humble/lib/python3.10/site-packages:${PYTHONPATH:-}"
nohup python3 "${WS}/scripts/il_hub_obs_feed.py" >/tmp/il_hub_obs.log 2>&1 &
echo "obs feed started pid=$! log=/tmp/il_hub_obs.log"
sleep 2
tail -20 /tmp/il_hub_obs.log || true
