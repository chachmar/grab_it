#!/usr/bin/env bash
# Open-loop replay of one lookreach episode on the twin (no ACT).
# Proves hub + relay + twin can execute a known-good grasp trajectory.
set -eo pipefail
WS="${ISAAC_ROS_WS:-$HOME/dev-ws}"
EP="${1:-$WS/datasets/tabletop_pick_lookreach/episode_000050}"
LOG="${2:-/tmp/il_replay_episode.log}"

if [[ ! -f "$EP/actions.npy" ]]; then
  echo "ERROR: missing $EP/actions.npy"
  exit 1
fi

set +u
source /opt/ros/humble/setup.bash
source "$HOME/lerobot-venv/bin/activate" 2>/dev/null || true
set -u

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
unset CYCLONEDDS_URI || true
unset FASTRTPS_DEFAULT_PROFILES_FILE || true
export PYTHONPATH="${WS}/install/interbotix_xs_msgs/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages:${PYTHONPATH:-}"
export PYTHONPATH="${WS}/src/grab_it_il:${PYTHONPATH}"
export AMENT_PREFIX_PATH="${WS}/install/interbotix_xs_msgs:${WS}/install/grab_it_il:${AMENT_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="${WS}/install/interbotix_xs_msgs/lib:${LD_LIBRARY_PATH:-}"

echo "[il_replay_episode] episode=$EP" | tee "$LOG"
# Omit empty checkpoint:= — ROS rejects '-p checkpoint:=' / empty override.
exec python3 "${WS}/src/grab_it_il/grab_it_il/play_policy.py" --ros-args \
  -p replay_episode:="$EP" \
  -p autoplay:=true 2>&1 | tee -a "$LOG"
