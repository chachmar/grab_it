#!/bin/bash
# il_play.sh — play ACT checkpoint or latest recorded episode.
#
# Usage (Isaac container):
#   bash /workspaces/isaac_ros-dev/scripts/il_play.sh
#   bash .../il_play.sh /path/to/pretrained_model
WS=/workspaces/isaac_ros-dev
SRC="$WS/src/grab_it_il"
XPROF=${XPROF:-$WS/rtabmap_profile.xml}
CKPT="${1:-}"

source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash" 2>/dev/null || true
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
unset CYCLONEDDS_URI
export FASTRTPS_DEFAULT_PROFILES_FILE="$XPROF"
export GRAB_IT_IL_DATASET="${GRAB_IT_IL_DATASET:-$WS/datasets/tabletop_pick}"

if [ ! -f "$SRC/package.xml" ]; then
  echo "[il] missing $SRC/package.xml — copy src/grab_it_il onto this workspace first."
  exit 1
fi

if ! ros2 pkg prefix grab_it_il >/dev/null 2>&1; then
  echo "[il] building grab_it_il…"
  cd "$WS"
  colcon build --packages-select grab_it_il --symlink-install
  source "$WS/install/setup.bash"
fi

if ros2 pkg prefix grab_it_il >/dev/null 2>&1; then
  if [ -n "$CKPT" ]; then
    exec ros2 launch grab_it_il play_policy.launch.py checkpoint:="$CKPT"
  fi
  exec ros2 launch grab_it_il play_policy.launch.py
fi

export PYTHONPATH="$SRC:${PYTHONPATH:-}"
if [ -n "$CKPT" ]; then
  exec python3 "$SRC/grab_it_il/play_policy.py" --ros-args -p checkpoint:="$CKPT"
fi
exec python3 "$SRC/grab_it_il/play_policy.py"
