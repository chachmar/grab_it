#!/bin/bash
# il_record.sh — start the tabletop-pick demo recorder (from source if needed).
#
# Usage (Isaac container):
#   bash /workspaces/isaac_ros-dev/scripts/il_record.sh
WS=/workspaces/isaac_ros-dev
SRC="$WS/src/grab_it_il"
XPROF=${XPROF:-$WS/rtabmap_profile.xml}

source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash" 2>/dev/null || true
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
unset CYCLONEDDS_URI
export FASTRTPS_DEFAULT_PROFILES_FILE="$XPROF"
export GRAB_IT_IL_DATASET="${GRAB_IT_IL_DATASET:-$WS/datasets/tabletop_pick}"

if [ ! -f "$SRC/package.xml" ]; then
  echo "[il] missing $SRC/package.xml"
  echo "     This AGX tree does not have grab_it_il yet."
  echo "     Copy it from the Windows workspace:"
  echo "       scp -r src/grab_it_il <agx>:~/dev-ws/src/"
  echo "     then re-run this script (or colcon build --packages-select grab_it_il)."
  exit 1
fi

if ! ros2 pkg prefix grab_it_il >/dev/null 2>&1; then
  echo "[il] building grab_it_il…"
  cd "$WS"
  colcon build --packages-select grab_it_il --symlink-install
  source "$WS/install/setup.bash"
fi

if ros2 pkg prefix grab_it_il >/dev/null 2>&1; then
  exec ros2 launch grab_it_il record_demos.launch.py
fi

echo "[il] colcon install missing — running from source"
export PYTHONPATH="$SRC:${PYTHONPATH:-}"
exec python3 "$SRC/grab_it_il/record_demos.py"
