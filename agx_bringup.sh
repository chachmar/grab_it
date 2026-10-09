#!/bin/bash
# agx_bringup.sh — GUI + MJPEG relay only (no Nav2).
# For full session use: bash grabit_session.sh
#
# Isaac twin on Windows (same ROS_DOMAIN_ID=0 LAN):
#   export GRAB_IT_SIM=1
#   bash /workspaces/isaac_ros-dev/agx_bringup.sh
#   # or: bash agx_bringup.sh --sim
set -m
WS=/workspaces/isaac_ros-dev
SIM=0
SKIP_MJPEG=0
if [ "${GRAB_IT_SIM:-0}" = "1" ] || [ "${GRAB_IT_SIM:-}" = "true" ]; then
  SIM=1
fi
if [ "${GRAB_IT_SKIP_MJPEG:-0}" = "1" ]; then
  SKIP_MJPEG=1
fi
for arg in "$@"; do
  case "$arg" in
    --sim) SIM=1 ;;
    --no-mjpeg) SKIP_MJPEG=1 ;;
    -h|--help)
      echo "Usage: $0 [--sim] [--no-mjpeg]"
      echo "  --sim or GRAB_IT_SIM=1  GUI on LAN domain 0 (Isaac Windows)"
      echo "  --no-mjpeg              leave :8081 to agx_isaac_hub (grabit_mode.sh sim)"
      exit 0
      ;;
  esac
done
NEED_APT=""
dpkg -s ros-humble-irobot-create-msgs          >/dev/null 2>&1 || NEED_APT="$NEED_APT ros-humble-irobot-create-msgs"
dpkg -s ros-humble-dynamixel-workbench-toolbox  >/dev/null 2>&1 || NEED_APT="$NEED_APT ros-humble-dynamixel-workbench-toolbox"
if [ -n "$NEED_APT" ]; then
  echo "[grab_it] installing:$NEED_APT"
  sudo apt-get update --allow-releaseinfo-change -qq 2>/dev/null || true
  sudo apt-get install -y $NEED_APT || echo "[grab_it] WARN: apt install failed"
fi
python3 -c 'import websockets' 2>/dev/null || pip3 install -q websockets --break-system-packages 2>/dev/null || pip3 install -q websockets
if [ ! -f "$WS/install/interbotix_xs_msgs/share/interbotix_xs_msgs/package.xml" ]; then
  echo "[grab_it] building workspace (one-time)…"
  cd "$WS"; source /opt/ros/humble/setup.bash
  colcon build --packages-up-to interbotix_xsarm_control grab_it_cameras grab_it_description \
    --packages-skip interbotix_slate_driver interbotix_slate_msgs \
                    interbotix_footswitch_driver interbotix_footswitch_msgs --symlink-install
  colcon build --packages-select grab_it_gui --symlink-install
fi
source /opt/ros/humble/setup.bash 2>/dev/null
source "$WS/install/setup.bash"
unset CYCLONEDDS_URI
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0

# GUI + MJPEG on bridge_both (LAN 192.168.50.207 + crossover 10.42.0.1).
# Drive/arm also hop UDP → wifi relays so Windows Isaac sees commands even
# if a leftover GUI process is still crossover-only.
BPROF=$WS/bridge_both.xml
XPROF=$WS/crossover_only.xml
[ -f "$WS/rtabmap_profile.xml" ] && XPROF=$WS/rtabmap_profile.xml
# LAN + crossover so Windows Isaac and Nano are both visible. Drive/arm
# still hop UDP → wifi relays (same as Create 3).
GUI_PROF="$BPROF"
if [ "$SIM" = "1" ]; then
  export GRAB_IT_SIM=1
fi

# Real grasp sessions use grabit_mode.sh real (crossover profile, no Isaac JPEG).
# Sim sessions leave :8081 to agx_isaac_hub so PLAY and cameras share one port.
if [ "$SKIP_MJPEG" = "1" ]; then
  echo "[grab_it] MJPEG skipped — agx_isaac_hub owns :8081"
  RELAY_PID=""
else
  FASTRTPS_DEFAULT_PROFILES_FILE="$BPROF" python3 "$WS/mjpeg_relay.py" > /tmp/relay.log 2>&1 &
  RELAY_PID=$!
fi
FASTRTPS_DEFAULT_PROFILES_FILE="$BPROF" python3 "$WS/scripts/cmd_vel_wifi_relay.py" > /tmp/cmd_vel_wifi.log 2>&1 &
VEL_PID=$!
FASTRTPS_DEFAULT_PROFILES_FILE="$BPROF" python3 "$WS/scripts/arm_cmd_wifi_relay.py" > /tmp/arm_cmd_wifi.log 2>&1 &
ARM_PID=$!
FASTRTPS_DEFAULT_PROFILES_FILE="$GUI_PROF" python3 "$WS/src/grab_it_gui/grab_it_gui/gui_node.py" > /tmp/gui.log 2>&1 &
GUI_PID=$!
if [ "$SIM" = "1" ]; then
  echo "[grab_it] SIM mode: GUI + MJPEG on LAN domain 0 (Isaac). Confirm from this container:"
  echo "         ros2 topic hz /px150_no_elbow/joint_states"
else
  echo "[grab_it] Note: domain_bridge must be running for Create3 (nav_bringup / grabit_session)."
fi
echo "[grab_it] AGX up: relay=$RELAY_PID vel=$VEL_PID arm=$ARM_PID gui=$GUI_PID → GUI http://192.168.50.207:8765 (Ctrl-C stops all)"
trap "kill $RELAY_PID $VEL_PID $ARM_PID $GUI_PID 2>/dev/null; wait" INT TERM
wait
