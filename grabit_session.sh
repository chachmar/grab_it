#!/bin/bash
# grabit_session.sh — AGX operator session: Nav2 + arm/base GUI + RViz + keyboard teleop
#
# Prereqs (start separately; this script does not bring up Nano/Create3):
#   1. Nano:  sudo docker start grab_it_nano && docker exec -it grab_it_nano /root/nano_bringup.sh
#      (cameras + PX150 arm on /dev/ttyDXL, 12V on)
#   2. Create 3: Wi‑Fi, ROS domain 5, undocked, publishing /odom
#   3. AGX:   bash ~/dev-ws/run_grabit_dev.sh   (Isaac container up)
#
# Run INSIDE the Isaac container:
#   bash /workspaces/isaac_ros-dev/grabit_session.sh
#
# Options:
#   --skip-nav     assume nav_bringup already running
#   --no-gui       skip web GUI + mjpeg relay
#   --no-rviz      skip Nav2 RViz
#   --no-teleop    skip keyboard teleop (session waits; Ctrl‑C to stop)
#   --kill-nav     on exit, also tear down nav_bringup helpers (default: leave nav up)
#   --no-moveit    skip MoveIt move_group + grab_it robot_description
#   --moveit-rviz  also open MoveIt RViz (default: move_group only; Nav2 RViz still separate)
#
# Notes:
#   - Teleop + GUI + Nav2 all use /cmd_vel — cancel Nav2 goals before driving manually.
#   - Keep teleop on domain 0 (bridge). Never put extra clients on domain 5.
#   - GUI: http://<AGX-IP>:8765   MJPEG: http://<AGX-IP>:8081/d435i

WS=/workspaces/isaac_ros-dev
XPROF=$WS/rtabmap_profile.xml
BPROF=$WS/bridge_both.xml
LOGDIR=/tmp/grabit_session
mkdir -p "$LOGDIR"

SKIP_NAV=0
NO_GUI=0
NO_RVIZ=0
NO_TELEOP=0
KILL_NAV=0
NO_MOVEIT=0
MOVEIT_RVIZ=0
for arg in "$@"; do
  case "$arg" in
    --skip-nav) SKIP_NAV=1 ;;
    --no-gui) NO_GUI=1 ;;
    --no-rviz) NO_RVIZ=1 ;;
    --no-teleop) NO_TELEOP=1 ;;
    --kill-nav) KILL_NAV=1 ;;
    --no-moveit) NO_MOVEIT=1 ;;
    --moveit-rviz) MOVEIT_RVIZ=1 ;;
    -h|--help)
      sed -n '2,32p' "$0"
      exit 0
      ;;
    *)
      echo "[session] unknown arg: $arg (try --help)"
      exit 1
      ;;
  esac
done

# Do not enable `set -u` before sourcing ROS — ament setup refs unbound vars.
source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash" 2>/dev/null || true
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
unset CYCLONEDDS_URI
set -u

PIDS=()
NAV_PID=""

cleanup() {
  echo ""
  echo "[session] stopping GUI / RViz / teleop helpers..."
  for p in "${PIDS[@]:-}"; do
    kill "$p" 2>/dev/null || true
  done
  pkill -f 'teleop_twist_keyboard' 2>/dev/null || true
  pkill -f 'nav2_bringup/rviz_launch' 2>/dev/null || true
  pkill -f '/opt/ros/humble/lib/rviz2/rviz2' 2>/dev/null || true
  pkill -f 'grab_it_gui|gui_node.py' 2>/dev/null || true
  pkill -f 'mjpeg_relay.py' 2>/dev/null || true
  pkill -f 'cmd_vel_wifi_relay.py' 2>/dev/null || true
  pkill -f 'xsarm_moveit.launch.py' 2>/dev/null || true
  pkill -f 'grab_it_launch.py' 2>/dev/null || true
  pkill -f 'grab_it_robot_state_publisher' 2>/dev/null || true
  pkill -f 'arm_joint_states_bridge.py' 2>/dev/null || true
  pkill -f 'filter_arm_joint_states.py' 2>/dev/null || true
  pkill -f 'grasp_from_click' 2>/dev/null || true
  pkill -f 'move_group' 2>/dev/null || true
  if [ "$KILL_NAV" = "1" ] && [ -n "${NAV_PID}" ]; then
    echo "[session] --kill-nav: stopping nav_bringup (pid $NAV_PID)..."
    kill "$NAV_PID" 2>/dev/null || true
    # nav_bringup cleans its own helpers on exit; give it a moment then sweep
    sleep 2
    pkill -f 'navigation_launch.py' 2>/dev/null || true
    pkill -f 'domain_bridge.*create3_bridge' 2>/dev/null || true
  else
    echo "[session] leaving Nav2 / RTAB running (pass --kill-nav to tear down)."
  fi
}
trap cleanup EXIT INT TERM

wait_topic() {
  # OccupancyGrid /map: use rclpy probe. ros2cli daemon + topic list/echo are
  # unreliable with FastDDS interface whitelists (false FAIL while data flows).
  local topic=$1
  local seconds=${2:-90}
  local profile=${3:-$XPROF}
  echo "[session] waiting for $topic (≤${seconds}s)..."
  if [[ "$topic" == "/map" || "$topic" == "/rtabmap/map" ]]; then
    if FASTRTPS_DEFAULT_PROFILES_FILE="$profile" \
        python3 "$WS/scripts/wait_map_probe.py" "$topic" "$seconds"; then
      echo "  [OK] $topic"
      return 0
    fi
  elif FASTRTPS_DEFAULT_PROFILES_FILE="$profile" \
      timeout "$seconds" ros2 topic echo "$topic" --once >/dev/null 2>&1; then
    echo "  [OK] $topic"
    return 0
  fi
  echo "  [FAIL] $topic not seen"
  return 1
}

echo "[session] =============================================="
echo "[session] grab_it operator session"
echo "[session] logs: $LOGDIR"
echo "[session] =============================================="
echo "[session] Prereq reminder: Nano bringup (cam+arm) + Create3 undocked."

# --- Nav2 stack ---
if [ "$SKIP_NAV" = "1" ]; then
  echo "[session] --skip-nav: not starting nav_bringup"
else
  if pgrep -f 'navigation_launch.py' >/dev/null 2>&1; then
    echo "[session] Nav2 already running — not starting a second nav_bringup"
  else
    # Drop stale sim RViz that breaks real-time clocks
    pkill -f 'use_sim_time:=true' 2>/dev/null || true
    echo "[session] starting nav_bringup.sh (log: $LOGDIR/nav.log)..."
    bash "$WS/nav_bringup.sh" >"$LOGDIR/nav.log" 2>&1 &
    NAV_PID=$!
    PIDS+=("$NAV_PID")
  fi
fi

if ! wait_topic /map 120 "$XPROF"; then
  echo "[session] map never appeared — check $LOGDIR/nav.log and Nano camera."
  echo "         Camera stuck? On Nano:"
  echo "           docker exec grab_it_nano bash -lc 'pkill -9 -f realsense2_camera; sleep 2; /root/nano_bringup.sh'"
  exit 1
fi

# --- MoveIt + full robot_description (arm driver stays on Nano) ---
if [ "$NO_MOVEIT" = "1" ]; then
  echo "[session] --no-moveit: skipping MoveIt / grab_it_description"
else
  ARM_ARGS=()
  [ "$MOVEIT_RVIZ" = "1" ] && ARM_ARGS+=(--rviz)
  echo "[session] starting MoveIt + robot_description (log: $LOGDIR/moveit.log)..."
  if bash "$WS/scripts/grabit_arm_bringup.sh" "${ARM_ARGS[@]+"${ARM_ARGS[@]}"}"; then
    echo "  [OK]   MoveIt / description launch requested"
  else
    echo "  [WARN] MoveIt bringup failed — session continues (GUI/nav still useful)."
    echo "         Build once: cd $WS && colcon build --packages-up-to interbotix_xsarm_moveit --symlink-install"
  fi
fi

# --- GUI (arm + base web controls) ---
if [ "$NO_GUI" = "1" ]; then
  echo "[session] --no-gui: skipping"
else
  python3 -c 'import websockets' 2>/dev/null || pip3 install -q websockets --break-system-packages 2>/dev/null || true
  if [ -f "$WS/mjpeg_relay.py" ]; then
    echo "[session] starting mjpeg_relay → http://0.0.0.0:8081/d435i"
    FASTRTPS_DEFAULT_PROFILES_FILE="$XPROF" \
      python3 "$WS/mjpeg_relay.py" >"$LOGDIR/mjpeg_relay.log" 2>&1 &
    PIDS+=($!)
  fi
  echo "[session] starting grab_it GUI → http://0.0.0.0:8765  (ws :8766)"
  # Crossover-only: GUI is a joint_states *reader*. Dual-iface (bridge_both)
  # made Fast DDS bind the Wi‑Fi locator; Nano xs_sdk cannot send there, so
  # arm ovals flapped. HTTP/WS still bind 0.0.0.0 for the laptop.
  # Create3 /cmd_vel: UDP hop → cmd_vel_wifi_relay (bridge_both writer).
  # Prefer source tree so gripper/teleop edits apply without a colcon rebuild.
  # Never start both ros2-run and raw python — they fight on joint_single.
  pkill -f 'grab_it_gui/gui_node.py' 2>/dev/null || true
  pkill -f 'ros2 run grab_it_gui' 2>/dev/null || true
  pkill -f 'cmd_vel_wifi_relay.py' 2>/dev/null || true
  sleep 0.5
  if [ -f "$WS/scripts/cmd_vel_wifi_relay.py" ]; then
    echo "[session] starting cmd_vel_wifi_relay (Create3 via Wi‑Fi DDS)"
    FASTRTPS_DEFAULT_PROFILES_FILE="$BPROF" \
      python3 "$WS/scripts/cmd_vel_wifi_relay.py" >"$LOGDIR/cmd_vel_wifi.log" 2>&1 &
    PIDS+=($!)
  fi
  if [ -f "$WS/src/grab_it_gui/grab_it_gui/gui_node.py" ]; then
    FASTRTPS_DEFAULT_PROFILES_FILE="$XPROF" \
      python3 "$WS/src/grab_it_gui/grab_it_gui/gui_node.py" >"$LOGDIR/gui.log" 2>&1 &
  elif ros2 pkg prefix grab_it_gui >/dev/null 2>&1; then
    FASTRTPS_DEFAULT_PROFILES_FILE="$XPROF" \
      ros2 run grab_it_gui gui_node >"$LOGDIR/gui.log" 2>&1 &
  else
    echo "  [WARN] grab_it_gui not found"
  fi
  PIDS+=($!)
  sleep 1
fi

# --- RViz (Nav2) ---
if [ "$NO_RVIZ" = "1" ]; then
  echo "[session] --no-rviz: skipping"
else
  echo "[session] starting Nav2 RViz (software GL, full grab_it RobotModel)..."
  export LIBGL_ALWAYS_SOFTWARE=1
  # Stock nav2_default_view has RobotModel disabled + Volatile /robot_description
  # (misses TransientLocal latched URDF). Use grabit config instead.
  NAV2_RVIZ=$WS/config/grabit_nav2.rviz
  if [ ! -f "$NAV2_RVIZ" ]; then
    echo "  [WARN] $NAV2_RVIZ missing — falling back to nav2_default_view"
    FASTRTPS_DEFAULT_PROFILES_FILE="$XPROF" \
      ros2 launch nav2_bringup rviz_launch.py use_sim_time:=false \
      >"$LOGDIR/rviz.log" 2>&1 &
  else
    # bridge_both: must see grab_it /robot_description (TransientLocal) + Nav2 topics
    FASTRTPS_DEFAULT_PROFILES_FILE="$BPROF" \
      ros2 launch nav2_bringup rviz_launch.py use_sim_time:=false \
      rviz_config:="$NAV2_RVIZ" \
      >"$LOGDIR/rviz.log" 2>&1 &
  fi
  PIDS+=($!)
  sleep 2
fi

echo ""
echo "[session] ready."
echo "  Map/Nav2 : live (see $LOGDIR/nav.log)"
echo "  RViz     : Fixed Frame=map; uncheck Amcl; 2D Pose Estimate → Nav2 Goal"
echo "  MoveIt   : move_group on px150_no_elbow (driver on Nano; see $LOGDIR/moveit.log)"
echo "  Robot    : grab_it_description RSP (see $LOGDIR/robot_description.log)"
echo "  GUI      : http://192.168.50.207:8765  (arm + base)"
echo "  Named    : bash $WS/scripts/go_to_named.sh kitchen"
echo "  /cmd_vel : Nav2, GUI, and teleop share it — cancel goals before manual drive"
echo ""

# --- Keyboard teleop (foreground; needs TTY) ---
if [ "$NO_TELEOP" = "1" ]; then
  echo "[session] --no-teleop: waiting (Ctrl‑C to stop session helpers)..."
  wait
else
  if [ ! -t 0 ]; then
    echo "[session] stdin is not a TTY — cannot run keyboard teleop."
    echo "         Re-run in an interactive shell, or use --no-teleop and teleop separately."
    wait
    exit 0
  fi
  echo "[session] keyboard teleop (domain 0 + bridge_both). Ctrl‑C ends teleop + session helpers."
  export FASTRTPS_DEFAULT_PROFILES_FILE="$BPROF"
  ros2 run teleop_twist_keyboard teleop_twist_keyboard
fi
