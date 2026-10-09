#!/bin/bash
# grabit_arm_bringup.sh — robot_description (full grab_it) + MoveIt move_group on AGX.
#
# Prereq: Nano already running xsarm_control (nano_bringup). This launch does NOT
# open /dev/ttyDXL — use_arm_driver:=false so GUI + xs_sdk on Nano keep the bus.
#
# Usage (inside Isaac container):
#   bash /workspaces/isaac_ros-dev/scripts/grabit_arm_bringup.sh           # daemon (default)
#   bash .../grabit_arm_bringup.sh --foreground   # wait until Ctrl-C
#   bash .../grabit_arm_bringup.sh --rviz
#   bash .../grabit_arm_bringup.sh --no-description
WS=/workspaces/isaac_ros-dev
BPROF=${BPROF:-$WS/bridge_both.xml}
LOGDIR=${LOGDIR:-/tmp/grabit_session}
mkdir -p "$LOGDIR"

USE_MOVEIT_RVIZ=false
USE_DESCRIPTION=true
FOREGROUND=false
for arg in "$@"; do
  case "$arg" in
    --rviz) USE_MOVEIT_RVIZ=true ;;
    --no-description) USE_DESCRIPTION=false ;;
    --foreground) FOREGROUND=true ;;
    -h|--help)
      sed -n '2,16p' "$0"
      exit 0
      ;;
    *)
      echo "[arm] unknown arg: $arg"
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

if ! ros2 pkg prefix interbotix_xsarm_moveit >/dev/null 2>&1; then
  echo "[arm] ERROR: interbotix_xsarm_moveit not installed."
  echo "       Inside container:"
  echo "         cd $WS && colcon build --packages-up-to interbotix_xsarm_moveit --symlink-install"
  echo "         source install/setup.bash"
  exit 1
fi

# URDF expects px150_N_*.stl; package meshes use short names — keep compat links fresh.
if [ -x "$WS/scripts/fix_px150_mesh_links.sh" ]; then
  bash "$WS/scripts/fix_px150_mesh_links.sh" >/dev/null 2>&1 || true
fi
# Stock no_elbow xacro skips upper_arm mesh (arm looks disconnected in MoveIt).
if [ -f "$WS/scripts/fix_px150_no_elbow_urdf.py" ]; then
  python3 "$WS/scripts/fix_px150_no_elbow_urdf.py" >/dev/null 2>&1 || true
fi
if [ -f "$WS/scripts/fix_xsarm_moveit_launch.py" ]; then
  python3 "$WS/scripts/fix_xsarm_moveit_launch.py" >/dev/null 2>&1 || true
fi
if [ -f "$WS/scripts/fix_grab_it_launch.py" ]; then
  python3 "$WS/scripts/fix_grab_it_launch.py" >/dev/null 2>&1 || true
fi
# 4-DOF arm: position-only IK + longer solver timeout (click-to-grasp).
if [ -f "$WS/scripts/fix_px150_moveit_position_ik.py" ]; then
  python3 "$WS/scripts/fix_px150_moveit_position_ik.py" >/dev/null 2>&1 || true
fi

# Kill MoveIt binaries without matching unrelated shells that mention them.
_kill_moveit() {
  local self=$$
  local pid cmd
  for pid in $(pgrep -f 'moveit_ros_move_group|/lib/.*/move_group' 2>/dev/null || true); do
    [ "$pid" = "$self" ] && continue
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
    case "$cmd" in
      *bash*|*docker*|*grabit_arm_bringup*) continue ;;
    esac
    echo "[arm] kill moveit pid=$pid"
    /bin/kill -9 "$pid" 2>/dev/null || true
  done
  pkill -9 -f 'xsarm_moveit.launch.py' 2>/dev/null || true
}

PIDS=()
cleanup() {
  echo "[arm] stopping MoveIt / robot_description..."
  for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done
  _kill_moveit
  pkill -f 'grab_it_launch.py' 2>/dev/null || true
}

if [ "$FOREGROUND" = "true" ]; then
  trap cleanup INT TERM EXIT
fi

# One authoritative full-robot RSP on AGX. Kill stale duplicates first (they split TF
# and crash MoveIt RViz). Include root-started leftovers from docker exec.
# Always kill MoveIt too — daemon re-runs used to stack solvers on /compute_ik.
_kill_moveit
pkill -9 -f 'grab_it_launch.py' 2>/dev/null || true
pkill -9 -f 'grab_it_robot_state_publisher' 2>/dev/null || true
pkill -9 -f 'arm_joint_states_bridge.py' 2>/dev/null || true
pkill -9 -f 'filter_arm_joint_states.py' 2>/dev/null || true
pkill -9 -f 'grasp_from_click' 2>/dev/null || true
sleep 1

# Bridge first: RSP needs /joint_states_rsp even when Nano/U2D2 is down.
echo "[arm] bridging joint_states for robot_state_publisher..."
FASTRTPS_DEFAULT_PROFILES_FILE="$BPROF" \
  python3 "$WS/scripts/arm_joint_states_bridge.py" \
  >"$LOGDIR/joint_bridge.log" 2>&1 &
PIDS+=($!)
sleep 1

# Full Create3+arm+stack model for RViz / TF consumers on AGX.
if [ "$USE_DESCRIPTION" = "true" ]; then
  if ros2 pkg prefix grab_it_description >/dev/null 2>&1; then
    echo "[arm] starting grab_it_description (robot_state_publisher)..."
    FASTRTPS_DEFAULT_PROFILES_FILE="$BPROF" \
      ros2 launch grab_it_description grab_it_launch.py \
      use_gui:=false \
      hardware_type:=actual \
      robot_name:=px150_no_elbow \
      joint_states_topic:=/px150_no_elbow/joint_states_rsp \
      >"$LOGDIR/robot_description.log" 2>&1 &
    PIDS+=($!)
    sleep 2
  else
    echo "[arm] WARN: grab_it_description not found — skipping full robot_description"
  fi
fi

# Strip pan (and other non-arm joints) so MoveIt can track the real pose.
echo "[arm] filtering joint_states for MoveIt..."
FASTRTPS_DEFAULT_PROFILES_FILE="$BPROF" \
  python3 "$WS/scripts/filter_arm_joint_states.py" \
  >"$LOGDIR/joint_filter.log" 2>&1 &
PIDS+=($!)
sleep 1

echo "[arm] starting MoveIt (px150_no_elbow, no local DXL driver)..."
echo "      use_moveit_rviz=$USE_MOVEIT_RVIZ  log=$LOGDIR/moveit.log"
# Fixed frame must be the arm model root (not Create3 base_link), or the arm
# floats disconnected in MoveIt RViz.
FASTRTPS_DEFAULT_PROFILES_FILE="$BPROF" \
  ros2 launch interbotix_xsarm_moveit xsarm_moveit.launch.py \
  robot_model:=px150_no_elbow \
  robot_name:=px150_no_elbow \
  hardware_type:=actual \
  use_arm_driver:=false \
  use_moveit_rviz:="$USE_MOVEIT_RVIZ" \
  use_world_frame:=false \
  rviz_frame:=px150_no_elbow/base_link \
  >"$LOGDIR/moveit.log" 2>&1 &
PIDS+=($!)

# Click-to-grasp (D435i depth → MoveIt IK). Safe to start after move_group.
if [ -x "$WS/scripts/grasp_bringup.sh" ]; then
  echo "[arm] starting grasp_from_click…"
  BPROF="$BPROF" LOGDIR="$LOGDIR" bash "$WS/scripts/grasp_bringup.sh" \
    >"$LOGDIR/grasp_bringup.log" 2>&1 &
  PIDS+=($!)
fi

echo "[arm] started pids: ${PIDS[*]}"
if [ "$FOREGROUND" = "true" ]; then
  wait
fi
