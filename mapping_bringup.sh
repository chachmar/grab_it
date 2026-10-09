#!/bin/bash
# mapping_bringup.sh - AGX-side cold start for grab_it RTAB-Map mapping stack.
# Frame naming: camera uses DEFAULT frames (camera_link, camera_color_optical_frame)
# i.e. the Nano camera launch has NO tf_prefix. Static TF connects base_link->camera_link.
#
# Restored to original working baseline. Only operational fix kept: TF discovery retries.
#
# PREREQS FIRST:
#   1. Nano: camera up (848x480x30 color+depth, align_depth, enable_sync, NO tf_prefix)
#   2. Create3 on domain 5, undocked, publishing /odom
#   3. Clocks synced (timedatectl set-ntp true on AGX host AND Nano host)
#
# Run inside the Isaac container:  bash mapping_bringup.sh
set -m
WS=/workspaces/isaac_ros-dev
XPROF=$WS/rtabmap_profile.xml
BPROF=$WS/bridge_both.xml
BYAML=$WS/create3_bridge.yaml
CAM_FRAME=camera_link                      # camera root frame (no tf_prefix)
CAM_OPTICAL=camera_color_optical_frame     # frame RTAB-Map needs connected

source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash" 2>/dev/null
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
unset CYCLONEDDS_URI

# clean any leftovers from prior runs (bridge, static TF, leftover nav nodes)
pkill -9 -f domain_bridge 2>/dev/null
pkill -9 -f static_transform_publisher 2>/dev/null
pkill -9 -f depthimage_to_laserscan 2>/dev/null
pkill -9 -f 'rtabmap.launch.py' 2>/dev/null
pkill -9 -f 'rtabmap_slam/rtabmap' 2>/dev/null
pkill -9 -f scan_stamp_fix 2>/dev/null
pkill -9 -f map_to_nav 2>/dev/null
pkill -9 -f odom_to_tf.py 2>/dev/null
sleep 1

echo "[map] starting domain_bridge (Create3 d5 -> d0; odom+tf_static+cmd_vel, no /tf)..."
FASTRTPS_DEFAULT_PROFILES_FILE="$BPROF" \
  ros2 run domain_bridge domain_bridge "$BYAML" > /tmp/bridge.log 2>&1 &
BRIDGE_PID=$!
sleep 3

ODOM_TF=/tmp/grabit_nav/odom_to_tf.py
mkdir -p /tmp/grabit_nav
cp -f "$WS/scripts/odom_to_tf.py" "$ODOM_TF"
chmod +x "$ODOM_TF"
echo "[map] publishing TF from /odom (avoids lagged Create3 /tf)..."
FASTRTPS_DEFAULT_PROFILES_FILE="$BPROF" \
  python3 "$ODOM_TF" > /tmp/map_odom_tf.log 2>&1 &
ODOM_TF_PID=$!
sleep 1

echo "[map] starting static TF base_link -> $CAM_FRAME..."
FASTRTPS_DEFAULT_PROFILES_FILE="$XPROF" \
  ros2 run tf2_ros static_transform_publisher \
  --x 0.15 --y 0 --z 0.30 --roll 0 --pitch 0.35 --yaw 0 \
  --frame-id base_link --child-frame-id "$CAM_FRAME" > /tmp/static_tf.log 2>&1 &
TF_PID=$!
sleep 4   # let static TF register in the graph before verifying

export FASTRTPS_DEFAULT_PROFILES_FILE="$XPROF"

echo "[map] waiting for TF base_link -> $CAM_OPTICAL..."
TF_OK=0
for i in $(seq 1 20); do
  if timeout 3 ros2 run tf2_ros tf2_echo base_link "$CAM_OPTICAL" 2>&1 \
      | grep -q "Translation:"; then
    echo "  [OK]   TF tree connected (try $i)"
    TF_OK=1
    break
  fi
  sleep 1
done
if [ "$TF_OK" != "1" ]; then
  echo "  [FAIL] TF tree NOT connected."
  echo "         - is the camera running? (frames must be camera_* i.e. no tf_prefix)"
  echo "         - check /tmp/static_tf.log"
  echo "         bridge=$BRIDGE_PID static_tf=$TF_PID left running."
  exit 1
fi

echo "[map] pre-flight input checks..."
# Depth QoS varies by RealSense launch; try several profiles. Prefer echo over hz
# (hz has no QoS flags on this ROS build and false-fails on Best Effort/RELIABLE mismatch).
check_image_topic () {
  local topic="$1" label="$2"
  local i qos
  for i in 1 2 3 4 5; do
    for qos in \
      "--qos-profile sensor_data" \
      "--qos-reliability reliable --qos-durability volatile" \
      "--qos-reliability reliable --qos-durability transient_local" \
      "--qos-reliability best_effort --qos-durability volatile"
    do
      if timeout 6 ros2 topic echo "$topic" --once --no-arr --no-str $qos 2>&1 \
          | grep -qE "header:|stamp:|height:|width:"; then
        echo "  [OK]   $label"
        return 0
      fi
    done
    sleep 1
  done
  echo "  [FAIL] $label ($topic) — topic listed but no samples (restart Nano camera / align_depth:=true)"
  return 1
}
FAIL=0
export FASTRTPS_DEFAULT_PROFILES_FILE="$XPROF"
check_image_topic /d435i/d435i/color/image_raw                  "camera color"  || FAIL=1
check_image_topic /d435i/d435i/aligned_depth_to_color/image_raw "aligned depth" || FAIL=1
# odom often needs wifi+crossover profile
export FASTRTPS_DEFAULT_PROFILES_FILE="$BPROF"
if timeout 10 ros2 topic echo /odom --once --no-arr --no-str 2>&1 | grep -qE "header:|child_frame_id:"; then
  echo "  [OK]   odom (bridge)"
elif timeout 10 ros2 topic hz /odom 2>&1 | grep -q "average rate"; then
  echo "  [OK]   odom (bridge)"
else
  echo "  [FAIL] odom (bridge)"; FAIL=1
fi
export FASTRTPS_DEFAULT_PROFILES_FILE="$XPROF"
if [ "$FAIL" = "1" ]; then
  echo "[map] PRE-FLIGHT FAILED"
  echo "  On Nano restart camera with align_depth:=true enable_sync:=true"
  echo "  Then: ros2 topic echo /d435i/d435i/aligned_depth_to_color/image_raw --once --qos-profile sensor_data"
  echo "  odom? -> Create3 domain5+undock; /tmp/bridge.log"
  echo "  bridge=$BRIDGE_PID static_tf=$TF_PID left running."
  exit 1
fi

echo "[map] all inputs live. Launching RTAB-Map (bridge=$BRIDGE_PID tf=$TF_PID)..."
# Remove --delete_db_on_start to PRESERVE/EXTEND an existing map.
ros2 launch rtabmap_launch rtabmap.launch.py \
  rgb_topic:=/d435i/d435i/color/image_raw \
  depth_topic:=/d435i/d435i/aligned_depth_to_color/image_raw \
  camera_info_topic:=/d435i/d435i/color/camera_info \
  frame_id:=base_link \
  odom_topic:=/odom \
  visual_odometry:=false \
  approx_sync:=true \
  approx_sync_max_interval:=0.1 \
  queue_size:=30 \
  wait_for_transform:=1.0 \
  database_path:=$WS/rtabmap.db \
  rtabmap_args:="--delete_db_on_start" \
  rviz:=false \
  rtabmap_viz:=false

echo "[map] RTAB-Map exited. kill bridge/tf: kill $BRIDGE_PID $TF_PID"
kill $BRIDGE_PID $TF_PID 2>/dev/null
