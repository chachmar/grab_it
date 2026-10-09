#!/bin/bash
# nav_bringup.sh — AGX Nav2 stack on top of an EXISTING RTAB-Map database.
#
# Uses RTAB-Map in localization mode (map frozen) + depth→scan obstacles + Nav2 (RPP).
#
# PREREQS:
#   1. You already remapped: bash mapping_bringup.sh  (deletes/rebuilds rtabmap.db)
#      Or: bash remap_map.sh to backup old db then remap.
#   2. Nano camera up (848x480 sync + aligned depth, crossover profile)
#   3. Create3 on domain 5, undocked, publishing /odom
#   4. Clocks synced on AGX + Nano
#   5. Image rebuilt with Dockerfile.grabit (nav2 + depthimage_to_laserscan)
#
# Run inside Isaac container:
#   bash /workspaces/isaac_ros-dev/nav_bringup.sh
#
# Then seed pose (if needed) and send a goal:
#   ros2 topic pub --once /initialpose geometry_msgs/PoseWithCovarianceStamped "{...}"
#   ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose "{...}"
set -m
WS=/workspaces/isaac_ros-dev
XPROF=$WS/rtabmap_profile.xml
BPROF=$WS/bridge_both.xml
BYAML=$WS/create3_bridge.yaml
PARAMS=$WS/config/nav2_params_create3.yaml
DB=$WS/rtabmap.db
# shellcheck source=/dev/null
source "$WS/config/camera_tf.env"
MAP_RELAY=/tmp/grabit_nav/map_to_nav.py
SCAN_FIX=/tmp/grabit_nav/scan_stamp_fix.py
POSE_RELAY=/tmp/grabit_nav/initialpose_to_rtab.py
ODOM_TF=/tmp/grabit_nav/odom_to_tf.py
WAIT_MAP=/tmp/grabit_nav/wait_map_probe.py
mkdir -p /tmp/grabit_nav
# Prefer workspace copies when present
if [ -f "$WS/scripts/odom_to_tf.py" ]; then
  cp -f "$WS/scripts/odom_to_tf.py" "$ODOM_TF"
  chmod +x "$ODOM_TF"
fi
if [ -f "$WS/scripts/wait_map_probe.py" ]; then
  cp -f "$WS/scripts/wait_map_probe.py" "$WAIT_MAP"
  chmod +x "$WAIT_MAP"
fi

source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash" 2>/dev/null
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
unset CYCLONEDDS_URI

if [ ! -f "$DB" ]; then
  echo "[nav] ERROR: no map database at $DB"
  echo "       Remap first: bash $WS/mapping_bringup.sh   (or bash $WS/remap_map.sh)"
  exit 1
fi
if [ ! -f "$PARAMS" ]; then
  echo "[nav] ERROR: missing $PARAMS"
  exit 1
fi

# Package check (fails clearly if Dockerfile.grabit not rebuilt yet)
for pkg in nav2_bringup depthimage_to_laserscan rtabmap_launch domain_bridge; do
  if ! ros2 pkg prefix "$pkg" >/dev/null 2>&1; then
    echo "[nav] ERROR: missing package '$pkg'"
    echo "       Rebuild Isaac image with Dockerfile.grabit (nav2 lines), then re-run run_dev.sh"
    exit 1
  fi
done

# Ensure map relay helper exists (RTAB publishes /rtabmap/map; Nav2 wants /map)
# Also erodes mapped obstacles by a few cells so RTAB halls aren't tighter than reality.
cat > "$MAP_RELAY" <<'PY'
#!/usr/bin/env python3
"""Relay OccupancyGrid /rtabmap/map -> /map for Nav2, with light obstacle erosion.

Process name must NOT contain 'rtabmap'.
Erosion: peel ~ERODE_CELLS of occupied fringe into free space so corridors
that mapped slightly tight stay navigable (Create3 still protected by inflation).
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import OccupancyGrid

MAP_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
)

# 1 cell @ 5 cm ≈ slight widen of mapped halls (2 was over-opening + wall-hugging plans)
ERODE_CELLS = 1
OCCUPIED_THRESH = 50


def erode_obstacles(msg: OccupancyGrid, iters: int) -> OccupancyGrid:
    w, h = msg.info.width, msg.info.height
    data = list(msg.data)
    if w < 3 or h < 3 or not data:
        return msg
    neigh = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for _ in range(iters):
        out = data[:]
        for y in range(h):
            row = y * w
            for x in range(w):
                i = row + x
                if data[i] < OCCUPIED_THRESH:
                    continue
                for dx, dy in neigh:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and data[ny * w + nx] == 0:
                        out[i] = 0
                        break
        data = out
    msg.data = data
    return msg


class MapToNav(Node):
    def __init__(self):
        super().__init__('map_to_nav')
        self.pub = self.create_publisher(OccupancyGrid, '/map', MAP_QOS)
        self.create_subscription(OccupancyGrid, '/rtabmap/map', self._cb, MAP_QOS)
        self._sent = False

    def _cb(self, msg):
        out = erode_obstacles(msg, ERODE_CELLS)
        self.pub.publish(out)
        if not self._sent:
            self.get_logger().info(
                f'Relaying map {out.info.width}x{out.info.height} @ '
                f'{out.info.resolution:.3f} m (erode={ERODE_CELLS} cells)'
            )
            self._sent = True


def main():
    rclpy.init()
    node = MapToNav()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
PY
chmod +x "$MAP_RELAY"

# Republish /scan_raw -> /scan in base_link (prefer workspace script)
if [ -f "$WS/scripts/scan_stamp_fix.py" ]; then
  cp -f "$WS/scripts/scan_stamp_fix.py" "$SCAN_FIX"
else
  echo "[nav] ERROR: missing $WS/scripts/scan_stamp_fix.py"
  exit 1
fi
chmod +x "$SCAN_FIX"

# RViz "2D Pose Estimate" publishes /initialpose; RTAB localization needs /rtabmap/initialpose
# Inflate covariance so RTAB can snap to nearby graph nodes instead of a brittle exact pose.
cat > "$POSE_RELAY" <<'PY'
#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseWithCovarianceStamped

QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.VOLATILE,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
)

# ~0.5 m XY, ~25 deg yaw — helps RTAB lock when arrow is roughly right
COV_X = 0.25
COV_Y = 0.25
COV_YAW = (25.0 * math.pi / 180.0) ** 2


class PoseRelay(Node):
    def __init__(self):
        super().__init__('initialpose_to_rtab')
        self.pub = self.create_publisher(PoseWithCovarianceStamped, '/rtabmap/initialpose', QOS)
        self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self._cb, QOS)
        self.get_logger().info(
            'Relaying /initialpose -> /rtabmap/initialpose '
            f'(cov xy={COV_X:.2f} yaw_deg=25)'
        )

    def _cb(self, msg):
        out = PoseWithCovarianceStamped()
        out.header = msg.header
        out.pose.pose = msg.pose.pose
        cov = [0.0] * 36
        cov[0] = COV_X
        cov[7] = COV_Y
        cov[35] = COV_YAW
        out.pose.covariance = cov
        self.pub.publish(out)
        self.get_logger().info('Forwarded initial pose to RTAB-Map (inflated covariance)')


def main():
    rclpy.init()
    node = PoseRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.try_shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
PY
chmod +x "$POSE_RELAY"

# Narrow but thorough pkills — kill ALL stale domain_bridges (duplicates corrupt TF/odom)
pkill -9 -f domain_bridge 2>/dev/null
pkill -9 -f static_transform_publisher 2>/dev/null
pkill -9 -f depthimage_to_laserscan 2>/dev/null
pkill -9 -f 'rtabmap.launch.py' 2>/dev/null
pkill -9 -f 'rtabmap_slam/rtabmap' 2>/dev/null
pkill -9 -f 'map_to_nav.py' 2>/dev/null
pkill -9 -f 'scan_stamp_fix.py' 2>/dev/null
pkill -9 -f 'initialpose_to_rtab.py' 2>/dev/null
pkill -9 -f 'odom_to_tf.py' 2>/dev/null
pkill -9 -f 'navigation_launch.py' 2>/dev/null
pkill -9 -f controller_server 2>/dev/null
pkill -9 -f planner_server 2>/dev/null
pkill -9 -f bt_navigator 2>/dev/null
pkill -9 -f lifecycle_manager 2>/dev/null
sleep 2

echo "[nav] starting domain_bridge (Create3 d5 <-> d0; odom+tf_static+cmd_vel, no /tf)..."
FASTRTPS_DEFAULT_PROFILES_FILE="$BPROF" \
  ros2 run domain_bridge domain_bridge "$BYAML" > /tmp/nav_bridge.log 2>&1 &
BRIDGE_PID=$!
sleep 3

# Ensure odom_to_tf helper exists even on older checkouts
if [ ! -f "$ODOM_TF" ]; then
  cp -f "$WS/scripts/odom_to_tf.py" "$ODOM_TF" 2>/dev/null || true
  chmod +x "$ODOM_TF" 2>/dev/null || true
fi
echo "[nav] publishing TF from /odom (avoids lagged Create3 /tf)..."
FASTRTPS_DEFAULT_PROFILES_FILE="$BPROF" \
  python3 "$ODOM_TF" > /tmp/nav_odom_tf.log 2>&1 &
ODOM_TF_PID=$!
sleep 1

# Pick TF/camera DDS profile: crossover-only needs 10.42.0.1 up
TF_PROF="$XPROF"
if FASTRTPS_DEFAULT_PROFILES_FILE="$XPROF" timeout 3 ros2 topic list >/tmp/nav_xprof_probe.log 2>&1; then
  if grep -q 'All whitelist interfaces were filtered out' /tmp/nav_xprof_probe.log 2>/dev/null; then
    TF_PROF="$BPROF"
    echo "  [warn] crossover DDS profile unusable (10.42.0.1 down?) — using bridge_both.xml"
  fi
else
  if grep -q 'All whitelist interfaces were filtered out' /tmp/nav_xprof_probe.log 2>/dev/null; then
    TF_PROF="$BPROF"
    echo "  [warn] crossover DDS profile unusable (10.42.0.1 down?) — using bridge_both.xml"
  fi
fi

echo "[nav] checking Nano camera (needs crossover 10.42.0.2 + nano_bringup)..."
# 15s: discovery after RealSense restart can exceed 8s; "Terminated" = timeout, not Ctrl+C
if ! FASTRTPS_DEFAULT_PROFILES_FILE="$XPROF" \
    timeout 15 ros2 topic hz /d435i/d435i/color/image_raw 2>/dev/null | grep -q "average rate"; then
  echo "  [FAIL] no D435i color on crossover."
  echo "         AGX host: ip -br addr | grep 10.42 ; ping -c2 10.42.0.2"
  echo "         Nano:     sudo docker start grab_it_nano && docker exec -it grab_it_nano /root/nano_bringup.sh"
  echo "         If bringup already running but camera stuck (device busy):"
  echo "           docker exec grab_it_nano bash -lc 'pkill -9 -f realsense2_camera; sleep 2; /root/nano_bringup.sh'"
  echo "         Check: docker exec grab_it_nano tail -30 /tmp/cameras.log"
  echo "         (eno1/crossover is currently required — WiFi alone is not enough for images)"
  exit 1
fi
echo "  [OK]   camera color visible"

# Publish static camera TF on bridge_both so grasp/GUI (also BPROF) always see
# it. Images still use TF_PROF/XPROF below for optical-frame connectivity.
echo "[nav] starting static TF base_link -> $CAM_FRAME" \
  "(x=$CAM_TF_X z=$CAM_TF_Z pitch=$CAM_TF_PITCH; from config/camera_tf.env; DDS=bridge_both)"
FASTRTPS_DEFAULT_PROFILES_FILE="$BPROF" \
  ros2 run tf2_ros static_transform_publisher \
  --x "$CAM_TF_X" --y "$CAM_TF_Y" --z "$CAM_TF_Z" \
  --roll "$CAM_TF_ROLL" --pitch "$CAM_TF_PITCH" --yaw "$CAM_TF_YAW" \
  --frame-id base_link --child-frame-id "$CAM_FRAME" > /tmp/nav_static_tf.log 2>&1 &
TF_PID=$!
sleep 3

export FASTRTPS_DEFAULT_PROFILES_FILE="$TF_PROF"

echo "[nav] waiting for camera TF + base_link -> $CAM_OPTICAL..."
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
  echo "  [FAIL] TF tree NOT connected after retries."
  echo "         Need RealSense TF ($CAM_FRAME -> $CAM_OPTICAL) from Nano (no tf_prefix)."
  echo "         See /tmp/nav_static_tf.log"
  timeout 5 ros2 topic list 2>/dev/null | grep -E 'd435i|tf' | head -15
  exit 1
fi

echo "[nav] pre-flight..."
check_topic () {
  local topic="$1" label="$2" profile="${3:-$TF_PROF}"
  if FASTRTPS_DEFAULT_PROFILES_FILE="$profile" \
      timeout 6 ros2 topic hz "$topic" 2>/dev/null | grep -q "average rate"; then
    echo "  [OK]   $label"
    return 0
  fi
  echo "  [FAIL] $label ($topic)"; return 1
}
FAIL=0
check_topic /d435i/d435i/color/image_raw                  "camera color"  "$XPROF" || \
  check_topic /d435i/d435i/color/image_raw                "camera color"  "$TF_PROF" || FAIL=1
check_topic /d435i/d435i/aligned_depth_to_color/image_raw "aligned depth" "$XPROF" || \
  check_topic /d435i/d435i/aligned_depth_to_color/image_raw "aligned depth" "$TF_PROF" || FAIL=1
if ! check_topic /odom "odom (bridge)" "$BPROF"; then
  check_topic /odom "odom (bridge)" "$TF_PROF" || FAIL=1
fi
if [ "$FAIL" = "1" ]; then
  echo "[nav] PRE-FLIGHT FAILED — bridge=$BRIDGE_PID tf=$TF_PID left running"
  exit 1
fi

# Rest of stack uses TF_PROF so it can see static TF + (when up) crossover peers
export FASTRTPS_DEFAULT_PROFILES_FILE="$TF_PROF"

echo "[nav] depth -> /scan_raw -> /scan (restamped for TF)..."
# output_frame MUST be camera_link (X-forward), NOT optical — otherwise LaserScan
# x=r*cos/y=r*sin is TF'd wrongly and the local costmap stays empty.
# scan_height>1 stacks several depth rows so obstacles aren't a 1-pixel slice.
FASTRTPS_DEFAULT_PROFILES_FILE="$XPROF" \
  ros2 run depthimage_to_laserscan depthimage_to_laserscan_node \
  --ros-args \
  -r depth:=/d435i/d435i/aligned_depth_to_color/image_raw \
  -r depth_camera_info:=/d435i/d435i/color/camera_info \
  -r scan:=/scan_raw \
  -p scan_height:=31 \
  -p range_min:=0.25 \
  -p range_max:=3.5 \
  -p output_frame:=$CAM_FRAME \
  > /tmp/nav_scan.log 2>&1 &
SCAN_PID=$!
FASTRTPS_DEFAULT_PROFILES_FILE="$BPROF" \
  python3 "$SCAN_FIX" > /tmp/nav_scan_fix.log 2>&1 &
SCANFIX_PID=$!
sleep 2
echo "[nav] waiting for restamped /scan..."
if ! timeout 30 bash -c 'until ros2 topic hz /scan 2>/dev/null | grep -q "average rate"; do sleep 1; done'; then
  echo "  [FAIL] /scan not publishing — see /tmp/nav_scan_fix.log /tmp/nav_scan.log"
  tail -40 /tmp/nav_scan_fix.log 2>/dev/null
  exit 1
fi
echo "  [OK]   /scan live (pid=$SCANFIX_PID)"

echo "[nav] RTAB-Map LOCALIZATION (keep $DB, no incremental mapping)..."
# Same extrinsic + Force3DoF as mapping. InitWMWithAllNodes helps first lock after pose estimate.
FASTRTPS_DEFAULT_PROFILES_FILE="$XPROF" \
  ros2 launch rtabmap_launch rtabmap.launch.py \
  rgb_topic:=/d435i/d435i/color/image_raw \
  depth_topic:=/d435i/d435i/aligned_depth_to_color/image_raw \
  camera_info_topic:=/d435i/d435i/color/camera_info \
  rgb_image_transport:=compressed \
  frame_id:=base_link \
  odom_frame_id:=odom \
  map_frame_id:=map \
  odom_topic:=/odom \
  visual_odometry:=false \
  approx_sync:=true \
  approx_sync_max_interval:=0.15 \
  queue_size:=5 \
  sync_queue_size:=5 \
  topic_queue_size:=2 \
  wait_for_transform:=2.0 \
  database_path:=$DB \
  localization:=true \
  rtabmap_args:="--Vis/MinInliers 15 --RGBD/OptimizeMaxError 3.0 --RGBD/Force3DoF true --RGBD/ProximityBySpace true --RGBD/LocalRadius 10 --RGBD/ProximityPathMaxNeighbors 8 --RGBD/ProximityMaxGraphDepth 50 --Rtabmap/DetectionRate 2 --Kp/MaxFeatures 500 --Mem/ImagePreDecimation 2 --Mem/InitWMWithAllNodes true" \
  rviz:=false \
  rtabmap_viz:=false \
  > /tmp/nav_rtabmap.log 2>&1 &
RTAB_PID=$!
sleep 5

echo "[nav] waiting for /rtabmap/map (rclpy probe; ros2cli daemon/list flaky on crossover)..."
# Ensure probe exists even on older checkouts
if [ ! -f "$WAIT_MAP" ] && [ -f "$WS/scripts/wait_map_probe.py" ]; then
  cp -f "$WS/scripts/wait_map_probe.py" "$WAIT_MAP"
  chmod +x "$WAIT_MAP"
fi
if ! FASTRTPS_DEFAULT_PROFILES_FILE="$XPROF" \
    python3 "$WAIT_MAP" /rtabmap/map 90; then
  echo "  [FAIL] /rtabmap/map not published — is DB valid? See /tmp/nav_rtabmap.log"
  exit 1
fi
echo "  [OK]   /rtabmap/map present"

echo "[nav] relaying /rtabmap/map -> /map for Nav2..."
FASTRTPS_DEFAULT_PROFILES_FILE="$XPROF" \
  python3 "$MAP_RELAY" > /tmp/nav_map_relay.log 2>&1 &
RELAY_PID=$!
sleep 2
if ! FASTRTPS_DEFAULT_PROFILES_FILE="$XPROF" \
    python3 "$WAIT_MAP" /map 45; then
  echo "  [FAIL] /map relay did not come up — see /tmp/nav_map_relay.log"
  exit 1
fi
echo "  [OK]   /map present (relay pid=$RELAY_PID)"

echo "[nav] relaying /initialpose -> /rtabmap/initialpose (for RViz 2D Pose Estimate)..."
FASTRTPS_DEFAULT_PROFILES_FILE="$XPROF" \
  python3 "$POSE_RELAY" > /tmp/nav_pose_relay.log 2>&1 &
POSE_PID=$!
sleep 1

echo "[nav] launching Nav2 (RPP controller, static+/scan costmaps)..."
echo "      bridge=$BRIDGE_PID odomtf=$ODOM_TF_PID tf=$TF_PID scan=$SCAN_PID scanfix=$SCANFIX_PID rtab=$RTAB_PID maprelay=$RELAY_PID pose=$POSE_PID"
echo "      Tip: Fixed Frame=map. UNCHECK Amcl Particle Swarm (unused)."
echo "      Use TF axes / RobotModel as truth. 2D Pose Estimate -> wait 3s -> Nav2 Goal."
echo "      Named goals: bash scripts/go_to_named.sh kitchen   (see config/named_locations.yaml)"

FASTRTPS_DEFAULT_PROFILES_FILE="$XPROF" \
  ros2 launch nav2_bringup navigation_launch.py \
  use_sim_time:=false \
  params_file:="$PARAMS" \
  autostart:=true

echo "[nav] Nav2 exited. Cleaning helpers..."
kill $BRIDGE_PID $ODOM_TF_PID $TF_PID $SCAN_PID $SCANFIX_PID $RTAB_PID $RELAY_PID $POSE_PID 2>/dev/null
pkill -f 'map_to_nav.py' 2>/dev/null
pkill -f 'scan_stamp_fix.py' 2>/dev/null
pkill -f 'initialpose_to_rtab.py' 2>/dev/null
pkill -f 'odom_to_tf.py' 2>/dev/null
