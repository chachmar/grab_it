#!/bin/bash
# nano_bringup.sh — cameras (D435i + wide USB) + PX150 arm on the Nano.
# Run inside grab_it_nano:
#   /root/nano_bringup.sh
set -m
source /opt/ros/humble/setup.bash
source /root/grab_it_ws/install/setup.bash 2>/dev/null || true
unset CYCLONEDDS_URI
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
export FASTRTPS_DEFAULT_PROFILES_FILE=/root/crossover_only.xml

PIDS=()
cleanup() {
  echo "[nano] stopping..."
  for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done
  pkill -f 'realsense2_camera_node|usb_cam_node|xs_sdk|xsarm_control|image_transport' 2>/dev/null || true
  pkill -f 'republish raw compressed' 2>/dev/null || true
}
trap cleanup INT TERM EXIT

# Drop stale republishers from prior bringups — duplicates saturate Nano CPU and
# flood the crossover with N copies of the same JPEG stream (GUI looks seconds late).
pkill -9 -f 'image_transport/republish' 2>/dev/null || true
pkill -9 -f 'republish raw compressed' 2>/dev/null || true
sleep 1

# --- /dev/ttyDXL (U2D2) ---
ensure_ttydxl() {
  if [ -e /dev/ttyDXL ]; then
    return 0
  fi
  # udev rule: serial FT6S4H3P -> ttyDXL; try by-id / refresh
  if [ -d /dev/serial/by-id ]; then
    local link
    link=$(ls /dev/serial/by-id/*FT6S4H3P* 2>/dev/null | head -1 || true)
    if [ -n "$link" ]; then
      ln -sf "$(readlink -f "$link")" /dev/ttyDXL
      chmod 666 /dev/ttyDXL 2>/dev/null || true
    fi
  fi
  if [ ! -e /dev/ttyDXL ]; then
    udevadm trigger -c add -s tty 2>/dev/null || true
    sleep 1
  fi
  [ -e /dev/ttyDXL ]
}

echo "[nano] starting dual cameras (D435i + wide)..."
ros2 launch grab_it_cameras dual_camera.launch.py > /tmp/cameras.log 2>&1 &
PIDS+=($!)
sleep 2

# Ensure compressed topics for AGX mjpeg_relay (GUI video needs these over crossover).
# Exactly ONE republisher per stream — never stack on top of leftovers.
if ros2 pkg prefix image_transport >/dev/null 2>&1; then
  echo "[nano] republishing /wide_cam/image_raw -> compressed (single)"
  ros2 run image_transport republish raw compressed \
    --ros-args -r in:=/wide_cam/image_raw \
    -r out/compressed:=/wide_cam/image_raw/compressed \
    > /tmp/wide_compress.log 2>&1 &
  PIDS+=($!)
  echo "[nano] republishing /d435i/.../color/image_raw -> compressed (single)"
  ros2 run image_transport republish raw compressed \
    --ros-args -r in:=/d435i/d435i/color/image_raw \
    -r out/compressed:=/d435i/d435i/color/image_raw/compressed \
    > /tmp/d435i_compress.log 2>&1 &
  PIDS+=($!)
fi

ARM_PID=""
if ensure_ttydxl; then
  echo "[nano] /dev/ttyDXL OK - starting PX150 (no RViz)..."
  # Only one xs_sdk may own the bus - duplicates cause syncRead failures,
  # purple GUI blocks, and RViz arm links pulsing.
  pkill -9 -f xsarm_control.launch.py 2>/dev/null || true
  pkill -9 -f 'interbotix_xs_sdk/xs_sdk' 2>/dev/null || true
  sleep 1
  # xsarm_control always starts robot_state_publisher; that fights AGX grab_it
  # TF (arm under top_plate). Keep xs_sdk only - AGX publishes the full model.
  ros2 launch interbotix_xsarm_control xsarm_control.launch.py \
    robot_model:=px150_no_elbow \
    hardware_type:=actual \
    use_rviz:=false \
    > /tmp/arm.log 2>&1 &
  ARM_PID=$!
  PIDS+=($ARM_PID)
  (
    # Launch may respawn RSP; suppress it while the arm launch lives.
    # Match /proc/PID/exe — /proc/PID/comm is truncated to 15 chars
    # ("robot_state_pu"), so name compares never match.
    while kill -0 "$ARM_PID" 2>/dev/null; do
      for pid in /proc/[0-9]*; do
        exe=$(readlink "$pid/exe" 2>/dev/null || true)
        case "$exe" in
          */robot_state_publisher)
            kill -9 "${pid#/proc/}" 2>/dev/null || true
            ;;
        esac
      done
      sleep 1
    done
  ) &
  PIDS+=($!)
  echo "[nano] suppressing Nano robot_state_publisher (AGX grab_it owns TF)"
else
  echo "[nano] WARNING: /dev/ttyDXL missing — arm NOT started."
  echo "       Plug in the U2D2 (FTDI serial FT6S4H3P), 12V servo power ON, then re-run."
  echo "       lsusb should show FTDI; udev rule is in /etc/udev/rules.d/99-grab-it.rules"
fi

echo "[nano] up: cameras + ${ARM_PID:+arm=$ARM_PID}${ARM_PID:-no arm}  (Ctrl-C stops all)"
echo "[nano] logs: /tmp/cameras.log /tmp/arm.log"
wait
