#!/bin/bash
# run_grabit_dev.sh — start the Isaac container from the docker-committed grab_it snapshot
# (grab_it_isaac_ros_dev:latest) instead of rebuilding from NVIDIA base layers.
#
# Usage (AGX host):
#   bash ~/dev-ws/run_grabit_dev.sh
#
# After installing more packages inside the container, refresh the snapshot:
#   docker commit isaac_ros_dev-aarch64-container grab_it_isaac_ros_dev:latest
#   docker tag grab_it_isaac_ros_dev:latest grab_it_isaac_ros_dev:$(date +%Y%m%d_%H%M%S)
#
# To rebuild from Dockerfile.grabit instead (apt layer), use stock run_dev:
#   cd ~/dev-ws/src/isaac_ros_common/scripts && ./run_dev.sh -d ~/dev-ws
set -euo pipefail

WS=${ISAAC_ROS_WS:-$HOME/dev-ws}
SNAP=${GRABIT_IMAGE:-grab_it_isaac_ros_dev:latest}
PLATFORM=$(uname -m)
BASE="isaac_ros_dev-${PLATFORM}"
SCRIPTS="$WS/src/isaac_ros_common/scripts"

if [[ ! -x "$SCRIPTS/run_dev.sh" ]]; then
  echo "[grabit] run_dev.sh not found at $SCRIPTS/run_dev.sh"
  exit 1
fi

if ! docker image inspect "$SNAP" >/dev/null 2>&1; then
  echo "[grabit] Missing snapshot image: $SNAP"
  echo "  Create with:"
  echo "    docker commit isaac_ros_dev-aarch64-container grab_it_isaac_ros_dev:latest"
  exit 1
fi

echo "[grabit] Using snapshot: $SNAP"
echo "[grabit] Tagging as $BASE:latest for run_dev.sh"
docker tag "$SNAP" "${BASE}:latest"

echo "[grabit] Launching (skip image build, workspace=$WS)"
cd "$SCRIPTS"
exec ./run_dev.sh --skip_image_build -d "$WS" "$@"
