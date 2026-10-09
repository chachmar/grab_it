#!/usr/bin/env bash
# Host: robot container + click-to-grasp + idle ACT player for the live GUI.
set -eo pipefail
NAME=grab_it_live
IMAGE=isaac_ros_dev-aarch64:latest
WS=/home/ubuntu/dev-ws

if [[ -d /home/ubuntu/lerobot-venv ]]; then
  EXTRA_VENV=(-v /home/ubuntu/lerobot-venv:/home/admin/lerobot-venv)
else
  EXTRA_VENV=()
fi
if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker run -d \
    --privileged \
    --network host \
    --ipc=host \
    --pid=host \
    --runtime nvidia \
    --name "$NAME" \
    -v "$WS/src":/workspaces/isaac_ros-dev/src \
    -v "$WS/scripts":/workspaces/isaac_ros-dev/scripts \
    -v "$WS/datasets":/workspaces/isaac_ros-dev/datasets \
    -v "$WS/config":/workspaces/isaac_ros-dev/config \
    -v "$WS/build":/workspaces/isaac_ros-dev/build \
    -v "$WS/install":/workspaces/isaac_ros-dev/install \
    -v "$WS/bridge_both.xml":/workspaces/isaac_ros-dev/bridge_both.xml \
    -v "$WS/rtabmap_profile.xml":/workspaces/isaac_ros-dev/rtabmap_profile.xml \
    "${EXTRA_VENV[@]}" \
    -v /etc/localtime:/etc/localtime:ro \
    -v /tmp:/tmp \
    -v /dev:/dev \
    -e NVIDIA_VISIBLE_DEVICES=all \
    --entrypoint /usr/local/bin/scripts/workspace-entrypoint.sh \
    --workdir /workspaces/isaac_ros-dev \
    "$IMAGE" \
    sleep infinity
  sleep 3
fi

if ! docker exec "$NAME" bash /workspaces/isaac_ros-dev/scripts/_il_real_alive.sh grasp; then
  echo "starting click-to-grasp (MoveIt + grasp_from_click)"
  cp "$WS/rtabmap_profile.xml" /tmp/rtabmap_profile.xml
  docker exec -d "$NAME" bash -lc "export BPROF=/tmp/rtabmap_profile.xml FASTRTPS_DEFAULT_PROFILES_FILE=/tmp/rtabmap_profile.xml; nohup bash /workspaces/isaac_ros-dev/scripts/grabit_arm_bringup.sh >/tmp/grabit_arm_bringup.log 2>&1"
fi

if ! pgrep -f 'grab_it_il/play_policy.py' >/dev/null; then
  echo "starting ACT player on the host (idle until PLAY)"
  nohup bash "$WS/scripts/_il_real_play_host.sh" >/tmp/il_real_play_host.log 2>&1 &
fi

echo "container=$NAME"
docker exec "$NAME" bash /workspaces/isaac_ros-dev/scripts/_il_real_alive.sh status || true
