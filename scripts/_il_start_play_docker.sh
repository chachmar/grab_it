#!/usr/bin/env bash
# Run ACT play_policy inside isaac_ros container with host lerobot-venv site-packages.
set -eo pipefail
WS_HOST="${HOME}/dev-ws"
CKPT_HOST="${1:-$WS_HOST/datasets/lerobot/outputs/act_tabletop_pick/checkpoints/last/pretrained_model}"
LOG="${2:-/tmp/il_play_act.log}"

# Expose host venv inside the container bind mount
if [[ ! -e "$WS_HOST/lerobot-venv" ]]; then
  ln -s "$HOME/lerobot-venv" "$WS_HOST/lerobot-venv"
fi

pkill -f "play_policy.py|grab_it_il_play" 2>/dev/null || true
docker exec isaac_ros_dev-aarch64-container bash -lc "pkill -f play_policy.py 2>/dev/null || true" || true
sleep 1

CKPT_CT="${CKPT_HOST/$WS_HOST/\/workspaces\/isaac_ros-dev}"
# Fix path replace for bash
CKPT_CT="/workspaces/isaac_ros-dev/datasets/lerobot/outputs/act_tabletop_pick/checkpoints/last/pretrained_model"

nohup docker exec -u admin -w /workspaces/isaac_ros-dev isaac_ros_dev-aarch64-container bash -lc "
set +u
source /opt/ros/humble/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
unset CYCLONEDDS_URI
# Default FastRTPS discovery (bridge_both whitelist broke play↔relay on localhost).
unset FASTRTPS_DEFAULT_PROFILES_FILE
export PYTHONPATH=/workspaces/isaac_ros-dev/lerobot-venv/lib/python3.10/site-packages:\${PYTHONPATH:-}
export LD_LIBRARY_PATH=/workspaces/isaac_ros-dev/lerobot-venv/lib/python3.10/site-packages/torch/lib:\${LD_LIBRARY_PATH:-}
echo deps:
python3 -c 'from interbotix_xs_msgs.msg import JointGroupCommand; import torch, lerobot; print(\"OK\", torch.__version__, torch.cuda.is_available())'
echo starting play_policy ckpt=$CKPT_CT
exec python3 /workspaces/isaac_ros-dev/src/grab_it_il/grab_it_il/play_policy.py --ros-args \
  -p checkpoint:=$CKPT_CT -p device:=cuda \
  -p n_action_steps:=1 -p temporal_ensemble_coeff:=0.01
" >"$LOG" 2>&1 &

echo "started docker play pid=$! log=$LOG"
sleep 12
tail -50 "$LOG" || true
