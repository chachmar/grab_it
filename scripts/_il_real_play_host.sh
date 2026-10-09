#!/usr/bin/env bash
# Host-side shoulder ACT for the real PX150. Idle until GUI PLAY.
set -eo pipefail
WS=/home/ubuntu/dev-ws
CKPT="${GRAB_IT_IL_CKPT:-}"
if [ -z "$CKPT" ] && [ -f /tmp/grabit_act_name ]; then
  _act_name=$(tr -d '\r\n' </tmp/grabit_act_name)
  _act_ckpt="$WS/datasets/lerobot/outputs/${_act_name}/checkpoints/last/pretrained_model"
  if [ -d "$_act_ckpt" ]; then
    CKPT="$_act_ckpt"
  fi
fi
CKPT="${CKPT:-$WS/datasets/lerobot/outputs/act_foam_pole_lookreach_office_shoulder/checkpoints/last/pretrained_model}"
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source /home/ubuntu/lerobot-venv/bin/activate
GEN="${WS}/build/interbotix_xs_msgs/rosidl_generator_py/interbotix_xs_msgs"
ROOT="${WS}/build/interbotix_xs_msgs"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
unset CYCLONEDDS_URI || true
export FASTRTPS_DEFAULT_PROFILES_FILE="$WS/rtabmap_profile.xml"
# Stay where the arm is. Policy motion is the change from its first command,
# not a move into the demo park (that fold hits the electronics).
export GRAB_IT_IL_FROM_CURRENT=1
unset GRAB_IT_IL_SLEEP_HOLD || true
unset GRAB_IT_IL_WAIST_LOCK || true
export GRAB_IT_IL_POLE_XY="${GRAB_IT_IL_POLE_XY:-0.22,0.00}"
export PYTHONPATH="${WS}/src/grab_it_il:${WS}/build/interbotix_xs_msgs/rosidl_generator_py:/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${GEN}:${ROOT}:${LD_LIBRARY_PATH:-}"
exec python3 "$WS/src/grab_it_il/grab_it_il/play_policy.py" --ros-args \
  -p checkpoint:="$CKPT" \
  -p device:=cuda \
  -p n_action_steps:=1 \
  -p temporal_ensemble_coeff:=0.01 \
  -p autoplay:=false \
  -p joint_envelope:="$CKPT/joint_envelope.json"
