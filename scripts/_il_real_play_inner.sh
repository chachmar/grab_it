#!/usr/bin/env bash
# Shoulder ACT on the real PX150. Stays idle until the GUI sends PLAY.
# Does not open the Dynamixel bus — Nano xs_sdk already owns it.
set -eo pipefail
WS=/workspaces/isaac_ros-dev
CKPT="${GRAB_IT_IL_CKPT:-$WS/datasets/lerobot/outputs/act_foam_pole_lookreach_office_shoulder/checkpoints/last/pretrained_model}"
source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"
for venv in /home/admin/lerobot-venv "$HOME/lerobot-venv" /opt/lerobot-venv; do
  if [[ -f "$venv/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "$venv/bin/activate"
    break
  fi
done
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
unset CYCLONEDDS_URI || true
export FASTRTPS_DEFAULT_PROFILES_FILE="$WS/bridge_both.xml"
export GRAB_IT_IL_WAIST_LOCK=hold
export GRAB_IT_IL_POLE_XY="${GRAB_IT_IL_POLE_XY:-0.22,0.00}"
export PYTHONPATH="$WS/src/grab_it_il:${PYTHONPATH:-}"
exec python3 "$WS/src/grab_it_il/grab_it_il/play_policy.py" --ros-args \
  -p checkpoint:="$CKPT" \
  -p device:=cuda \
  -p n_action_steps:=1 \
  -p temporal_ensemble_coeff:=0.01 \
  -p autoplay:=false \
  -p joint_envelope:="$CKPT/joint_envelope.json"
