#!/usr/bin/env bash
# Host-side ACT play for Isaac twin (lerobot-venv + Humble DDS).
# Avoids broken colcon local_setup paths by setting PYTHONPATH explicitly.
set -eo pipefail
WS="${ISAAC_ROS_WS:-$HOME/dev-ws}"
CKPT="${1:-$WS/datasets/lerobot/outputs/act_tabletop_pick/checkpoints/last/pretrained_model}"
LOG="${2:-/tmp/il_play_act.log}"

set +u
source /opt/ros/humble/setup.bash
source "$HOME/lerobot-venv/bin/activate"
set -u

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
unset CYCLONEDDS_URI || true
# Host twin play + relay + hub-obs all share default discovery. Whitelist
# profiles (rtabmap/bridge_both) break localhost pub↔sub for il_command.
unset FASTRTPS_DEFAULT_PROFILES_FILE || true

# Host install was built under /workspaces/... inside the container; point Python
# at the bind-mounted install tree instead of sourcing broken local_setup.bash.
export PYTHONPATH="${WS}/install/interbotix_xs_msgs/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages:${PYTHONPATH:-}"
if [[ -d "${WS}/install/grab_it_il/lib/python3.10/site-packages" ]]; then
  export PYTHONPATH="${WS}/install/grab_it_il/lib/python3.10/site-packages:${PYTHONPATH}"
fi
export PYTHONPATH="${WS}/src/grab_it_il:${PYTHONPATH}"
# AMENT prefix so rclpy can load message typesupport libs
export AMENT_PREFIX_PATH="${WS}/install/interbotix_xs_msgs:${WS}/install/grab_it_il:${AMENT_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="${WS}/install/interbotix_xs_msgs/lib:${LD_LIBRARY_PATH:-}"

echo "[il_play_host] ckpt=$CKPT" | tee "$LOG"
echo "[il_play_host] temporal_ensemble_coeff=0.01 n_action_steps=1" | tee -a "$LOG"
echo "[il_play_host] profile=(default discovery)" | tee -a "$LOG"
python3 -c "import torch, lerobot; print('deps OK', torch.__version__, 'cuda', torch.cuda.is_available())" | tee -a "$LOG"

ENV_JSON=""
for cand in \
  "${CKPT}/joint_envelope.json" \
  "$(dirname "$CKPT")/joint_envelope.json" \
  "$(dirname "$(dirname "$CKPT")")/joint_envelope.json"
do
  if [[ -f "$cand" ]]; then
    ENV_JSON="$cand"
    break
  fi
done
if [[ -n "$ENV_JSON" ]]; then
  echo "[il_play_host] joint_envelope=$ENV_JSON" | tee -a "$LOG"
  exec python3 "${WS}/src/grab_it_il/grab_it_il/play_policy.py" --ros-args \
    -p checkpoint:="$CKPT" \
    -p device:=cuda \
    -p n_action_steps:=1 \
    -p temporal_ensemble_coeff:=0.01 \
    -p joint_envelope:="$ENV_JSON" \
    -p autoplay:=true 2>&1 | tee -a "$LOG"
else
  echo "[il_play_host] WARN: no joint_envelope.json near ckpt" | tee -a "$LOG"
  exec python3 "${WS}/src/grab_it_il/grab_it_il/play_policy.py" --ros-args \
    -p checkpoint:="$CKPT" \
    -p device:=cuda \
    -p n_action_steps:=1 \
    -p temporal_ensemble_coeff:=0.01 \
    -p autoplay:=true 2>&1 | tee -a "$LOG"
fi
