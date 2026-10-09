#!/usr/bin/env bash
# Resume ACT train from last checkpoint (do not wipe output_dir).
set -euo pipefail
source "$HOME/lerobot-venv/bin/activate"
WS="${ISAAC_ROS_WS:-$HOME/dev-ws}"
OUT="${GRAB_IT_IL_OUT:-$WS/datasets/lerobot/outputs/act_tabletop_pick}"
DS="${GRAB_IT_IL_LEROBOT:-$WS/datasets/lerobot/grabit_tabletop_pick}"
STEPS="${GRAB_IT_IL_STEPS:-20000}"
CFG="$OUT/checkpoints/last/pretrained_model/train_config.json"

if [[ ! -f "$CFG" ]]; then
  echo "ERROR: missing $CFG — nothing to resume"
  exit 1
fi
if [[ ! -f "$DS/meta/info.json" ]]; then
  echo "ERROR: missing dataset $DS"
  exit 1
fi

echo "Resuming from $CFG"
echo "output_dir=$OUT steps=$STEPS"
# Prefer config_path + resume so optimizer/step restore from training_state.
exec lerobot-train \
  --config_path="$CFG" \
  --resume=true \
  --steps="$STEPS" \
  --dataset.root="$DS" \
  --dataset.video_backend=pyav \
  --output_dir="$OUT" \
  --policy.device=cuda
