#!/usr/bin/env bash
# Resume tutorial ACT train from last checkpoint through 80k steps.
set -euo pipefail
source "$HOME/lerobot-venv/bin/activate"
WS="${ISAAC_ROS_WS:-$HOME/dev-ws}"
OUT="${GRAB_IT_IL_OUT:-$WS/datasets/lerobot/outputs/act_foam_pole_lookreach_office_tutorial}"
DS="${GRAB_IT_IL_LEROBOT:-$WS/datasets/lerobot/grabit_lookreach_office_tutorial}"
STEPS="${GRAB_IT_IL_STEPS:-80000}"
CFG="$OUT/checkpoints/last/pretrained_model/train_config.json"

if [[ ! -f "$CFG" ]]; then
  echo "ERROR: missing $CFG"
  exit 1
fi
if [[ ! -f "$DS/meta/info.json" ]]; then
  echo "ERROR: missing dataset $DS"
  exit 1
fi

echo "Resuming tutorial ACT from $CFG"
echo "output_dir=$OUT steps=$STEPS"
lerobot-train \
  --config_path="$CFG" \
  --resume=true \
  --steps="$STEPS" \
  --dataset.root="$DS" \
  --dataset.video_backend=pyav \
  --output_dir="$OUT" \
  --policy.device=cuda

# Ensure envelope sits next to final checkpoint for play.
RAW="$WS/datasets/tabletop_pick_lookreach_office_tutorial"
if [[ -f "$RAW/joint_envelope.json" ]]; then
  mkdir -p "$OUT/checkpoints/last/pretrained_model"
  cp -f "$RAW/joint_envelope.json" "$OUT/joint_envelope.json"
  cp -f "$RAW/joint_envelope.json" "$OUT/checkpoints/last/pretrained_model/joint_envelope.json"
fi

echo "Done. Play checkpoint:"
echo "  $OUT/checkpoints/last/pretrained_model"
