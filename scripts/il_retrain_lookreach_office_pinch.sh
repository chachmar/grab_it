#!/usr/bin/env bash
# Fine-tune near-contact ACT on centered pinch-hold demos (~8k).
set -euo pipefail
source "$HOME/lerobot-venv/bin/activate"
WS="${ISAAC_ROS_WS:-$HOME/dev-ws}"
RAW="${GRAB_IT_IL_RAW:-$WS/datasets/tabletop_pick_lookreach_office_pinch}"
DS="${GRAB_IT_IL_LEROBOT:-$WS/datasets/lerobot/grabit_lookreach_office_pinch}"
OUT="${GRAB_IT_IL_OUT:-$WS/datasets/lerobot/outputs/act_foam_pole_lookreach_office_pinch}"
PRE="${GRAB_IT_IL_PRETRAINED:-$WS/datasets/lerobot/outputs/act_foam_pole_lookreach_office_near/checkpoints/last/pretrained_model}"
STEPS="${GRAB_IT_IL_STEPS:-8000}"

n=$(find "$RAW" -maxdepth 1 -type d -name 'episode_*' | wc -l)
echo "pinch raw episodes: $n"
if [[ "$n" -lt 15 ]]; then
  echo "ERROR: need >=15 enclosed pinch episodes; have $n"
  exit 1
fi
python3 "$WS/scripts/il_filter_loose_pad.py" --dataset "$RAW" --max-pad 0.034 || true
n=$(find "$RAW" -maxdepth 1 -type d -name 'episode_*' | wc -l)
echo "after filter: $n"
rm -rf "$DS"
python3 "$WS/scripts/il_to_lerobot.py" --input "$RAW" --output "$DS"
if [[ -d "$OUT" ]]; then
  mv "$OUT" "${OUT}_prev_$(date +%Y%m%d_%H%M%S)"
fi
# Prefer warm-start from the near-contact checkpoint when the CLI accepts it.
extra=()
if [[ ! -f "$PRE/model.safetensors" ]]; then
  echo "ERROR: missing warm-start $PRE/model.safetensors"
  exit 1
fi
if lerobot-train --help 2>/dev/null | grep -q -- '--policy.pretrained_path'; then
  extra+=(--policy.pretrained_path="$PRE")
  echo "warm-start policy.pretrained_path=$PRE"
elif lerobot-train --help 2>/dev/null | grep -q 'policy.path'; then
  extra+=(--policy.path="$PRE")
  echo "warm-start policy.path=$PRE"
else
  echo "ERROR: lerobot-train has no warm-start flag; refusing from-scratch pinch train"
  exit 1
fi
lerobot-train \
  --policy.type=act \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/grabit_lookreach_office_pinch \
  --dataset.root="$DS" \
  --dataset.video_backend=pyav \
  --output_dir="$OUT" \
  --policy.device=cuda \
  --batch_size=4 \
  --steps="$STEPS" \
  --save_freq=2000 \
  "${extra[@]}"
echo "Done. Play checkpoint:"
echo "  $OUT/checkpoints/last/pretrained_model"
