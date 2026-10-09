#!/usr/bin/env bash
# Train ACT on varied-start closed demos plus centered pinch-hold.
# Warm-start from the office lookreach run that still aimed at the pole.
set -euo pipefail
source "$HOME/lerobot-venv/bin/activate"
WS="${ISAAC_ROS_WS:-$HOME/dev-ws}"
VARIED="${GRAB_IT_IL_VARIED:-$WS/datasets/tabletop_pick_lookreach_office_varied}"
PINCH="${GRAB_IT_IL_PINCH:-$WS/datasets/tabletop_pick_lookreach_office_pinch}"
RAW="${GRAB_IT_IL_RAW:-$WS/datasets/tabletop_pick_lookreach_office_varied_mix}"
DS="${GRAB_IT_IL_LEROBOT:-$WS/datasets/lerobot/grabit_lookreach_office_varied}"
OUT="${GRAB_IT_IL_OUT:-$WS/datasets/lerobot/outputs/act_foam_pole_lookreach_office_varied}"
PRE="${GRAB_IT_IL_PRETRAINED:-$WS/datasets/lerobot/outputs/act_foam_pole_lookreach_office/checkpoints/last/pretrained_model}"
STEPS="${GRAB_IT_IL_STEPS:-12000}"

nv=$(find "$VARIED" -maxdepth 1 -type d -name 'episode_*' | wc -l)
np=$(find "$PINCH" -maxdepth 1 -type d -name 'episode_*' | wc -l)
echo "varied raw episodes: $nv"
echo "pinch raw episodes: $np"
if [[ "$nv" -lt 30 ]]; then
  echo "ERROR: need >=30 varied-start episodes; have $nv"
  exit 1
fi
if [[ "$np" -lt 15 ]]; then
  echo "ERROR: need >=15 pinch episodes; have $np"
  exit 1
fi
rm -rf "$RAW"
mkdir -p "$RAW"
i=0
for src in "$VARIED" "$PINCH"; do
  while IFS= read -r ep; do
    [[ -n "$ep" ]] || continue
    dest=$(printf "%s/episode_%06d" "$RAW" "$i")
    cp -a "$ep" "$dest"
    i=$((i + 1))
  done < <(find "$src" -maxdepth 1 -type d -name 'episode_*' | sort)
done
echo "mix episodes: $i"
python3 "$WS/scripts/il_filter_loose_pad.py" --dataset "$RAW" --max-pad 0.034 || true
n=$(find "$RAW" -maxdepth 1 -type d -name 'episode_*' | wc -l)
echo "after filter: $n"
if [[ "$n" -lt 40 ]]; then
  echo "ERROR: mix has $n episodes after filter; need >=40"
  exit 1
fi
rm -rf "$DS"
python3 "$WS/scripts/il_to_lerobot.py" --input "$RAW" --output "$DS"
if [[ -d "$OUT" ]]; then
  mv "$OUT" "${OUT}_prev_$(date +%Y%m%d_%H%M%S)"
fi
if [[ ! -f "$PRE/model.safetensors" ]]; then
  echo "ERROR: missing warm-start $PRE/model.safetensors"
  exit 1
fi
case "$PRE" in
  *office_pinch*|*office_near*)
    echo "ERROR: refusing warm-start from $PRE"
    exit 1
    ;;
esac
extra=()
if lerobot-train --help 2>/dev/null | grep -q -- '--policy.pretrained_path'; then
  extra+=(--policy.pretrained_path="$PRE")
  echo "warm-start policy.pretrained_path=$PRE"
else
  echo "ERROR: lerobot-train has no --policy.pretrained_path"
  exit 1
fi
lerobot-train \
  --policy.type=act \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/grabit_lookreach_office_varied \
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
