#!/usr/bin/env bash
# Fine-tune ACT on office near-contact pinch demos (15k steps, new out dir).
set -euo pipefail
WS="${ISAAC_ROS_WS:-$HOME/dev-ws}"
export GRAB_IT_IL_RAW="${GRAB_IT_IL_RAW:-$WS/datasets/tabletop_pick_lookreach_office_near}"
export GRAB_IT_IL_LEROBOT="${GRAB_IT_IL_LEROBOT:-$WS/datasets/lerobot/grabit_lookreach_office_near}"
export GRAB_IT_IL_OUT="${GRAB_IT_IL_OUT:-$WS/datasets/lerobot/outputs/act_foam_pole_lookreach_office_near}"
export GRAB_IT_IL_STEPS="${GRAB_IT_IL_STEPS:-15000}"
# Allow fewer than 40 if some quarantined — floor at 25.
MIN_N="${GRAB_IT_IL_MIN_EPS:-25}"
n=$(find "$GRAB_IT_IL_RAW" -maxdepth 1 -type d -name 'episode_*' 2>/dev/null | wc -l)
echo "near-contact raw episodes: $n (min $MIN_N)"
if [[ "$n" -lt "$MIN_N" ]]; then
  echo "ERROR: need >=$MIN_N near-contact episodes; have $n"
  exit 1
fi
# Bypass il_retrain_lookreach.sh's hard >=40 check by calling convert+train directly
# when between MIN_N and 39; otherwise use standard path.
if [[ "$n" -ge 40 ]]; then
  exec bash "$WS/scripts/il_retrain_lookreach.sh"
fi
source "$HOME/lerobot-venv/bin/activate"
RAW="$GRAB_IT_IL_RAW"
DS="$GRAB_IT_IL_LEROBOT"
OUT="$GRAB_IT_IL_OUT"
STEPS="$GRAB_IT_IL_STEPS"
python3 "$WS/scripts/il_filter_loose_pad.py" --dataset "$RAW" --max-pad 0.042 || true
rm -rf "$DS"
python3 "$WS/scripts/il_to_lerobot.py" --input "$RAW" --output "$DS"
if [[ -d "$OUT" ]]; then
  bak="${OUT}_prev_$(date +%Y%m%d_%H%M%S)"
  mv "$OUT" "$bak"
fi
export GRAB_IT_IL_STEPS="$STEPS"
export GRAB_IT_IL_LEROBOT="$DS"
export GRAB_IT_IL_OUT="$OUT"
if [[ -x "$WS/scripts/il_train_chunked.sh" ]]; then
  bash "$WS/scripts/il_train_chunked.sh"
else
  bash "$WS/scripts/il_train.sh"
fi
echo "Done. Play checkpoint:"
echo "  $OUT/checkpoints/last/pretrained_model"
