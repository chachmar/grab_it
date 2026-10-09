#!/usr/bin/env bash
# Train ACT from look-then-reach GT-IK demos (pan in state/action).
# Run on AGX HOST: bash ~/dev-ws/scripts/il_retrain_lookreach.sh
set -euo pipefail
# shellcheck disable=SC1091
source "$HOME/lerobot-venv/bin/activate"
WS="${ISAAC_ROS_WS:-$HOME/dev-ws}"
RAW="${GRAB_IT_IL_RAW:-$WS/datasets/tabletop_pick_lookreach}"
DS="${GRAB_IT_IL_LEROBOT:-$WS/datasets/lerobot/grabit_lookreach}"
OUT="${GRAB_IT_IL_OUT:-$WS/datasets/lerobot/outputs/act_foam_pole_lookreach}"
STEPS="${GRAB_IT_IL_STEPS:-20000}"

echo "python=$(which python3)"
python3 -c "import lerobot; print('lerobot ok')"

n=$(find "$RAW" -maxdepth 1 -type d -name 'episode_*' 2>/dev/null | wc -l)
echo "raw lookreach episodes under $RAW: $n"
if [[ "$n" -lt 40 ]]; then
  echo "ERROR: expected >=40 lookreach episodes; have $n — aborting"
  exit 1
fi

echo "Quarantine loose-pad demos ..."
python3 "$WS/scripts/il_filter_loose_pad.py" --dataset "$RAW" --max-pad 0.042 || true
n=$(find "$RAW" -maxdepth 1 -type d -name 'episode_*' 2>/dev/null | wc -l)
echo "episodes after filter: $n"

echo "Rebuilding LeRobot dataset at $DS ..."
rm -rf "$DS"
python3 "$WS/scripts/il_to_lerobot.py" --input "$RAW" --output "$DS"

if [[ -d "$OUT" ]]; then
  bak="${OUT}_prev_$(date +%Y%m%d_%H%M%S)"
  echo "Moving old output → $bak"
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
