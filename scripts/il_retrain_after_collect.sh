#!/usr/bin/env bash
# After Windows collect finishes: rebuild LeRobot dataset + retrain ACT.
# Run on AGX HOST with: source ~/lerobot-venv/bin/activate && bash ~/dev-ws/scripts/il_retrain_after_collect.sh
set -euo pipefail
WS="${ISAAC_ROS_WS:-$HOME/dev-ws}"
RAW="${GRAB_IT_IL_RAW:-$WS/datasets/tabletop_pick}"
DS="${GRAB_IT_IL_LEROBOT:-$WS/datasets/lerobot/grabit_tabletop_pick}"
OUT="${GRAB_IT_IL_OUT:-$WS/datasets/lerobot/outputs/act_tabletop_pick}"
STEPS="${GRAB_IT_IL_STEPS:-20000}"

n=$(find "$RAW" -maxdepth 1 -type d -name 'episode_*' 2>/dev/null | wc -l)
echo "raw episodes under $RAW: $n"
if [[ "$n" -lt 100 ]]; then
  echo "ERROR: expected >=100 episodes after +100 collect; have $n — aborting"
  exit 1
fi

echo "Quarantine loose-pad (early-close) demos ..."
python3 "$WS/scripts/il_filter_loose_pad.py" --dataset "$RAW" --max-pad 0.028
n=$(find "$RAW" -maxdepth 1 -type d -name 'episode_*' 2>/dev/null | wc -l)
echo "episodes after filter: $n"
if [[ "$n" -lt 80 ]]; then
  echo "ERROR: too few tight-kiss episodes after filter ($n) — aborting"
  exit 1
fi

echo "Rebuilding LeRobot dataset at $DS ..."
rm -rf "$DS"
python3 "$WS/scripts/il_to_lerobot.py" --input "$RAW" --output "$DS"

if [[ -d "$OUT" ]]; then
  bak="${OUT}_prev_$(date +%Y%m%d_%H%M%S)"
  echo "Moving old output → $bak"
  mv "$OUT" "$bak"
fi

export GRAB_IT_IL_STEPS="$STEPS"
# Chunked train survives AGX MAXN reboots better than one-shot il_train.sh.
if [[ -x "$WS/scripts/il_train_chunked.sh" ]]; then
  bash "$WS/scripts/il_train_chunked.sh"
else
  bash "$WS/scripts/il_train.sh"
fi
echo "Done. Play checkpoint:"
echo "  $OUT/checkpoints/last/pretrained_model"
