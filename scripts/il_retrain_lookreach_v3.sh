#!/usr/bin/env bash
# Retrain lookreach ACT on post gripper-readback episodes only (episode_000105+).
# New output dir — do not resume v2. Run on AGX host.
set -euo pipefail
WS="${ISAAC_ROS_WS:-$HOME/dev-ws}"
SRC="${GRAB_IT_IL_SRC:-$WS/datasets/tabletop_pick_lookreach}"
SUB="${GRAB_IT_IL_RAW:-$WS/datasets/tabletop_pick_lookreach_v3}"
MIN_EP="${GRAB_IT_IL_MIN_EP:-105}"

rm -rf "$SUB"
mkdir -p "$SUB"
n=0
for ep in "$SRC"/episode_*; do
  [[ -d "$ep" ]] || continue
  base=$(basename "$ep")
  num=$((10#${base#episode_}))
  if (( num < MIN_EP )); then
    continue
  fi
  # Real directory of hardlinks so find -type d and the loose-pad filter
  # see episodes without moving the originals.
  mkdir -p "$SUB/$base"
  cp -al "$ep/." "$SUB/$base/"
  n=$((n + 1))
done
echo "v3 subset episodes=$n under $SUB (from $MIN_EP)"
if (( n < 40 )); then
  echo "ERROR: expected >=40 episodes from ${MIN_EP}+; have $n"
  exit 1
fi

export GRAB_IT_IL_RAW="$SUB"
export GRAB_IT_IL_LEROBOT="${GRAB_IT_IL_LEROBOT:-$WS/datasets/lerobot/grabit_lookreach_v3}"
export GRAB_IT_IL_OUT="${GRAB_IT_IL_OUT:-$WS/datasets/lerobot/outputs/act_foam_pole_lookreach_v3}"
export GRAB_IT_IL_STEPS="${GRAB_IT_IL_STEPS:-30000}"
exec bash "$WS/scripts/il_retrain_lookreach.sh"
