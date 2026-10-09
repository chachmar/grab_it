#!/usr/bin/env bash
# Finish tutorial ACT to 80k with chunked cool-downs (AGX brown-out safe).
set -euo pipefail
source "$HOME/lerobot-venv/bin/activate"
WS="${ISAAC_ROS_WS:-$HOME/dev-ws}"
export GRAB_IT_IL_OUT="${GRAB_IT_IL_OUT:-$WS/datasets/lerobot/outputs/act_foam_pole_lookreach_office_tutorial}"
export GRAB_IT_IL_LEROBOT="${GRAB_IT_IL_LEROBOT:-$WS/datasets/lerobot/grabit_lookreach_office_tutorial}"
export GRAB_IT_IL_STEPS="${GRAB_IT_IL_STEPS:-80000}"
export GRAB_IT_IL_CHUNK="${GRAB_IT_IL_CHUNK:-5000}"
export GRAB_IT_IL_BATCH="${GRAB_IT_IL_BATCH:-8}"
export GRAB_IT_IL_LOG="${GRAB_IT_IL_LOG:-$GRAB_IT_IL_OUT/train_chunked.log}"

mkdir -p "$GRAB_IT_IL_OUT"
touch "$GRAB_IT_IL_LOG"
ln -sfn "$GRAB_IT_IL_LOG" /tmp/il_retrain_lookreach_office_tutorial.log 2>/dev/null || true

# Cooler than MAXN before the loop (chunked script also tries this).
if command -v nvpmodel >/dev/null 2>&1; then
  sudo nvpmodel -m 2 || sudo nvpmodel -m 1 || true
fi

exec bash "$WS/scripts/il_train_chunked.sh"
