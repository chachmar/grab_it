#!/usr/bin/env bash
# Kill all ACT train jobs and start exactly one.
set -euo pipefail
pkill -f '/lerobot-venv/bin/lerobot-train' 2>/dev/null || true
pkill -f 'lerobot-train --policy' 2>/dev/null || true
sleep 3
# If output dir exists from aborted multi-train, refresh it
OUT="$HOME/dev-ws/datasets/lerobot/outputs/act_tabletop_pick"
if [[ -d "$OUT" ]]; then
  bak="${OUT}_abort_$(date +%Y%m%d_%H%M%S)"
  mv "$OUT" "$bak"
  echo "moved partial → $bak"
fi
source "$HOME/lerobot-venv/bin/activate"
nohup bash "$HOME/dev-ws/scripts/il_train.sh" >/tmp/il_retrain.log 2>&1 &
echo "train_pid=$!"
sleep 6
pgrep -af 'lerobot-train' | head -5 || true
tail -20 /tmp/il_retrain.log || true
