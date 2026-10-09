#!/usr/bin/env bash
set -euo pipefail
OUT="$HOME/dev-ws/datasets/lerobot/outputs/act_foam_pole_lookreach_office_tutorial"
mkdir -p "$OUT"
touch "$OUT/train_chunked.log"
ln -sfn "$OUT/train_chunked.log" /tmp/il_retrain_lookreach_office_tutorial.log
# Do not pkill ourselves; only stop stray train if any.
pkill -f '/lerobot-train ' 2>/dev/null || true
sleep 2
nohup bash "$HOME/dev-ws/scripts/il_finish_lookreach_office_tutorial_chunked.sh" \
  >>"$OUT/train_chunked.log" 2>&1 &
echo "STARTED pid=$!"
sleep 20
pgrep -af 'lerobot-train|il_train_chunked|il_finish_lookreach' | head -8 || true
ls "$OUT/checkpoints" || true
tail -c 2500 "$OUT/train_chunked.log" | tr '\r' '\n' | tail -30 || true
