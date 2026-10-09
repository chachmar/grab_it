#!/usr/bin/env bash
set -euo pipefail
echo "=== before ==="
pgrep -af 'lerobot-venv/bin/lerobot-train' || true
pgrep -af 'il_train_chunked' || true
# Kill all train + chunked wrappers, then start ONE clean chunked finish.
pkill -f 'lerobot-venv/bin/lerobot-train' 2>/dev/null || true
pkill -f 'scripts/il_train_chunked.sh' 2>/dev/null || true
pkill -f 'il_finish_lookreach_office_tutorial_chunked' 2>/dev/null || true
sleep 3
echo "=== after kill ==="
pgrep -af 'lerobot-train|il_train_chunked' || echo none
OUT="$HOME/dev-ws/datasets/lerobot/outputs/act_foam_pole_lookreach_office_tutorial"
mkdir -p "$OUT"
touch "$OUT/train_chunked.log"
ln -sfn "$OUT/train_chunked.log" /tmp/il_retrain_lookreach_office_tutorial.log
nohup bash "$HOME/dev-ws/scripts/il_finish_lookreach_office_tutorial_chunked.sh" \
  >>"$OUT/train_chunked.log" 2>&1 &
echo "STARTED $!"
sleep 25
echo "=== running ==="
pgrep -af 'lerobot-train|il_train_chunked|il_finish' | head -10 || true
ls "$OUT/checkpoints"
tail -c 1500 "$OUT/train_chunked.log" | tr '\r' '\n' | tail -20
