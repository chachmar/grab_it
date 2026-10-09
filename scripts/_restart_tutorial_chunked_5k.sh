#!/usr/bin/env bash
set -euo pipefail
OUT="$HOME/dev-ws/datasets/lerobot/outputs/act_foam_pole_lookreach_office_tutorial"
pkill -f 'lerobot-venv/bin/lerobot-train' 2>/dev/null || true
pkill -f 'scripts/il_train_chunked.sh' 2>/dev/null || true
pkill -f 'il_finish_lookreach_office_tutorial' 2>/dev/null || true
sleep 2
# Truncate null-byte junk at end of log from last crash.
if [[ -f "$OUT/train_chunked.log" ]]; then
  python3 - <<'PY'
from pathlib import Path
from datetime import datetime
p = Path.home() / "dev-ws/datasets/lerobot/outputs/act_foam_pole_lookreach_office_tutorial/train_chunked.log"
data = p.read_bytes().rstrip(b"\x00")
p.write_bytes(data + f"\n=== restart {datetime.now().isoformat()} ===\n".encode())
PY
fi
mkdir -p "$OUT"
ln -sfn "$OUT/train_chunked.log" /tmp/il_retrain_lookreach_office_tutorial.log
nohup bash "$HOME/dev-ws/scripts/il_finish_lookreach_office_tutorial_chunked.sh" \
  >>"$OUT/train_chunked.log" 2>&1 &
echo "STARTED $!"
sleep 20
pgrep -af 'lerobot-train|il_train_chunked' | head -8 || true
ls "$OUT/checkpoints"
tail -c 1200 "$OUT/train_chunked.log" | tr '\r' '\n' | tail -20
