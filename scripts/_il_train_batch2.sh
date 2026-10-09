#!/usr/bin/env bash
set -euo pipefail
echo "=== dmesg ==="
dmesg -T 2>/dev/null | grep -iE 'oom|Out of memory|nvgpu|GPU' | tail -20 || true
echo "=== tegrastats sample ==="
timeout 3 tegrastats 2>/dev/null | head -2 || true
echo "=== partial ckpt ==="
ls -la "$HOME/dev-ws/datasets/lerobot/outputs/act_tabletop_pick/checkpoints/" 2>/dev/null || echo none
echo "=== start batch=2 ==="
pkill -f '/lerobot-venv/bin/lerobot-train' 2>/dev/null || true
sleep 2
source "$HOME/lerobot-venv/bin/activate"
export GRAB_IT_IL_BATCH=2
export GRAB_IT_IL_STEPS=20000
OUT="$HOME/dev-ws/datasets/lerobot/outputs/act_tabletop_pick"
if [[ -d "$OUT" ]] && [[ ! -f "$OUT/checkpoints/last/pretrained_model/model.safetensors" ]]; then
  mv "$OUT" "${OUT}_partial_$(date +%Y%m%d_%H%M%S)" || true
fi
# Prefer performance mode if available
sudo nvpmodel -m 0 2>/dev/null || true
sudo jetson_clocks 2>/dev/null || true
nohup env PYTHONUNBUFFERED=1 bash "$HOME/dev-ws/scripts/il_train.sh" >/tmp/il_retrain.log 2>&1 &
echo "train_pid=$!"
sleep 8
pgrep -af 'lerobot-venv/bin/lerobot-train' | head -3
tail -25 /tmp/il_retrain.log
