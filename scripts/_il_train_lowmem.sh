#!/usr/bin/env bash
# Single low-memory ACT train after AGX reboot (batch=4).
set -euo pipefail
pkill -f '/lerobot-venv/bin/lerobot-train' 2>/dev/null || true
sleep 2
source "$HOME/lerobot-venv/bin/activate"
export GRAB_IT_IL_BATCH=4
export GRAB_IT_IL_STEPS="${GRAB_IT_IL_STEPS:-20000}"
OUT="$HOME/dev-ws/datasets/lerobot/outputs/act_tabletop_pick"
if [[ -d "$OUT" ]] && [[ ! -f "$OUT/checkpoints/last/pretrained_model/model.safetensors" ]]; then
  mv "$OUT" "${OUT}_partial_$(date +%Y%m%d_%H%M%S)" || true
fi
nohup bash "$HOME/dev-ws/scripts/il_train.sh" >/tmp/il_retrain.log 2>&1 &
echo "train_pid=$! batch=$GRAB_IT_IL_BATCH steps=$GRAB_IT_IL_STEPS"
sleep 5
pgrep -af 'lerobot-venv/bin/lerobot-train' | head -3
tail -15 /tmp/il_retrain.log || true
