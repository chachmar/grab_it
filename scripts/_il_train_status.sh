#!/usr/bin/env bash
set -euo pipefail
echo "procs:"
pgrep -af 'lerobot-venv/bin/lerobot-train' | head -10 || echo none
echo "--- log tail ---"
if [[ -f /tmp/il_retrain.log ]]; then
  tr '\r' '\n' </tmp/il_retrain.log | grep -E 'Training:|Error|Traceback|steps=|Start offline|CUDA|killed' | tail -20
  echo "last lines:"
  tr '\r' '\n' </tmp/il_retrain.log | tail -8
fi
echo "--- ckpt ---"
ls -la "$HOME/dev-ws/datasets/lerobot/outputs/act_tabletop_pick/checkpoints/last/pretrained_model/model.safetensors" 2>&1 | head -2
