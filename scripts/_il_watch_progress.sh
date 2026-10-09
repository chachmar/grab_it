#!/usr/bin/env bash
# Live AGX progress for IL sync/train (run via: ssh jetson-agx 'bash -s' < thisfile
# or: ssh jetson-agx bash ~/dev-ws/scripts/_il_watch_progress.sh)
set -euo pipefail
echo "=== AGX IL progress (Ctrl+C to stop) ==="
while true; do
  printf '\n======== %s ========\n' "$(date '+%Y-%m-%d %H:%M:%S')"
  n=$(ls "$HOME/dev-ws/datasets/tabletop_pick" 2>/dev/null | grep -c '^episode_' || true)
  echo "episodes on AGX: ${n:-0}"
  echo
  echo "sync/train processes:"
  pgrep -af 'scp|rsync|il_train|lerobot-train|il_to_lerobot|il_retrain' 2>/dev/null | grep -v pgrep || echo none
  echo
  echo "--- last train log ---"
  if [[ -f /tmp/il_retrain.log ]]; then
    tail -8 /tmp/il_retrain.log
  elif [[ -f /tmp/il_train.log ]]; then
    tail -8 /tmp/il_train.log
  else
    echo "no train log yet"
  fi
  sleep 3
done
