#!/usr/bin/env bash
set -uo pipefail
# Fix hub import if truncated, start hub, run tutorial retest.
WS="$HOME/dev-ws"
CKPT="$WS/datasets/lerobot/outputs/act_foam_pole_lookreach_office_tutorial/checkpoints/last/pretrained_model"
LOG=/tmp/il_retest_tutorial.out
{
  echo "=== $(date -Is) retest boot ==="
  # Kill any dead hub attempts
  pkill -f '/scripts/agx_isaac_hub.py' 2>/dev/null || true
  pkill -f 'python3 .*agx_isaac_hub' 2>/dev/null || true
  sleep 1
  grep -n 'ThreadingHTTP' "$WS/scripts/agx_isaac_hub.py" | head -5
  nohup python3 "$WS/scripts/agx_isaac_hub.py" >/tmp/agx_isaac_hub.log 2>&1 &
  echo hub_pid=$!
  for i in 1 2 3 4 5 6 7 8 9 10; do
    code=$(curl -s -m 1 -o /dev/null -w '%{http_code}' http://127.0.0.1:8081/isaac/twin || echo 000)
    echo "hub_try=$i code=$code"
    if [[ "$code" == "200" ]]; then break; fi
    sleep 1
  done
  tail -20 /tmp/agx_isaac_hub.log
  export GRAB_IT_IL_CKPT="$CKPT"
  bash "$WS/scripts/_il_retest_tutorial.sh" "$CKPT"
  echo retest_exit=$?
} >>"$LOG" 2>&1
