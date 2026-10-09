#!/usr/bin/env bash
set -eo pipefail
pids=$(pgrep -f '/play_policy.py' || true)
if [[ -n "$pids" ]]; then
  # shellcheck disable=SC2086
  kill -9 $pids 2>/dev/null || true
fi
sleep 1
# Default lookreach PARK. Shoulder play sets GRAB_IT_IL_PARK_WAIST=0.
PARK_WAIST="${GRAB_IT_IL_PARK_WAIST:--0.55}"
curl -s -X POST http://127.0.0.1:8081/isaac/cmd \
  -H "Content-Type: application/json" \
  -d "{\"t\":\"arm\",\"n\":[\"waist\",\"shoulder\",\"wrist_angle\",\"wrist_rotate\",\"gripper\",\"pan\"],\"p\":[${PARK_WAIST},-1.9,1.4,0.0,1.65,1.10]}" \
  >/dev/null || true
echo "play stopped; hub PARK waist=${PARK_WAIST}"
curl -s http://127.0.0.1:8081/isaac/cmd | head -c 240
echo
pgrep -af '/play_policy.py' || echo no_play
