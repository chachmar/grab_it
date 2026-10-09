#!/usr/bin/env bash
# Start host ACT player in background; log to /tmp/il_play_act.log
set -eo pipefail
WS="${HOME}/dev-ws"
pkill -f "grab_it_il_play|play_policy.py|play_policy.launch" 2>/dev/null || true
sleep 1
nohup bash "${WS}/scripts/il_play_host.sh" \
  "${WS}/datasets/lerobot/outputs/act_tabletop_pick/checkpoints/last/pretrained_model" \
  /tmp/il_play_act.log \
  >/tmp/il_play_act.nohup 2>&1 &
echo "started pid=$!"
sleep 8
tail -40 /tmp/il_play_act.log 2>/dev/null || tail -40 /tmp/il_play_act.nohup
pgrep -af "play_policy|grab_it_il_play" | head -5 || true
