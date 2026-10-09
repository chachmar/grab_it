#!/usr/bin/env bash
set -uo pipefail
WS="$HOME/dev-ws"
CKPT="$WS/datasets/lerobot/outputs/act_foam_pole_lookreach_office_tutorial/checkpoints/last/pretrained_model"
LOG=/tmp/il_retest_tutorial.out
exec > >(tee -a "$LOG") 2>&1
echo "=== $(date -Is) kick retest ==="
echo "ckpt=$CKPT"
test -f "$CKPT/model.safetensors" || { echo missing ckpt; exit 1; }

# Envelope
RAW="$WS/datasets/tabletop_pick_lookreach_office_tutorial"
if [[ -f "$RAW/joint_envelope.json" ]]; then
  cp -f "$RAW/joint_envelope.json" "$WS/datasets/lerobot/outputs/act_foam_pole_lookreach_office_tutorial/joint_envelope.json"
  cp -f "$RAW/joint_envelope.json" "$CKPT/joint_envelope.json"
  echo envelope ok
fi

# Hub
if ! pgrep -f agx_isaac_hub.py >/dev/null; then
  nohup python3 "$WS/scripts/agx_isaac_hub.py" >/tmp/agx_isaac_hub.log 2>&1 &
  echo hub_started pid=$!
  sleep 3
else
  echo hub_already_running
fi
curl -s -m 2 -o /dev/null -w "hub_http=%{http_code}\n" http://127.0.0.1:8081/isaac/twin || echo hub_curl_fail

export GRAB_IT_IL_CKPT="$CKPT"
# Do not use set -e for the long replay path.
bash "$WS/scripts/_il_retest_tutorial.sh" "$CKPT"
echo "retest_exit=$?"
