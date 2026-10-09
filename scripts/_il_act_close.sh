#!/usr/bin/env bash
# Final close: ACT starts from the pose the click already reached.
set -eo pipefail
WS="${HOME}/dev-ws"
CKPT="${GRAB_IT_IL_CKPT:-$WS/datasets/lerobot/outputs/act_foam_pole_lookreach_office_shoulder/checkpoints/last/pretrained_model}"

pids=$(pgrep -f '/play_policy.py' || true)
if [[ -n "$pids" ]]; then
  # shellcheck disable=SC2086
  kill -9 $pids 2>/dev/null || true
  sleep 0.3
fi
pkill -f '/_il_wrist_finish.py' 2>/dev/null || true
# Hold the waist the approach already aimed. Do not sample the twin: that
# reading can still be the folded park pose and ACT then yaws off the pole.
export GRAB_IT_IL_WAIST_LOCK=hold
unset GRAB_IT_IL_WRIST_ROT_LOCK || true
echo "ACT close waist lock=hold"
export GRAB_IT_IL_SKIP_PARK=1
export GRAB_IT_IL_CKPT="$CKPT"
bash "${WS}/scripts/_il_replay_lookreach.sh" "$CKPT"
nohup python3 "${WS}/scripts/_il_wrist_finish.py" >>/tmp/il_wrist_servo.log 2>&1 &
echo "ACT close pid=$!"
