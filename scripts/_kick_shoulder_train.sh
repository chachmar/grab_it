#!/usr/bin/env bash
set -euo pipefail
export GRAB_IT_IL_OUT=/home/ubuntu/dev-ws/datasets/lerobot/outputs/act_foam_pole_lookreach_office_shoulder
export GRAB_IT_IL_LEROBOT=/home/ubuntu/dev-ws/datasets/lerobot/grabit_lookreach_office_shoulder
export GRAB_IT_IL_STEPS=80000
export GRAB_IT_IL_CHUNK=5000
export GRAB_IT_IL_BATCH=8
export GRAB_IT_IL_REPO=local/grabit_lookreach_office_shoulder
export GRAB_IT_IL_LOG="$GRAB_IT_IL_OUT/train_shoulder.log"
export GRAB_IT_IL_COOLDOWN_S=300
echo 300 > "$GRAB_IT_IL_OUT/cooldown_s.txt"
nohup bash /home/ubuntu/dev-ws/scripts/il_train_chunked.sh >> "$GRAB_IT_IL_LOG" 2>&1 &
echo "STARTED pid=$!"
sleep 4
tr '\r' '\n' < "$GRAB_IT_IL_LOG" | tail -12
ps -eo args | grep '[l]erobot-train' | head -1 || echo no_train
