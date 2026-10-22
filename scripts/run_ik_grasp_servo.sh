#!/usr/bin/env bash
# Closed-loop IK servo on the AGX (grab_it_live). Logs stall/lift into
# datasets/rl_goal/servo_trials/trials.jsonl — the baseline RL must beat.
#
# Static:   bash scripts/run_ik_grasp_servo.sh
# Nudged:   bash scripts/run_ik_grasp_servo.sh --note nudge
# With IL:  bash scripts/run_ik_grasp_servo.sh --ranker datasets/rl_goal/servo_trials/ranker.json
set -euo pipefail
WS="${WS:-$HOME/dev-ws}"
cd "$WS"
exec python3 scripts/ik_grasp_servo.py \
  --goal "${WS}/config/rl_goal.yaml" \
  --hub "http://127.0.0.1:8082" \
  "$@"
