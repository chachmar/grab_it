#!/usr/bin/env bash
# Play a timing (joint-delta) checkpoint on the real PX150.
# Gate: only after ik_grasp_servo static+nudged held rates are recorded, and
# only deploy if this policy's stall-and-lift rate matches the servo baseline.
set -euo pipefail
WS="${WS:-$HOME/dev-ws}"
CKPT="${1:-$WS/datasets/rl_goal/runs/timing_bc/model.pt}"
cd "$WS"
exec python3 scripts/play_goal_rl_on_real.py \
  --timing \
  --checkpoint "$CKPT" \
  --goal "${WS}/config/rl_goal.yaml" \
  --hub "http://127.0.0.1:8082"
