#!/usr/bin/env bash
# One grasp from the folded approach. Kills ACT first so it cannot fight.
set -u
pids=$(ps -eo pid,cmd | awk '/play_policy.py/ && !/awk/ {print $1}')
if [[ -n "${pids}" ]]; then
  # shellcheck disable=SC2086
  kill -9 ${pids} 2>/dev/null || true
  sleep 0.4
fi
pids=$(ps -eo pid,cmd | awk '/_il_wrist_finish.py/ && !/awk/ {print $1}')
if [[ -n "${pids}" ]]; then
  # shellcheck disable=SC2086
  kill -9 ${pids} 2>/dev/null || true
  sleep 0.2
fi
export GRAB_IT_FINISH_NOW=1
export GRAB_IT_RESET_APPROACH=1
exec python3 /home/ubuntu/dev-ws/scripts/_il_wrist_finish.py
