#!/usr/bin/env bash
# Usage: _il_real_alive.sh grasp|act|status
# Exit 0 when the requested process is running.
set +e
mode="${1:-status}"
grasp=1
act=1
if pgrep -f 'src/grab_it_grasp/.*/grasp_from_click.py' >/dev/null \
  || pgrep -f 'lib/grab_it_grasp/grasp_from_click' >/dev/null \
  || pgrep -f '/grasp_from_click' >/dev/null; then
  grasp=0
fi
if pgrep -f 'grab_it_il/play_policy.py' >/dev/null; then
  act=0
fi
case "$mode" in
  grasp) exit "$grasp" ;;
  act) exit "$act" ;;
  status)
    echo "grasp=$([[ $grasp -eq 0 ]] && echo up || echo down) act=$([[ $act -eq 0 ]] && echo up || echo down)"
    exit 0
    ;;
esac
exit 1
