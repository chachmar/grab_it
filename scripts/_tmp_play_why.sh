#!/usr/bin/env bash
set +e
echo "=== procs ==="
ps -eo pid,cmd | grep -E 'gui_node.py|mjpeg_relay.py|play_policy.py|grasp_from_click|move_group' | grep -v grep | head -20
echo "=== ports ==="
ss -ltn | grep -E ':8765|:8766|:8081' || true
echo "=== gui log ==="
grep -E 'IL cmd|PLAY|play|error|Error|Traceback' /tmp/gui_node.log 2>/dev/null | tail -30
echo "=== act log ==="
tail -40 /tmp/il_real_play_host.log 2>/dev/null || echo no_act_log
echo "=== ensure ==="
tail -20 /tmp/grabit_arm_bringup.log 2>/dev/null || true
