#!/usr/bin/env bash
echo "=== retest log tail ==="
tail -80 /tmp/il_retest_tutorial.out || true
echo "=== play procs ==="
pgrep -af play_policy.py | head -5 || echo no_play
echo "=== twin ==="
python3 - <<'PY'
import json, urllib.request
try:
    d = json.loads(urllib.request.urlopen("http://127.0.0.1:8081/isaac/twin", timeout=2).read())
except Exception as e:
    print("twin fail", e)
    raise SystemExit(1)
by = dict(zip(d.get("arm_n") or [], d.get("arm_p") or []))
print({k: round(float(by.get(k, 0)), 3) for k in ("waist", "shoulder", "wrist_angle", "gripper", "pan")})
print("pad_d", d.get("pad_d"), "close_gate", d.get("close_gate"), "pole_xy", d.get("pole_xy"))
PY
echo "=== play log ==="
grep -E 'autoplay|policy PLAY|temporal_ensemble|Error|Traceback|env_state|envelope' /tmp/il_play_lookreach.log 2>/dev/null | tail -25 || true
