#!/usr/bin/env bash
# IL PLAY: aim the waist, put the shaft on the jaw line, then ACT closes.
set -eo pipefail
WS="${HOME}/dev-ws"
CKPT="${1:-${GRAB_IT_IL_CKPT:-$WS/datasets/lerobot/outputs/act_foam_pole_lookreach_office_shoulder/checkpoints/last/pretrained_model}}"

pids=$(pgrep -f '/play_policy.py' || true)
if [[ -n "$pids" ]]; then
  # shellcheck disable=SC2086
  kill -9 $pids 2>/dev/null || true
  sleep 1
fi
pkill -f '/_il_wrist_finish.py' 2>/dev/null || true
# Already on the pole: do not fold and re-aim. That swing is what play was
# repeating before the grab. A fresh aim still runs when the waist is off.
already=$(python3 - <<'PY'
import json, math, urllib.request
try:
    raw = urllib.request.urlopen("http://127.0.0.1:8081/isaac/twin", timeout=2).read()
    d = json.loads(raw)
except Exception:
    print("no")
    raise SystemExit(0)
xy = d.get("pole_xy") or [None, None]
names = list(d.get("arm_n") or [])
pos = [float(x) for x in (d.get("arm_p") or [])]
arm = {n: pos[i] for i, n in enumerate(names) if i < len(pos)}
try:
    wx, wy = float(xy[0]), float(xy[1])
    waist = float(arm["waist"])
except (TypeError, ValueError, KeyError, IndexError):
    print("no")
    raise SystemExit(0)
if wx < 0.08:
    print("no")
    raise SystemExit(0)
bearing = max(-1.80, min(1.80, math.atan2(wy, wx)))
print("yes" if abs(waist - bearing) <= 0.08 else "no")
PY
)
if [[ "$already" == "yes" ]]; then
  echo "waist already on the pole — not re-centering"
else
  echo "centering waist on pole…"
  python3 "${WS}/scripts/_il_waist_center.py"
fi
echo "moving to the grasp point…"
set +e
GRAB_IT_FINISH_NOW=1 GRAB_IT_TO_POINT=1 python3 "${WS}/scripts/_il_wrist_finish.py"
point_rc=$?
set -e
if [[ "$point_rc" -ne 0 ]]; then
  echo "not at the grasp point — ACT not started"
  exit "$point_rc"
fi
# Hold the waist where the point servo left it. ACT's waist channel
# otherwise yaws back into the ±0.16 rad training box. The gripper stays
# open so ACT's close is the grasp.
export GRAB_IT_IL_WAIST_LOCK=hold
unset GRAB_IT_IL_WRIST_ROT_LOCK || true
echo "waist lock=hold"
export GRAB_IT_IL_SKIP_PARK=1
export GRAB_IT_IL_CKPT="$CKPT"
bash "${WS}/scripts/_il_replay_lookreach.sh" "$CKPT"
nohup python3 "${WS}/scripts/_il_wrist_finish.py" >>/tmp/il_wrist_servo.log 2>&1 &
echo "ACT at the grasp point, wrist finish pid=$!"
