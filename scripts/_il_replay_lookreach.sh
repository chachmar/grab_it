#!/usr/bin/env bash
# Stop ACT, wait until twin measured joints match lookreach PARK, then ACT play.
set -eo pipefail
WS="${HOME}/dev-ws"
CKPT="${1:-${GRAB_IT_IL_CKPT:-$WS/datasets/lerobot/outputs/act_foam_pole_lookreach_office_tutorial/checkpoints/last/pretrained_model}}"
# Default is lookreach PARK (waist -0.55). Shoulder demos use GRAB_IT_IL_PARK_WAIST=0.
PARK_WAIST="${GRAB_IT_IL_PARK_WAIST:--0.55}"
export GRAB_IT_IL_PARK_WAIST="$PARK_WAIST"
PARK_JSON="{\"t\":\"arm\",\"n\":[\"waist\",\"shoulder\",\"wrist_angle\",\"wrist_rotate\",\"gripper\",\"pan\"],\"p\":[${PARK_WAIST},-1.9,1.4,0.0,1.65,1.10]}"

# GRAB_IT_IL_SKIP_PARK=1 starts ACT from the current arm pose (folded-pose check).
SKIP_PARK="${GRAB_IT_IL_SKIP_PARK:-0}"
if [[ "$SKIP_PARK" == "1" || "$SKIP_PARK" == "true" ]]; then
  pids=$(pgrep -f '/play_policy.py' || true)
  if [[ -n "$pids" ]]; then
    kill -9 $pids 2>/dev/null || true
  fi
  sleep 1
  echo "play stopped; keeping current arm pose"
else
  bash "${WS}/scripts/_il_stop_play.sh"
fi

if ! pgrep -f agx_isaac_hub.py >/dev/null; then
  nohup python3 "${WS}/scripts/agx_isaac_hub.py" >/tmp/agx_isaac_hub.log 2>&1 &
  sleep 2
fi
bash "${WS}/scripts/_il_start_hub_obs_host.sh"
bash "${WS}/scripts/_il_start_twin_relay_host.sh"

park_ok() {
  python3 - <<'PY'
import json, urllib.request, sys
try:
    raw = urllib.request.urlopen("http://127.0.0.1:8081/isaac/twin", timeout=1.0).read()
    d = json.loads(raw.decode())
except Exception as exc:
    print(f"twin fetch fail: {exc}")
    sys.exit(1)
names = list(d.get("arm_n") or [])
pos = [float(x) for x in (d.get("arm_p") or [])]
if not names or len(pos) < 5:
    print(f"twin missing arm_p names={names} npos={len(pos)}")
    sys.exit(1)
by = {n: pos[i] for i, n in enumerate(names) if i < len(pos)}
import os
waist = float(os.environ.get("GRAB_IT_IL_PARK_WAIST", "-0.55"))
want = {"waist": waist, "shoulder": -1.9, "wrist_angle": 1.4}
tol = 0.08
bad = []
for k, v in want.items():
    got = float(by.get(k, 1e9))
    if abs(got - v) > tol:
        bad.append(f"{k}={got:.3f} want={v}")
# Pan: require look-down demo band. Exact 1.10 sometimes stalls in PhysX
# (~0.78) while still showing the pole; do not block restart on that.
pan = float(by.get("pan", 1e9))
if not (0.70 <= pan <= 1.35):
    bad.append(f"pan={pan:.3f} want look-down [0.70,1.35]")
if bad:
    print("not parked:", ", ".join(bad))
    sys.exit(1)
print("parked", {**{k: round(float(by.get(k, 0)), 3) for k in want}, "pan": round(pan, 3)})
sys.exit(0)
PY
}

if [[ "$SKIP_PARK" == "1" || "$SKIP_PARK" == "true" ]]; then
  echo "skip PARK — ACT starts from the current arm pose"
else
  echo "holding PARK until twin arm_p matches…"
  ok=0
  # Soft rate-limit (~0.08 rad/cmd) needs frequent seq bumps to reach PARK.
  for i in $(seq 1 200); do
    curl -s -X POST http://127.0.0.1:8081/isaac/cmd \
      -H "Content-Type: application/json" \
      -d "$PARK_JSON" >/dev/null || true
    if park_ok; then
      ok=1
      break
    fi
    sleep 0.1
  done
  if [[ "$ok" -ne 1 ]]; then
    echo "WARN: twin did not report PARK within timeout — starting ACT anyway"
    curl -s http://127.0.0.1:8081/isaac/twin | head -c 400; echo
  fi
fi

# Spot-check D435i before PLAY.
python3 - <<'PY'
import urllib.request
try:
    blob = urllib.request.urlopen("http://127.0.0.1:8081/d435i.jpg", timeout=1.0).read()
    print(f"d435i.jpg bytes={len(blob)}")
except Exception as exc:
    print(f"d435i.jpg FAIL: {exc}")
PY

nohup bash "${WS}/scripts/il_play_host.sh" "$CKPT" /tmp/il_play_lookreach.log \
  >/tmp/il_play_lookreach.nohup 2>&1 &
echo "ACT play started pid=$! ckpt=$CKPT"
sleep 12
grep -E 'autoplay|policy PLAY|temporal_ensemble|Error|Traceback' /tmp/il_play_lookreach.log | tail -25 || true
tail -8 /tmp/il_twin_relay.log || true
pgrep -af '/play_policy.py' | head -3 || echo no_play
