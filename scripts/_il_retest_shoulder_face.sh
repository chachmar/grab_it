#!/usr/bin/env bash
# Retest shoulder ACT starting with waist aimed at the live pole (collection start).
set -eo pipefail
WS="${HOME}/dev-ws"
OUT="$WS/datasets/lerobot/outputs/act_foam_pole_lookreach_office_shoulder"
CKPT="${1:-${GRAB_IT_IL_CKPT:-$OUT/checkpoints/last/pretrained_model}}"
RAW="$WS/datasets/tabletop_pick_lookreach_office_shoulder"
LOG=/tmp/il_retest_shoulder_face.log

if [[ ! -f "$CKPT/model.safetensors" ]]; then
  echo "ERROR: checkpoint missing: $CKPT"
  exit 1
fi
if [[ ! -f "$CKPT/joint_envelope.json" && -f "$RAW/joint_envelope.json" ]]; then
  cp -f "$RAW/joint_envelope.json" "$CKPT/joint_envelope.json"
  cp -f "$RAW/joint_envelope.json" "$OUT/joint_envelope.json"
  echo "copied joint_envelope.json"
fi

# Stop any prior ACT so park cmds are not overwritten.
# Use waist 0 (shoulder-park), not the old lookreach -0.55.
export GRAB_IT_IL_PARK_WAIST=0
bash "${WS}/scripts/_il_stop_play.sh" 2>/dev/null || true
sleep 1

AIM=$(python3 - <<'PY'
import json, math, time, urllib.request

def twin():
    raw = urllib.request.urlopen("http://127.0.0.1:8081/isaac/twin", timeout=2).read()
    return json.loads(raw.decode())

d = None
for _ in range(30):
    try:
        d = twin()
        break
    except Exception:
        time.sleep(0.3)
if d is None:
    raise SystemExit("no twin")
xy = d.get("pole_xy") or [0.28, 0.0]
wx, wy = float(xy[0]), float(xy[1])
waist = 0.0
if abs(wx) > 1e-3:
    waist = max(-0.15, min(0.15, math.atan2(wy, wx)))
pan = 1.10
print(f"pole_xy=({wx:.4f},{wy:.4f}) waist_aim={waist:.4f} pan={pan:.2f}", flush=True)
body = json.dumps({
    "t": "arm",
    "n": ["waist", "shoulder", "wrist_angle", "wrist_rotate", "gripper", "pan"],
    "p": [waist, -1.9, 1.4, 0.0, 1.65, pan],
}).encode()
req = urllib.request.Request(
    "http://127.0.0.1:8081/isaac/cmd",
    data=body,
    headers={"Content-Type": "application/json"},
    method="POST",
)
for _ in range(400):
    try:
        urllib.request.urlopen(req, timeout=1).read()
    except Exception:
        pass
    time.sleep(0.1)
# Confirm parked near aim (soft rate-limit needs many cmds)
ok = False
for _ in range(80):
    try:
        urllib.request.urlopen(req, timeout=1).read()
        d = twin()
        names = list(d.get("arm_n") or [])
        pos = [float(x) for x in (d.get("arm_p") or [])]
        by = {n: pos[i] for i, n in enumerate(names) if i < len(pos)}
        got = float(by.get("waist", 9.0))
        sh = float(by.get("shoulder", 9.0))
        pan_got = float(by.get("pan", 9.0))
        if abs(got - waist) <= 0.12 and abs(sh - (-1.9)) <= 0.25:
            print(
                f"aimed_park ok waist={got:.3f} shoulder={sh:.3f} pan={pan_got:.3f}",
                flush=True,
            )
            ok = True
            break
        if _ % 20 == 0:
            print(
                f"park wait waist={got:.3f}/{waist:.3f} "
                f"shoulder={sh:.3f}/-1.90 pan={pan_got:.3f}",
                flush=True,
            )
    except Exception:
        pass
    time.sleep(0.15)
if not ok:
    raise SystemExit("ERROR: aimed park not reached — aborting retest")
print(f"{waist:.6f}")
PY
)

# Last line is the waist float
WAIST=$(echo "$AIM" | tail -1)
echo "$AIM" | head -n -1 | tee "$LOG"
export GRAB_IT_IL_PARK_WAIST="$WAIST"
export GRAB_IT_IL_CKPT="$CKPT"
export GRAB_IT_IL_SKIP_PARK=1
echo "=== shoulder face-pole retest waist=$WAIST ckpt=$CKPT ===" | tee -a "$LOG"

# Skip the script's hardcoded park — we already aimed. Run play + score.
bash "${WS}/scripts/_il_replay_lookreach.sh" "$CKPT" 2>&1 | tee -a "$LOG"
python3 - <<'PY' | tee -a "$LOG"
import json, sys, time, urllib.request

def twin():
    raw = urllib.request.urlopen("http://127.0.0.1:8081/isaac/twin", timeout=2).read()
    return json.loads(raw.decode())

def arm(d):
    names = list(d.get("arm_n") or [])
    pos = [float(x) for x in (d.get("arm_p") or [])]
    return {n: pos[i] for i, n in enumerate(names) if i < len(pos)}

z0 = None
w0 = None
for _ in range(10):
    try:
        d = twin()
        if d.get("pole_z") is not None:
            z0 = float(d["pole_z"])
            w0 = float(arm(d).get("waist", 0.0))
            break
    except Exception:
        pass
    time.sleep(0.2)
print(f"pole_z0={z0} waist0={w0}")
time.sleep(35)
d = twin()
by = arm(d)
z1 = d.get("pole_z")
w1 = float(by.get("waist", 9.0))
grip = float(by.get("gripper", 9.0))
rose = z0 is not None and z1 is not None and float(z1) - float(z0) >= 0.02
ok_grip = grip <= 0.20
yaw = w0 is not None and abs(w1 - float(w0)) <= 0.20
print("arm", {k: round(float(by.get(k, 0)), 3) for k in ("waist", "shoulder", "wrist_angle", "gripper", "pan")})
print(
    f"pole_z1={z1} rose={rose} grip_ok={ok_grip} waist_held={yaw} "
    f"dw={None if w0 is None else round(w1 - float(w0), 3)}"
)
print("SUCCESS" if rose and ok_grip and yaw else "FAIL")
sys.exit(0 if rose and ok_grip and yaw else 1)
PY
