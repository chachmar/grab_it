#!/usr/bin/env bash
# Retest shoulder-reach ACT from waist 0 + shoulder PARK.
# Success: pole lifts, gripper stays closed, waist does not yaw into the shaft.
set -eo pipefail
WS="${HOME}/dev-ws"
CKPT="${1:-${GRAB_IT_IL_CKPT:-$WS/datasets/lerobot/outputs/act_foam_pole_lookreach_office_shoulder/checkpoints/last/pretrained_model}}"
LOG=/tmp/il_retest_shoulder.log
export GRAB_IT_IL_PARK_WAIST=0
export GRAB_IT_IL_CKPT="$CKPT"

if [[ ! -f "$CKPT/model.safetensors" ]]; then
  echo "ERROR: checkpoint missing: $CKPT"
  exit 1
fi

echo "=== shoulder retest ckpt=$CKPT park_waist=0 ===" | tee "$LOG"
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
print(f"pole_z1={z1} rose={rose} grip_ok={ok_grip} waist_held={yaw} dw={None if w0 is None else round(w1 - float(w0), 3)}")
print("SUCCESS" if rose and ok_grip and yaw else "FAIL")
sys.exit(0 if rose and ok_grip and yaw else 1)
PY
