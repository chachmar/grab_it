#!/usr/bin/env bash
# Retest the hold checkpoint twice: PARK, then a small left/right lean.
# Success is the pole centre rising and the gripper staying closed.
set -eo pipefail
WS="${HOME}/dev-ws"
CKPT="${1:-${GRAB_IT_IL_CKPT:-$WS/datasets/lerobot/outputs/act_foam_pole_lookreach_office_tutorial/checkpoints/last/pretrained_model}}"
LOG=/tmp/il_retest_hold.log

if [[ ! -f "$CKPT/model.safetensors" ]]; then
  echo "ERROR: checkpoint missing: $CKPT"
  exit 1
fi

echo "=== hold retest ckpt=$CKPT ===" | tee "$LOG"

one_trial() {
  local name="$1"
  local skip="$2"
  echo "---- trial $name skip_park=$skip ----" | tee -a "$LOG"
  if [[ "$skip" == "1" ]]; then
    export GRAB_IT_IL_SKIP_PARK=1
  else
    unset GRAB_IT_IL_SKIP_PARK || true
  fi
  export GRAB_IT_IL_CKPT="$CKPT"
  bash "${WS}/scripts/_il_replay_lookreach.sh" "$CKPT" 2>&1 | tee -a "$LOG"
  python3 - "$name" <<'PY' | tee -a "$LOG"
import json, sys, time, urllib.request
name = sys.argv[1]

def twin():
    raw = urllib.request.urlopen("http://127.0.0.1:8081/isaac/twin", timeout=2).read()
    return json.loads(raw.decode())

z0 = None
for _ in range(10):
    try:
        d = twin()
        if d.get("pole_z") is not None:
            z0 = float(d["pole_z"])
            break
    except Exception:
        pass
    time.sleep(0.2)
print(f"{name} pole_z0={z0}")
time.sleep(35)
d = twin()
names = list(d.get("arm_n") or [])
pos = [float(x) for x in (d.get("arm_p") or [])]
by = {n: pos[i] for i, n in enumerate(names) if i < len(pos)}
z1 = d.get("pole_z")
grip = float(by.get("gripper", 9.0))
rose = z0 is not None and z1 is not None and float(z1) - float(z0) >= 0.02
ok_grip = grip <= 0.20
print("arm", {k: round(float(by.get(k, 0)), 3) for k in ("waist", "shoulder", "wrist_angle", "gripper", "pan")})
print(f"pole_z1={z1} rose={rose} grip_ok={ok_grip} close_gate={d.get('close_gate')}")
print("SUCCESS" if rose and ok_grip else "FAIL")
sys.exit(0 if rose and ok_grip else 1)
PY
}

fail=0
one_trial park 0 || fail=1
bash "${WS}/scripts/_il_stop_play.sh" 2>&1 | tee -a "$LOG" || true
sleep 1
LEAN='{"t":"arm","n":["waist","shoulder","wrist_angle","wrist_rotate","gripper","pan"],"p":[-0.35,-1.75,1.25,0.0,1.65,1.10]}'
echo "driving lean start…" | tee -a "$LOG"
for i in $(seq 1 80); do
  curl -s -X POST http://127.0.0.1:8081/isaac/cmd \
    -H "Content-Type: application/json" \
    -d "$LEAN" >/dev/null || true
  sleep 0.1
done
one_trial lean 1 || fail=1
echo "hold_retest_exit=$fail" | tee -a "$LOG"
exit "$fail"
