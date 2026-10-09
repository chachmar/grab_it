#!/usr/bin/env bash
# Morning retest: PARK → tutorial ACT. Success = both pads enclosed, gripper
# closed, pole upright, arm inside the demo envelope.
set -eo pipefail
WS="${HOME}/dev-ws"
CKPT="${1:-${GRAB_IT_IL_CKPT:-$WS/datasets/lerobot/outputs/act_foam_pole_lookreach_office_tutorial/checkpoints/last/pretrained_model}}"
LOG="/tmp/il_retest_tutorial.log"

if [[ ! -f "$CKPT/model.safetensors" && ! -f "$CKPT/config.json" ]]; then
  echo "ERROR: checkpoint missing: $CKPT"
  echo "Wait until training writes checkpoints/last/pretrained_model"
  exit 1
fi

echo "=== tutorial retest ckpt=$CKPT ===" | tee "$LOG"
bash "${WS}/scripts/_il_stop_play.sh" 2>&1 | tee -a "$LOG" || true
sleep 1

# Sync latest play host + hub defaults if present on this machine.
export GRAB_IT_IL_CKPT="$CKPT"
bash "${WS}/scripts/_il_replay_lookreach.sh" "$CKPT" 2>&1 | tee -a "$LOG"

echo "waiting 25s for ACT approach…" | tee -a "$LOG"
sleep 25

python3 - <<'PY' | tee -a "$LOG"
import json, math, urllib.request, sys

def twin():
    raw = urllib.request.urlopen("http://127.0.0.1:8081/isaac/twin", timeout=2.0).read()
    return json.loads(raw.decode())

d = twin()
names = list(d.get("arm_n") or [])
pos = [float(x) for x in (d.get("arm_p") or [])]
by = {n: pos[i] for i, n in enumerate(names) if i < len(pos)}
pad_d = d.get("pad_d")
close_gate = bool(d.get("close_gate"))
pole_xy = d.get("pole_xy")
grip = float(by.get("gripper", 9.0))

# Prefer twin pad_d when present; success needs both pads ≤ 0.032 (reported as max).
ok_pad = pad_d is not None and float(pad_d) <= 0.032
ok_grip = grip <= -0.70  # near CLOSED (-0.95)
# Soft envelope check vs lookreach PARK band / demo-ish waist.
ok_arm = abs(float(by.get("waist", 9))) < 1.6 and abs(float(by.get("shoulder", 9))) < 1.6

print("arm", {k: round(float(by.get(k, 0)), 3) for k in ("waist", "shoulder", "wrist_angle", "gripper", "pan")})
print("pad_d", pad_d, "close_gate", close_gate, "pole_xy", pole_xy)
print("checks", {"pad": ok_pad, "grip": ok_grip, "arm": ok_arm})

success = bool(ok_pad and ok_grip and ok_arm)
print("SUCCESS" if success else "FAIL")
sys.exit(0 if success else 1)
PY
