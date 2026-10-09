#!/usr/bin/env bash
# Convert shoulder-reach demos only and chunk-train ACT from scratch to 80k.
set -euo pipefail
source "$HOME/lerobot-venv/bin/activate"
WS="${ISAAC_ROS_WS:-$HOME/dev-ws}"
RAW="${GRAB_IT_IL_RAW:-$WS/datasets/tabletop_pick_lookreach_office_shoulder}"
DS="${GRAB_IT_IL_LEROBOT:-$WS/datasets/lerobot/grabit_lookreach_office_shoulder}"
OUT="${GRAB_IT_IL_OUT:-$WS/datasets/lerobot/outputs/act_foam_pole_lookreach_office_shoulder}"
STEPS="${GRAB_IT_IL_STEPS:-80000}"
CHUNK="${GRAB_IT_IL_CHUNK:-5000}"
BATCH="${GRAB_IT_IL_BATCH:-8}"
TASK="grab the upright foam pole and hold it"
REPO_ID="local/grabit_lookreach_office_shoulder"

n=$(find "$RAW" -maxdepth 1 -type d -name 'episode_*' | wc -l)
echo "shoulder raw episodes: $n"
if [[ "$n" -lt 50 ]]; then
  echo "ERROR: need >=50 shoulder episodes; have $n"
  exit 1
fi

echo "Rebuilding LeRobot dataset at $DS"
rm -rf "$DS"
python3 "$WS/scripts/il_to_lerobot.py" \
  --input "$RAW" \
  --output "$DS" \
  --task "$TASK" \
  --repo-id "$REPO_ID"

python3 - "$RAW" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
raw = Path(sys.argv[1])
names = ["waist", "shoulder", "wrist_angle", "wrist_rotate", "gripper", "pan"]
stacks = []
for ep in sorted(raw.glob("episode_*")):
    a = ep / "actions.npy"
    if a.is_file():
        stacks.append(np.load(a))
if not stacks:
    raise SystemExit("no actions.npy")
arr = np.concatenate(stacks, axis=0).astype(np.float32)
margin = 0.05
env = {}
for i, name in enumerate(names):
    col = arr[:, i]
    lo, hi = float(col.min()) - margin, float(col.max()) + margin
    if lo > hi:
        lo, hi = hi, lo
    env[name] = [lo, hi]
payload = {"margin": margin, "limits": env}
(raw / "joint_envelope.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print("wrote", raw / "joint_envelope.json")
PY

if [[ -d "$OUT/checkpoints" ]]; then
  echo "ERROR: shoulder train is from scratch; remove $OUT first"
  exit 1
fi

export GRAB_IT_IL_OUT="$OUT"
export GRAB_IT_IL_LEROBOT="$DS"
export GRAB_IT_IL_STEPS="$STEPS"
export GRAB_IT_IL_CHUNK="$CHUNK"
export GRAB_IT_IL_BATCH="$BATCH"
export GRAB_IT_IL_REPO="$REPO_ID"
export GRAB_IT_IL_LOG="${GRAB_IT_IL_LOG:-/tmp/il_train_shoulder.log}"
export GRAB_IT_IL_COOLDOWN_S="${GRAB_IT_IL_COOLDOWN_S:-300}"
# lerobot-train refuses a pre-existing output_dir on the first chunk.
bash "$WS/scripts/il_train_chunked.sh"

if [[ -f "$RAW/joint_envelope.json" ]]; then
  cp -f "$RAW/joint_envelope.json" "$OUT/joint_envelope.json"
  mkdir -p "$OUT/checkpoints/last/pretrained_model"
  cp -f "$RAW/joint_envelope.json" "$OUT/checkpoints/last/pretrained_model/joint_envelope.json"
fi
echo "Done. Checkpoint: $OUT/checkpoints/last/pretrained_model"
