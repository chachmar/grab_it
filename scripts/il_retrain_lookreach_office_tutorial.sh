#!/usr/bin/env bash
# Train ACT from scratch on tutorial success-only office demos (80k, batch 8).
# No warm-start. Converts pole_xy → observation.environment_state.
set -euo pipefail
source "$HOME/lerobot-venv/bin/activate"
WS="${ISAAC_ROS_WS:-$HOME/dev-ws}"
RAW="${GRAB_IT_IL_RAW:-$WS/datasets/tabletop_pick_lookreach_office_tutorial}"
DS="${GRAB_IT_IL_LEROBOT:-$WS/datasets/lerobot/grabit_lookreach_office_tutorial}"
OUT="${GRAB_IT_IL_OUT:-$WS/datasets/lerobot/outputs/act_foam_pole_lookreach_office_tutorial}"
STEPS="${GRAB_IT_IL_STEPS:-80000}"
BATCH="${GRAB_IT_IL_BATCH:-8}"
SAVE_FREQ="${GRAB_IT_IL_SAVE_FREQ:-10000}"
TASK="grab the upright foam pole and hold it"
REPO_ID="local/grabit_lookreach_office_tutorial"

echo "python=$(which python3)"
python3 -c "import lerobot; print('lerobot ok')"

n=$(find "$RAW" -maxdepth 1 -type d -name 'episode_*' 2>/dev/null | wc -l)
echo "tutorial raw episodes under $RAW: $n"
if [[ "$n" -lt 50 ]]; then
  echo "ERROR: need >=50 tutorial episodes; have $n"
  exit 1
fi

# Keep only success episodes that recorded pole_xy (env state).
python3 - "$RAW" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
kept = dropped = 0
for ep in sorted(root.glob("episode_*")):
    if not ep.is_dir():
        continue
    meta = {}
    meta_p = ep / "meta.json"
    if meta_p.is_file():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
    ok = meta.get("success", True) is not False
    has_xy = (ep / "pole_xy.npy").is_file()
    if ok and has_xy:
        kept += 1
        continue
    dest = root / "_dropped" / ep.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    ep.rename(dest)
    dropped += 1
print(f"kept={kept} dropped={dropped}")
PY

n=$(find "$RAW" -maxdepth 1 -type d -name 'episode_*' 2>/dev/null | wc -l)
echo "episodes after pole_xy filter: $n"
if [[ "$n" -lt 50 ]]; then
  echo "ERROR: need >=50 episodes with pole_xy; have $n"
  exit 1
fi

echo "Rebuilding LeRobot dataset at $DS (with environment_state) ..."
rm -rf "$DS"
python3 "$WS/scripts/il_to_lerobot.py" \
  --input "$RAW" \
  --output "$DS" \
  --task "$TASK" \
  --repo-id "$REPO_ID"

# Dataset joint envelope for play clamp (min/max + margin).
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
    col = arr[:, i] if arr.shape[1] > i else np.zeros(len(arr), dtype=np.float32)
    lo = float(col.min()) - margin
    hi = float(col.max()) + margin
    if lo > hi:
        lo, hi = hi, lo
    env[name] = [lo, hi]
payload = {"margin": margin, "limits": env}
(raw / "joint_envelope.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print("wrote", raw / "joint_envelope.json")
print(json.dumps(env, indent=2))
PY

if [[ -d "$OUT" ]]; then
  bak="${OUT}_prev_$(date +%Y%m%d_%H%M%S)"
  echo "Moving old output → $bak"
  mv "$OUT" "$bak"
fi

# Refuse accidental warm-start.
if [[ -n "${GRAB_IT_IL_PRETRAINED:-}" ]]; then
  echo "ERROR: tutorial train is from-scratch; unset GRAB_IT_IL_PRETRAINED"
  exit 1
fi

echo "Training ACT from scratch steps=$STEPS batch=$BATCH save_freq=$SAVE_FREQ"
# Do not mkdir $OUT first — lerobot-train refuses a pre-existing output_dir.
lerobot-train \
  --policy.type=act \
  --policy.push_to_hub=false \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="$DS" \
  --dataset.video_backend=pyav \
  --output_dir="$OUT" \
  --policy.device=cuda \
  --batch_size="$BATCH" \
  --steps="$STEPS" \
  --save_freq="$SAVE_FREQ"

# Copy envelope next to output + last checkpoint for play.
if [[ -f "$RAW/joint_envelope.json" ]]; then
  mkdir -p "$OUT" "$OUT/checkpoints/last/pretrained_model"
  cp -f "$RAW/joint_envelope.json" "$OUT/joint_envelope.json"
  cp -f "$RAW/joint_envelope.json" "$OUT/checkpoints/last/pretrained_model/joint_envelope.json"
fi

echo "Done. Play checkpoint:"
echo "  $OUT/checkpoints/last/pretrained_model"
