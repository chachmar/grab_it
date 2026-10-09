#!/usr/bin/env bash
# il_smoke.sh — short ACT train to verify the pipeline (AGX HOST, not isaac_ros_dev).
#
# Usage (AGX host, after 10–15 saved episodes):
#   bash ~/dev-ws/scripts/il_smoke.sh
set -euo pipefail

WS="${ISAAC_ROS_WS:-$HOME/dev-ws}"
# Container bind-mount: /workspaces/isaac_ros-dev == ~/dev-ws
if [[ -d /workspaces/isaac_ros-dev/datasets/tabletop_pick && ! -d "$WS/datasets/tabletop_pick" ]]; then
  WS=/workspaces/isaac_ros-dev
fi
RAW="${GRAB_IT_IL_RAW:-$WS/datasets/tabletop_pick}"
DS="${GRAB_IT_IL_LEROBOT:-$WS/datasets/lerobot/grabit_tabletop_pick}"
OUT="${GRAB_IT_IL_OUT:-$WS/datasets/lerobot/outputs/act_tabletop_pick_smoke}"
STEPS="${GRAB_IT_IL_STEPS:-1500}"
BATCH="${GRAB_IT_IL_BATCH:-4}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONVERT="${SCRIPT_DIR}/il_to_lerobot.py"

if [[ -d /workspaces/isaac_ros-dev && -f /opt/ros/humble/setup.bash ]]; then
  echo "WARNING: this looks like isaac_ros_dev. Ctrl-C and run on the AGX host:"
  echo "  bash ~/dev-ws/scripts/il_smoke.sh"
  echo "Continuing in 5s anyway (convert is OK here; CUDA train may not be)..."
  sleep 5
fi

n=0
if [[ -d "$RAW" ]]; then
  n=$(find "$RAW" -maxdepth 1 -type d -name 'episode_*' | wc -l)
fi
echo "[smoke] episodes in $RAW: $n"
if [[ "$n" -lt 5 ]]; then
  echo "[smoke] need several SAVE'd episode_* folders (you said 15 — check the path)."
  ls -la "$RAW" 2>/dev/null || true
  exit 1
fi
vid=$(find "$RAW" -maxdepth 2 \( -name 'd435i.mp4' -o -name 'wrist.mp4' -o -name '000000.jpg' \) | head -5)
if [[ -z "$vid" ]]; then
  echo "[smoke] WARNING: no d435i/wrist mp4 or jpeg frames found."
  echo "         ACT smoke will be joints-only and will not learn alignment from the jaw cam."
fi

ver=""
if [[ -f "$DS/meta/info.json" ]]; then
  ver=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('codebase_version',''))" "$DS/meta/info.json")
fi
if [[ "$ver" != "v3.0" ]]; then
  echo "[smoke] converting (need LeRobot v3.0, have '${ver:-none}') → $DS"
  python3 "$CONVERT" --input "$RAW" --output "$DS"
else
  echo "[smoke] dataset already v3.0 at $DS"
fi

if ! command -v lerobot-train >/dev/null 2>&1; then
  echo "[smoke] lerobot-train not on PATH."
  echo "  python3 -m venv ~/lerobot-venv && source ~/lerobot-venv/bin/activate"
  echo "  pip install 'lerobot' pandas pyarrow"
  echo "Then re-run: bash $SCRIPT_DIR/il_smoke.sh"
  exit 1
fi

if [[ -d "$OUT" ]]; then
  echo "[smoke] removing previous $OUT (lerobot will not overwrite)"
  rm -rf "$OUT"
fi

echo "[smoke] training ACT steps=$STEPS batch=$BATCH cuda → $OUT"
lerobot-train \
  --policy.type=act \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/grabit_tabletop_pick \
  --dataset.root="$DS" \
  --output_dir="$OUT" \
  --policy.device=cuda \
  --batch_size="$BATCH" \
  --steps="$STEPS"

CKPT="$OUT/checkpoints/last/pretrained_model"
echo "[smoke] done. checkpoint: $CKPT"
echo "Play (container, same FastRTPS as GUI) — needs lerobot/torch in that env:"
echo "  ros2 launch grab_it_il play_policy.launch.py checkpoint:=$CKPT"
echo "Or host venv + same DDS profile, then GUI PLAY."
