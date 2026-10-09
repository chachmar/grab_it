#!/usr/bin/env bash
# Train a LeRobot ACT policy for grab_it tabletop pick.
# Run on the AGX HOST (or jetson-containers lerobot), NOT inside isaac_ros_dev.
set -euo pipefail

WS="${ISAAC_ROS_WS:-$HOME/dev-ws}"
RAW="${GRAB_IT_IL_RAW:-$WS/datasets/tabletop_pick}"
DS="${GRAB_IT_IL_LEROBOT:-$WS/datasets/lerobot/grabit_tabletop_pick}"
OUT="${GRAB_IT_IL_OUT:-$WS/datasets/lerobot/outputs/act_tabletop_pick}"
STEPS="${GRAB_IT_IL_STEPS:-20000}"
BATCH="${GRAB_IT_IL_BATCH:-8}"
SAVE_FREQ="${GRAB_IT_IL_SAVE_FREQ:-2500}"

if [[ -d /workspaces/isaac_ros-dev && -f /opt/ros/humble/setup.bash ]]; then
  echo "WARNING: this looks like isaac_ros_dev. Train on the AGX host venv instead."
fi

ver=""
if [[ -f "$DS/meta/info.json" ]]; then
  ver=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('codebase_version',''))" "$DS/meta/info.json")
fi
if [[ "$ver" != "v3.0" ]]; then
  echo "Converting episodes → LeRobot v3.0 (have '${ver:-none}')..."
  python3 "$WS/scripts/il_to_lerobot.py" --input "$RAW" --output "$DS"
fi

if ! command -v lerobot-train >/dev/null 2>&1; then
  echo "lerobot-train not on PATH."
  echo "  python3 -m venv ~/lerobot-venv && source ~/lerobot-venv/bin/activate"
  echo "  pip install lerobot pandas pyarrow"
  echo "Or: jetson-containers run \$(autotag lerobot)"
  exit 1
fi

lerobot-train \
  --policy.type=act \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/grabit_tabletop_pick \
  --dataset.root="$DS" \
  --dataset.video_backend=pyav \
  --output_dir="$OUT" \
  --policy.device=cuda \
  --batch_size="$BATCH" \
  --steps="$STEPS" \
  --save_freq="$SAVE_FREQ"

echo "checkpoint dir: $OUT"
echo "play on the robot:"
echo "  ros2 launch grab_it_il play_policy.launch.py checkpoint:=$OUT/checkpoints/last/pretrained_model"
