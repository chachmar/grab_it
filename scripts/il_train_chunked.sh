#!/usr/bin/env bash
# Chunked ACT train for AGX boards that reboot under sustained MAXN load.
# Trains 2500 steps at a time (matches save_freq), then loops / resumes.
# Run on the AGX desktop (not required via SSH):
#   source ~/lerobot-venv/bin/activate && bash ~/dev-ws/scripts/il_train_chunked.sh
set -euo pipefail
source "$HOME/lerobot-venv/bin/activate"
WS="${ISAAC_ROS_WS:-$HOME/dev-ws}"
OUT="${GRAB_IT_IL_OUT:-$WS/datasets/lerobot/outputs/act_tabletop_pick}"
DS="${GRAB_IT_IL_LEROBOT:-$WS/datasets/lerobot/grabit_tabletop_pick}"
TARGET="${GRAB_IT_IL_STEPS:-20000}"
CHUNK="${GRAB_IT_IL_CHUNK:-2500}"
BATCH="${GRAB_IT_IL_BATCH:-4}"
LOG="${GRAB_IT_IL_LOG:-/tmp/il_retrain.log}"

# Prefer a cooler power mode so the board is less likely to brown-out/reboot.
# MAXN (0) is what was killing ~25% runs. Mode 2 is typically 30W-class on Orin.
if command -v nvpmodel >/dev/null 2>&1; then
  echo "Setting nvpmodel mode 2 (cooler than MAXN)..."
  sudo nvpmodel -m 2 || sudo nvpmodel -m 1 || true
  nvpmodel -q || true
fi
if command -v jetson_clocks >/dev/null 2>&1; then
  sudo jetson_clocks --restore 2>/dev/null || true
fi

current_step() {
  local link target base
  link="$OUT/checkpoints/last"
  if [[ ! -L "$link" && ! -d "$link" ]]; then
    echo 0
    return
  fi
  target=$(readlink -f "$link" 2>/dev/null || echo "$link")
  base=$(basename "$target")
  if [[ "$base" =~ ^[0-9]+$ ]]; then
    echo $((10#$base))
  else
    echo 0
  fi
}

if [[ ! -f "$DS/meta/info.json" ]]; then
  echo "ERROR: missing LeRobot dataset at $DS"
  exit 1
fi

exec > >(tee -a "$LOG") 2>&1
echo "=== chunked train $(date -Is) target=$TARGET chunk=$CHUNK batch=$BATCH ==="

step=$(current_step)
echo "current checkpoint step=$step"

while (( step < TARGET )); do
  next=$(( step + CHUNK ))
  if (( next > TARGET )); then
    next=$TARGET
  fi
  echo "---- chunk: $step → $next ($(date -Is)) ----"
  CFG="$OUT/checkpoints/last/pretrained_model/train_config.json"
  if [[ -f "$CFG" && "$step" -gt 0 ]]; then
    lerobot-train \
      --config_path="$CFG" \
      --resume=true \
      --steps="$next" \
      --batch_size="$BATCH" \
      --save_freq="$CHUNK" \
      --dataset.root="$DS" \
      --dataset.video_backend=pyav \
      --output_dir="$OUT" \
      --policy.device=cuda \
      --policy.push_to_hub=false
  else
    lerobot-train \
      --policy.type=act \
      --policy.push_to_hub=false \
      --dataset.repo_id="${GRAB_IT_IL_REPO:-local/grabit_tabletop_pick}" \
      --dataset.root="$DS" \
      --dataset.video_backend=pyav \
      --output_dir="$OUT" \
      --policy.device=cuda \
      --batch_size="$BATCH" \
      --steps="$next" \
      --save_freq="$CHUNK"
  fi
  step=$(current_step)
  echo "chunk done; now at step=$step"
  # Cool-down between chunks (AGX brown-outs after sustained load).
  # Prefer $OUT/cooldown_s.txt so you can raise it without editing the script.
  cool_file="${OUT}/cooldown_s.txt"
  if [[ -f "$cool_file" ]]; then
    cool=$(tr -cd '0-9' < "$cool_file" | head -c 6)
  fi
  cool="${cool:-${GRAB_IT_IL_COOLDOWN_S:-300}}"
  if [[ -z "$cool" || "$cool" -lt 30 ]]; then
    cool=300
  fi
  echo "cooling ${cool}s…"
  sleep "$cool"
done

echo "Done. Checkpoint: $OUT/checkpoints/last/pretrained_model"
