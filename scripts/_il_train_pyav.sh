#!/bin/bash
# Install pyav if needed, then train with pyav video backend (skip reconvert).
set -euo pipefail
source "$HOME/lerobot-venv/bin/activate"
WS="$HOME/dev-ws"
DS="$WS/datasets/lerobot/grabit_tabletop_pick"
OUT="$WS/datasets/lerobot/outputs/act_tabletop_pick"
STEPS="${GRAB_IT_IL_STEPS:-20000}"

python3 - <<'PY'
try:
    import av
    print("pyav ok", av.__version__)
except Exception as e:
    raise SystemExit(f"need pyav: {e}")
PY

if [[ ! -f "$DS/meta/info.json" ]]; then
  echo "ERROR: missing dataset $DS — run il_retrain_after_collect.sh first"
  exit 1
fi

if [[ -d "$OUT" ]]; then
  bak="${OUT}_fail_$(date +%Y%m%d_%H%M%S)"
  echo "Moving failed/partial output → $bak"
  mv "$OUT" "$bak"
fi

exec bash "$WS/scripts/il_train.sh"
