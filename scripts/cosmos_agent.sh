#!/bin/bash
# cosmos_agent.sh — run Cosmos3-Edge planner against grab_it GUI WebSocket.
# Prereqs on AGX host:
#   - ~/cosmos3-venv with torch + transformers (already set up)
#   - GUI up (grabit_session / agx_bringup) so ws://127.0.0.1:8766 answers
#   - For camera: mjpeg_relay on :8081/d435i
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${COSMOS3_VENV:-$HOME/cosmos3-venv}"

if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "Missing venv at $VENV — create it first (see Cosmos install notes)."
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

python - <<'PY' || pip install -q websockets requests pillow
import websockets, requests, PIL  # noqa: F401
PY

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export GRAB_IT_WS_URL="${GRAB_IT_WS_URL:-ws://127.0.0.1:8766}"
export GRAB_IT_CAMERA_URL="${GRAB_IT_CAMERA_URL:-http://127.0.0.1:8081/d435i}"

exec python "$ROOT/scripts/cosmos_agent.py" "$@"
