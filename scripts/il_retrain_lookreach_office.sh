#!/usr/bin/env bash
# Train ACT on office lookreach recovery demos only (no room mix).
# Run on AGX host after syncing datasets/tabletop_pick_lookreach_office.
set -euo pipefail
WS="${ISAAC_ROS_WS:-$HOME/dev-ws}"
export GRAB_IT_IL_RAW="${GRAB_IT_IL_RAW:-$WS/datasets/tabletop_pick_lookreach_office}"
export GRAB_IT_IL_LEROBOT="${GRAB_IT_IL_LEROBOT:-$WS/datasets/lerobot/grabit_lookreach_office}"
export GRAB_IT_IL_OUT="${GRAB_IT_IL_OUT:-$WS/datasets/lerobot/outputs/act_foam_pole_lookreach_office}"
export GRAB_IT_IL_STEPS="${GRAB_IT_IL_STEPS:-30000}"
exec bash "$WS/scripts/il_retrain_lookreach.sh"
