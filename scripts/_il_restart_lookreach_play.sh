#!/usr/bin/env bash
# Restart lookreach ACT play — always via PARK-wait path.
set -eo pipefail
WS="${HOME}/dev-ws"
CKPT="${1:-${GRAB_IT_IL_CKPT:-$WS/datasets/lerobot/outputs/act_foam_pole_lookreach_office_tutorial/checkpoints/last/pretrained_model}}"
exec bash "${WS}/scripts/_il_replay_lookreach.sh" "$CKPT"
