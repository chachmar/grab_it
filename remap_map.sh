#!/bin/bash
# remap_map.sh — backup current RTAB-Map DB (if any), then start a FRESH mapping run.
#
# mapping_bringup.sh already uses --delete_db_on_start. This wrapper just saves the
# previous database so you can remap without losing the old file forever.
#
# Usage (inside Isaac container):
#   bash /workspaces/isaac_ros-dev/remap_map.sh
set -e
WS=/workspaces/isaac_ros-dev
DB=$WS/rtabmap.db
STAMP=$(date +%Y%m%d_%H%M%S)

if [ -f "$DB" ]; then
  BAK="$WS/rtabmap_backup_${STAMP}.db"
  echo "[remap] backing up $DB -> $BAK"
  cp -v "$DB" "$BAK"
else
  echo "[remap] no existing DB — starting clean"
fi

echo "[remap] launching mapping_bringup.sh (will delete_db_on_start and rebuild map)"
echo "[remap] drive slowly with teleop; when done Ctrl-C, then use nav_bringup.sh"
exec bash "$WS/mapping_bringup.sh"
