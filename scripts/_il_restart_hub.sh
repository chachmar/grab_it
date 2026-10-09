#!/usr/bin/env bash
set -eo pipefail
# Restart only the hub; leave play_policy / relay alone.
pkill -f '/home/ubuntu/dev-ws/scripts/agx_isaac_hub.py' 2>/dev/null || true
pkill -f 'python3 scripts/agx_isaac_hub.py' 2>/dev/null || true
sleep 1
cd /home/ubuntu/dev-ws
nohup python3 /home/ubuntu/dev-ws/scripts/agx_isaac_hub.py >/tmp/agx_isaac_hub.log 2>&1 &
echo "hub_pid=$!"
sleep 3
if ! pgrep -f agx_isaac_hub.py >/dev/null; then
  echo "HUB FAILED"
  cat /tmp/agx_isaac_hub.log || true
  exit 1
fi
tail -15 /tmp/agx_isaac_hub.log || true
curl -s -o /dev/null -w "gui=%{http_code} bytes=%{size_download}\n" http://127.0.0.1:8765/ || true
curl -s -o /dev/null -w "jpg=%{http_code}\n" http://127.0.0.1:8081/d435i.jpg || true
