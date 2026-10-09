#!/usr/bin/env bash
set -euo pipefail
from_tmp() {
  local src="$1" dest="$2"
  python3 - "$src" "$dest" <<'PY'
import sys
from pathlib import Path
src, dest = Path(sys.argv[1]), Path(sys.argv[2])
data = src.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
dest.write_bytes(data)
print(dest, len(data))
PY
}
from_tmp /tmp/agx_isaac_hub.py "$HOME/dev-ws/scripts/agx_isaac_hub.py"
from_tmp /tmp/_boot_hub_and_retest.sh "$HOME/dev-ws/scripts/_boot_hub_and_retest.sh"
from_tmp /tmp/_il_retest_tutorial.sh "$HOME/dev-ws/scripts/_il_retest_tutorial.sh"
from_tmp /tmp/il_play_host.sh "$HOME/dev-ws/scripts/il_play_host.sh"
from_tmp /tmp/_il_replay_lookreach.sh "$HOME/dev-ws/scripts/_il_replay_lookreach.sh"
from_tmp /tmp/play_policy.py "$HOME/dev-ws/src/grab_it_il/grab_it_il/play_policy.py"
from_tmp /tmp/common.py "$HOME/dev-ws/src/grab_it_il/grab_it_il/common.py"
chmod +x "$HOME/dev-ws/scripts/_boot_hub_and_retest.sh" \
  "$HOME/dev-ws/scripts/_il_retest_tutorial.sh" \
  "$HOME/dev-ws/scripts/il_play_host.sh" \
  "$HOME/dev-ws/scripts/_il_replay_lookreach.sh"
grep -n 'ThreadingHTTPServer' "$HOME/dev-ws/scripts/agx_isaac_hub.py" | head -3
rm -f /tmp/il_retest_tutorial.out
nohup bash "$HOME/dev-ws/scripts/_boot_hub_and_retest.sh" >/dev/null 2>&1 &
echo BOOT_STARTED
sleep 12
head -50 /tmp/il_retest_tutorial.out || true
curl -s -m 2 -o /dev/null -w "hub=%{http_code}\n" http://127.0.0.1:8081/isaac/twin || true
