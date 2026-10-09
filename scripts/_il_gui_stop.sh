#!/usr/bin/env bash
# IL STOP: kill ACT and the wrist finish. Leave the arm where it is.
set -eo pipefail
pids=$(pgrep -f '/play_policy.py' || true)
if [[ -n "$pids" ]]; then
  # shellcheck disable=SC2086
  kill -9 $pids 2>/dev/null || true
  sleep 0.3
fi
pids=$(pgrep -f '/_il_wrist_finish.py' || true)
if [[ -n "$pids" ]]; then
  # shellcheck disable=SC2086
  kill -9 $pids 2>/dev/null || true
  sleep 0.2
fi
pids=$(pgrep -f '/il_twin_cmd_relay.py' || true)
if [[ -n "$pids" ]]; then
  # shellcheck disable=SC2086
  kill -9 $pids 2>/dev/null || true
  sleep 0.2
fi
python3 - <<'PY'
import json
import urllib.request

hub = "http://127.0.0.1:8081"
d = json.loads(urllib.request.urlopen(hub + "/isaac/twin", timeout=2).read())
names = list(d.get("arm_n") or [])
pos = [float(x) for x in (d.get("arm_p") or [])]
body = json.dumps({"t": "arm", "n": names, "p": pos, "clear_gate": True}).encode()
for _ in range(8):
    req = urllib.request.Request(
        hub + "/isaac/cmd",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    urllib.request.urlopen(req, timeout=2).read()
print("held", {n: round(p, 3) for n, p in zip(names, pos)})
PY
echo "stopped · arm held"
