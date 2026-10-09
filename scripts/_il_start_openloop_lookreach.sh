#!/usr/bin/env bash
# Stop any player, post lookreach PARK until twin measured joints match, then
# start open-loop episode replay (stack proof).
set -eo pipefail
WS="${HOME}/dev-ws"
EP="${1:-$WS/datasets/tabletop_pick_lookreach/episode_000050}"
LOG="${2:-/tmp/il_replay_episode.log}"
PARK_JSON='{"t":"arm","n":["waist","shoulder","wrist_angle","wrist_rotate","gripper","pan"],"p":[-0.55,-1.9,1.4,0.0,1.65,1.10]}'
PARK=(-0.55 -1.9 1.4 0.0 1.65 1.10)

bash "${WS}/scripts/_il_stop_play.sh"

# Ensure hub + obs + relay (no ACT yet).
if ! pgrep -f agx_isaac_hub.py >/dev/null; then
  nohup python3 "${WS}/scripts/agx_isaac_hub.py" >/tmp/agx_isaac_hub.log 2>&1 &
  sleep 2
fi
bash "${WS}/scripts/_il_start_hub_obs_host.sh"
bash "${WS}/scripts/_il_start_twin_relay_host.sh"

park_ok() {
  python3 - <<'PY'
import json, urllib.request, sys
try:
    raw = urllib.request.urlopen("http://127.0.0.1:8081/isaac/twin", timeout=1.0).read()
    d = json.loads(raw.decode())
except Exception as exc:
    print(f"twin fetch fail: {exc}")
    sys.exit(1)
names = list(d.get("arm_n") or [])
pos = [float(x) for x in (d.get("arm_p") or [])]
if not names or len(pos) < 5:
    print(f"twin missing arm_p names={names} npos={len(pos)}")
    sys.exit(1)
by = {n: pos[i] for i, n in enumerate(names) if i < len(pos)}
want = {"waist": -0.55, "shoulder": -1.9, "wrist_angle": 1.4}
tol = 0.08
bad = []
for k, v in want.items():
    got = float(by.get(k, 1e9))
    if abs(got - v) > tol:
        bad.append(f"{k}={got:.3f} want={v}")
# Pan: require look-down demo band. Exact 1.10 sometimes stalls in PhysX
# (~0.78) while still showing the pole; do not block restart on that.
pan = float(by.get("pan", 1e9))
if not (0.70 <= pan <= 1.35):
    bad.append(f"pan={pan:.3f} want look-down [0.70,1.35]")
if bad:
    print("not parked:", ", ".join(bad))
    sys.exit(1)
print("parked", {**{k: round(float(by.get(k, 0)), 3) for k in want}, "pan": round(pan, 3)})
sys.exit(0)
PY
}

echo "holding PARK until twin arm_p matches…"
ok=0
# Soft rate-limit (~0.08 rad/cmd) needs frequent seq bumps to reach PARK.
for i in $(seq 1 200); do
  curl -s -X POST http://127.0.0.1:8081/isaac/cmd \
    -H "Content-Type: application/json" \
    -d "$PARK_JSON" >/dev/null || true
  if park_ok; then
    ok=1
    break
  fi
  sleep 0.1
done
if [[ "$ok" -ne 1 ]]; then
  echo "WARN: twin did not report PARK within timeout — starting replay anyway"
  curl -s http://127.0.0.1:8081/isaac/twin | head -c 400; echo
fi

# Kill any leftover play before open-loop.
pids=$(pgrep -f '/play_policy.py' || true)
if [[ -n "$pids" ]]; then
  # shellcheck disable=SC2086
  kill -9 $pids 2>/dev/null || true
  sleep 1
fi

nohup bash "${WS}/scripts/_il_replay_episode_host.sh" "$EP" "$LOG" \
  >/tmp/il_replay_episode.nohup 2>&1 &
echo "open-loop replay started pid=$! ep=$EP log=$LOG"
sleep 8
grep -E 'policy PLAY|autoplay|IL player|Error|Traceback|episode done' "$LOG" | tail -20 || true
pgrep -af '/play_policy.py' | head -3 || echo no_play
