#!/usr/bin/env bash
# Pulse lookreach PARK (waist -0.55) until close-gate clears and shoulder folds.
# Shoulder-face park (waist≈0) does NOT clear the IL close-gate latch.
set -eo pipefail
pkill -9 -f '/play_policy.py' 2>/dev/null || true
sleep 1
python3 - <<'PY'
import json, time, urllib.request

def twin():
    return json.loads(
        urllib.request.urlopen("http://127.0.0.1:8081/isaac/twin", timeout=2).read().decode()
    )

def arm(d):
    n = list(d.get("arm_n") or [])
    p = [float(x) for x in (d.get("arm_p") or [])]
    return {a: p[i] for i, a in enumerate(n) if i < len(p)}

def post(waist):
    body = json.dumps({
        "t": "arm",
        "n": ["waist", "shoulder", "wrist_angle", "wrist_rotate", "gripper", "pan"],
        "p": [waist, -1.9, 1.4, 0.0, 1.65, 1.10],
    }).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8081/isaac/cmd",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    urllib.request.urlopen(req, timeout=1).read()

print("clearing close-gate with lookreach PARK…", flush=True)
for i in range(300):
    post(-0.55)
    if i % 30 == 0:
        try:
            d = twin()
            by = arm(d)
            print(
                f"  t={i} waist={by.get('waist', 9):.3f} "
                f"shoulder={by.get('shoulder', 9):.3f} "
                f"cg={d.get('close_gate')}",
                flush=True,
            )
        except Exception as e:
            print(f"  twin {e}", flush=True)
    try:
        by = arm(twin())
        if (
            abs(float(by.get("shoulder", 9)) + 1.9) < 0.25
            and abs(float(by.get("waist", 9)) + 0.55) < 0.15
        ):
            print(
                f"folded at lookreach park "
                f"waist={by['waist']:.3f} shoulder={by['shoulder']:.3f}",
                flush=True,
            )
            break
    except Exception:
        pass
    time.sleep(0.1)
else:
    try:
        print(f"WARN still not folded: {arm(twin())}", flush=True)
    except Exception as e:
        print(f"WARN fold check failed: {e}", flush=True)
PY
pgrep -af '/play_policy.py' || echo no_play
