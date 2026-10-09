#!/usr/bin/env bash
# Preflight: measured twin joints + D435i present (lookreach PARK / play).
set -eo pipefail
python3 - <<'PY'
import json, urllib.request, sys

base = "http://127.0.0.1:8081"
ok = True

# Twin measured joints
try:
    twin = json.loads(urllib.request.urlopen(base + "/isaac/twin", timeout=1.5).read().decode())
except Exception as exc:
    print(f"FAIL twin: {exc}")
    sys.exit(2)
names = list(twin.get("arm_n") or [])
pos = [float(x) for x in (twin.get("arm_p") or [])]
by = {n: pos[i] for i, n in enumerate(names) if i < len(pos)}
print("twin_ok", twin.get("ok"), "arm", {k: round(float(by.get(k, 0)), 3) for k in ("waist", "shoulder", "wrist_angle", "gripper", "pan")})
if not names or len(pos) < 5:
    print("FAIL: twin missing arm_p (obs would fall back to commanded hub)")
    ok = False
else:
    print("PASS: measured arm_p present")

# JPEG
try:
    jpg = urllib.request.urlopen(base + "/d435i.jpg", timeout=1.5).read()
    print(f"d435i.jpg bytes={len(jpg)}")
    if len(jpg) < 800:
        print("FAIL: d435i.jpg too small")
        ok = False
    else:
        print("PASS: d435i.jpg live")
except Exception as exc:
    print(f"FAIL d435i: {exc}")
    ok = False

# Lookreach PARK cue: pan look-down (pole in frame). Ceiling = pan negative.
pan = float(by.get("pan", 0.0)) if by else 0.0
if pan < 0.50:
    print(f"WARN: pan={pan:.3f} not look-down (want ≥0.70 at PARK) — pole may be out of frame")
elif pan < 0.70:
    print(f"WARN: pan={pan:.3f} below PARK band [0.70, 1.35]")
else:
    print(f"PASS: pan look-down ({pan:.3f})")

# Hub cmd (authoritative command snapshot)
try:
    cmd = json.loads(urllib.request.urlopen(base + "/isaac/cmd", timeout=1.5).read().decode())
    n = list(cmd.get("n") or [])
    p = [float(x) for x in (cmd.get("p") or [])]
    cby = {a: p[i] for i, a in enumerate(n) if i < len(p)}
    print("hub_cmd", {k: round(float(cby.get(k, 0)), 3) for k in ("waist", "shoulder", "pan")})
except Exception as exc:
    print(f"hub_cmd note: {exc}")

sys.exit(0 if ok else 1)
PY
