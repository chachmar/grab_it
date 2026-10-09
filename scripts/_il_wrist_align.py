#!/usr/bin/env python3
"""After the waist is on the pole, align the jaws, then let ACT grasp.

The wrist camera is mounted on the wrist housing, so wrist_rotate does not
move the shaft in the image. The shaft's tilt is an absolute jaw angle:
command that angle once and stop. Pitch is a real visual loop because
wrist_angle does move the image; it stops when the shaft is centered or the
error gets worse. Shoulder only creeps far enough for the wrist camera to
see the pole. The reach and the close belong to ACT.
"""
from __future__ import annotations

import importlib.util
import json
import math
import time
import urllib.request
from pathlib import Path

HUB = "http://127.0.0.1:8081"
NAMES = ["waist", "shoulder", "wrist_angle", "wrist_rotate", "gripper", "pan"]
GRIP_HOLD_OPEN = 1.20
# Jaw window in the wrist image. A shaft already here is lined up; pitching
# it toward image center walks it back out of the gripper.
JAW_U = (0.35, 0.65)
JAW_V = (0.55, 0.95)
ROT_TOL = 0.08
# Do not unfold into the grasp. ACT starts from here.
SHOULDER_SEE = -1.50
WRIST_ANG_LIM = (-1.6, 1.45)
WRIST_ROT_LIM = (-1.2, 1.2)
ALIGN_S = 22.0


def _load_shaft():
    path = Path(__file__).resolve().with_name("_il_wrist_finish.py")
    spec = importlib.util.spec_from_file_location("il_wrist_finish", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.shaft_pose


shaft_pose = _load_shaft()


def _get(path: str) -> bytes:
    return urllib.request.urlopen(HUB + path, timeout=2.0).read()


def twin() -> dict:
    return json.loads(_get("/isaac/twin").decode())


def arm_of(d: dict) -> dict:
    names = list(d.get("arm_n") or [])
    pos = [float(x) for x in (d.get("arm_p") or [])]
    return {n: pos[i] for i, n in enumerate(names) if i < len(pos)}


def post(pose: dict) -> None:
    body = json.dumps(
        {
            "t": "arm",
            "n": NAMES,
            "p": [float(pose[n]) for n in NAMES],
            "clear_gate": True,
        }
    ).encode()
    req = urllib.request.Request(
        HUB + "/isaac/cmd",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    urllib.request.urlopen(req, timeout=1.0).read()


def _clip(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(val)))


def _bearing(d: dict) -> float | None:
    xy = d.get("pole_xy")
    if not xy or len(xy) < 2:
        return None
    wx, wy = float(xy[0]), float(xy[1])
    if wx < 0.08:
        return None
    return max(-1.8, min(1.8, math.atan2(wy, wx)))


def _hold(cmd: dict, seconds: float = 3.5) -> dict | None:
    """Post until the twin advances and shoulder / wrist have arrived."""
    last = None
    try:
        prev_t = twin().get("t")
    except Exception:
        prev_t = None
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        try:
            post(cmd)
        except Exception as exc:
            print(f"post fail: {exc}", flush=True)
        time.sleep(0.15)
        try:
            last = twin()
        except Exception:
            continue
        if last.get("t") == prev_t:
            continue
        meas = arm_of(last)
        arrived = True
        for joint in ("shoulder", "wrist_angle", "wrist_rotate", "waist"):
            if joint not in meas or joint not in cmd:
                continue
            if abs(float(meas[joint]) - float(cmd[joint])) > 0.10:
                arrived = False
                break
        if arrived:
            return last
    return last


def main() -> int:
    try:
        d0 = twin()
    except Exception as exc:
        print(f"ERROR: no twin: {exc}", flush=True)
        return 1
    arm = arm_of(d0)
    for n in NAMES:
        arm.setdefault(n, 0.0)
    bearing = _bearing(d0)
    cmd = dict(arm)
    cmd["gripper"] = GRIP_HOLD_OPEN
    cmd["pan"] = 0.78
    if bearing is not None:
        cmd["waist"] = float(bearing)
    print(
        f"wrist align start waist={float(cmd['waist']):.3f} "
        f"shoulder={float(cmd['shoulder']):.3f} "
        f"wrist_angle={float(cmd['wrist_angle']):.3f}",
        flush=True,
    )
    pending = None
    rot_target = None
    seen = False
    t0 = time.monotonic()
    while time.monotonic() - t0 < ALIGN_S:
        d = _hold(cmd)
        try:
            wide = _get("/wide.jpg")
        except Exception:
            wide = b""
        shaft = shaft_pose(wide)
        if shaft is None:
            if rot_target is not None:
                cmd["wrist_rotate"] = float(rot_target)
                meas = arm_of(d) if d else {}
                rot_now = float(meas.get("wrist_rotate", 9.0))
                print(
                    f"wrist hold rotate measured={rot_now:+.3f} "
                    f"target={float(rot_target):+.3f}",
                    flush=True,
                )
                if abs(rot_now - float(rot_target)) <= ROT_TOL:
                    print(
                        f"wrist aligned rot={rot_now:+.3f} "
                        f"wrist_angle={float(cmd['wrist_angle']):+.3f}",
                        flush=True,
                    )
                    return 0
                continue
            if not seen and float(cmd["shoulder"]) < SHOULDER_SEE - 0.02:
                cmd["shoulder"] = min(SHOULDER_SEE, float(cmd["shoulder"]) + 0.12)
                print(f"wrist seek shoulder={float(cmd['shoulder']):+.2f}", flush=True)
                continue
            try:
                eye = shaft_pose(_get("/d435i.jpg"))
            except Exception:
                eye = None
            if eye is not None:
                rot_target = _clip(-float(eye["from_vert"]), *WRIST_ROT_LIM)
                cmd["wrist_rotate"] = rot_target
                settled = _hold(cmd, 5.0)
                got = arm_of(settled) if settled else {}
                print(
                    f"wrist aligned from d435i tilt={eye['from_vert']:+.3f} "
                    f"wrist_rotate={float(got.get('wrist_rotate', rot_target)):+.3f}",
                    flush=True,
                )
                return 0
            print("wrist align: shaft not in the wrist camera — leaving rotate", flush=True)
            return 0
        seen = True
        tilt = float(shaft["from_vert"])
        in_jaws = JAW_U[0] <= float(shaft["u"]) <= JAW_U[1] and JAW_V[0] <= float(shaft["v"]) <= JAW_V[1]
        if rot_target is None:
            if pending is None or abs(tilt - pending) > 0.35:
                pending = tilt
                print(
                    f"wrist shaft u={shaft['u']:.2f} v={shaft['v']:.2f} "
                    f"tilt={tilt:+.3f} in_jaws={in_jaws}",
                    flush=True,
                )
                continue
            rot_target = _clip(-0.5 * (pending + tilt), *WRIST_ROT_LIM)
            print(f"wrist rotate target={rot_target:+.3f}", flush=True)
        cmd["wrist_rotate"] = float(rot_target)
        # Pitch only to bring a shaft that is outside the jaw window back in.
        # A shaft already between the jaws stays put.
        if not in_jaws:
            if float(shaft["v"]) > JAW_V[1]:
                cmd["wrist_angle"] = _clip(float(cmd["wrist_angle"]) - 0.06, *WRIST_ANG_LIM)
            elif float(shaft["v"]) < JAW_V[0]:
                cmd["wrist_angle"] = _clip(float(cmd["wrist_angle"]) + 0.06, *WRIST_ANG_LIM)
        meas = arm_of(d) if d else {}
        rot_now = float(meas.get("wrist_rotate", cmd["wrist_rotate"]))
        print(
            f"wrist servo u={shaft['u']:.2f} v={shaft['v']:.2f} "
            f"in_jaws={in_jaws} rot={rot_now:+.3f} target={float(rot_target):+.3f}",
            flush=True,
        )
        if abs(rot_now - float(rot_target)) <= ROT_TOL and (in_jaws or rot_target is not None):
            # rot_now is from the sample taken before this command. Hold once more.
            settled = _hold(cmd, 5.0)
            got = arm_of(settled) if settled else {}
            rot_got = float(got.get("wrist_rotate", 9.0))
            if abs(rot_got - float(rot_target)) <= ROT_TOL:
                print(
                    f"wrist aligned rot={rot_got:+.3f} "
                    f"wrist_angle={float(got.get('wrist_angle', 0)):+.3f} "
                    f"v={shaft['v']:.2f}",
                    flush=True,
                )
                return 0
    print(
        f"wrist align timeout rot_cmd={float(cmd['wrist_rotate']):+.3f} "
        f"wrist_angle={float(cmd['wrist_angle']):+.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
