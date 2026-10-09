#!/usr/bin/env python3
"""Stop ACT when the gripper stalls on the pole, then leave the arm there."""
from __future__ import annotations

import json
import math
import os
import subprocess
import time
import urllib.request

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

HUB = "http://127.0.0.1:8081"
NAMES = ["waist", "shoulder", "wrist_angle", "wrist_rotate", "gripper", "pan"]
# Open enough that the jaws stay clear, and far enough from the hub's
# GRIP_OPEN sentinel (1.65) that Isaac actually applies the command.
GRIP_HOLD_OPEN = 1.20
GRIP_CLOSED = -0.95
# both_pads_enclosed latches at 0.032. IL_PAD_CLOSE_READY is 0.042:
# a live miss at 0.033–0.036 used to keep moving instead of closing.
PAD_CLOSE = 0.042
# Take over while the jaws are still open, before the pads seat.
PAD_TAKEOVER = 0.065
# Jaw window in the wrist image. Pitch only until the shaft is inside it.
JAW_U = (0.35, 0.65)
JAW_V = (0.55, 0.95)
WRIST_ANG_LIM = (-1.6, 1.45)
WRIST_ROT_LIM = (-1.2, 1.2)
# Foam reads cyan-gray. Carpet is orange.
WAIT_S = 25.0
# A foam shaft in the wrist view is a tall thin streak, not a fat blob.


def _get(path: str) -> bytes:
    return urllib.request.urlopen(HUB + path, timeout=2.0).read()


def twin() -> dict:
    return json.loads(_get("/isaac/twin").decode())


def arm_of(d: dict) -> dict:
    names = list(d.get("arm_n") or [])
    pos = [float(x) for x in (d.get("arm_p") or [])]
    return {n: pos[i] for i, n in enumerate(names) if i < len(pos)}


def post(pose: dict, *, clear_gate: bool) -> None:
    body = json.dumps(
        {
            "t": "arm",
            "n": NAMES,
            "p": [float(pose[n]) for n in NAMES],
            "clear_gate": bool(clear_gate),
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


def kill_act() -> None:
    subprocess.run(
        ["pkill", "-f", "/play_policy.py"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["pkill", "-f", "/il_twin_cmd_relay.py"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.3)


def shaft_pose(jpeg: bytes) -> dict | None:
    """Foam shaft: centroid, tilt from vertical, pixel count.

    A shaft in the wrist view is often only a few dozen pale pixels spread
    down the frame. A contour-area test treated that as nothing. A mask that
    lights the whole D435i frame is clutter, not a pole, and is rejected.
    """
    if cv2 is None or not jpeg:
        return None
    img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if h < 40 or w < 40:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hh, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = (hh > 85) & (hh < 130) & (s > 25) & (s < 140) & (v > 70) & (v < 230)
    # The wrist view often blows the foam out to a white column. That column
    # sits under the gripper, below the housing, and stays narrow.
    if int(mask.sum()) < 40:
        y0 = int(0.34 * h)
        pale = np.zeros((h, w), dtype=bool)
        pale[y0:] = (s[y0:] <= 35) & (v[y0:] >= 200)
        mask = pale
    ys, xs = np.where(mask)
    if xs.size < 40:
        return None
    col = mask.sum(axis=0)
    peak = int(col.max())
    lit = int(np.count_nonzero(col))
    # Scattered cyan (blinds, walls) spans the frame without a tall column.
    if lit > int(0.45 * w) and peak < int(0.18 * h):
        return None
    bw = int(xs.max() - xs.min())
    bh = int(ys.max() - ys.min())
    if bh < int(0.12 * h):
        return None
    # A pale wall lights a wide band. A shaft does not.
    if int(mask.sum()) >= 40 and bw > int(0.22 * w) and peak < int(0.35 * h):
        return None
    thresh = max(2, int(0.45 * peak))
    sel = np.where(col >= thresh)[0]
    if sel.size == 0:
        sel = np.array([int(np.argmax(col))])
    # Stray pale pixels make the whole-frame box wide and flip the tilt.
    # The shaft is the bright column.
    col_lo = max(0, int(sel.min()) - 2)
    col_hi = min(w, int(sel.max()) + 3)
    shaft = np.zeros_like(mask)
    shaft[:, col_lo:col_hi] = mask[:, col_lo:col_hi]
    ys, xs = np.where(shaft)
    if xs.size < 40:
        return None
    bw = int(xs.max() - xs.min())
    bh = int(ys.max() - ys.min())
    rows = np.where(shaft[:, sel].any(axis=1))[0]
    u = float(np.median(sel)) / float(w)
    v = float(np.median(rows if rows.size else ys)) / float(h)
    # A tall streak is an upright shaft. PCA on a few dozen pixels flips
    # between +90° and -90°, which slammed wrist_rotate into its stops.
    tall = bh > int(1.3 * max(bw, 1))
    if tall:
        from_vert = 0.0
    else:
        pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
        pts -= pts.mean(axis=0)
        cov = np.cov(pts.T)
        if not np.isfinite(cov).all():
            from_vert = 0.0
        else:
            _vals, vecs = np.linalg.eigh(cov)
            axis = vecs[:, 1]
            from_vert = math.atan2(float(axis[0]), float(axis[1]))
            if from_vert > math.pi / 2:
                from_vert -= math.pi
            if from_vert < -math.pi / 2:
                from_vert += math.pi
            # Near ±90° the long axis was swapped. Fold it back toward vertical.
            if abs(from_vert) > 1.05:
                from_vert = 0.0
    return {
        "u": u,
        "v": v,
        "from_vert": float(from_vert),
        "area": float(xs.size),
        "tall": tall,
    }


def _bearing(d: dict) -> float | None:
    xy = d.get("pole_xy")
    if not xy or len(xy) < 2:
        return None
    wx, wy = float(xy[0]), float(xy[1])
    if wx < 0.08:
        return None
    return max(-1.8, min(1.8, math.atan2(wy, wx)))


def _pad(d: dict | None) -> float | None:
    if not d:
        return None
    raw = d.get("pad_d")
    return None if raw is None else float(raw)


def _aim_waist(cmd: dict) -> float | None:
    """Point the waist at the pole once. Do not refresh while the base yaws."""
    try:
        d = twin()
    except Exception:
        return None
    bearing = _bearing(d)
    if bearing is None:
        return None
    cmd["waist"] = float(bearing)
    return float(bearing)


def _hold(cmd: dict, seconds: float) -> dict | None:
    """Post until the twin timestamp advances and the arm is near the command.

    The hub twin is only rewritten every couple of seconds. Scoring a move
    before that returns the previous pad distance for every candidate.
    """
    last = None
    try:
        prev_t = twin().get("t")
    except Exception:
        prev_t = None
    t0 = time.monotonic()
    while time.monotonic() - t0 < max(seconds, 4.5):
        try:
            post(cmd, clear_gate=True)
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
        for joint in ("shoulder", "wrist_angle", "waist"):
            if joint not in meas:
                continue
            if abs(float(meas[joint]) - float(cmd[joint])) > 0.08:
                arrived = False
                break
        if arrived:
            return last
    if last is not None:
        meas = arm_of(last)
        print(
            f"aim short waist={float(meas.get('waist', 0)):.3f} "
            f"cmd={float(cmd.get('waist', 0)):.3f}",
            flush=True,
        )
    return last


def _snap(cmd: dict) -> dict:
    return {n: float(cmd[n]) for n in NAMES}


def _ready(pad: float | None) -> bool:
    return pad is not None and float(pad) <= PAD_CLOSE


def _base_level(d: dict | None) -> bool:
    if not d or d.get("z") is None:
        return True
    return float(d["z"]) < 0.08


def _wide_shaft() -> dict | None:
    try:
        return shaft_pose(_get("/wide.jpg"))
    except Exception:
        return None


def _in_jaws(shaft: dict | None) -> bool:
    if not shaft:
        return False
    return (
        JAW_U[0] <= float(shaft["u"]) <= JAW_U[1]
        and JAW_V[0] <= float(shaft["v"]) <= JAW_V[1]
    )


def _align_wrist_camera(cmd: dict) -> None:
    """Set jaw roll from the shaft tilt, then pitch until it sits on the jaw line.

    The wrist camera does not spin with wrist_rotate, so the tilt is an
    absolute roll target. Pitch does move the image: tipping down raises the
    shaft in the frame. Stop when it is inside the jaw window.
    """
    shaft = None
    for _ in range(5):
        shaft = _wide_shaft()
        if shaft is not None:
            break
        if float(cmd["shoulder"]) >= -1.05:
            break
        cmd["shoulder"] = min(-1.05, float(cmd["shoulder"]) + 0.12)
        print(f"seek shoulder={float(cmd['shoulder']):+.2f}", flush=True)
        _hold(cmd, 3.0)
    if shaft is None:
        print("wrist camera: no shaft for roll/pitch", flush=True)
        return
    roll = _clip(-float(shaft["from_vert"]), *WRIST_ROT_LIM)
    cmd["wrist_rotate"] = roll
    print(
        f"roll tilt={float(shaft['from_vert']):+.3f} wrist_rotate={roll:+.3f} "
        f"u={shaft['u']:.2f} v={shaft['v']:.2f}",
        flush=True,
    )
    for _ in range(6):
        d = _hold(cmd, 2.0)
        shaft = _wide_shaft()
        if shaft is None:
            print("pitch lost the shaft — holding this wrist", flush=True)
            return
        meas = arm_of(d) if d else {}
        print(
            f"pitch u={shaft['u']:.2f} v={shaft['v']:.2f} "
            f"in_jaws={_in_jaws(shaft)} "
            f"wrist_angle={float(meas.get('wrist_angle', cmd['wrist_angle'])):+.2f} "
            f"rot={float(meas.get('wrist_rotate', 0)):+.3f}",
            flush=True,
        )
        v = float(shaft["v"])
        u = float(shaft["u"])
        # Tip-down (larger wrist_angle) moves the shaft up in the image.
        if v > JAW_V[1]:
            cmd["wrist_angle"] = _clip(float(cmd["wrist_angle"]) + 0.06, *WRIST_ANG_LIM)
        elif v < JAW_V[0]:
            cmd["wrist_angle"] = _clip(float(cmd["wrist_angle"]) - 0.06, *WRIST_ANG_LIM)
        rot_err = abs(float(meas.get("wrist_rotate", 0.0)) - float(cmd["wrist_rotate"]))
        if (
            JAW_U[0] <= u <= JAW_U[1]
            and JAW_V[0] <= v <= JAW_V[1]
            and rot_err <= 0.10
        ):
            return
        if not (JAW_U[0] <= u <= JAW_U[1]):
            # One lateral step. Keep it only if the shaft moves toward center
            # and the waist joint actually arrives. Stacking steps while the
            # joint lags winds the waist off the pole.
            prev_u = u
            prev = float(cmd["waist"])
            step = _clip((0.50 - u) * 0.35, -0.08, 0.08)
            moved = False
            for signed in (step, -step):
                cmd["waist"] = _clip(prev + signed, -1.8, 1.8)
                print(
                    f"waist trim {prev:+.3f} → {float(cmd['waist']):+.3f} u={prev_u:.2f}",
                    flush=True,
                )
                dtry = _hold(cmd, 3.0)
                got_w = float(arm_of(dtry).get("waist", 9.0)) if dtry else 9.0
                if abs(got_w - float(cmd["waist"])) > 0.12:
                    print(f"waist trim not arrived got={got_w:+.3f}", flush=True)
                    continue
                nxt = _wide_shaft()
                if nxt is None:
                    continue
                nu = float(nxt["u"])
                print(f"waist trim u {prev_u:.2f} → {nu:.2f}", flush=True)
                if abs(nu - 0.50) < abs(prev_u - 0.50) - 0.03:
                    moved = True
                    break
            if not moved:
                cmd["waist"] = prev
                print("waist trim undone", flush=True)
                _hold(cmd, 2.0)
                return
            continue
    print("jaw align stopped at the step limit", flush=True)


def _close_and_lift(cmd: dict) -> int:
    """Shut the gripper on the seated pose, then lift. No more searching.

    Fingers stall on the foam, so the measured gripper stays near 0 even
    while the command is fully closed. A stall with the pads on the shaft
    is a close. The lift is a small fold so the jaws stay on the pole.

    GRAB_IT_TO_POINT leaves the jaws open here. ACT does the final close.
    """
    if os.environ.get("GRAB_IT_TO_POINT", "").strip() in ("1", "true"):
        cmd["gripper"] = GRIP_HOLD_OPEN
        d = _hold(cmd, 1.5)
        print(
            f"at grasp point pad={_pad(d)} "
            f"shoulder={float(cmd['shoulder']):+.2f} "
            f"wrist_angle={float(cmd['wrist_angle']):+.2f} "
            f"— ACT does the close",
            flush=True,
        )
        return 0
    cmd["gripper"] = GRIP_CLOSED
    print(
        f"close now shoulder={float(cmd['shoulder']):+.2f} "
        f"wrist_angle={float(cmd['wrist_angle']):+.2f} "
        f"wrist_rotate={float(cmd['wrist_rotate']):+.3f}",
        flush=True,
    )
    d = None
    t0 = time.monotonic()
    while time.monotonic() - t0 < 6.0:
        d = _hold(cmd, 1.5)
        grip = float(arm_of(d).get("gripper", 1.0)) if d else 1.0
        print(f"closing grip={grip:.2f} pad={_pad(d)}", flush=True)
        pad_now = _pad(d)
        # Open command is 1.20. Anything well below that has started to close,
        # and a foam stall often reads near 0 rather than -0.95.
        if grip < 0.45 and pad_now is not None and pad_now <= PAD_CLOSE:
            break
    if _pad(d) is None or _pad(d) > 0.032:
        print(
            f"grasp MISS pad={_pad(d)} grip={float(arm_of(d).get('gripper', 1)) if d else 1:.2f} "
            f"— not on the shaft, not lifting",
            flush=True,
        )
        return 1
    z0 = None if not d else d.get("pole_z")
    before = arm_of(d) if d else {}
    # Raise the jaws. Folding the shoulder pulls the tip back into the sleeve,
    # so the pole stays planted. A little more shoulder and a lot less
    # wrist tip moves the jaws up without dragging the shaft sideways.
    for _ in range(5):
        cmd["wrist_angle"] = _clip(float(cmd["wrist_angle"]) - 0.08, *WRIST_ANG_LIM)
        cmd["shoulder"] = _clip(float(cmd["shoulder"]) + 0.03, -1.9, 0.8)
        cmd["gripper"] = GRIP_CLOSED
        d = _hold(cmd, 2.0)
        meas = arm_of(d) if d else {}
        arrived = (
            abs(float(meas.get("shoulder", 9.0)) - float(cmd["shoulder"])) < 0.10
            and abs(float(meas.get("wrist_angle", 9.0)) - float(cmd["wrist_angle"])) < 0.10
        )
        pad_now = _pad(d)
        z_now = None if not d else d.get("pole_z")
        print(f"lift step pad={pad_now} pole_z={z_now} arrived={arrived}", flush=True)
        if not arrived:
            continue
        if pad_now is None or pad_now > 0.032:
            cmd["wrist_angle"] = _clip(float(cmd["wrist_angle"]) + 0.08, *WRIST_ANG_LIM)
            cmd["shoulder"] = _clip(float(cmd["shoulder"]) - 0.03, -1.9, 0.8)
            cmd["gripper"] = GRIP_CLOSED
            d = _hold(cmd, 2.0)
            print("lift step undone — pads left the shaft", flush=True)
            break
    cmd["gripper"] = GRIP_CLOSED
    d = _hold(cmd, 3.0)
    z1 = None if not d else d.get("pole_z")
    after = arm_of(d) if d else {}
    pad = _pad(d)
    grip = float(after.get("gripper", 1.0)) if d else 1.0
    rose = (
        z0 is not None
        and z1 is not None
        and float(z1) > float(z0) + 0.025
    )
    enclosed = pad is not None and pad <= 0.032 and grip < 0.55
    ok = bool(enclosed and rose)
    print(
        f"lift shoulder {float(before.get('shoulder', 0)):+.2f}→"
        f"{float(after.get('shoulder', 0)):+.2f} "
        f"wrist {float(before.get('wrist_angle', 0)):+.2f}→"
        f"{float(after.get('wrist_angle', 0)):+.2f}",
        flush=True,
    )
    print(
        f"grasp {'OK' if ok else 'MISS'} pad={pad} grip={grip:.2f} "
        f"pole_z {z0} → {z1} rose={rose}",
        flush=True,
    )
    return 0 if ok else 1


def _seat_and_lift(cmd: dict) -> int:
    """Walk the shoulder toward the shaft and close as soon as a grab is possible.

    Pad distance is the worse of the two jaw pads. Once it is inside the
    close-ready band, further shoulder and wrist steps push the shaft out.
    """
    # Turn the waist onto the pole before the arm reaches. Refreshing the
    # bearing while the base yaws chases the joint away from the shaft.
    aimed = _aim_waist(cmd)
    d = _hold(cmd, 6.0)
    meas = arm_of(d) if d else {}
    print(
        f"waist aim measured={float(meas.get('waist', 0)):.3f} "
        f"bearing={aimed} pad={_pad(d)}",
        flush=True,
    )
    best = _pad(d)
    best_cmd = _snap(cmd)
    print(f"seat start pad={best} pz={None if not d else d.get('pole_z')}", flush=True)
    shaft = _wide_shaft()
    if _ready(best) and _in_jaws(shaft):
        cmd["wrist_rotate"] = _clip(-float(shaft["from_vert"]), *WRIST_ROT_LIM)
        print(f"already in jaws pad={best} u={shaft['u']:.2f} v={shaft['v']:.2f}", flush=True)
        return _close_and_lift(cmd)
    _align_wrist_camera(cmd)
    d = _hold(cmd, 2.0)
    best = _pad(d)
    best_cmd = _snap(cmd)
    print(f"after align pad={best}", flush=True)
    if _ready(best):
        print(f"pads near shaft pad={best} — close", flush=True)
        return _close_and_lift(cmd)
    # Coarse reach. Stop at the first pose the pads can close on.
    sh = float(cmd["shoulder"])
    while sh < 0.45:
        sh = min(0.45, sh + 0.10)
        cmd["shoulder"] = sh
        d2 = _hold(cmd, 0.4)
        p2 = _pad(d2)
        got = arm_of(d2).get("shoulder") if d2 else None
        print(f"coarse shoulder {sh:+.2f} got={got} pad={p2}", flush=True)
        if not _base_level(d2):
            print(f"base tipped z={None if not d2 else d2.get('z')} — holding", flush=True)
            return 1
        if got is None or abs(float(got) - sh) > 0.12:
            continue
        if p2 is None:
            continue
        if best is None or p2 < best:
            best = p2
            best_cmd = _snap(cmd)
        if _ready(p2):
            print(f"pads on shaft pad={p2:.4f} — stop reach", flush=True)
            cmd.update(best_cmd)
            return _close_and_lift(cmd)
        if p2 > best + 0.008:
            break
    # Stay at the pose where the reach stopped. Snapping back to an earlier
    # shoulder pulls the jaws off the shaft.
    d = _hold(cmd, 0.4)
    settled = _pad(d)
    if settled is not None and (best is None or settled < best):
        best = settled
    if _ready(best):
        print(f"pads near shaft pad={best} — close", flush=True)
        return _close_and_lift(cmd)
    print(
        f"grasp MISS pad={best} — jaws never reached the shaft",
        flush=True,
    )
    return 1


def main() -> int:
    if cv2 is None:
        print("ERROR: cv2 missing — cannot wrist-servo", flush=True)
        return 1
    if os.environ.get("GRAB_IT_FINISH_NOW", "").strip() in ("1", "true"):
        print("wrist finish: seating now", flush=True)
        try:
            d = twin()
        except Exception as exc:
            print(f"ERROR: no twin: {exc}", flush=True)
            return 1
        arm = arm_of(d)
        for n in NAMES:
            arm.setdefault(n, 0.0)
        shaft = None
        try:
            shaft = shaft_pose(_get("/wide.jpg"))
        except Exception:
            shaft = None
        # Already seated: do not open, fold, or re-aim. A shaft in the picture
        # with the pads still a few centimetres off is not seated yet.
        if (
            _pad(d) is not None
            and float(_pad(d)) <= 0.025
            and _in_jaws(shaft)
        ):
            cmd = dict(arm)
            cmd["wrist_rotate"] = _clip(-float(shaft["from_vert"]), *WRIST_ROT_LIM)
            print(
                f"already in jaws pad={_pad(d)} u={shaft['u']:.2f} v={shaft['v']:.2f} — close in place",
                flush=True,
            )
            kill_act()
            return _close_and_lift(cmd)
        bearing = _bearing(d)
        cmd = dict(arm)
        cmd["gripper"] = GRIP_HOLD_OPEN
        cmd["pan"] = 0.78
        if os.environ.get("GRAB_IT_RESET_APPROACH", "").strip() in ("1", "true"):
            cmd["shoulder"] = -1.35
            cmd["wrist_angle"] = 1.15
        if bearing is not None:
            cmd["waist"] = bearing
        kill_act()
        return _seat_and_lift(cmd)

    print("wrist finish: waiting until the gripper stalls on the pole", flush=True)
    t0 = time.monotonic()
    while time.monotonic() - t0 < WAIT_S:
        try:
            d = twin()
        except Exception:
            time.sleep(0.15)
            continue
        if d.get("grasp_stall"):
            print(
                f"grasp stall meas={d.get('grip_meas')} cmd={d.get('grip_cmd')} "
                f"err={d.get('grip_err')} — stop ACT, hold the arm",
                flush=True,
            )
            kill_act()
            return 0
        time.sleep(0.12)
    print("wrist finish: no grasp stall — leaving ACT running", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
