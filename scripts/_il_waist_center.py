#!/usr/bin/env python3
"""Aim the waist at the foam pole and fold into the shoulder-demo start.

The D435i is fixed to the base pan joint. Yawing the waist does not move the
pole in the image, so this is not a closed visual loop. The aim is the pole's
bearing in the arm base, atan2(pole_y, pole_x), from the live twin measurement.

A camera column is used only when that measurement is missing. The look-down
view often contains a tall pale pillar and not the foam shaft, and a linear
pixel-to-angle gain plus the ACT training clamp (±0.16 rad) was reporting
"centered" while the arm was still ~20° off the pole.

Pan is tipped down so a later ACT frame can see the workspace. Shoulder and
wrist fold to the demo park. clear_gate drops a latched IL close-gate.
"""
from __future__ import annotations

import json
import math
import sys
import time
import urllib.request

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

HUB = "http://127.0.0.1:8081"
# Look-down saturates near the pan stop (~45 deg). 1.10 never arrives.
PAN_DOWN = 0.78
# Image column of a pole on the arm's forward axis (measured at pan≈0.78).
U0 = 0.565
# f_px / width. focal 24 mm, horizontal aperture 33.3 mm. u is 0..1.
FX_NORM = 24.0 / 33.3 
# Camera is aft of the arm base (nano mount behind, arm mount ahead).
CAM_AFT_M = 0.20
# URDF waist is about ±2.0 rad. The ACT dataset box (±0.16) is not an aim limit.
WAIST_MECH = 1.80
# Folded start the shoulder policy was trained from.
SHOULDER_PARK = -1.90
WRIST_PARK = 1.40
WRIST_ROT_PARK = 0.0
TOL = 0.06
FOLD_TOL = 0.08
TIMEOUT_S = 30.0
NAMES = ["waist", "shoulder", "wrist_angle", "wrist_rotate", "gripper", "pan"]


def _get(path: str) -> bytes:
    return urllib.request.urlopen(HUB + path, timeout=2.0).read()


def twin() -> dict:
    return json.loads(_get("/isaac/twin").decode())


def arm_of(d: dict) -> dict:
    names = list(d.get("arm_n") or [])
    pos = [float(x) for x in (d.get("arm_p") or [])]
    return {n: pos[i] for i, n in enumerate(names) if i < len(pos)}


def _clamp_waist(angle: float) -> float:
    return max(-WAIST_MECH, min(WAIST_MECH, float(angle)))


def geometric_waist(d: dict) -> float | None:
    """Arm-base bearing of the pole. None if it is not in front of the arm."""
    xy = d.get("pole_xy")
    if not xy or len(xy) < 2:
        return None
    wx, wy = float(xy[0]), float(xy[1])
    if wx < 0.08:
        return None
    return _clamp_waist(math.atan2(wy, wx))


def pole_column_u(jpeg: bytes) -> float | None:
    """Normalized u of a narrow foam shaft. None if it is not in frame.

    A full-height pale column is usually a pillar or window mullion. The foam
    shaft, under look-down, is a compact vertical blob in the lower image and
    not glued to the left or right edge.
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
    # Foam reads cyan-gray. The carpet is orange and must not count.
    pole = (hh > 85) & (hh < 130) & (s > 25) & (s < 140) & (v > 70) & (v < 230)
    band = pole[int(0.25 * h) :, :]
    col = band.sum(axis=0).astype(np.float64)
    if col.size < 5:
        return None
    # Drop the outer columns. Pillars sit there; a reachable pole does not.
    edge = max(2, int(0.08 * w))
    col[:edge] = 0.0
    col[-edge:] = 0.0
    need = 0.35 * float(band.shape[0])
    if float(col.max()) < need:
        return None
    kernel = np.ones(5, dtype=np.float64) / 5.0
    smooth = np.convolve(col, kernel, mode="same")
    peak = float(smooth.max())
    xs = np.where(smooth >= 0.60 * peak)[0]
    if xs.size == 0:
        return None
    # A shaft is narrow. A wall or window spans a large fraction of the width.
    if float(xs.max() - xs.min()) > 0.12 * w:
        return None
    return float(xs.min() + xs.max()) * 0.5 / float(w)


def visual_waist(u: float, x_forward: float) -> float:
    """Camera-column → arm waist. +u is image-right, which is -Y (waist < 0)."""
    # Pinhole bearing. A linear gain under-reads once the pole leaves the center.
    alpha = math.atan((U0 - float(u)) / FX_NORM)
    x = max(0.12, float(x_forward))
    y = (x + CAM_AFT_M) * math.tan(alpha)
    return _clamp_waist(math.atan2(y, x))


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


def main() -> int:
    if cv2 is None:
        print("ERROR: cv2 missing — cannot visual-servo waist", flush=True)
        return 1
    d0 = None
    for _ in range(20):
        try:
            d0 = twin()
            if d0.get("arm_p"):
                break
        except Exception:
            time.sleep(0.2)
    if not d0 or not d0.get("arm_p"):
        print("ERROR: no twin arm", flush=True)
        return 1
    hold = arm_of(d0)
    for n in NAMES:
        hold.setdefault(n, 0.0)
    t0 = time.monotonic()
    target = None
    source = "none"
    last_u = None
    while time.monotonic() - t0 < TIMEOUT_S:
        try:
            d = twin()
        except Exception:
            time.sleep(0.1)
            continue
        geo = geometric_waist(d)
        if geo is not None:
            # Twin pole_xy is the stage measurement. The camera cannot yaw, so
            # a pixel column must not override a bearing that is already known.
            target = geo
            source = "bearing"
        else:
            try:
                jpeg = _get("/d435i.jpg")
            except Exception:
                jpeg = b""
            u = pole_column_u(jpeg)
            xy = d.get("pole_xy") or [0.24, 0.0]
            x_forward = float(xy[0]) if xy and float(xy[0]) > 0.08 else 0.24
            if u is not None:
                target = visual_waist(u, x_forward)
                source = "visual"
                last_u = u
        if target is None:
            time.sleep(0.1)
            continue
        pose = dict(hold)
        pose["waist"] = float(target)
        pose["shoulder"] = SHOULDER_PARK
        pose["wrist_angle"] = WRIST_PARK
        pose["wrist_rotate"] = WRIST_ROT_PARK
        pose["pan"] = PAN_DOWN
        try:
            post(pose)
        except Exception as exc:
            print(f"post fail: {exc}", flush=True)
        got_arm = arm_of(d)
        got = float(got_arm.get("waist", 9.0))
        sh = float(got_arm.get("shoulder", 9.0))
        wr = float(got_arm.get("wrist_angle", 9.0))
        if (
            abs(got - float(target)) <= TOL
            and abs(sh - SHOULDER_PARK) <= FOLD_TOL
            and abs(wr - WRIST_PARK) <= FOLD_TOL
        ):
            xy = d.get("pole_xy") or [None, None]
            print(
                f"waist centered source={source} "
                f"pole_xy=({xy[0]},{xy[1]}) "
                f"target={float(target):.3f} waist={got:.3f} "
                f"shoulder={sh:.3f} wrist={wr:.3f}",
                flush=True,
            )
            return 0
        time.sleep(0.1)
    got = None
    try:
        got = float(arm_of(twin()).get("waist", 9.0))
    except Exception:
        pass
    print(
        f"ERROR: arm not at aimed fold source={source} u={last_u} "
        f"target={target} waist={got}",
        flush=True,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
