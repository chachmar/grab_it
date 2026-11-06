#!/usr/bin/env python3
"""Independent transit depth gate for NexArm approach.

Pure evaluation helpers + ``_self_check()``. The grasp node owns the thread
that polls depth and publishes a hold on trip.
"""
from __future__ import annotations

import math
from typing import Any, Optional, Sequence, Tuple

import numpy as np

GATE_TRIP_M = 0.20
GATE_MIN_PIX = 150
GATE_HOLE_JUMP = 0.25
# Envelope past fingertips (metres, tool frame): forward, lateral, vertical.
ENV_S0 = 0.06
ENV_S1 = 0.30
ENV_LAT = 0.07
ENV_Z_LO = -0.05
ENV_Z_HI = 0.10


def _optical(
    p: Sequence[float],
    origin: Sequence[float],
    R_opt: Sequence[Sequence[float]],
) -> Optional[Tuple[float, float, float]]:
    d0 = float(p[0]) - float(origin[0])
    d1 = float(p[1]) - float(origin[1])
    d2 = float(p[2]) - float(origin[2])
    x = float(R_opt[0][0]) * d0 + float(R_opt[1][0]) * d1 + float(R_opt[2][0]) * d2
    y = float(R_opt[0][1]) * d0 + float(R_opt[1][1]) * d1 + float(R_opt[2][1]) * d2
    z = float(R_opt[0][2]) * d0 + float(R_opt[1][2]) * d1 + float(R_opt[2][2]) * d2
    if z <= 1e-4:
        return None
    return (x, y, z)


def project_envelope_roi(
    *,
    tip: Sequence[float],
    axis: Sequence[float],
    origin: Sequence[float],
    R_opt: Sequence[Sequence[float]],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    s0: float = ENV_S0,
    s1: float = ENV_S1,
    lat: float = ENV_LAT,
    z_lo: float = ENV_Z_LO,
    z_hi: float = ENV_Z_HI,
) -> Optional[Tuple[int, int, int, int]]:
    """Project the gripper forward envelope into an image AABB.

    Returns (u0, v0, u1, v1) inclusive-exclusive pixel box, or None on failure.
    Fails when the tip is far outside the image (wrist cam looks past the
    jaws) or the projected box covers most of the frame — callers then use
    ``fallback_roi``.
    """
    tip_o = _optical(tip, origin, R_opt)
    if tip_o is None or tip_o[2] < 0.04:
        return None
    tip_u = float(fx) * tip_o[0] / tip_o[2] + float(cx)
    tip_v = float(fy) * tip_o[1] / tip_o[2] + float(cy)
    # Tip well outside the frame → envelope AABB is meaningless.
    if (
        tip_u < -0.25 * float(width)
        or tip_u > 1.25 * float(width)
        or tip_v < -0.25 * float(height)
        or tip_v > 1.25 * float(height)
    ):
        return None

    ax = np.array(
        [float(axis[0]), float(axis[1]), float(axis[2])], dtype=np.float64
    )
    n = float(np.linalg.norm(ax))
    if n < 1e-9:
        return None
    ax = ax / n
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    lat_v = np.cross(up, ax)
    ln = float(np.linalg.norm(lat_v))
    if ln < 1e-6:
        lat_v = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        lat_v = lat_v / ln
    vert = np.cross(ax, lat_v)
    vn = float(np.linalg.norm(vert))
    if vn < 1e-6:
        vert = up
    else:
        vert = vert / vn
    tip_a = np.array(
        [float(tip[0]), float(tip[1]), float(tip[2])], dtype=np.float64
    )
    us, vs = [], []
    for s in (float(s0), float(s1)):
        for ly in (-float(lat), float(lat)):
            for vz in (float(z_lo), float(z_hi)):
                p = tip_a + s * ax + ly * lat_v + vz * vert
                opt = _optical(p, origin, R_opt)
                if opt is None or opt[2] < 0.06:
                    continue
                us.append(float(fx) * opt[0] / opt[2] + float(cx))
                vs.append(float(fy) * opt[1] / opt[2] + float(cy))
    if len(us) < 3:
        return None
    u0 = int(max(0, math.floor(min(us))))
    u1 = int(min(int(width), math.ceil(max(us)) + 1))
    v0 = int(max(0, math.floor(min(vs))))
    v1 = int(min(int(height), math.ceil(max(vs)) + 1))
    if u1 - u0 < 8 or v1 - v0 < 8:
        return None
    area = float(u1 - u0) * float(v1 - v0)
    if area > 0.55 * float(width) * float(height):
        return None
    return (u0, v0, u1, v1)


def fallback_roi(width: int, height: int) -> Tuple[int, int, int, int]:
    """Lower-central band when envelope projection fails (tip often out of frame)."""
    w, h = int(width), int(height)
    return (
        int(0.20 * w),
        int(0.20 * h),
        int(0.80 * w),
        int(0.75 * h),
    )


def support_z_floor(
    cloud_z: Sequence[float], margin_m: float = 0.01
) -> float:
    """Lowest points of the click cloud, less margin — ignore table pixels."""
    if cloud_z is None or len(cloud_z) == 0:
        return float("-inf")
    arr = np.asarray(cloud_z, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("-inf")
    return float(np.percentile(arr, 5)) - float(margin_m)


def evaluate_frame(
    depth: np.ndarray,
    roi: Tuple[int, int, int, int],
    *,
    trip_m: float = GATE_TRIP_M,
    min_pix: int = GATE_MIN_PIX,
    hole_jump: float = GATE_HOLE_JUMP,
    baseline_invalid_frac: Optional[float] = None,
    support_z: float = float("-inf"),
    origin: Optional[Sequence[float]] = None,
    R_opt: Optional[Sequence[Sequence[float]]] = None,
    fx: float = 600.0,
    fy: float = 600.0,
    cx: float = 320.0,
    cy: float = 240.0,
    ignore_s_near: Optional[float] = None,
    ignore_u_hat: Optional[Sequence[float]] = None,
    ignore_width_m: float = 0.10,
    ignore_s_min: Optional[float] = None,
) -> dict[str, Any]:
    """Evaluate one depth frame inside the ROI.

    Returns dict with keys: trip (bool), reason (str|None), n_close, invalid_frac,
    near_z (optional live near-face estimate along look).

    When ``ignore_s_near`` / ``ignore_u_hat`` are set, points on the planned
    object column (s >= s_near - 4 cm within the width band) do not count as
    close obstacles — otherwise the target itself trips the gate on approach.
    ``ignore_s_min`` drops points at/behind the current tip (gripper self).
    """
    u0, v0, u1, v1 = roi
    h, w = depth.shape[:2]
    u0 = max(0, min(w - 1, int(u0)))
    u1 = max(u0 + 1, min(w, int(u1)))
    v0 = max(0, min(h - 1, int(v0)))
    v1 = max(v0 + 1, min(h, int(v1)))
    sub = np.asarray(depth[v0:v1, u0:u1], dtype=np.float64)
    total = int(sub.size)
    if total <= 0:
        return {
            "trip": False,
            "reason": None,
            "n_close": 0,
            "invalid_frac": 1.0,
            "near_z": None,
        }
    finite = np.isfinite(sub)
    invalid_frac = float((~finite).sum()) / float(total)
    # Trip band: ignore the stereo near-field / gripper self (<12 cm) and
    # anything beyond gate_trip_m. Drop support-surface pixels via FK.
    z_self = 0.12
    valid = finite & (sub > z_self) & (sub < float(trip_m))
    n_close = 0
    near_z = None
    use_xyz = origin is not None and R_opt is not None
    if use_xyz:
        ys, xs = np.where(valid)
        keep = 0
        zs_keep = []
        ux = uy = None
        s_allow = None
        if ignore_s_near is not None and ignore_u_hat is not None:
            ux, uy = float(ignore_u_hat[0]), float(ignore_u_hat[1])
            s_allow = float(ignore_s_near) - 0.04
        for yy, xx in zip(ys, xs):
            z = float(sub[yy, xx])
            u = u0 + int(xx)
            v = v0 + int(yy)
            x = (u - float(cx)) * z / float(fx)
            y = (v - float(cy)) * z / float(fy)
            bx = (
                float(origin[0])
                + float(R_opt[0][0]) * x
                + float(R_opt[0][1]) * y
                + float(R_opt[0][2]) * z
            )
            by = (
                float(origin[1])
                + float(R_opt[1][0]) * x
                + float(R_opt[1][1]) * y
                + float(R_opt[1][2]) * z
            )
            bz = (
                float(origin[2])
                + float(R_opt[2][0]) * x
                + float(R_opt[2][1]) * y
                + float(R_opt[2][2]) * z
            )
            if float(support_z) > -1e8 and bz < float(support_z):
                continue
            if ux is not None:
                s = bx * ux + by * uy
                if ignore_s_min is not None and s < float(ignore_s_min):
                    continue
                if s_allow is not None:
                    lat = abs(-uy * bx + ux * by)
                    if s >= s_allow and lat <= float(ignore_width_m):
                        continue
            keep += 1
            zs_keep.append(z)
        n_close = keep
        if zs_keep:
            near_z = float(np.percentile(np.asarray(zs_keep), 10))
    else:
        n_close = int(valid.sum())
        if n_close > 0:
            near_z = float(np.percentile(sub[valid], 10))

    reason = None
    trip = False
    if n_close >= int(min_pix):
        trip = True
        reason = f"close_pix n={n_close} <{trip_m:.2f}m"
    elif baseline_invalid_frac is not None:
        jump = invalid_frac - float(baseline_invalid_frac)
        if jump >= float(hole_jump):
            trip = True
            reason = (
                f"holes frac={invalid_frac:.2f} "
                f"(base={float(baseline_invalid_frac):.2f}, +{jump:.2f})"
            )
    return {
        "trip": trip,
        "reason": reason,
        "n_close": n_close,
        "invalid_frac": invalid_frac,
        "near_z": near_z,
    }


class SafetyGateState:
    """Consecutive-frame latch for the transit gate."""

    def __init__(
        self,
        *,
        trip_m: float = GATE_TRIP_M,
        min_pix: int = GATE_MIN_PIX,
        hole_jump: float = GATE_HOLE_JUMP,
    ) -> None:
        self.trip_m = float(trip_m)
        self.min_pix = int(min_pix)
        self.hole_jump = float(hole_jump)
        self.baseline_invalid: Optional[float] = None
        self.baseline_n_close: int = 0
        self.baseline_near_z: Optional[float] = None
        self.support_z = float("-inf")
        self.ignore_s_near: Optional[float] = None
        self.ignore_u_hat: Optional[Tuple[float, float]] = None
        self.ignore_s_min: Optional[float] = None
        self.roi: Optional[Tuple[int, int, int, int]] = None
        self._pending: Optional[str] = None
        self.tripped = False
        self.reason: Optional[str] = None
        self.last: Optional[dict[str, Any]] = None

    def arm(
        self,
        *,
        depth: np.ndarray,
        roi: Tuple[int, int, int, int],
        support_z: float = float("-inf"),
        origin=None,
        R_opt=None,
        fx: float = 600.0,
        fy: float = 600.0,
        cx: float = 320.0,
        cy: float = 240.0,
        ignore_s_near: Optional[float] = None,
        ignore_u_hat: Optional[Sequence[float]] = None,
        ignore_s_min: Optional[float] = None,
    ) -> None:
        self.roi = roi
        self.support_z = float(support_z)
        self.ignore_s_near = (
            float(ignore_s_near) if ignore_s_near is not None else None
        )
        self.ignore_u_hat = (
            (float(ignore_u_hat[0]), float(ignore_u_hat[1]))
            if ignore_u_hat is not None
            else None
        )
        self.ignore_s_min = (
            float(ignore_s_min) if ignore_s_min is not None else None
        )
        self.tripped = False
        self.reason = None
        self._pending = None
        ev = evaluate_frame(
            depth,
            roi,
            trip_m=self.trip_m,
            min_pix=self.min_pix,
            hole_jump=self.hole_jump,
            baseline_invalid_frac=None,
            support_z=self.support_z,
            origin=origin,
            R_opt=R_opt,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            ignore_s_near=self.ignore_s_near,
            ignore_u_hat=self.ignore_u_hat,
            ignore_s_min=self.ignore_s_min,
        )
        self.baseline_invalid = float(ev["invalid_frac"])
        self.baseline_n_close = int(ev["n_close"])
        self.baseline_near_z = (
            float(ev["near_z"]) if ev.get("near_z") is not None else None
        )
        self.last = ev

    def set_tip_s_min(self, tip_s: float, margin_m: float = 0.05) -> None:
        """Ignore points at/behind the live fingertip along the approach axis."""
        self.ignore_s_min = float(tip_s) + float(margin_m)

    def update(
        self,
        depth: np.ndarray,
        *,
        origin=None,
        R_opt=None,
        fx: float = 600.0,
        fy: float = 600.0,
        cx: float = 320.0,
        cy: float = 240.0,
    ) -> Optional[str]:
        if self.tripped:
            return self.reason
        if self.roi is None:
            return None
        ev = evaluate_frame(
            depth,
            self.roi,
            trip_m=self.trip_m,
            min_pix=self.min_pix,
            hole_jump=self.hole_jump,
            baseline_invalid_frac=self.baseline_invalid,
            support_z=self.support_z,
            origin=origin,
            R_opt=R_opt,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            ignore_s_near=self.ignore_s_near,
            ignore_u_hat=self.ignore_u_hat,
            ignore_s_min=self.ignore_s_min,
        )
        self.last = ev
        reason = None
        kind = None
        # Close trip only on NEW near content vs the arm-time scene (static
        # clutter already in the ROI must not fire on the first frames).
        n_close = int(ev["n_close"])
        near_z = ev.get("near_z")
        new_close = n_close >= int(self.min_pix) and (
            n_close >= int(self.baseline_n_close) + max(80, int(self.min_pix) // 2)
            or (
                near_z is not None
                and self.baseline_near_z is not None
                and float(near_z) < float(self.baseline_near_z) - 0.03
                and n_close >= int(self.min_pix)
            )
        )
        if new_close:
            reason = (
                f"close_pix n={n_close} <{self.trip_m:.2f}m "
                f"(base_n={self.baseline_n_close})"
            )
            kind = "close"
        elif ev["trip"] and "holes" in str(ev.get("reason")):
            reason = str(ev["reason"])
            kind = "holes"
        if reason is None:
            self._pending = None
            return None
        if self._pending == kind:
            self.tripped = True
            self.reason = reason
            return self.reason
        self._pending = kind
        return None


def _self_check() -> None:
    h, w = 48, 64
    # Baseline: far scene.
    far = np.full((h, w), 0.80, dtype=np.float64)
    roi = (8, 8, 56, 40)
    st = SafetyGateState(trip_m=0.20, min_pix=40, hole_jump=0.25)
    st.arm(depth=far, roi=roi)
    assert abs(float(st.baseline_invalid) - 0.0) < 1e-9
    # Still far — no trip.
    assert st.update(far) is None
    assert st.update(far) is None
    # New close content vs a clear baseline.
    close = np.full((h, w), 0.15, dtype=np.float64)
    assert st.update(close) is None  # first frame pending
    why = st.update(close)
    assert why is not None and "close_pix" in why, why
    assert st.tripped
    # Same scene as arm baseline must not trip.
    st3 = SafetyGateState(trip_m=0.20, min_pix=40, hole_jump=0.25)
    st3.arm(depth=close, roi=roi)
    assert st3.update(close) is None
    assert st3.update(close) is None
    assert not st3.tripped

    # Holes: invalidate most of the ROI above baseline.
    st2 = SafetyGateState(trip_m=0.20, min_pix=9999, hole_jump=0.25)
    st2.arm(depth=far, roi=roi)
    holes = far.copy()
    holes[8:40, 8:56] = np.nan
    assert st2.update(holes) is None
    why2 = st2.update(holes)
    assert why2 is not None and "holes" in why2, why2

    # Envelope projection with a simple look-along-+X camera at origin.
    R = ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))  # look +X
    # Actually optical: columns = right, down, look. Put look = +X.
    R = ((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0))
    tip = (0.20, 0.0, 0.10)
    axis = (1.0, 0.0, 0.0)
    box = project_envelope_roi(
        tip=tip,
        axis=axis,
        origin=(0.0, 0.0, 0.15),
        R_opt=R,
        fx=300.0,
        fy=300.0,
        cx=320.0,
        cy=240.0,
        width=640,
        height=480,
    )
    assert box is not None, box
    u0, v0, u1, v1 = box
    assert 0 <= u0 < u1 <= 640
    assert 0 <= v0 < v1 <= 480
    print("nexarm_safety_gate self-check OK")


if __name__ == "__main__":
    _self_check()
