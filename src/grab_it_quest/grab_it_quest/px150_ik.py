"""Analytic 4-DOF IK for px150_no_elbow (waist, shoulder, wrist_angle, wrist_rotate).

Link lengths from grab_it_isaac.urdf / Interbotix px150_no_elbow:
  waist origin Z = 0.06566
  shoulder origin = (0, 0, 0.03891) in shoulder_link
  forearm (shoulder -> wrist_angle) = 0.15
  wrist_angle -> ee_gripper_link along X = 0.065 + 0.043 + 0.023 + 0.027575
"""
from __future__ import annotations

import math
from typing import Optional

JOINT_NAMES = ("waist", "shoulder", "wrist_angle", "wrist_rotate")
JOINT_LIMITS = {
    "waist": (-2.0246, 2.0944),
    "shoulder": (-1.9722, 1.9373),
    "wrist_angle": (-2.0769, 1.4486),
    "wrist_rotate": (-1.5708, 1.5708),
}

# Shoulder pivot in px150_base_link at waist=0
SHOULDER_X = 0.0
SHOULDER_Z = 0.06566 + 0.03891
L1 = 0.15
L2 = 0.065 + 0.043 + 0.023 + 0.027575  # 0.158575


def clamp(name: str, value: float) -> float:
    lo, hi = JOINT_LIMITS[name]
    return max(lo, min(hi, float(value)))


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def fk_xyz(waist: float, shoulder: float, wrist_angle: float) -> tuple[float, float, float]:
    """EE position in px150_base_link (wrist_rotate does not move the origin)."""
    c = math.cos(waist)
    s = math.sin(waist)
    r = L1 * math.cos(shoulder) + L2 * math.cos(shoulder + wrist_angle)
    z = -L1 * math.sin(shoulder) - L2 * math.sin(shoulder + wrist_angle)
    x = c * SHOULDER_X + c * r
    y = s * SHOULDER_X + s * r
    return x, y, SHOULDER_Z + z


def _in_limits(q: dict[str, float], extra_shoulder_min: Optional[float] = None) -> bool:
    for name in JOINT_NAMES:
        lo, hi = JOINT_LIMITS[name]
        if name == "shoulder" and extra_shoulder_min is not None:
            lo = max(lo, extra_shoulder_min)
        if not (lo - 1e-6 <= q[name] <= hi + 1e-6):
            return False
    return True


def ik_position(
    x: float,
    y: float,
    z: float,
    wrist_rotate: float,
    seed: Optional[dict[str, float]] = None,
    waist_offset: float = 0.0,
    shoulder_min: Optional[float] = None,
) -> Optional[dict[str, float]]:
    """Position-only IK. wrist_rotate is passed through (clamped)."""
    seed = seed or {n: 0.0 for n in JOINT_NAMES}
    waist = _wrap_pi(math.atan2(y, x) + float(waist_offset))
    c = math.cos(waist)
    s = math.sin(waist)
    rel_x = x - c * SHOULDER_X
    rel_y = y - s * SHOULDER_X
    r = c * rel_x + s * rel_y
    zz = z - SHOULDER_Z
    # Standard 2R on (r, -zz) so Ry convention matches FK
    px = r
    py = -zz
    d2 = px * px + py * py
    reach = L1 + L2
    if d2 < 1e-10 or d2 > (reach * 1.02) ** 2:
        return None
    cos_w = (d2 - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    cos_w = max(-1.0, min(1.0, cos_w))
    wr = float(clamp("wrist_rotate", wrist_rotate))
    candidates: list[dict[str, float]] = []
    for sign in (1.0, -1.0):
        w_ang = sign * math.acos(cos_w)
        k1 = L1 + L2 * math.cos(w_ang)
        k2 = L2 * math.sin(w_ang)
        sh = math.atan2(py, px) - math.atan2(k2, k1)
        q = {
            "waist": clamp("waist", waist),
            "shoulder": sh,
            "wrist_angle": w_ang,
            "wrist_rotate": wr,
        }
        if _in_limits(q, extra_shoulder_min=shoulder_min):
            q["shoulder"] = clamp("shoulder", q["shoulder"])
            q["wrist_angle"] = clamp("wrist_angle", q["wrist_angle"])
            candidates.append(q)
    if not candidates:
        return None

    def _dist(q: dict[str, float]) -> float:
        return sum((q[n] - float(seed.get(n, 0.0))) ** 2 for n in JOINT_NAMES)

    return min(candidates, key=_dist)


if __name__ == "__main__":
    home = {"waist": 0.0, "shoulder": 0.1745, "wrist_angle": -1.309, "wrist_rotate": 0.0}
    x, y, z = fk_xyz(home["waist"], home["shoulder"], home["wrist_angle"])
    sol = ik_position(x, y, z, 0.2, seed=home)
    assert sol is not None, "home IK failed"
    fx, fy, fz = fk_xyz(sol["waist"], sol["shoulder"], sol["wrist_angle"])
    err = math.sqrt((fx - x) ** 2 + (fy - y) ** 2 + (fz - z) ** 2)
    assert err < 2e-3, err
    assert abs(sol["wrist_rotate"] - 0.2) < 1e-6
    # tabletop-ish point
    sol2 = ik_position(0.28, 0.05, 0.18, 0.0, seed=home)
    assert sol2 is not None, "tabletop IK failed"
    fx2, fy2, fz2 = fk_xyz(sol2["waist"], sol2["shoulder"], sol2["wrist_angle"])
    err2 = math.sqrt((fx2 - 0.28) ** 2 + (fy2 - 0.05) ** 2 + (fz2 - 0.18) ** 2)
    assert err2 < 3e-3, err2
    assert ik_position(1.5, 0.0, 0.2, 0.0, seed=home) is None
    print("px150_ik ok", sol, sol2, "fk_err", err, "table_err", err2)
