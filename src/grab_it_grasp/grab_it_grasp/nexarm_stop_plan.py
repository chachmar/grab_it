#!/usr/bin/env python3
"""Click-time stop-pose planner for NexArm approach-only grasps.

Builds a fingertip stop pose from the click-time depth cloud (near face +
height), solves a full IK line from pre to stop, and checks the whole path
against the click-time depth cloud. Pure functions + ``_self_check()``.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Optional, Sequence

import numpy as np

# Defaults match grasp_params.yaml / plan.
TCP_M = 0.025
FINGERTIP_M = 0.046
STOP_GAP_M = 0.03
PRE_BACK_M = 0.15
CORRIDOR_MARGIN_M = 0.015
JOINT_JUMP_MAX = 0.30
GRIPPER_HALF_W = 0.052
DEPTH_BAND_M = 0.05
OBJ_WINDOW_HALF_PX = 18
CLOUD_STRIDE = 4


def unproject_pixel(
    u_px: float,
    v_px: float,
    z: float,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    origin: Sequence[float],
    R_opt: Sequence[Sequence[float]],
) -> tuple[float, float, float]:
    """Optical (u,v,z) → base-frame XYZ using FK camera pose."""
    x = (float(u_px) - float(cx)) * float(z) / float(fx)
    y = (float(v_px) - float(cy)) * float(z) / float(fy)
    ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])
    # R columns = optical axes in base → base = origin + R @ optical.
    bx = ox + float(R_opt[0][0]) * x + float(R_opt[0][1]) * y + float(R_opt[0][2]) * z
    by = oy + float(R_opt[1][0]) * x + float(R_opt[1][1]) * y + float(R_opt[1][2]) * z
    bz = oz + float(R_opt[2][0]) * x + float(R_opt[2][1]) * y + float(R_opt[2][2]) * z
    return (bx, by, bz)


def object_points_from_click(
    depth: np.ndarray,
    u_px: int,
    v_px: int,
    click_z: float,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    origin: Sequence[float],
    R_opt: Sequence[Sequence[float]],
    band_m: float = DEPTH_BAND_M,
    half_px: int = OBJ_WINDOW_HALF_PX,
    z_min: float = 0.08,
    z_max: float = 2.0,
) -> list[tuple[float, float, float]]:
    """Window around the click, keep pixels within ±band of click depth."""
    h, w = depth.shape[:2]
    half = max(2, int(half_px))
    y0, y1 = max(0, int(v_px) - half), min(h, int(v_px) + half + 1)
    x0, x1 = max(0, int(u_px) - half), min(w, int(u_px) + half + 1)
    pts: list[tuple[float, float, float]] = []
    lo = float(click_z) - float(band_m)
    hi = float(click_z) + float(band_m)
    for vv in range(y0, y1):
        for uu in range(x0, x1):
            z = float(depth[vv, uu])
            if not math.isfinite(z) or z < z_min or z > z_max:
                continue
            if z < lo or z > hi:
                continue
            pts.append(
                unproject_pixel(
                    uu, vv, z, fx=fx, fy=fy, cx=cx, cy=cy, origin=origin, R_opt=R_opt
                )
            )
    return pts


def near_face_and_height(
    pts: Sequence[Sequence[float]],
    click_xy: Sequence[float],
) -> tuple[float, float, float, tuple[float, float]]:
    """Return (s_near, z_lo, z_hi, u_hat) from object points.

    ``u_hat`` is the horizontal unit vector from the arm base toward the click.
    ``s_near`` is the 5th percentile of ``p · u_hat``.
    """
    hx, hy = float(click_xy[0]), float(click_xy[1])
    hn = math.hypot(hx, hy)
    if hn < 1e-6:
        u_hat = (1.0, 0.0)
    else:
        u_hat = (hx / hn, hy / hn)
    if not pts:
        s_click = hx * u_hat[0] + hy * u_hat[1]
        return float(s_click), float("nan"), float("nan"), u_hat
    dots = np.array(
        [float(p[0]) * u_hat[0] + float(p[1]) * u_hat[1] for p in pts],
        dtype=np.float64,
    )
    zs = np.array([float(p[2]) for p in pts], dtype=np.float64)
    s_near = float(np.percentile(dots, 5))
    z_lo = float(np.percentile(zs, 5))
    z_hi = float(np.percentile(zs, 95))
    return s_near, z_lo, z_hi, u_hat


def stop_and_pre_poses(
    *,
    s_near: float,
    z_lo: float,
    z_hi: float,
    click_z: float,
    u_hat: Sequence[float],
    gap_m: float = STOP_GAP_M,
    pre_back_m: float = PRE_BACK_M,
    tcp_m: float = TCP_M,
    fingertip_m: float = FINGERTIP_M,
) -> dict[str, Any]:
    """Build stop / pre TCP poses (pitch 0, approach along u_hat)."""
    z_app = float(np.clip(float(click_z), float(z_lo) + 0.02, float(z_hi) - 0.02))
    # Degenerate height band: fall back to click_z.
    if not math.isfinite(z_lo) or not math.isfinite(z_hi) or (z_hi - z_lo) < 0.03:
        z_app = float(click_z)
    s_tip = float(s_near) - float(gap_m)
    s_tcp = s_tip - (float(fingertip_m) - float(tcp_m))
    s_pre = s_tcp - float(pre_back_m)
    ux, uy = float(u_hat[0]), float(u_hat[1])

    def _xy(s: float) -> tuple[float, float, float]:
        return (s * ux, s * uy, z_app)

    stop = _xy(s_tcp)
    pre = _xy(s_pre)
    tip = _xy(s_tip)
    return {
        "z_app": z_app,
        "s_near": float(s_near),
        "s_tip": s_tip,
        "s_tcp": s_tcp,
        "s_pre": s_pre,
        "u_hat": (ux, uy),
        "stop_xyz": stop,
        "pre_xyz": pre,
        "tip_xyz": tip,
        "gap_m": float(gap_m),
        "pre_back_m": float(pre_back_m),
        "z_lo": float(z_lo),
        "z_hi": float(z_hi),
    }


def _joint_jump(a: Sequence[float], b: Sequence[float]) -> float:
    return max(abs(float(x) - float(y)) for x, y in zip(a, b))


def solve_line_ik(
    pre_xyz: Sequence[float],
    stop_xyz: Sequence[float],
    *,
    solve_ik: Callable[..., Optional[list[float]]],
    seed: Sequence[float],
    step_m: float = 0.01,
    max_jump: float = JOINT_JUMP_MAX,
    z_alts: Sequence[float] = (0.0, 0.02, -0.02),
    z_lo: float = float("-inf"),
    z_hi: float = float("inf"),
) -> tuple[Optional[dict[str, Any]], str]:
    """IK pre + every step_m along the straight line. Full success required.

    ``solve_ik(xyz, seed) -> joint list | None``.
    Tries z offsets in ``z_alts`` (clamped to [z_lo, z_hi]) if the nominal
    height has no solution.
    """
    for dz in z_alts:
        z = float(pre_xyz[2]) + float(dz)
        if math.isfinite(z_lo) and math.isfinite(z_hi):
            if z < float(z_lo) + 0.01 or z > float(z_hi) - 0.01:
                if abs(float(dz)) > 1e-9:
                    continue
        pre = (float(pre_xyz[0]), float(pre_xyz[1]), z)
        stop = (float(stop_xyz[0]), float(stop_xyz[1]), z)
        q_pre = solve_ik(pre, list(seed))
        if q_pre is None:
            continue
        span = math.dist(pre, stop)
        n_step = max(1, int(math.ceil(max(span, 1e-3) / max(float(step_m), 0.005))))
        waypoints: list[list[float]] = [list(q_pre)]
        cart: list[tuple[float, float, float]] = [pre]
        seed_q = list(q_pre)
        ok = True
        for i in range(1, n_step + 1):
            a = i / float(n_step)
            xyz = (
                float(pre[0] + a * (stop[0] - pre[0])),
                float(pre[1] + a * (stop[1] - pre[1])),
                float(pre[2] + a * (stop[2] - pre[2])),
            )
            solved = solve_ik(xyz, seed_q)
            if solved is None:
                ok = False
                break
            if _joint_jump(seed_q, solved) > float(max_jump):
                ok = False
                break
            waypoints.append(list(solved))
            cart.append(xyz)
            seed_q = list(solved)
        if ok and len(waypoints) >= 2:
            return {
                "q_pre": list(q_pre),
                "q_stop": list(waypoints[-1]),
                "waypoints": waypoints,
                "cart": cart,
                "pre_xyz": pre,
                "stop_xyz": stop,
                "n_step": n_step,
            }, "ok"
    return None, "unreachable at object height"


def depth_cloud_base(
    depth: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    origin: Sequence[float],
    R_opt: Sequence[Sequence[float]],
    stride: int = CLOUD_STRIDE,
    z_min: float = 0.08,
    z_max: float = 2.5,
) -> np.ndarray:
    """Unproject every Nth depth pixel to base frame. Shape (N, 3)."""
    h, w = depth.shape[:2]
    pts: list[tuple[float, float, float]] = []
    s = max(1, int(stride))
    for vv in range(0, h, s):
        row = depth[vv]
        for uu in range(0, w, s):
            z = float(row[uu])
            if not math.isfinite(z) or z < z_min or z > z_max:
                continue
            pts.append(
                unproject_pixel(
                    uu, vv, z, fx=fx, fy=fy, cx=cx, cy=cy, origin=origin, R_opt=R_opt
                )
            )
    if not pts:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(pts, dtype=np.float64)


def body_points_at(
    q_model: dict,
    pan: float,
    *,
    ik_mod: Any,
    tcp_m: float = TCP_M,
    fingertip_m: float = FINGERTIP_M,
    half_w: float = GRIPPER_HALF_W,
) -> list[tuple[float, float, float]]:
    """Fingertips, gripper body corners, camera housing."""
    tip = ik_mod.fk_xyz_q(q_model, tool=float(tcp_m))
    axis = ik_mod.tool_axis_q(q_model)
    # Lateral = cross(world_up, axis) roughly; fall back to world Y.
    ux, uy, uz = float(axis[0]), float(axis[1]), float(axis[2])
    lx, ly, lz = -uy, ux, 0.0
    ln = math.hypot(lx, ly)
    if ln < 1e-6:
        lx, ly, lz = 0.0, 1.0, 0.0
    else:
        lx, ly = lx / ln, ly / ln
    tip_ext = float(fingertip_m) - float(tcp_m)
    tip_pt = (
        float(tip[0]) + tip_ext * ux,
        float(tip[1]) + tip_ext * uy,
        float(tip[2]) + tip_ext * uz,
    )
    pts = [tip_pt]
    for sign in (-1.0, 1.0):
        pts.append(
            (
                tip_pt[0] + sign * float(half_w) * lx,
                tip_pt[1] + sign * float(half_w) * ly,
                tip_pt[2] + sign * float(half_w) * lz,
            )
        )
        # Gripper body slightly behind fingertips.
        pts.append(
            (
                float(tip[0]) + sign * float(half_w) * lx,
                float(tip[1]) + sign * float(half_w) * ly,
                float(tip[2]) + sign * float(half_w) * lz,
            )
        )
    pts.extend(ik_mod.camera_body_points_q(q_model, float(pan)))
    return pts


def clear_path(
    cloud: np.ndarray,
    samples_body: Sequence[Sequence[Sequence[float]]],
    *,
    s_near: float,
    u_hat: Sequence[float],
    margin_m: float = CORRIDOR_MARGIN_M,
    width_band_m: float = 0.08,
) -> tuple[bool, str]:
    """True if clear. Target face (at/past s_near-1cm inside width band) allowed."""
    if cloud is None or len(cloud) == 0:
        return True, "empty cloud"
    ux, uy = float(u_hat[0]), float(u_hat[1])
    # Lateral direction in XY.
    lx, ly = -uy, ux
    # Allow the object's own face: anything at/past the stop gap in front of
    # s_near (fingertips plan to sit gap_m short). A tight s_near-1cm band
    # false-blocked the intentional 3 cm hold against the face cloud.
    s_allow = float(s_near) - 0.04
    m2 = float(margin_m) ** 2
    for body in samples_body:
        for bp in body:
            bx, by, bz = float(bp[0]), float(bp[1]), float(bp[2])
            d = cloud - np.array([bx, by, bz], dtype=np.float64)
            d2 = np.einsum("ij,ij->i", d, d)
            close = np.where(d2 <= m2)[0]
            for idx in close:
                p = cloud[idx]
                s = float(p[0]) * ux + float(p[1]) * uy
                lat = abs(float(p[0]) * lx + float(p[1]) * ly)
                if s >= s_allow and lat <= float(width_band_m):
                    continue
                return (
                    False,
                    f"blocked near body "
                    f"({bx:.3f},{by:.3f},{bz:.3f}) "
                    f"cloud=({float(p[0]):.3f},{float(p[1]):.3f},{float(p[2]):.3f}) "
                    f"dist={math.sqrt(float(d2[idx])):.3f}",
                )
    return True, "clear"


def interpolate_q(
    q0: Sequence[float], q1: Sequence[float], n: int = 10
) -> list[list[float]]:
    out: list[list[float]] = []
    n = max(2, int(n))
    for i in range(n):
        a = i / float(n - 1)
        out.append([float(a0) + a * (float(a1) - float(a0)) for a0, a1 in zip(q0, q1)])
    return out


def build_stop_plan(
    *,
    depth: np.ndarray,
    u_px: int,
    v_px: int,
    click_z: float,
    click_xyz: Sequence[float],
    origin: Sequence[float],
    R_opt: Sequence[Sequence[float]],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    q_now: Sequence[float],
    pan: float,
    wrist_roll: float,
    solve_ik: Callable[..., Optional[list[float]]],
    q_to_model: Callable[[Sequence[float]], dict],
    ik_mod: Any,
    gap_m: float = STOP_GAP_M,
    pre_back_m: float = PRE_BACK_M,
    tcp_m: float = TCP_M,
    fingertip_m: float = FINGERTIP_M,
    corridor_margin_m: float = CORRIDOR_MARGIN_M,
) -> tuple[Optional[dict[str, Any]], str]:
    """Full click → stop plan + clear-path. Returns (plan, reason)."""
    pts = object_points_from_click(
        depth,
        u_px,
        v_px,
        click_z,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        origin=origin,
        R_opt=R_opt,
    )
    if len(pts) < 8:
        # Fall back to the single click point.
        pts = [
            (float(click_xyz[0]), float(click_xyz[1]), float(click_xyz[2]))
        ]
    s_near, z_lo, z_hi, u_hat = near_face_and_height(pts, click_xyz)
    poses = stop_and_pre_poses(
        s_near=s_near,
        z_lo=z_lo if math.isfinite(z_lo) else float(click_xyz[2]) - 0.05,
        z_hi=z_hi if math.isfinite(z_hi) else float(click_xyz[2]) + 0.05,
        click_z=float(click_xyz[2]),
        u_hat=u_hat,
        gap_m=gap_m,
        pre_back_m=pre_back_m,
        tcp_m=tcp_m,
        fingertip_m=fingertip_m,
    )

    def _solve(xyz, seed):
        return solve_ik(xyz, seed)

    line, why = solve_line_ik(
        poses["pre_xyz"],
        poses["stop_xyz"],
        solve_ik=_solve,
        seed=q_now,
        z_lo=poses["z_lo"],
        z_hi=poses["z_hi"],
    )
    if line is None:
        return None, why

    cloud = depth_cloud_base(
        depth, fx=fx, fy=fy, cx=cx, cy=cy, origin=origin, R_opt=R_opt
    )
    # Samples: current→pre joint interp + every final-line waypoint.
    transit = interpolate_q(q_now, line["q_pre"], n=10)
    sample_qs = transit + list(line["waypoints"])
    bodies: list[list[tuple[float, float, float]]] = []
    for q in sample_qs:
        q_m = q_to_model(q)
        bodies.append(
            body_points_at(
                q_m,
                pan,
                ik_mod=ik_mod,
                tcp_m=tcp_m,
                fingertip_m=fingertip_m,
            )
        )
    ok, cmsg = clear_path(
        cloud,
        bodies,
        s_near=poses["s_near"],
        u_hat=u_hat,
        margin_m=corridor_margin_m,
    )
    plan = {
        **poses,
        **line,
        "object_n": len(pts),
        "cloud_n": int(cloud.shape[0]),
        "clear": ok,
        "clear_msg": cmsg,
        "wrist_roll": float(wrist_roll),
        "pan": float(pan),
        "pitch": 0.0,
    }
    if not ok:
        return plan, cmsg
    return plan, "ok"


def _self_check() -> None:
    """Synthetic cylinder → expected stop; clear/blocked corridor cases."""
    # Cylinder in front of base at s≈0.40, radius 0.035, z 0.05..0.25.
    rng = np.random.default_rng(0)
    u_hat = (1.0, 0.0)
    pts = []
    for _ in range(200):
        th = float(rng.uniform(-math.pi, math.pi))
        r = float(rng.uniform(0.0, 0.035))
        z = float(rng.uniform(0.05, 0.25))
        # Face toward base: points near s=0.40 - r*|cos| approx.
        s = 0.40 - r * abs(math.cos(th))
        pts.append((s, r * math.sin(th), z))
    s_near, z_lo, z_hi, uh = near_face_and_height(pts, (0.40, 0.0))
    assert abs(uh[0] - 1.0) < 1e-9
    assert 0.36 < s_near < 0.40, s_near
    assert 0.04 < z_lo < 0.08, z_lo
    assert 0.22 < z_hi < 0.26, z_hi
    poses = stop_and_pre_poses(
        s_near=s_near, z_lo=z_lo, z_hi=z_hi, click_z=0.15, u_hat=uh
    )
    # Fingertips at s_near - gap; TCP shorter by fingertip-tcp.
    assert abs(poses["s_tip"] - (s_near - STOP_GAP_M)) < 1e-9
    assert abs(poses["s_tcp"] - (poses["s_tip"] - (FINGERTIP_M - TCP_M))) < 1e-9
    assert abs(poses["s_pre"] - (poses["s_tcp"] - PRE_BACK_M)) < 1e-9
    assert abs(poses["stop_xyz"][2] - poses["z_app"]) < 1e-9

    # Clear path: body far from cloud.
    cloud = np.asarray(pts, dtype=np.float64)
    body_clear = [[(0.10, 0.0, 0.15)]]
    ok, msg = clear_path(cloud, body_clear, s_near=s_near, u_hat=uh)
    assert ok, msg
    # Blocked: cloud obstacle on the approach path (well short of the face).
    obstacle = np.vstack(
        [cloud, np.array([[0.22, 0.0, 0.15]], dtype=np.float64)]
    )
    body_hit = [[(0.22, 0.0, 0.15)]]
    ok2, msg2 = clear_path(obstacle, body_hit, s_near=s_near, u_hat=uh)
    assert not ok2, msg2

    # Fake IK line: identity seed, always returns seed+epsilon*i.
    seed = [0.0, 0.0, 0.0, 0.0, 0.0]

    def _fake_ik(xyz, s):
        # Encode progress in first joint; reject big jumps via solve_line_ik.
        t = float(xyz[0])
        return [0.01 * t, float(s[1]), float(s[2]), float(s[3]), float(s[4])]

    line, why = solve_line_ik(
        poses["pre_xyz"], poses["stop_xyz"], solve_ik=_fake_ik, seed=seed
    )
    assert line is not None, why
    assert line["n_step"] >= 10
    print("nexarm_stop_plan self-check OK")


if __name__ == "__main__":
    _self_check()
