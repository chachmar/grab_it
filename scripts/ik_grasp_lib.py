#!/usr/bin/env python3
"""IK-centered grasp pool, jaw-window check, and trial logging.

Inverse kinematics is the only executor. This module builds a small pool of
shaft-frame grasp poses, drops those ``ik_eef`` cannot solve under the taught
wrist cap, ranks the rest (geometry now, IL scorer later), and measures whether
the wide USB camera still sees the shaft inside the jaw window.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rl_goal_lib import (
    ARM_OFFSET,
    CLOSED_GRIP,
    DEFAULT_WRIST_MAX,
    dataset_root,
    ik_eef,
    load_goal,
    wrap_pi,
)

# Jaw window in the wrist (wide USB) image. Matches _il_wrist_finish.
JAW_U = (0.35, 0.65)
JAW_V = (0.55, 0.95)
JAW_CENTER_U = 0.50
JAW_CENTER_V = 0.75

# Prefers a slightly open approach, then hard close.
GRIP_OPEN = 1.65
PREGRASP_BACK_M = 0.025
HEIGHTS_M = (-0.02, 0.0, 0.02)


@dataclass
class GraspCandidate:
    """One tip pose in the create3 / analytic tip frame."""

    x: float
    y: float
    z: float
    yaw: float
    phase: str  # "pregrasp" | "grasp"
    height_off: float = 0.0
    ik_err: float | None = None
    joints: dict[str, float] | None = None
    geo_score: float = 0.0
    il_score: float | None = None

    @property
    def tip(self) -> list[float]:
        return [float(self.x), float(self.y), float(self.z)]


@dataclass
class JawView:
    u: float
    v: float
    from_vert: float = 0.0
    in_window: bool = False
    du: float = 0.0
    dv: float = 0.0
    lost: bool = True
    detected: bool = False

    @classmethod
    def from_shaft(cls, shaft: dict | None) -> "JawView":
        if not shaft or shaft.get("u") is None or shaft.get("v") is None:
            return cls(u=0.5, v=0.5, lost=True, in_window=False, detected=False)
        u = float(shaft["u"])
        v = float(shaft["v"])
        in_win = JAW_U[0] <= u <= JAW_U[1] and JAW_V[0] <= v <= JAW_V[1]
        return cls(
            u=u,
            v=v,
            from_vert=float(shaft.get("from_vert") or 0.0),
            in_window=in_win,
            du=u - JAW_CENTER_U,
            dv=v - JAW_CENTER_V,
            lost=not in_win,
            detected=True,
        )


@dataclass
class ServoTrial:
    """One closed-loop IK grasp attempt — the IL label source."""

    t: float = field(default_factory=time.time)
    shaft_xy: list[float] = field(default_factory=list)
    shaft_z: float = 0.0
    shaft_yaw: float = 0.0
    chosen: dict[str, Any] = field(default_factory=dict)
    pool_size: int = 0
    reachable: int = 0
    wide_in_view: bool = False
    wide_lost_any: bool = False
    ik_live: bool = True
    stalled: bool = False
    lifted: bool = False
    held: bool = False
    steps: int = 0
    note: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=None)


def trials_dir() -> Path:
    root = dataset_root() / "servo_trials"
    root.mkdir(parents=True, exist_ok=True)
    return root


def append_trial(trial: ServoTrial, path: Path | None = None) -> Path:
    path = path or (trials_dir() / "trials.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(trial.to_json() + "\n")
    return path


def load_trials(path: Path | None = None) -> list[dict]:
    path = path or (trials_dir() / "trials.jsonl")
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _radial_back(x: float, y: float, back_m: float) -> tuple[float, float]:
    r = math.hypot(x, y)
    if r < 1e-4:
        return x, y
    scale = max(0.05, r - float(back_m)) / r
    return x * scale, y * scale


def build_grasp_pool(
    tip: list[float],
    shaft_yaw: float,
    *,
    wrist_max: float,
    seed: dict | list | None = None,
    heights: tuple[float, ...] = HEIGHTS_M,
    pregrasp_back_m: float = PREGRASP_BACK_M,
) -> list[GraspCandidate]:
    """Candidates in shaft frame: heights × {pregrasp, grasp}, IK-filtered.

    Tries shaft yaw and yaw+π (other pad side) so a wrist-capped solve can
    pick the reachable side.
    """
    gx, gy, gz = float(tip[0]), float(tip[1]), float(tip[2])
    yaws = [float(shaft_yaw), wrap_pi(float(shaft_yaw) + math.pi)]
    pool: list[GraspCandidate] = []
    for yaw in yaws:
        for hz in heights:
            z = gz + float(hz)
            grasp = GraspCandidate(
                x=gx, y=gy, z=z, yaw=yaw, phase="grasp", height_off=float(hz)
            )
            px, py = _radial_back(gx, gy, pregrasp_back_m)
            pre = GraspCandidate(
                x=px, y=py, z=z + 0.01, yaw=yaw, phase="pregrasp", height_off=float(hz)
            )
            pool.extend([pre, grasp])

    reachable: list[GraspCandidate] = []
    for cand in pool:
        grip = GRIP_OPEN if cand.phase == "pregrasp" else CLOSED_GRIP
        sol = ik_eef(
            cand.tip,
            float(cand.yaw),
            grip,
            float(wrist_max),
            seed=seed,
        )
        if sol is None:
            continue
        cand.ik_err = float(sol.get("err", 0.0))
        cand.joints = {
            "waist": float(sol["waist"]),
            "shoulder": float(sol["shoulder"]),
            "wrist_angle": float(sol["wrist_angle"]),
            "wrist_rotate": float(sol["wrist_rotate"]),
            "gripper": float(sol["gripper"]),
        }
        # Prefer mid-height, small IK residual, grasp over far pregrasp.
        cand.geo_score = (
            1.0
            - min(1.0, abs(cand.height_off) / 0.04)
            - 0.5 * min(1.0, float(cand.ik_err) / 0.03)
            + (0.15 if cand.phase == "grasp" else 0.0)
        )
        reachable.append(cand)
    return reachable


def rank_candidates(
    cands: list[GraspCandidate],
    *,
    current_wrist_rotate: float | None = None,
    scorer=None,
) -> list[GraspCandidate]:
    """Sort reachable candidates. Optional IL scorer overrides geo_score."""
    scored: list[GraspCandidate] = []
    for c in cands:
        score = float(c.geo_score)
        if current_wrist_rotate is not None:
            score -= 0.2 * abs(wrap_pi(float(c.yaw) - float(current_wrist_rotate)))
            c.geo_score = score
        if scorer is not None:
            pred = float(scorer(candidate_features(c)))
            c.il_score = pred
        scored.append(c)
    if any(c.il_score is not None for c in scored):
        scored.sort(key=lambda c: float(c.il_score if c.il_score is not None else -1e9), reverse=True)
    else:
        scored.sort(key=lambda c: float(c.geo_score), reverse=True)
    return scored


def candidate_features(c: GraspCandidate) -> list[float]:
    """Fixed-length feature vector for the IL ranker."""
    j = c.joints or {}
    return [
        float(c.x),
        float(c.y),
        float(c.z),
        float(c.yaw),
        float(c.height_off),
        1.0 if c.phase == "grasp" else 0.0,
        float(c.ik_err or 1.0),
        float(c.geo_score),
        float(j.get("shoulder", 0.0)),
        float(j.get("wrist_angle", 0.0)),
    ]


def pick_grasp(
    tip: list[float],
    shaft_yaw: float,
    *,
    wrist_max: float,
    seed: dict | list | None = None,
    current_wrist_rotate: float | None = None,
    scorer=None,
    prefer_phase: str | None = None,
) -> tuple[GraspCandidate | None, list[GraspCandidate]]:
    pool = build_grasp_pool(tip, shaft_yaw, wrist_max=wrist_max, seed=seed)
    ranked = rank_candidates(
        pool, current_wrist_rotate=current_wrist_rotate, scorer=scorer
    )
    if prefer_phase:
        for c in ranked:
            if c.phase == prefer_phase:
                return c, ranked
    return (ranked[0] if ranked else None), ranked


def jaw_bias_joints(
    q: dict[str, float],
    view: JawView,
    *,
    shoulder_step: float = 0.05,
    wrist_step: float = 0.05,
    roll_gain: float = 1.0,
    u_gain: float = 0.35,
    wrist_max: float | None = None,
) -> dict[str, float]:
    """Bias joints to keep the shaft in the wide-cam jaw window.

    Never fold ``wrist_angle`` past the taught ``wrist_max`` toward the floor.
    When the shaft sits too high in the image (low ``v``), unfold the wrist
    and/or reach the shoulder toward the pole — do not dive the tip down.
    """
    out = dict(q)
    # Taught stop is an upper bound on fold: wrist_angle must stay >= wrist_max
    # when wrist_max is negative (e.g. -1.48). Never command more-negative.
    wmax = float(wrist_max) if wrist_max is not None else -0.22
    wr_floor = float(wmax)  # most-folded allowed
    wr_ceil = 1.40

    if not view.detected or not math.isfinite(view.u):
        # Blind seek: unfold shoulder toward a reachable mid pose, keep wrist at floor.
        cur = float(out.get("shoulder", 0.0))
        out["shoulder"] = min(1.0, cur + shoulder_step)
        out["wrist_angle"] = max(wr_floor, float(out.get("wrist_angle", wr_floor)))
        return out

    # Absolute roll from tilt (camera does not spin with wrist_rotate stator).
    out["wrist_rotate"] = max(-1.2, min(1.2, -float(view.from_vert) * float(roll_gain)))
    # Horizontal: waist nudge toward jaw center (same sign as _align_wrist_camera).
    waist = float(out.get("waist", 0.0)) + float(u_gain) * (JAW_CENTER_U - float(view.u))
    out["waist"] = max(-1.8, min(1.8, waist))

    wr = float(out.get("wrist_angle", wr_floor))
    sh = float(out.get("shoulder", 0.0))
    # If v is already in the jaw band, only trim waist (caller keeps IK tip on pole).
    if JAW_V[0] <= view.v <= JAW_V[1]:
        out["wrist_angle"] = max(wr_floor, wr)
        return out
    # Shaft low in image (high v) → slight fold, still >= taught floor.
    if view.v > JAW_V[1]:
        wr = max(wr_floor, wr - wrist_step)
    # Shaft high in image (low v) → tip wrist UP (unfold / more positive) so the
    # pads can wrap the pole; optional small shoulder reach. Never dive past floor.
    elif view.v < JAW_V[0]:
        wr = min(wr_ceil, wr + wrist_step * (1.5 if view.v < (JAW_V[0] - 0.08) else 1.0))
        sh = min(1.2, sh + 0.5 * shoulder_step)
        wr = max(wr_floor, wr)
    out["wrist_angle"] = max(wr_floor, min(wr_ceil, wr))
    out["shoulder"] = sh
    return out


def blend_toward_ik(
    current: dict[str, float],
    target: dict[str, float],
    alpha: float,
) -> dict[str, float]:
    a = max(0.0, min(1.0, float(alpha)))
    keys = ("waist", "shoulder", "wrist_angle", "wrist_rotate", "gripper")
    return {k: (1.0 - a) * float(current[k]) + a * float(target[k]) for k in keys}


def tip_from_twin(
    twin: dict,
    goal: dict | None = None,
    *,
    use_live_z: bool = False,
) -> tuple[list[float], float]:
    """Hub twin pole_* → analytic tip + yaw.

    Grasp height defaults to the taught yaml ``z`` (mid-shaft). Live ``pole_z``
    is usually the shaft root/top and is too high for the wrist-capped IK.
    Live XY and yaw still track the D435i shaft.
    """
    goal = goal or load_goal()
    tip = [float(goal["x"]), float(goal["y"]), float(goal["z"])]
    yaw = float(goal.get("wrist_rotate", 0.0))
    xy = twin.get("pole_xy") if twin else None
    if isinstance(xy, (list, tuple)) and len(xy) >= 2:
        tip[0] = float(xy[0]) + float(ARM_OFFSET[0])
        tip[1] = float(xy[1]) + float(ARM_OFFSET[1])
    if use_live_z and twin and twin.get("pole_z") is not None:
        tip[2] = float(twin["pole_z"]) + float(ARM_OFFSET[2])
    if twin and twin.get("pole_yaw") is not None:
        yaw = float(twin["pole_yaw"])
    return tip, yaw


def project_reachable_tip(
    tip: list[float],
    shaft_yaw: float,
    *,
    wrist_max: float,
    goal: dict | None = None,
    steps: int = 8,
) -> tuple[list[float], float, str]:
    """Pull tip toward the taught yaml until ``ik_eef`` finds a grasp.

    Live poles that drift outside the wrist-capped envelope otherwise empty
    the pool. Returns (tip, yaw, note).
    """
    goal = goal or load_goal()
    taught = [float(goal["x"]), float(goal["y"]), float(goal["z"])]
    yaw = float(shaft_yaw)
    taught_yaw = float(goal.get("wrist_rotate", yaw))
    # Prefer live tip if already reachable at taught Z.
    probe = [float(tip[0]), float(tip[1]), float(taught[2])]
    if build_grasp_pool(probe, yaw, wrist_max=wrist_max):
        return probe, yaw, "live"
    if build_grasp_pool(probe, taught_yaw, wrist_max=wrist_max):
        return probe, taught_yaw, "live_taught_yaw"
    # Blend toward the taught click (known reachable under this wrist cap).
    for i in range(1, steps + 1):
        a = i / float(steps)
        blended = [
            probe[0] * (1.0 - a) + taught[0] * a,
            probe[1] * (1.0 - a) + taught[1] * a,
            taught[2],
        ]
        for y in (yaw, taught_yaw, wrap_pi(yaw + math.pi)):
            if build_grasp_pool(blended, y, wrist_max=wrist_max):
                return blended, y, f"blend_{a:.2f}"
    return list(taught), taught_yaw, "yaml_fallback"


def wrist_max_of(goal: dict | None = None) -> float:
    goal = goal or load_goal()
    return float(goal.get("wrist_angle_max", DEFAULT_WRIST_MAX))
