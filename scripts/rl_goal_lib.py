#!/usr/bin/env python3
"""Shared GrabIt-Goal yaml, analytic tip, and real-wrist IK.

Sim and real do not share joint zeros. A sim example is playable on the real
arm only as an end-effector track: FK the sim joints, then IK under the real
wrist cap. Never send Isaac joint numbers to the PX150.
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import yaml

ARM_OFFSET = (0.06, 0.0, 0.096)
TOOL = 0.008
# Operator positive stop on the real wrist (rad). Teach may set this tighter.
DEFAULT_WRIST_MAX = -0.22
REAL_WRIST_MIN = -1.90
# Factory sleep wrist is -1.73 rad. Home is -1.31 and a reach is often near -1.1.
# Only the fold itself is sleep — not every wrist below the positive stop.
# Interbotix factory sleep is ~-1.8..-pi. Taught pole clicks often use
# wrist_angle_max ≈ -1.48 — that must NOT count as sleep.
SLEEP_WRIST = -1.70

_SHOULDER_Z = 0.06566 + 0.03891
_L1 = 0.15
_L2 = 0.065 + 0.043 + 0.023 + 0.027575

JOINTS = ("waist", "shoulder", "wrist_angle", "wrist_rotate", "gripper")

# Policy observation. v1 is the original 21-D tip vector. v2 appends a
# geometric stage, shaft-yaw error, and a single object-class slot.
OBS_DIM_V1 = 21
# 21-D tip vector + stage(5) + yaw error + object class.
OBS_DIM_V2 = 28
# EARL-style timing obs: tip, grasp, tip-grasp, grasp motion, joints, jaw,
# wide-cam jaw-window error, lost-view bit. Used by the approach-timing policy;
# IK still executes the chosen grasp tip.
OBS_DIM_TIMING = 21
STAGES = ("approach", "align", "descend", "close", "lift")
APPROACH_M = 0.08
ALIGN_WAIST = 0.25
ALIGN_YAW = 0.35
DESCEND_XY = 0.04
DESCEND_Z = 0.03
PAD_RADIUS = 0.025
OBJECT_CLASS_POLE = 1.0
LIFT_M = 0.04
POLE_SPAWN_Z = 0.16


def goal_yaml_path() -> Path:
    for cand in (
        Path("/home/ubuntu/dev-ws/config/rl_goal.yaml"),
        Path("/workspaces/isaac_ros-dev/config/rl_goal.yaml"),
        Path(__file__).resolve().parents[1] / "config" / "rl_goal.yaml",
    ):
        if cand.parent.is_dir():
            return cand
    return Path(__file__).resolve().parents[1] / "config" / "rl_goal.yaml"


def dataset_root() -> Path:
    for cand in (
        Path("/home/ubuntu/dev-ws/datasets/rl_goal"),
        Path("/workspaces/isaac_ros-dev/datasets/rl_goal"),
        Path(__file__).resolve().parents[1] / "datasets" / "rl_goal",
    ):
        if cand.parent.is_dir():
            cand.mkdir(parents=True, exist_ok=True)
            return cand
    root = Path(__file__).resolve().parents[1] / "datasets" / "rl_goal"
    root.mkdir(parents=True, exist_ok=True)
    return root


def runs_root() -> Path:
    root = dataset_root() / "runs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def sanitize_run_name(name: str) -> str:
    text = "".join(c if c.isalnum() or c in "-_ " else "_" for c in str(name).strip())
    text = "_".join(text.split())
    return text[:64] or "unnamed"


def list_runs() -> list[dict]:
    """Named Isaac train folders under datasets/rl_goal/runs/."""
    root = runs_root()
    out = []
    for p in sorted(root.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not p.is_dir():
            continue
        meta_p = p / "meta.json"
        ckpt = None
        for name in ("model.pt", "model_50.pt", "model_25.pt", "model_75.pt"):
            cand = p / name
            if cand.is_file():
                ckpt = cand.name
                break
        eps = sorted(
            e.name
            for e in p.glob("episodes/episode_*")
            if e.is_dir() and (e / "meta.json").is_file()
        )
        label = p.name
        if meta_p.is_file():
            try:
                import json

                meta = json.loads(meta_p.read_text(encoding="utf-8"))
                label = str(meta.get("label") or meta.get("name") or p.name)
            except Exception:
                pass
        out.append(
            {
                "name": p.name,
                "label": label,
                "checkpoint": ckpt,
                "episodes": eps,
                "has_goal": (p / "rl_goal.yaml").is_file(),
            }
        )
    return out


def fk_tip(waist: float, shoulder: float, wrist: float, tool: float = TOOL):
    """Analytic tip in px150_base (same model as px150_ik.fk_xyz)."""
    c, s = math.cos(waist), math.sin(waist)
    l2 = _L2 + float(tool)
    r = _L1 * math.cos(shoulder) + l2 * math.cos(shoulder + wrist)
    z = -_L1 * math.sin(shoulder) - l2 * math.sin(shoulder + wrist)
    return c * r, s * r, _SHOULDER_Z + z


def eef_of(q) -> list[float]:
    tip = fk_tip(float(q[0]), float(q[1]), float(q[2]))
    return [float(tip[i]) + ARM_OFFSET[i] for i in range(3)]


def _seed_joints(seed) -> tuple[float, float] | None:
    """Shoulder and wrist_angle from a joint vector or dict, if given."""
    if seed is None:
        return None
    if isinstance(seed, dict):
        return float(seed["shoulder"]), float(seed["wrist_angle"])
    return float(seed[1]), float(seed[2])


def ik_eef(
    eef,
    wrist_rotate: float,
    gripper: float,
    wrist_max: float,
    *,
    wrist_min: float | None = None,
    seed=None,
    seed_weight: float = 0.15,
) -> dict | None:
    """Real-playable IK near the taught positive stop, not down in factory sleep.

    Default band is 0.40 rad below wrist_max. A sim tip that only solves by
    folding toward sleep is rejected so it is not played on the PX150.

    When ``seed`` is set, solutions inside 3 cm of the tip prefer the shoulder
    and wrist closest to that pose (Visual-IRL min joint change).
    """
    if wrist_min is None:
        wrist_min = max(REAL_WRIST_MIN, float(wrist_max) - 0.40)
    arm = [float(eef[i]) - ARM_OFFSET[i] for i in range(3)]
    radial = math.hypot(arm[0], arm[1])
    if radial < 1e-4:
        waist = 0.0
    else:
        waist = math.atan2(arm[1], arm[0])
    hi = min(1.4486, float(wrist_max))
    lo = float(wrist_min)
    if hi < lo:
        return None
    seeded = _seed_joints(seed)
    best = None
    best_score = 1e9
    best_err = 1e9
    sh = -1.97
    while sh <= 1.90:
        wa = lo
        while wa <= hi + 1e-9:
            tip = fk_tip(0.0, sh, wa)
            err = math.hypot(tip[0] - radial, tip[2] - arm[2])
            if err < best_err:
                best_err = err
            if err <= 0.03:
                score = err
                if seeded is not None:
                    score = err + float(seed_weight) * (
                        (sh - seeded[0]) ** 2 + (wa - seeded[1]) ** 2
                    )
                if score < best_score:
                    best_score = score
                    best = (sh, wa, err)
            wa += 0.02
        sh += 0.02
    if best is None or best[2] > 0.03:
        return None
    return {
        "waist": float(waist),
        "shoulder": float(best[0]),
        "wrist_angle": float(best[1]),
        "wrist_rotate": float(wrist_rotate),
        "gripper": float(gripper),
        "err": float(best[2]),
    }


# Isaac twin prefers the SIM_HOME branch (shoulder back, wrist positive).
SIM_SHOULDER_PREF = -1.45
SIM_WRIST_PREF = 1.448
SIM_WRIST_LO = 0.05
SIM_WRIST_HI = 1.4486


def ik_sim_eef(
    eef,
    wrist_rotate: float,
    gripper: float,
) -> dict | None:
    """Tip → Isaac joint numbers for twin mirror. Never send these to the PX150."""
    arm = [float(eef[i]) - ARM_OFFSET[i] for i in range(3)]
    radial = math.hypot(arm[0], arm[1])
    if radial < 1e-4:
        waist = 0.0
    else:
        waist = math.atan2(arm[1], arm[0])
    best = None
    best_score = 1e9
    sh = -1.97
    while sh <= 1.90:
        wa = SIM_WRIST_LO
        while wa <= SIM_WRIST_HI + 1e-9:
            tip = fk_tip(0.0, sh, wa)
            err = math.hypot(tip[0] - radial, tip[2] - arm[2])
            if err > 0.04:
                wa += 0.02
                continue
            score = err + 0.15 * (
                (sh - SIM_SHOULDER_PREF) ** 2 + (wa - SIM_WRIST_PREF) ** 2
            )
            if score < best_score:
                best_score = score
                best = (sh, wa, err)
            wa += 0.02
        sh += 0.02
    if best is None or best[2] > 0.04:
        return None
    return {
        "waist": float(waist),
        "shoulder": float(best[0]),
        "wrist_angle": float(best[1]),
        "wrist_rotate": float(wrist_rotate),
        "gripper": float(gripper),
        "err": float(best[2]),
    }


def load_goal(path: Path | None = None) -> dict:
    path = path or goal_yaml_path()
    data = {}
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    wrist_max = float(data.get("wrist_angle_max", DEFAULT_WRIST_MAX))
    return {
        "x": float(data.get("x", 0.22)),
        "y": float(data.get("y", 0.0)),
        "z": float(data.get("z", 0.26)),
        "gripper": float(data.get("gripper", -0.95)),
        "wrist_rotate": float(data.get("wrist_rotate", 0.0)),
        "wrist_angle_max": wrist_max,
        "frame": str(data.get("frame") or "analytic_eef"),
        "source": str(data.get("source") or ""),
        "play_on_real": True,
    }


CLOSED_GRIP = -0.95
POLE_XY_JITTER = 0.03


def write_goal(
    eef: list[float],
    joints: dict,
    *,
    source: str,
    wrist_max: float,
    path: Path | None = None,
    xy_jitter: float = 0.0,
) -> Path:
    """Write a tip goal that has a real IK under wrist_max."""
    path = path or goal_yaml_path()
    grip = float(joints.get("gripper", CLOSED_GRIP))
    wr = float(joints.get("wrist_rotate", 0.0))
    real = ik_eef(eef, wr, grip, wrist_max)
    if real is None:
        raise ValueError(
            f"tip ({eef[0]:.3f},{eef[1]:.3f},{eef[2]:.3f}) has no real IK "
            f"with wrist_angle <= {wrist_max:.3f}"
        )
    text = f"""# GrabIt-Goal. Canonical target is the analytic tip (create3 frame).
# Sim joint numbers are not the real arm. Replay a sim take via eef.npy
# re-IK under wrist_angle_max — do not stream Isaac joints to the PX150.
#
# eef = fk_tip(waist, shoulder, wrist_angle) + {ARM_OFFSET}

frame: analytic_eef
play_on_real: true
source: {source}

x: {float(eef[0]):.4f}
y: {float(eef[1]):.4f}
z: {float(eef[2]):.4f}
gripper: {grip:.4f}
wrist_rotate: {wr:.4f}
wrist_angle_max: {float(wrist_max):.4f}

joints_real:
  waist: {real['waist']:.4f}
  shoulder: {real['shoulder']:.4f}
  wrist_angle: {real['wrist_angle']:.4f}
  wrist_rotate: {real['wrist_rotate']:.4f}
  gripper: {real['gripper']:.4f}

xy_jitter: {float(xy_jitter):.4f}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def factory_sleep(wrist_angle: float) -> bool:
    return float(wrist_angle) < SLEEP_WRIST


def wrap_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def stage_index(
    *,
    dist: float,
    xy: float,
    z_err: float,
    waist_err: float,
    yaw_err: float,
    grasped: bool,
) -> int:
    """Geometric pick stage. Grasped wins; otherwise far → approach."""
    if grasped:
        return 4
    if float(dist) > APPROACH_M:
        return 0
    if abs(float(waist_err)) > ALIGN_WAIST or abs(float(yaw_err)) > ALIGN_YAW:
        return 1
    if float(xy) <= DESCEND_XY and abs(float(z_err)) > DESCEND_Z:
        return 2
    return 3


def stage_one_hot(**kwargs) -> list[float]:
    hot = [0.0] * len(STAGES)
    hot[stage_index(**kwargs)] = 1.0
    return hot


def lift_term(
    grasped: bool,
    pole_z: float,
    spawn_z: float = POLE_SPAWN_Z,
    lift_m: float = LIFT_M,
) -> float:
    """0 until the object is grasped, then how far it has risen, saturating at 1."""
    if not grasped:
        return 0.0
    risen = max(0.0, float(pole_z) - float(spawn_z))
    return min(1.0, risen / max(float(lift_m), 1e-6))


def observable_grasp(dist: float, grip: float, closed: float = CLOSED_GRIP) -> bool:
    """Jaw is around the shaft: in the pad, closed past zero, not slammed shut."""
    return (
        float(dist) < PAD_RADIUS
        and float(grip) < 0.0
        and float(grip) > (float(closed) + 0.15)
    )


def build_goal_obs(
    eef,
    aim,
    grip: float,
    goal_grip: float,
    joints,
    last_action,
    *,
    shaft_yaw: float = 0.0,
    grasped: bool | None = None,
    obs_dim: int = OBS_DIM_V2,
) -> list[float]:
    """Tip policy vector. v1 is 21-D. v2 appends stage, yaw error, class.

    Order is fixed: eef, aim, tip-to-aim, grip, goal grip, joints, last action,
    then (v2) stage one-hot, wrap(wrist_rotate - shaft_yaw), object class.
    """
    eef_l = [float(eef[i]) for i in range(3)]
    aim_l = [float(aim[i]) for i in range(3)]
    delta = [aim_l[i] - eef_l[i] for i in range(3)]
    q = [float(joints[i]) for i in range(5)]
    last = [float(last_action[i]) for i in range(5)]
    base = eef_l + aim_l + delta + [float(grip), float(goal_grip)] + q + last
    dim = int(obs_dim)
    if dim == OBS_DIM_V1:
        if len(base) != OBS_DIM_V1:
            raise ValueError(f"v1 obs is {len(base)}, expected {OBS_DIM_V1}")
        return base
    if dim != OBS_DIM_V2:
        raise ValueError(f"obs_dim must be {OBS_DIM_V1} or {OBS_DIM_V2}, got {dim}")
    dist = math.sqrt(sum(d * d for d in delta))
    xy = math.hypot(delta[0], delta[1])
    if grasped is None:
        grasped = observable_grasp(dist, q[4])
    arm_x = aim_l[0] - ARM_OFFSET[0]
    arm_y = aim_l[1] - ARM_OFFSET[1]
    taught_waist = math.atan2(arm_y, arm_x if abs(arm_x) > 1e-6 else 1e-6)
    yaw_err = wrap_pi(q[3] - float(shaft_yaw))
    waist_err = wrap_pi(q[0] - taught_waist)
    hot = stage_one_hot(
        dist=dist,
        xy=xy,
        z_err=delta[2],
        waist_err=waist_err,
        yaw_err=yaw_err,
        grasped=bool(grasped),
    )
    out = base + hot + [yaw_err, OBJECT_CLASS_POLE]
    if len(out) != OBS_DIM_V2:
        raise ValueError(f"v2 obs is {len(out)}, expected {OBS_DIM_V2}")
    return out


def build_timing_obs(
    tip,
    grasp,
    grasp_prev,
    joints,
    *,
    jaw_closed: bool,
    wide_du: float = 0.0,
    wide_dv: float = 0.0,
    lost_view: bool = False,
) -> list[float]:
    """EARL-style approach-timing vector (OBS_DIM_TIMING).

    tip / grasp are analytic tips. grasp_prev is the previous tick's grasp
    (zeros on the first step). The policy outputs joint deltas + close; IK
    still owns the grasp tip.
    """
    tip_l = [float(tip[i]) for i in range(3)]
    grasp_l = [float(grasp[i]) for i in range(3)]
    prev_l = [float(grasp_prev[i]) for i in range(3)]
    delta = [grasp_l[i] - tip_l[i] for i in range(3)]
    motion = [grasp_l[i] - prev_l[i] for i in range(3)]
    q = [float(joints[i]) for i in range(5)]
    out = (
        tip_l
        + grasp_l
        + delta
        + motion
        + q
        + [1.0 if jaw_closed else 0.0, float(wide_du), float(wide_dv), 1.0 if lost_view else 0.0]
    )
    if len(out) != OBS_DIM_TIMING:
        raise ValueError(f"timing obs is {len(out)}, expected {OBS_DIM_TIMING}")
    return out


def apply_joint_delta(
    current,
    delta,
    *,
    scale: float = 0.15,
    close_thresh: float = 0.5,
) -> tuple[list[float], bool]:
    """Apply a 5-D timing action: joint deltas + close gate on channel 4.

    Channels 0..3 are added to waist/shoulder/wrist_angle/wrist_rotate.
    Channel 4 > close_thresh means start the foam-stall close.
    """
    cur = [float(current[i]) for i in range(5)]
    d = [float(delta[i]) for i in range(min(5, len(delta)))]
    while len(d) < 5:
        d.append(0.0)
    out = list(cur)
    for i in range(4):
        out[i] = cur[i] + float(scale) * d[i]
    close = d[4] > float(close_thresh)
    out[4] = CLOSED_GRIP if close else cur[4]
    return out, close


def actor_obs_dim(state_dict) -> int:
    """Read the actor's first linear layer. 21 keeps the old builder."""
    weight = None
    for key in ("mlp.0.weight", "actor.mlp.0.weight"):
        if key in state_dict:
            weight = state_dict[key]
            break
    if weight is None:
        for key, value in state_dict.items():
            if key.endswith("mlp.0.weight"):
                weight = value
                break
    if weight is None:
        raise KeyError("actor mlp.0.weight missing")
    dim = int(weight.shape[1])
    if dim not in (OBS_DIM_V1, OBS_DIM_V2, OBS_DIM_TIMING):
        raise ValueError(
            f"checkpoint obs dim {dim} is not "
            f"{OBS_DIM_V1}, {OBS_DIM_V2}, or {OBS_DIM_TIMING}"
        )
    return dim


def fk_horizon_command(
    action,
    current,
    aim,
    wrist_max: float,
    *,
    n_samples: int = 7,
    horizon_s: float = 0.4,
    hz: float = 12.0,
    rng: random.Random | None = None,
) -> list[float]:
    """Sample joint targets around ``action``, roll analytic FK, return the first step.

    The wrist of every candidate is held at or under ``wrist_max``. Only the
    first fraction of the winning sample is returned so the caller can replan.
    """
    rng = rng or random.Random()
    current_l = [float(current[i]) for i in range(5)]
    action_l = [float(action[i]) for i in range(5)]
    action_l[2] = min(action_l[2], float(wrist_max))
    aim_l = [float(aim[i]) for i in range(3)]
    steps = max(2, int(round(float(horizon_s) * float(hz))))
    cands = [action_l]
    for _ in range(max(0, int(n_samples) - 1)):
        cand = list(action_l)
        cand[0] += rng.uniform(-0.12, 0.12)
        cand[1] += rng.uniform(-0.15, 0.15)
        cand[2] = min(cand[2] + rng.uniform(-0.15, 0.15), float(wrist_max))
        cands.append(cand)

    def _final_tip(cand: list[float]) -> list[float]:
        q = list(current_l)
        for s in range(1, steps + 1):
            a = s / steps
            q = [(1.0 - a) * current_l[i] + a * cand[i] for i in range(5)]
            q[2] = min(q[2], float(wrist_max))
        return eef_of(q)

    best = action_l
    best_d = 1e9
    for cand in cands:
        tip = _final_tip(cand)
        dist = math.sqrt(sum((tip[i] - aim_l[i]) ** 2 for i in range(3)))
        if dist < best_d:
            best_d = dist
            best = cand
    frac = 1.0 / steps
    first = [(1.0 - frac) * current_l[i] + frac * best[i] for i in range(5)]
    first[2] = min(first[2], float(wrist_max))
    return first
