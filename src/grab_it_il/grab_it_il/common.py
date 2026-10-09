"""Shared tabletop-pick IL constants (match gui_node / grasp_params)."""
from __future__ import annotations

import json
import os
from pathlib import Path

ARM_JOINTS = ["waist", "shoulder", "wrist_angle", "wrist_rotate"]
STATE_NAMES = ARM_JOINTS + ["gripper", "pan"]
JOINT_LIMITS = {
    "waist": (-2.0246, 2.0944),
    "shoulder": (-1.9722, 1.9373),
    "wrist_angle": (-2.0769, 1.4486),
    "wrist_rotate": (-1.5708, 1.5708),
    # D435i tilt (Dynamixel); match agx_isaac_hub PAN_MIN/MAX.
    "pan": (-2.0944, 1.6406),
}
GRIPPER_CLOSED = float(os.environ.get("GRAB_IT_GRIPPER_JOINT_CLOSED", "-0.95"))
GRIPPER_OPEN = float(os.environ.get("GRAB_IT_GRIPPER_JOINT_OPEN", "1.65"))
HZ = 20
MIN_EPISODE_S = 1.0
TASK = "tabletop_pick"

JOINT_STATES_TOPIC = "/px150_no_elbow/joint_states"
JOINT_GROUP_TOPIC = "/px150_no_elbow/commands/joint_group"
JOINT_SINGLE_TOPIC = "/px150_no_elbow/commands/joint_single"
IL_COMMAND_TOPIC = "/grab_it/il_command"
IL_STATUS_TOPIC = "/grab_it/il_status"
GRASP_STATUS_TOPIC = "/grab_it/grasp_status"
D435I_COMPRESSED = "/d435i/d435i/color/image_raw/compressed"
D435I_RAW = "/d435i/d435i/color/image_raw"
WRIST_COMPRESSED = "/wide_cam/image_raw/compressed"
WRIST_RAW = "/wide_cam/image_raw"

DEFAULT_DATASET = Path(
    os.environ.get(
        "GRAB_IT_IL_DATASET",
        "/workspaces/isaac_ros-dev/datasets/tabletop_pick",
    )
)


def clamp_action(action: list[float] | tuple[float, ...]) -> list[float]:
    out = []
    for i, name in enumerate(STATE_NAMES):
        val = float(action[i]) if i < len(action) else 0.0
        if name == "gripper":
            lo, hi = GRIPPER_CLOSED, GRIPPER_OPEN
            if lo > hi:
                lo, hi = hi, lo
        else:
            lo, hi = JOINT_LIMITS[name]
        out.append(max(lo, min(hi, val)))
    return out


def load_joint_envelope(path: Path | str | None) -> dict[str, tuple[float, float]] | None:
    """Load dataset joint min/max envelope JSON → {name: (lo, hi)}."""
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    limits = data.get("limits") if isinstance(data, dict) else None
    if not isinstance(limits, dict):
        return None
    out: dict[str, tuple[float, float]] = {}
    for name in STATE_NAMES:
        pair = limits.get(name)
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue
        lo, hi = float(pair[0]), float(pair[1])
        if lo > hi:
            lo, hi = hi, lo
        out[name] = (lo, hi)
    return out or None


def clamp_action_envelope(
    action: list[float] | tuple[float, ...],
    envelope: dict[str, tuple[float, float]] | None,
    *,
    last_ok: list[float] | None = None,
) -> tuple[list[float], list[float], bool]:
    """Clamp each joint into the dataset envelope.

    A command past the envelope used to keep the previous pose, so PLAY could
    sit still when the policy asked for a reach the demos never showed.
    Those joints now stop at the envelope edge and the arm still moves.
    Returns (clamped_action, new_last_ok, held).
    """
    hard = clamp_action(action)
    if not envelope:
        return hard, list(hard), False
    held = False
    out: list[float] = []
    for i, name in enumerate(STATE_NAMES):
        val = float(hard[i]) if i < len(hard) else 0.0
        lim = envelope.get(name)
        if lim is None:
            out.append(val)
            continue
        lo, hi = lim
        if val < lo or val > hi:
            held = True
            val = max(lo, min(hi, val))
        out.append(val)
    return out, list(out), held


def parse_json_cmd(raw: str) -> dict:
    raw = (raw or "").strip()
    if not raw:
        return {}
    if raw[0] == "{":
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {"cmd": raw}
        except json.JSONDecodeError:
            return {"cmd": raw}
    return {"cmd": raw.lower()}


def next_episode_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    existing = []
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith("episode_"):
            try:
                existing.append(int(p.name.split("_")[1]))
            except (IndexError, ValueError):
                continue
    n = (max(existing) + 1) if existing else 0
    return root / f"episode_{n:06d}"


def count_episodes(root: Path) -> int:
    if not root.is_dir():
        return 0
    return sum(1 for p in root.iterdir() if p.is_dir() and p.name.startswith("episode_"))
