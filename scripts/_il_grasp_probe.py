#!/usr/bin/env python3
"""Nudge one joint and report pad distance plus wrist-image shaft position."""
from __future__ import annotations

import importlib.util
import sys
import time

spec = importlib.util.spec_from_file_location(
    "wf", "/home/ubuntu/dev-ws/scripts/_il_wrist_finish.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

NAMES = m.NAMES


def snap() -> None:
    d = m.twin()
    arm = m.arm_of(d)
    wide = m._get("/wide.jpg")
    pose = m.shaft_pose(wide)
    print(
        "pad",
        None if d.get("pad_d") is None else round(float(d["pad_d"]), 4),
        "pz",
        None if d.get("pole_z") is None else round(float(d["pole_z"]), 3),
        "arm",
        {k: round(arm.get(k, 0.0), 3) for k in NAMES},
        "shaft",
        None
        if pose is None
        else {k: round(pose[k], 3) if isinstance(pose[k], float) else pose[k] for k in pose},
        flush=True,
    )


def nudge(joint: str, delta: float) -> None:
    d = m.twin()
    arm = m.arm_of(d)
    for n in NAMES:
        arm.setdefault(n, 0.0)
    arm["gripper"] = m.GRIP_HOLD_OPEN
    arm[joint] = float(arm[joint]) + delta
    print(f"nudge {joint} {delta:+.2f}", flush=True)
    for _ in range(12):
        m.post(arm, clear_gate=True)
        time.sleep(0.15)
    snap()


if __name__ == "__main__":
    joint = sys.argv[1]
    delta = float(sys.argv[2])
    snap()
    nudge(joint, delta)
