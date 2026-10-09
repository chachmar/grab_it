#!/usr/bin/env python3
"""Harden xsarm_moveit.launch.py for grab_it.

- Wrap robot_description in ParameterValue(..., value_type=str) so URDF XML
  is not parsed as YAML (breaks MoveIt launch after URDF edits).
- Prefer /<robot>/joint_states_arm for the planning scene monitor.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

CANDIDATES = [
    Path(
        "/workspaces/isaac_ros-dev/src/interbotix/interbotix_ros_manipulators/"
        "interbotix_ros_xsarms/interbotix_xsarm_moveit/launch/xsarm_moveit.launch.py"
    ),
    Path(
        "/workspaces/isaac_ros-dev/install/interbotix_xsarm_moveit/share/"
        "interbotix_xsarm_moveit/launch/xsarm_moveit.launch.py"
    ),
]


def patch(path: Path) -> None:
    t = path.read_text()
    orig = t
    if "parameter_descriptions import ParameterValue" not in t:
        t = t.replace(
            "from launch_ros.actions import Node",
            "from launch_ros.actions import Node\n"
            "from launch_ros.parameter_descriptions import ParameterValue",
            1,
        )
    t = t.replace(
        "robot_description = {'robot_description': robot_description_launch_arg}",
        "robot_description = {'robot_description': "
        "ParameterValue(robot_description_launch_arg, value_type=str)}",
    )
    t = t.replace(
        "f'/{robot_name_launch_arg.perform(context)}/joint_states'",
        "f'/{robot_name_launch_arg.perform(context)}/joint_states_arm'",
    )
    # Don't spam Home joint_group commands when Nano owns the DXL bus.
    if "move_to_home = TimerAction(" in t and "use_arm_driver" in t:
        old_home = """    move_to_home = TimerAction(
        period=2.5,
        actions=["""
        new_home = """    move_to_home = TimerAction(
        period=2.5,
        condition=IfCondition(LaunchConfiguration('use_arm_driver')),
        actions=["""
        if "condition=IfCondition(LaunchConfiguration('use_arm_driver'))" not in t[
            t.find("move_to_home") : t.find("move_to_home") + 250
        ]:
            if old_home in t:
                t = t.replace(old_home, new_home, 1)

    if t == orig and "ParameterValue(robot_description_launch_arg" in t:
        print(f"[ok] {path}")
        return
    if t != orig:
        path.write_text(t)
        print(f"[patched] {path}")
    else:
        print(f"[warn] no changes: {path}")


def main() -> None:
    found = False
    seen = set()
    for p in CANDIDATES:
        if not p.exists():
            continue
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        found = True
        patch(p)
    if not found:
        print("xsarm_moveit.launch.py not found", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
