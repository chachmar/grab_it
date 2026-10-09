#!/usr/bin/env python3
"""Make grab_it_launch honor joint_states_topic (was declared but unused)."""
from __future__ import annotations

import sys
from pathlib import Path

CANDIDATES = [
    Path(
        "/workspaces/isaac_ros-dev/src/grab_it_description/launch/grab_it_launch.py"
    ),
    Path(
        "/workspaces/isaac_ros-dev/install/grab_it_description/share/"
        "grab_it_description/launch/grab_it_launch.py"
    ),
]

OLD = """        remappings=[
            ('joint_states', '/px150_no_elbow/joint_states'),
        ],"""

NEW = """        remappings=[
            ('joint_states', LaunchConfiguration('joint_states_topic')),
        ],"""


def patch(path: Path) -> None:
    t = path.read_text()
    if "LaunchConfiguration('joint_states_topic')" in t and "joint_states_topic" in t:
        # ensure remapping uses the arg
        if OLD in t:
            path.write_text(t.replace(OLD, NEW, 1))
            print(f"[patched] {path}")
        else:
            print(f"[ok] {path}")
        return
    if OLD in t:
        path.write_text(t.replace(OLD, NEW, 1))
        print(f"[patched] {path}")
    else:
        print(f"[warn] no remap block: {path}", file=sys.stderr)


def main() -> None:
    found = False
    for p in CANDIDATES:
        if p.exists():
            found = True
            patch(p)
    if not found:
        print("grab_it_launch.py not found", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
