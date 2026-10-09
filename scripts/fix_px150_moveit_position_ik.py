#!/usr/bin/env python3
"""Enable position-only IK for px150_no_elbow (4-DOF cannot meet full 6D poses)."""
from __future__ import annotations

from pathlib import Path

CANDIDATES = [
    Path(
        "/workspaces/isaac_ros-dev/install/interbotix_xsarm_moveit/share/"
        "interbotix_xsarm_moveit/config/kinematics.yaml"
    ),
    Path(
        "/workspaces/isaac_ros-dev/src/interbotix/interbotix_ros_manipulators/"
        "interbotix_ros_xsarms/interbotix_xsarm_moveit/config/kinematics.yaml"
    ),
]


def patch(text: str) -> str:
    out = text
    if "position_only_ik:" in out:
        out = out.replace("position_only_ik: false", "position_only_ik: true")
        out = out.replace("position_only_ik: False", "position_only_ik: true")
    else:
        # insert under interbotix_arm block
        needle = "kinematics_solver_timeout:"
        if needle in out:
            lines = out.splitlines(True)
            new_lines = []
            inserted = False
            for line in lines:
                new_lines.append(line)
                if (not inserted) and "kinematics_solver_timeout:" in line:
                    indent = line[: len(line) - len(line.lstrip())]
                    new_lines.append(f"{indent}position_only_ik: true\n")
                    inserted = True
            out = "".join(new_lines)

    # Give LMA more than 5ms — full-pose search was failing spuriously.
    out = out.replace(
        "kinematics_solver_timeout: 0.005",
        "kinematics_solver_timeout: 0.05",
    )
    return out


def main() -> int:
    changed = 0
    for path in CANDIDATES:
        if not path.is_file():
            continue
        old = path.read_text(encoding="utf-8")
        new = patch(old)
        if new != old:
            path.write_text(new, encoding="utf-8")
            print(f"[ik] patched {path}")
            changed += 1
        else:
            print(f"[ik] already ok {path}")
    if not changed:
        # still success if files already correct
        if any(p.is_file() for p in CANDIDATES):
            return 0
        print("[ik] no kinematics.yaml found")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
