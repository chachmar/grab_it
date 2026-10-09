#!/usr/bin/env python3
"""Restore the validated px150_no_elbow kinematics (no upper-arm boom).

Hardware removed the elbow servos AND the upper-arm boom. The working model
(from image my_isaac_perceptor:interbotix-moveit-no-elbow) is:

  - shoulder -> forearm_link directly (no upper_arm / elbow joint)
  - shoulder origin rpy="0 -1.5708 0"  (frame fix for removed boom)
  - shoulder axis stays "0 1 0"        (never flip — inverts commands)
  - wrist_angle axis "0 -1 0"          (correct bend after forearm rotate)

Do NOT re-insert stock upper_arm + fixed_elbow — that makes FK too long and
drives Sleep/Home into the AGX/Nano stack.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

CANDIDATES = [
    Path(
        "/workspaces/isaac_ros-dev/src/interbotix/interbotix_ros_manipulators/"
        "interbotix_ros_xsarms/interbotix_xsarm_descriptions/urdf/"
        "px150_no_elbow.urdf.xacro"
    ),
    Path(
        "/workspaces/isaac_ros-dev/install/interbotix_xsarm_descriptions/share/"
        "interbotix_xsarm_descriptions/urdf/px150_no_elbow.urdf.xacro"
    ),
    Path(
        "/home/ubuntu/dev-ws/src/interbotix/interbotix_ros_manipulators/"
        "interbotix_ros_xsarms/interbotix_xsarm_descriptions/urdf/"
        "px150_no_elbow.urdf.xacro"
    ),
]

# Validated shoulder block: short arm, -pi/2 pitch, stock axis sign.
SHOULDER_JOINT = """  <joint name="shoulder" type="revolute">
    <axis xyz="0 1 0"/>
    <limit effort="18"
           lower="${radians(-113)}" upper="${radians(111)}"
           velocity="${pi}"/>
    <!-- Removed upper-arm boom: pitch frame so servo-zero matches hardware. -->
    <origin rpy="0 -1.5708 0" xyz="0 0 0.03891"/>
    <parent link="$(arg robot_name)/shoulder_link"/>
    <child  link="$(arg robot_name)/forearm_link"/>
    <dynamics friction="0.1"/>
  </joint>
"""


def strip_upper_arm_chain(text: str) -> str:
    """Remove inserted upper_arm_link + fixed/elbow joint if present."""
    # shoulder..fixed_elbow/elbow block that includes upper_arm_link
    pat = re.compile(
        r'  <joint name="shoulder" type="revolute">[\s\S]*?'
        r'  <joint name="(?:elbow|fixed_elbow)" type="fixed">[\s\S]*?</joint>\n',
        re.M,
    )
    m = pat.search(text)
    if m:
        text = text[: m.start()] + SHOULDER_JOINT + "\n" + text[m.end() :]
        print("[strip] removed upper_arm/fixed_elbow chain")
        return text

    # orphan upper_arm link left behind
    text2 = re.sub(
        r'  <link name="\$\(arg robot_name\)/upper_arm_link">[\s\S]*?</link>\n+',
        "",
        text,
        count=1,
    )
    if text2 != text:
        print("[strip] removed orphan upper_arm_link")
        text = text2
    return text


def ensure_shoulder(text: str) -> str:
    m = re.search(
        r'  <joint name="shoulder" type="revolute">[\s\S]*?</joint>\n',
        text,
    )
    if not m:
        raise SystemExit("shoulder revolute joint not found")
    block = m.group(0)
    if (
        'rpy="0 -1.5708 0"' in block
        and 'child  link="$(arg robot_name)/forearm_link"' in block
        and 'axis xyz="0 1 0"' in block
        and "upper_arm_link" not in block
    ):
        print("[ok] shoulder geometry already correct")
        return text
    text = text[: m.start()] + SHOULDER_JOINT + "\n" + text[m.end() :]
    print("[fix] shoulder -> forearm with rpy 0 -1.5708 0, axis 0 1 0")
    return text


def ensure_wrist_axis(text: str) -> str:
    m = re.search(
        r'  <joint name="wrist_angle" type="revolute">[\s\S]*?</joint>\n',
        text,
    )
    if not m:
        raise SystemExit("wrist_angle joint not found")
    block = m.group(0)
    if 'axis xyz="0 -1 0"' in block:
        print("[ok] wrist_angle axis already 0 -1 0")
        return text
    block2 = re.sub(
        r'<axis xyz="0\s+1\s+0"/>',
        '<axis xyz="0 -1 0"/>',
        block,
        count=1,
    )
    if block2 == block:
        # already something else — force replace first axis line
        block2 = re.sub(
            r'<axis xyz="[^"]+"/>',
            '<axis xyz="0 -1 0"/>',
            block,
            count=1,
        )
    text = text[: m.start()] + block2 + text[m.end() :]
    print("[fix] wrist_angle axis -> 0 -1 0")
    return text


def patch_file(path: Path) -> None:
    text = path.read_text()
    orig = text
    text = strip_upper_arm_chain(text)
    text = ensure_shoulder(text)
    text = ensure_wrist_axis(text)
    # safety: never leave upper_arm in the model
    if "upper_arm_link" in text:
        text = re.sub(
            r'  <link name="\$\(arg robot_name\)/upper_arm_link">[\s\S]*?</link>\n+',
            "",
            text,
        )
        text = re.sub(
            r'  <joint name="(?:elbow|fixed_elbow)" type="fixed">[\s\S]*?</joint>\n+',
            "",
            text,
        )
        print("[strip] cleaned residual upper_arm/elbow")
    if text != orig:
        path.write_text(text)
        print(f"[wrote] {path}")
    else:
        print(f"[ok] {path}")


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
        patch_file(p)
    if not found:
        print("px150_no_elbow.urdf.xacro not found", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
