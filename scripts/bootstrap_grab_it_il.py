#!/usr/bin/env python3
"""Write src/grab_it_il onto this workspace (AGX ~/dev-ws or /workspaces/isaac_ros-dev).

Run on the AGX host or inside the Isaac container:
  python3 bootstrap_grab_it_il.py
  # or: python3 bootstrap_grab_it_il.py /workspaces/isaac_ros-dev
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspaces/isaac_ros-dev")
PKG = ROOT / "src" / "grab_it_il"

FILES = {
    "package.xml": """<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>grab_it_il</name>
  <version>0.1.0</version>
  <description>Imitation-learning recorder and policy player for tabletop pick.</description>
  <maintainer email="ubuntu@todo.todo">grab_it</maintainer>
  <license>Apache-2.0</license>
  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>interbotix_xs_msgs</depend>
  <depend>cv_bridge</depend>
  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
""",
    "setup.cfg": """[develop]
script_dir=$base/lib/grab_it_il
[install]
install_scripts=$base/lib/grab_it_il
""",
    "setup.py": '''from setuptools import setup
from glob import glob

package_name = "grab_it_il"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="grab_it",
    maintainer_email="ubuntu@todo.todo",
    description="Imitation-learning recorder and policy player for tabletop pick.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "record_demos = grab_it_il.record_demos:main",
            "play_policy = grab_it_il.play_policy:main",
        ],
    },
)
''',
    "resource/grab_it_il": "grab_it_il\n",
    "grab_it_il/__init__.py": '"""grab_it imitation learning (tabletop pick)."""\n',
}


def main() -> None:
    # Copy live sources if this script sits next to the Windows/repo tree.
    here = Path(__file__).resolve().parent
    live = here.parent / "src" / "grab_it_il"
    if live.is_dir() and (live / "package.xml").is_file():
        import shutil

        PKG.parent.mkdir(parents=True, exist_ok=True)
        if PKG.exists():
            shutil.rmtree(PKG)
        shutil.copytree(live, PKG)
        print(f"copied {live} → {PKG}")
        return

    PKG.mkdir(parents=True, exist_ok=True)
    (PKG / "grab_it_il").mkdir(exist_ok=True)
    (PKG / "launch").mkdir(exist_ok=True)
    (PKG / "resource").mkdir(exist_ok=True)
    for rel, text in FILES.items():
        dest = PKG / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
    print(f"wrote skeleton {PKG}")
    print("This bootstrap only writes package metadata.")
    print("Copy the full src/grab_it_il from the Windows repo (record_demos.py, play_policy.py).")


if __name__ == "__main__":
    main()
