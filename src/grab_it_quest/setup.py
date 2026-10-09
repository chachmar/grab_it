from setuptools import setup
from glob import glob

package_name = "grab_it_quest"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="grab_it",
    maintainer_email="ubuntu@todo.todo",
    description="Quest 3 4-DOF end-effector teleop for the PX150 (sim + real).",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "quest_ee_teleop = grab_it_quest.quest_ee_teleop:main",
        ],
    },
)
