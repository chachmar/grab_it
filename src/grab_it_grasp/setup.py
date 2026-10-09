from setuptools import setup
import os
from glob import glob

package_name = "grab_it_grasp"

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
    description="Click-to-grasp from D435i depth using MoveIt IK + arm trajectory.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "grasp_from_click = grab_it_grasp.grasp_from_click:main",
        ],
    },
)
