from glob import glob

from setuptools import setup

package_name = "grab_it_nav"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.py")),
        (f"share/{package_name}/config", glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="grab_it",
    maintainer_email="ubuntu@todo.todo",
    description="Create 3 Nav2 localization stack and RTAB-Map mapping for grab_it.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "odom_to_tf = grab_it_nav.odom_to_tf:main",
            "scan_stamp_fix = grab_it_nav.scan_stamp_fix:main",
            "wait_map_probe = grab_it_nav.wait_map_probe:main",
            "go_to_named = grab_it_nav.go_to_named:main",
            "map_to_nav = grab_it_nav.map_to_nav:main",
            "initialpose_to_rtab = grab_it_nav.initialpose_to_rtab:main",
        ],
    },
)
