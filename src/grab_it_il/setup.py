from setuptools import setup
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
