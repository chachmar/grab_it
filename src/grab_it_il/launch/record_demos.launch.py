from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_WS = Path("/workspaces/isaac_ros-dev")
_XPROF = (
    _WS / "rtabmap_profile.xml"
    if (_WS / "rtabmap_profile.xml").is_file()
    else _WS / "crossover_only.xml"
)


def generate_launch_description():
    dataset = LaunchConfiguration("dataset_dir")
    source = LaunchConfiguration("source")
    return LaunchDescription(
        [
            SetEnvironmentVariable("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp"),
            SetEnvironmentVariable("ROS_DOMAIN_ID", "0"),
            SetEnvironmentVariable("FASTRTPS_DEFAULT_PROFILES_FILE", str(_XPROF)),
            DeclareLaunchArgument(
                "dataset_dir",
                default_value="/workspaces/isaac_ros-dev/datasets/tabletop_pick",
            ),
            DeclareLaunchArgument("source", default_value="teleop"),
            Node(
                package="grab_it_il",
                executable="record_demos",
                name="grab_it_il_record",
                output="screen",
                parameters=[{"dataset_dir": dataset, "source": source}],
            ),
        ]
    )
