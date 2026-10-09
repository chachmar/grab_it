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
    return LaunchDescription(
        [
            SetEnvironmentVariable("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp"),
            SetEnvironmentVariable("ROS_DOMAIN_ID", "0"),
            SetEnvironmentVariable("FASTRTPS_DEFAULT_PROFILES_FILE", str(_XPROF)),
            DeclareLaunchArgument("checkpoint", default_value=""),
            DeclareLaunchArgument("replay_episode", default_value=""),
            DeclareLaunchArgument(
                "dataset_dir",
                default_value="/workspaces/isaac_ros-dev/datasets/tabletop_pick",
            ),
            DeclareLaunchArgument("autoplay", default_value="false"),
            DeclareLaunchArgument("device", default_value="cuda"),
            Node(
                package="grab_it_il",
                executable="play_policy",
                name="grab_it_il_play",
                output="screen",
                parameters=[
                    {
                        "checkpoint": LaunchConfiguration("checkpoint"),
                        "replay_episode": LaunchConfiguration("replay_episode"),
                        "dataset_dir": LaunchConfiguration("dataset_dir"),
                        "autoplay": LaunchConfiguration("autoplay"),
                        "device": LaunchConfiguration("device"),
                    }
                ],
            ),
        ]
    )
