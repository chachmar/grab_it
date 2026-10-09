from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    cfg = str(
        Path(get_package_share_directory("grab_it_quest")) / "config" / "quest_teleop.yaml"
    )
    return LaunchDescription(
        [
            SetEnvironmentVariable("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp"),
            SetEnvironmentVariable("ROS_DOMAIN_ID", "0"),
            # Do not pin FASTRTPS_DEFAULT_PROFILES_FILE. The Isaac twin is on
            # Wi-Fi / default interfaces; rtabmap/crossover profiles hide
            # /grab_it/quest/* from that graph.
            DeclareLaunchArgument("target", default_value="sim"),
            DeclareLaunchArgument("allow_both", default_value="false"),
            DeclareLaunchArgument("clutch_mode", default_value="grip"),
            DeclareLaunchArgument("scale", default_value="1.0"),
            Node(
                package="grab_it_quest",
                executable="quest_ee_teleop",
                name="quest_teleop",
                output="screen",
                parameters=[
                    cfg,
                    {
                        "target": LaunchConfiguration("target"),
                        "allow_both": LaunchConfiguration("allow_both"),
                        "clutch_mode": LaunchConfiguration("clutch_mode"),
                        "scale": LaunchConfiguration("scale"),
                    },
                ],
            ),
        ]
    )
