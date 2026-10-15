"""Localization Nav2 stack. Same order as nav_bringup.sh.

1. domain_bridge                 DDS bridge_both.xml     create3_bridge.yaml
2. odom_to_tf.py                 DDS bridge_both.xml
3. static_transform_publisher    DDS bridge_both.xml     camera_tf.env
4. depthimage_to_laserscan       DDS rtabmap_profile.xml scan_height 31, range 0.25–3.5
5. scan_stamp_fix.py             DDS bridge_both.xml
6. rtabmap.launch.py             DDS rtabmap_profile.xml localization:=true
7. map_to_nav.py                 DDS rtabmap_profile.xml /rtabmap/map -> /map
8. initialpose_to_rtab.py        DDS rtabmap_profile.xml
9. navigation_launch.py          DDS rtabmap_profile.xml use_sim_time:=false autostart:=true

Gates between those steps (camera hz, TF, preflight, /scan, /rtabmap/map, /map)
live in grab_it_nav.stack and follow nav_bringup.sh. This file is the
`ros2 launch grab_it_nav nav.launch.py` entry point. nav_bringup.sh is a wrapper.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction

from grab_it_nav.bringup_spec import WS_DEFAULT
from grab_it_nav.stack import nav_main

_WS = WS_DEFAULT


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("workspace", default_value=_WS),
        DeclareLaunchArgument("database_path", default_value=f"{_WS}/rtabmap.db"),
        DeclareLaunchArgument(
            "params_file",
            default_value=f"{_WS}/config/nav2_params_create3.yaml",
        ),
        DeclareLaunchArgument("bridge_yaml", default_value=f"{_WS}/create3_bridge.yaml"),
        DeclareLaunchArgument(
            "crossover_profile",
            default_value=f"{_WS}/rtabmap_profile.xml",
        ),
        DeclareLaunchArgument("bridge_profile", default_value=f"{_WS}/bridge_both.xml"),
        DeclareLaunchArgument(
            "camera_tf_env",
            default_value=f"{_WS}/config/camera_tf.env",
        ),
        OpaqueFunction(function=nav_main),
    ])
