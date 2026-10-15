"""RTAB-Map mapping. Same nodes and arguments as mapping_bringup.sh.

1. domain_bridge                 DDS bridge_both.xml
2. odom_to_tf.py                 DDS bridge_both.xml
3. static_transform_publisher    DDS rtabmap_profile.xml
   x=0.15 y=0 z=0.30 roll=0 pitch=0.35 yaw=0  (literals from mapping_bringup.sh)
4. rtabmap.launch.py             DDS rtabmap_profile.xml  rtabmap_args:=--delete_db_on_start

That last argument deletes rtabmap.db. remap_map.sh still calls mapping_bringup.sh,
which is unchanged. This launch is the package entry for the same mapping stack.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction

from grab_it_nav.bringup_spec import WS_DEFAULT
from grab_it_nav.stack import mapping_main

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
        OpaqueFunction(function=mapping_main),
    ])
