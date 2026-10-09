from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="grab_it_grasp",
                executable="grasp_from_click",
                name="grasp_from_click",
                output="screen",
            )
        ]
    )
