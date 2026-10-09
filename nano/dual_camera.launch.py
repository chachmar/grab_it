from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # Wide USB cam (Microdia /dev/video_wide -> video0) + D435i.
    # D435i: matched 848x480 color+depth, aligned, hardware-synced.
    wide_cam = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='wide_cam',
        namespace='wide_cam',
        parameters=[{
            'video_device': '/dev/video_wide',
            'image_width': 640,
            'image_height': 480,
            'pixel_format': 'mjpeg2rgb',
            'camera_frame_id': 'wide_cam',
            'framerate': 30.0,
            'io_method': 'mmap',
            'camera_name': 'wide_cam',
        }],
        output='screen',
    )
    realsense = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='d435i',
        namespace='d435i',
        parameters=[{
            'serial_no': '934222070174',
            'enable_color': True,
            'enable_depth': True,
            'enable_infra1': False,
            'enable_infra2': False,
            'enable_gyro': True,
            'enable_accel': True,
            'unite_imu_method': 1,
            'depth_module.depth_profile': '848x480x15',
            'rgb_camera.color_profile': '848x480x15',
            'align_depth.enable': True,
            'enable_sync': True,
        }],
        output='screen',
    )
    return LaunchDescription([wide_cam, realsense])
