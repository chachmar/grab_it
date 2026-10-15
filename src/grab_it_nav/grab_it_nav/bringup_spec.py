"""Node and parameter record for nav_bringup.sh and mapping_bringup.sh.

stack.py starts processes from these values. Nothing here imports ROS.
"""
from __future__ import annotations

from pathlib import Path

WS_DEFAULT = "/workspaces/isaac_ros-dev"

COLOR_TOPIC = "/d435i/d435i/color/image_raw"
DEPTH_TOPIC = "/d435i/d435i/aligned_depth_to_color/image_raw"
CAMERA_INFO_TOPIC = "/d435i/d435i/color/camera_info"

# Shell: for pkg in nav2_bringup depthimage_to_laserscan rtabmap_launch domain_bridge
NAV_REQUIRED_PACKAGES = (
    "nav2_bringup",
    "depthimage_to_laserscan",
    "rtabmap_launch",
    "domain_bridge",
)

# pkill -9 -f <pattern>, in nav_bringup.sh order, then sleep 2
NAV_PKILL = (
    "domain_bridge",
    "static_transform_publisher",
    "depthimage_to_laserscan",
    "rtabmap.launch.py",
    "rtabmap_slam/rtabmap",
    "map_to_nav.py",
    "scan_stamp_fix.py",
    "initialpose_to_rtab.py",
    "odom_to_tf.py",
    "navigation_launch.py",
    "controller_server",
    "planner_server",
    "bt_navigator",
    "lifecycle_manager",
)
NAV_PKILL_SLEEP_S = 2

# mapping_bringup.sh pkill list, then sleep 1
MAPPING_PKILL = (
    "domain_bridge",
    "static_transform_publisher",
    "depthimage_to_laserscan",
    "rtabmap.launch.py",
    "rtabmap_slam/rtabmap",
    "scan_stamp_fix",
    "map_to_nav",
    "odom_to_tf.py",
)
MAPPING_PKILL_SLEEP_S = 1

# depthimage_to_laserscan_node --ros-args (nav_bringup.sh)
DEPTH_SCAN_HEIGHT = 31
DEPTH_RANGE_MIN = 0.25
DEPTH_RANGE_MAX = 3.5

# RTAB localization args, same order and values as nav_bringup.sh
RTABMAP_NAV_ARGS = (
    ("rgb_topic", COLOR_TOPIC),
    ("depth_topic", DEPTH_TOPIC),
    ("camera_info_topic", CAMERA_INFO_TOPIC),
    ("rgb_image_transport", "compressed"),
    ("frame_id", "base_link"),
    ("odom_frame_id", "odom"),
    ("map_frame_id", "map"),
    ("odom_topic", "/odom"),
    ("visual_odometry", "false"),
    ("approx_sync", "true"),
    ("approx_sync_max_interval", "0.15"),
    ("queue_size", "5"),
    ("sync_queue_size", "5"),
    ("topic_queue_size", "2"),
    ("wait_for_transform", "2.0"),
    # database_path inserted here by rtabmap_nav_cmd()
    ("localization", "true"),
    (
        "rtabmap_args",
        "--Vis/MinInliers 15 --RGBD/OptimizeMaxError 3.0 --RGBD/Force3DoF true "
        "--RGBD/ProximityBySpace true --RGBD/LocalRadius 10 "
        "--RGBD/ProximityPathMaxNeighbors 8 --RGBD/ProximityMaxGraphDepth 50 "
        "--Rtabmap/DetectionRate 2 --Kp/MaxFeatures 500 "
        "--Mem/ImagePreDecimation 2 --Mem/InitWMWithAllNodes true",
    ),
    ("rviz", "false"),
    ("rtabmap_viz", "false"),
)

# mapping_bringup.sh rtabmap.launch.py arguments (no localization flag)
RTABMAP_MAPPING_ARGS = (
    ("rgb_topic", COLOR_TOPIC),
    ("depth_topic", DEPTH_TOPIC),
    ("camera_info_topic", CAMERA_INFO_TOPIC),
    ("frame_id", "base_link"),
    ("odom_topic", "/odom"),
    ("visual_odometry", "false"),
    ("approx_sync", "true"),
    ("approx_sync_max_interval", "0.1"),
    ("queue_size", "30"),
    ("wait_for_transform", "1.0"),
    # database_path inserted by rtabmap_mapping_cmd()
    ("rtabmap_args", "--delete_db_on_start"),
    ("rviz", "false"),
    ("rtabmap_viz", "false"),
)

# mapping_bringup.sh static TF literals (does not source camera_tf.env)
MAPPING_STATIC_TF = {
    "x": "0.15",
    "y": "0",
    "z": "0.30",
    "roll": "0",
    "pitch": "0.35",
    "yaw": "0",
    "frame_id": "base_link",
    "child_frame_id": "camera_link",
}

NAV2_USE_SIM_TIME = "false"
NAV2_AUTOSTART = "true"

# Gate timeouts from nav_bringup.sh
CAMERA_HZ_TIMEOUT_S = 15
TF_ATTEMPTS = 20
TF_ECHO_TIMEOUT_S = 3
TOPIC_HZ_TIMEOUT_S = 6
SCAN_WAIT_TIMEOUT_S = 30
MAP_WAIT_TIMEOUT_S = 90
MAP_RELAY_WAIT_TIMEOUT_S = 45
SLEEP_AFTER_BRIDGE_S = 3
SLEEP_AFTER_ODOM_S = 1
SLEEP_AFTER_STATIC_TF_NAV_S = 3
SLEEP_AFTER_STATIC_TF_MAPPING_S = 4
SLEEP_AFTER_SCAN_START_S = 2
SLEEP_AFTER_RTABMAP_S = 5
SLEEP_AFTER_MAP_RELAY_S = 2
SLEEP_AFTER_POSE_RELAY_S = 1

# DDS profile used for each long-running process.
# "crossover" = rtabmap_profile.xml, "bridge" = bridge_both.xml
NAV_NODE_DDS = (
    ("domain_bridge", "bridge"),
    ("odom_to_tf", "bridge"),
    ("static_transform_publisher", "bridge"),
    ("depthimage_to_laserscan", "crossover"),
    ("scan_stamp_fix", "bridge"),
    ("rtabmap_localization", "crossover"),
    ("map_to_nav", "crossover"),
    ("initialpose_to_rtab", "crossover"),
    ("navigation_launch", "crossover"),
)

MAPPING_NODE_DDS = (
    ("domain_bridge", "bridge"),
    ("odom_to_tf", "bridge"),
    ("static_transform_publisher", "crossover"),
    ("rtabmap_mapping", "crossover"),
)


def _launch_pairs(pairs: tuple[tuple[str, str], ...], database_path: str, insert_before: str) -> list[str]:
    out: list[str] = []
    for key, value in pairs:
        if key == insert_before:
            out.append(f"database_path:={database_path}")
        out.append(f"{key}:={value}")
    return out


def domain_bridge_cmd(bridge_yaml: str) -> list[str]:
    return ["ros2", "run", "domain_bridge", "domain_bridge", bridge_yaml]


def static_tf_cmd(
    x: str, y: str, z: str, roll: str, pitch: str, yaw: str, child_frame: str
) -> list[str]:
    return [
        "ros2", "run", "tf2_ros", "static_transform_publisher",
        "--x", x, "--y", y, "--z", z,
        "--roll", roll, "--pitch", pitch, "--yaw", yaw,
        "--frame-id", "base_link", "--child-frame-id", child_frame,
    ]


def depth_scan_cmd(output_frame: str) -> list[str]:
    return [
        "ros2", "run", "depthimage_to_laserscan", "depthimage_to_laserscan_node",
        "--ros-args",
        "-r", f"depth:={DEPTH_TOPIC}",
        "-r", f"depth_camera_info:={CAMERA_INFO_TOPIC}",
        "-r", "scan:=/scan_raw",
        "-p", f"scan_height:={DEPTH_SCAN_HEIGHT}",
        "-p", f"range_min:={DEPTH_RANGE_MIN}",
        "-p", f"range_max:={DEPTH_RANGE_MAX}",
        "-p", f"output_frame:={output_frame}",
    ]


def rtabmap_nav_cmd(database_path: str) -> list[str]:
    return [
        "ros2", "launch", "rtabmap_launch", "rtabmap.launch.py",
        *_launch_pairs(RTABMAP_NAV_ARGS, database_path, "localization"),
    ]


def rtabmap_mapping_cmd(database_path: str) -> list[str]:
    return [
        "ros2", "launch", "rtabmap_launch", "rtabmap.launch.py",
        *_launch_pairs(RTABMAP_MAPPING_ARGS, database_path, "rtabmap_args"),
    ]


def nav2_cmd(params_file: str) -> list[str]:
    return [
        "ros2", "launch", "nav2_bringup", "navigation_launch.py",
        f"use_sim_time:={NAV2_USE_SIM_TIME}",
        f"params_file:={params_file}",
        f"autostart:={NAV2_AUTOSTART}",
    ]


def python_cmd(script_path: str, *args: str) -> list[str]:
    return ["python3", script_path, *args]


def load_camera_tf(path: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def verify() -> None:
    """Assert the spec still matches the shell values this package was taken from."""
    scan = depth_scan_cmd("camera_link")
    joined = " ".join(scan)
    assert "depth:=/d435i/d435i/aligned_depth_to_color/image_raw" in joined
    assert "depth_camera_info:=/d435i/d435i/color/camera_info" in joined
    assert "scan:=/scan_raw" in joined
    assert "scan_height:=31" in joined
    assert "range_min:=0.25" in joined
    assert "range_max:=3.5" in joined
    assert "output_frame:=camera_link" in joined

    nav = " ".join(rtabmap_nav_cmd("/tmp/rtabmap.db"))
    assert "rtabmap.launch.py" in nav
    assert "localization:=true" in nav
    assert "approx_sync_max_interval:=0.15" in nav
    assert "queue_size:=5" in nav
    assert "rgb_image_transport:=compressed" in nav
    assert "visual_odometry:=false" in nav
    assert "--Mem/InitWMWithAllNodes true" in nav
    assert "database_path:=/tmp/rtabmap.db" in nav
    assert "rviz:=false" in nav and "rtabmap_viz:=false" in nav

    mapping = " ".join(rtabmap_mapping_cmd("/tmp/rtabmap.db"))
    assert "approx_sync_max_interval:=0.1" in mapping
    assert "queue_size:=30" in mapping
    assert "wait_for_transform:=1.0" in mapping
    assert 'rtabmap_args:=--delete_db_on_start' in mapping
    assert "localization:=" not in mapping

    nav2 = nav2_cmd("/tmp/nav2.yaml")
    assert nav2[:4] == ["ros2", "launch", "nav2_bringup", "navigation_launch.py"]
    assert "use_sim_time:=false" in nav2
    assert "params_file:=/tmp/nav2.yaml" in nav2
    assert "autostart:=true" in nav2

    tf = static_tf_cmd("0.15", "0", "0.30", "0", "0.35", "0", "camera_link")
    assert tf[0:4] == ["ros2", "run", "tf2_ros", "static_transform_publisher"]
    assert "--pitch" in tf and tf[tf.index("--pitch") + 1] == "0.35"

    names = [name for name, _dds in NAV_NODE_DDS]
    assert names == [
        "domain_bridge",
        "odom_to_tf",
        "static_transform_publisher",
        "depthimage_to_laserscan",
        "scan_stamp_fix",
        "rtabmap_localization",
        "map_to_nav",
        "initialpose_to_rtab",
        "navigation_launch",
    ]
    assert dict(NAV_NODE_DDS)["static_transform_publisher"] == "bridge"
    assert dict(NAV_NODE_DDS)["depthimage_to_laserscan"] == "crossover"
    assert dict(NAV_NODE_DDS)["scan_stamp_fix"] == "bridge"
    assert dict(MAPPING_NODE_DDS)["static_transform_publisher"] == "crossover"

    here = Path(__file__).resolve().parent
    map_src = (here / "map_to_nav.py").read_text(encoding="utf-8")
    pose_src = (here / "initialpose_to_rtab.py").read_text(encoding="utf-8")
    assert "ERODE_CELLS = 1" in map_src
    assert "OCCUPIED_THRESH = 50" in map_src
    assert "'/rtabmap/map'" in map_src and "'/map'" in map_src
    assert "COV_X = 0.25" in pose_src
    assert "COV_Y = 0.25" in pose_src
    assert "/initialpose" in pose_src and "/rtabmap/initialpose" in pose_src

    env_path = here.parent / "config" / "camera_tf.env"
    env = load_camera_tf(str(env_path))
    assert env["CAM_TF_X"] == MAPPING_STATIC_TF["x"]
    assert float(env["CAM_TF_Y"]) == float(MAPPING_STATIC_TF["y"])
    assert env["CAM_TF_Z"] == MAPPING_STATIC_TF["z"]
    assert float(env["CAM_TF_ROLL"]) == float(MAPPING_STATIC_TF["roll"])
    assert env["CAM_TF_PITCH"] == MAPPING_STATIC_TF["pitch"]
    assert float(env["CAM_TF_YAW"]) == float(MAPPING_STATIC_TF["yaw"])
    assert env["CAM_FRAME"] == "camera_link"
    assert env["CAM_OPTICAL"] == "camera_color_optical_frame"


def mapping_table(
    database_path: str,
    params_file: str,
    bridge_yaml: str,
    cam: dict[str, str],
) -> str:
    """Human-readable node map for the nav stack. Used by the adoption notes."""
    cam_frame = cam.get("CAM_FRAME", "camera_link")
    static = " ".join(
        static_tf_cmd(
            cam["CAM_TF_X"],
            cam["CAM_TF_Y"],
            cam["CAM_TF_Z"],
            cam["CAM_TF_ROLL"],
            cam["CAM_TF_PITCH"],
            cam["CAM_TF_YAW"],
            cam_frame,
        )
    )
    rows = [
        ("domain_bridge", "bridge_both.xml", " ".join(domain_bridge_cmd(bridge_yaml))),
        ("odom_to_tf", "bridge_both.xml", "python3 odom_to_tf.py"),
        ("static_transform_publisher", "bridge_both.xml", static),
        ("depthimage_to_laserscan", "rtabmap_profile.xml", " ".join(depth_scan_cmd(cam_frame))),
        ("scan_stamp_fix", "bridge_both.xml", "python3 scan_stamp_fix.py"),
        ("rtabmap_localization", "rtabmap_profile.xml", " ".join(rtabmap_nav_cmd(database_path))),
        ("map_to_nav", "rtabmap_profile.xml", "python3 map_to_nav.py"),
        ("initialpose_to_rtab", "rtabmap_profile.xml", "python3 initialpose_to_rtab.py"),
        ("navigation_launch", "rtabmap_profile.xml", " ".join(nav2_cmd(params_file))),
    ]
    lines = ["| order | node | DDS profile | command |", "|---|---|---|---|"]
    for i, (name, dds, cmd) in enumerate(rows, start=1):
        lines.append(f"| {i} | {name} | {dds} | `{cmd}` |")
    return "\n".join(lines)
