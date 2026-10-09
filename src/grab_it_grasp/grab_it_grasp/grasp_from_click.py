#!/usr/bin/env python3
"""Click-to-grasp: D435i depth pixel → TF → MoveIt IK → approach/close/retreat."""
from __future__ import annotations

import json
import math
import os
import re
import threading
import time
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rclpy
import yaml
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, Pose, PoseStamped, Quaternion
from interbotix_xs_msgs.msg import JointGroupCommand, JointSingleCommand
from interbotix_xs_msgs.srv import OperatingModes, RegisterValues, TorqueEnable
from moveit_msgs.msg import PositionIKRequest, RobotState
from moveit_msgs.srv import GetPositionIK
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, JointState

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import tf2_geometry_msgs  # noqa: F401  # registers PoseStamped conversion


DEFAULT_PARAMS = Path(
    os.environ.get(
        "GRAB_IT_GRASP_PARAMS",
        "/workspaces/isaac_ros-dev/config/grasp_params.yaml",
    )
)

# BEST_EFFORT + keep 1: never backpressure usb_cam / mjpeg_relay.
# RELIABLE keep-1 on this topic freezes the GUI wide pane.
_WIDE_JPEG_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

# Nano D435i pubs are RELIABLE. BEST_EFFORT sensor_data often never connects
# across the AGX Fast-DDS bridge (depth snapshot looks empty).
_DEPTH_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
)


def _rpy_to_quat(roll: float, pitch: float, yaw: float) -> Quaternion:
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q


def _rpy_to_mat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rz = np.array(
        [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    ry = np.array(
        [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64
    )
    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64
    )
    return rz @ ry @ rx


def _axis_angle_mat(axis: np.ndarray, ang: float) -> np.ndarray:
    n = float(np.linalg.norm(axis))
    if n < 1e-9 or abs(ang) < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = (axis / n).tolist()
    c, s = math.cos(ang), math.sin(ang)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


# Wireframe cube: 0:--0 1:--1 2:-+0 3:-+1 4:+-0 5:+-1 6:++0 7:++1
# Nested loops: sx in (-h,+h), sy, sz. -X = gripper-approach face.
_CUBE_EDGES = [
    (0, 1), (0, 2), (0, 4),
    (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6),
    (5, 7),
    (6, 7),
]
_CUBE_APPROACH_FACE = [0, 1, 3, 2]  # sx=-hx, toward gripper
_CUBE_OPENING_FACE = [4, 5, 7, 6]   # sx=+hx, out the fingers
# On each solid face: red = +Y (jaw span), blue = +Z (gripper top).
_CUBE_FACE_ARROWS = {
    "approach": {"red": (0, 2), "blue": (0, 1)},
    "opening": {"red": (4, 6), "blue": (4, 5)},
}
# Semantic keypoints on the black band (D435i ↔ wide).
_MARKER_KEYS = ("center", "top", "bottom", "left", "right")
_MARKER_RMSE_PX = 12.0


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _load_cfg(path: Path) -> dict:
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("grasp", data)
    return {}


class GraspFromClick(Node):
    def __init__(self):
        super().__init__("grasp_from_click")
        self._cg = ReentrantCallbackGroup()
        cfg = _load_cfg(DEFAULT_PARAMS)

        self.depth_topic = cfg.get(
            "depth_topic", "/d435i/d435i/aligned_depth_to_color/image_raw"
        )
        self.info_topic = cfg.get(
            "camera_info_topic", "/d435i/d435i/color/camera_info"
        )
        self.camera_frame = cfg.get("camera_frame", "camera_color_optical_frame")
        self.planning_frame = cfg.get("planning_frame", "px150_no_elbow/base_link")
        self.ee_link = cfg.get("ee_link", "px150_no_elbow/ee_gripper_link")
        self.planning_group = cfg.get("planning_group", "interbotix_arm")
        self.arm_joints = list(
            cfg.get(
                "arm_joints",
                ["waist", "shoulder", "wrist_angle", "wrist_rotate"],
            )
        )
        self.traj_action = cfg.get(
            "follow_joint_trajectory_action",
            "/px150_no_elbow/arm_controller/follow_joint_trajectory",
        )
        self.ik_service = cfg.get("compute_ik_service", "/compute_ik")
        self.pan_lock = float(cfg.get("pan_lock_rad", 0.0))
        self.approach_z = float(cfg.get("approach_offset_z", 0.02))
        self.grasp_z = float(cfg.get("grasp_offset_z", 0.00))
        self.retreat_z = float(cfg.get("retreat_offset_z", 0.10))
        self.grasp_standoff_m = float(cfg.get("grasp_standoff_m", 0.05))
        self.gripper_bar_offset_x = float(
            cfg.get("gripper_bar_offset_x", 0.0)
        )
        rpy = cfg.get("grasp_rpy", [0.0, 0.0, 0.0])
        self.grasp_quat = _rpy_to_quat(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        self.use_current_ee_orient = bool(cfg.get("use_current_ee_orientation", True))
        self.avoid_collisions = bool(cfg.get("avoid_collisions", False))
        self.subscribe_depth = bool(cfg.get("subscribe_depth", False))
        self.depth_window = int(cfg.get("depth_window", 7))
        self.min_depth = float(cfg.get("min_depth_m", 0.12))
        self.max_depth = float(cfg.get("max_depth_m", 1.20))
        self.depth_near_pct = float(cfg.get("depth_near_percentile", 20))
        self.use_table_plane = bool(cfg.get("use_table_plane", True))
        self.table_z = float(cfg.get("table_z", 0.04))
        self.object_height = float(cfg.get("object_height", 0.07))
        # Extra lift after plane hit (TCP vs fingertip / residual).
        self.grasp_z_bias = float(cfg.get("grasp_z_bias", 0.02))
        # If set, override TCP Z after plane XY (hand-calibrated object height).
        self.grasp_tcp_z = float(cfg.get("grasp_tcp_z", 0.0))
        # URDF d435i origin is aft/high vs measured nav camera_link. Without
        # this, table hits land ~20cm too close ("always below the click").
        off = cfg.get("click_origin_offset", [0.0, 0.0, 0.0])
        self.click_origin_offset = np.array(
            [float(off[0]), float(off[1]), float(off[2])], dtype=np.float64
        )
        # Push the grasp XY farther than the D435i projection so the fingers
        # surround the object instead of stopping short in front of it.
        self.grasp_farther_m = float(cfg.get("grasp_farther_m", 0.0))
        # Gripper landing box (half-extents, m) drawn on the D435i after a click.
        # Oriented: forward = radial toward object, lateral = jaw span, up = Z.
        # True cube whose edge matches the inner jaw opening (~40 mm).
        cube_m = float(cfg.get("landing_cube_m", 0.040))
        half = 0.5 * cube_m
        self.landing_box_half = (
            float(cfg.get("landing_box_depth_m", half)),
            float(cfg.get("landing_box_width_m", half)),
            float(cfg.get("landing_box_height_m", half)),
        )
        self.show_landing_cube = bool(cfg.get("show_landing_cube", True))
        self.d435i_reach_overlay = bool(cfg.get("d435i_reach_overlay", True))
        self.d435i_reach_grid_u = max(4, int(cfg.get("d435i_reach_grid_u", 32)))
        self.d435i_reach_grid_v = max(4, int(cfg.get("d435i_reach_grid_v", 24)))
        self.d435i_reach_period_s = float(cfg.get("d435i_reach_period_s", 1.5))
        # Only paint points this far above table_z (objects, not the tabletop).
        self.d435i_reach_min_above_table = float(
            cfg.get("d435i_reach_min_above_table", 0.025)
        )
        self.d435i_reach_max_above_table = float(
            cfg.get("d435i_reach_max_above_table", 0.28)
        )
        self.gripper_cube_offset_x = float(cfg.get("gripper_cube_offset_x", 0.040))
        # Extra along cube +X after the opening meets the approach face.
        self.grasp_deeper_m = float(cfg.get("grasp_deeper_m", 0.03))
        ext = cfg.get("wide_extrinsic_xyz", [0.0, 0.0, 0.0])
        self.wide_extrinsic_xyz = np.array(
            [float(ext[0]), float(ext[1]), float(ext[2])], dtype=np.float64
        )
        erpy = cfg.get("wide_extrinsic_rpy", [0.0, 0.0, 0.0])
        self.wide_extrinsic_rpy = (
            float(erpy[0]),
            float(erpy[1]),
            float(erpy[2]),
        )
        # Positive shifts the click down in the image (nearer); negative = farther.
        self.click_v_bias = float(cfg.get("click_v_bias", 0.0))
        # URDF pan+cam_pitch looks too steep. 1.0 = trust TF; 0.4 = use 40% of pan.
        self.click_pan_scale = float(cfg.get("click_pan_scale", 1.0))
        self.click_pan_z_ref = float(cfg.get("click_pan_z_ref", 0.85))
        self.ws_min = list(cfg.get("workspace_min", [0.05, -0.35, 0.00]))
        self.ws_max = list(cfg.get("workspace_max", [0.42, 0.35, 0.38]))
        self.max_reach = float(cfg.get("max_horizontal_reach", 0.42))
        self.seg_dur = float(cfg.get("segment_duration_s", 2.5))
        self.ik_timeout = float(cfg.get("ik_timeout_s", 3.0))
        # Gripper joint (rad), same mapping as the GUI slider.
        self.grip_joint_closed = float(
            os.environ.get(
                "GRAB_IT_GRIPPER_JOINT_CLOSED",
                cfg.get("gripper_joint_closed", -0.95),
            )
        )
        self.grip_joint_open = float(
            os.environ.get(
                "GRAB_IT_GRIPPER_JOINT_OPEN",
                cfg.get("gripper_joint_open", 1.65),
            )
        )
        # Legacy PWM scale kept only for config compatibility; grasp now
        # commands position (the GUI slider already flipped the servo).
        self.grip_open = float(cfg.get("gripper_open", 1.0))
        self.grip_close = float(cfg.get("gripper_close", -0.35))
        self.grip_settle = float(cfg.get("gripper_settle_s", 0.8))
        self.grip_close_s = float(cfg.get("gripper_close_s", 4.0))
        self.grip_pwm_scale = float(cfg.get("gripper_pwm_scale", 650.0))
        self.grip_current_limit = int(cfg.get("gripper_current_limit", 900))
        self.grip_hold_extra = float(cfg.get("gripper_hold_extra_rad", 0.12))
        # Max joint step per 50ms tick while pinching. Full-closed goals every
        # tick made the XL430 velocity profile cruise after a slow search pulse.
        self.grip_close_step_rad = float(cfg.get("gripper_close_step_rad", 0.055))
        self.grip_profile_velocity = int(cfg.get("gripper_profile_velocity", 40))
        self.grip_profile_acceleration = int(
            cfg.get("gripper_profile_acceleration", 15)
        )
        # XL430 Present_Load magnitude (0–1023). Contact during squeeze / pinch.
        self.grip_load_contact = int(cfg.get("gripper_load_contact", 160))
        self.grip_load_grab = int(cfg.get("gripper_load_grab", 220))
        self.grip_empty_closed = float(cfg.get("gripper_empty_closed_rad", 0.12))
        self.grasp_require_load = bool(cfg.get("grasp_require_load", True))
        self.grasp_empty_retries = int(cfg.get("grasp_empty_retries", 3))
        self.grasp_retry_forward_m = float(cfg.get("grasp_retry_forward_m", 0.01))
        self.grasp_lift_check_z = float(cfg.get("grasp_lift_check_z", 0.025))
        self.grasp_retry_d_shoulder = float(
            cfg.get("grasp_retry_shoulder_rad", 0.20)
        )
        self.grasp_retry_d_wrist = float(cfg.get("grasp_retry_wrist_rad", 0.22))
        self.grasp_retry_d_wrist_floor = float(
            cfg.get("grasp_retry_wrist_floor_rad", -0.22)
        )
        # Click moves to the grasp point. act_closes hands that pose to ACT;
        # otherwise the scripted pinch closes here.
        act_env = os.environ.get("GRAB_IT_ACT_CLOSES", "").strip().lower()
        if act_env in ("0", "false", "no"):
            self._act_closes = False
        elif act_env in ("1", "true", "yes"):
            self._act_closes = True
        else:
            self._act_closes = bool(cfg.get("act_closes", True))
        self._act_handed = False
        sh_lim = cfg.get("shoulder_limits", [-1.70, 1.70])
        self.shoulder_lim = (float(sh_lim[0]), float(sh_lim[1]))
        # During close-in, never increase shoulder (fold back). Forward = more negative.
        self._shoulder_forward_only = False
        self._grip_load_idle = 0
        self._gripper_mode_ok = False
        # Wide USB cam: watch-only during close — stall ROI motion → cut PWM.
        self.use_wide = bool(cfg.get("use_wide_servo", True))
        self.wide_topic = cfg.get(
            "wide_topic", "/wide_cam/image_raw/compressed"
        )
        self.wide_info_topic = cfg.get(
            "wide_camera_info_topic", "/wide_cam/camera_info"
        )
        self.wide_frame = cfg.get("wide_frame", "wide_cam")
        self.wide_overlay = bool(cfg.get("wide_overlay", True))
        self.wide_cube_servo = bool(cfg.get("wide_cube_servo", False))
        self.d435i_cube_servo = bool(cfg.get("d435i_cube_servo", False))
        self.d435i_cube_servo_gain_uv = float(
            cfg.get("d435i_cube_servo_gain_uv", 0.55)
        )
        self.wide_cube_servo_iters = int(cfg.get("wide_cube_servo_iters", 20))
        self.wide_cube_servo_tol = float(cfg.get("wide_cube_servo_tol", 0.05))
        self.wide_cube_servo_timeout_s = float(
            cfg.get("wide_cube_servo_timeout_s", 12.0)
        )
        self.wide_cube_servo_gain_uv = float(cfg.get("wide_cube_servo_gain_uv", 0.08))
        self.wide_cube_servo_gain_roll = float(
            cfg.get("wide_cube_servo_gain_roll", 0.80)
        )
        self.wide_cube_servo_gain_pitch = float(
            cfg.get("wide_cube_servo_gain_pitch", 0.60)
        )
        self.wide_cube_servo_gain_range = float(
            cfg.get("wide_cube_servo_gain_range", 0.04)
        )
        self.wide_cube_servo_max_wr = float(cfg.get("wide_cube_servo_max_wr", 0.18))
        self.wide_cube_servo_max_wa = float(cfg.get("wide_cube_servo_max_wa", 0.12))
        self.wide_cube_servo_wr_sign = float(cfg.get("wide_cube_servo_wr_sign", -1.0))
        self.wide_cube_servo_wa_sign = float(cfg.get("wide_cube_servo_wa_sign", 1.0))
        self.wide_hfov = float(cfg.get("wide_hfov_rad", 1.222))  # ~70° 640x480 crop
        self.wide_u_bias = float(cfg.get("wide_u_bias", 0.0))
        self.wide_v_bias = float(cfg.get("wide_v_bias", 0.0))
        self._wide_look_cache: Optional[Tuple[float, Tuple[float, float, float]]] = None
        self._wide_look_logged = False
        self.wide_ring_enable = bool(cfg.get("wide_ring_enable", True))
        # Ring is a one-shot projection calibrator, not a grasp tracker.
        self.wide_ring_white_pct = float(cfg.get("wide_ring_white_pct", 82.0))
        self.wide_ring_white_min = float(cfg.get("wide_ring_white_min", 155.0))
        self.wide_ring_dark_frac = float(cfg.get("wide_ring_dark_frac", 0.62))
        self.wide_jpeg_url = str(
            cfg.get("wide_jpeg_url", "http://127.0.0.1:8081/wide.jpg")
        )
        # Image-based visual servo: center closest blob on the jaw pixel.
        self.wide_grasp_u = float(cfg.get("wide_grasp_u", 0.50))
        self.wide_grasp_v = float(cfg.get("wide_grasp_v", 0.78))
        self.wide_servo_tol = float(cfg.get("wide_servo_tol", 0.06))
        self.wide_servo_gain_u = float(cfg.get("wide_servo_gain_u", 0.10))
        self.wide_servo_gain_v = float(cfg.get("wide_servo_gain_v", 0.12))
        self.wide_servo_z_est = float(cfg.get("wide_servo_z_est", 0.14))
        self.wide_servo_max_step = float(cfg.get("wide_servo_max_step", 0.025))
        self.wide_servo_max_iters = int(cfg.get("wide_servo_max_iters", 6))
        self.wide_servo_nudge_s = float(cfg.get("wide_servo_nudge_s", 0.65))
        # Image-u wrist/waist chase twists jaws off the D435i crosshair.
        # Default: hold waist facing the click and wrist_rotate=0.
        self.wide_servo_wrist = bool(cfg.get("wide_servo_wrist", False))
        self.wide_servo_gain_wr = float(cfg.get("wide_servo_gain_wr", 0.90))
        self.wide_servo_max_wr = float(cfg.get("wide_servo_max_wr", 0.22))
        self.wide_servo_wr_sign = float(cfg.get("wide_servo_wr_sign", -1.0))
        self.wide_servo_wrist_u_scale = float(
            cfg.get("wide_servo_wrist_u_scale", 0.35)
        )
        self.wide_range_waist = bool(cfg.get("wide_range_waist", False))
        self.wide_range_gain_waist = float(cfg.get("wide_range_gain_waist", 0.50))
        self.wide_range_max_waist = float(cfg.get("wide_range_max_waist", 0.07))
        self.wide_range_waist_sign = float(
            cfg.get("wide_range_waist_sign", cfg.get("wide_servo_wr_sign", -1.0))
        )
        self.wide_range_lat_gain = float(cfg.get("wide_range_lat_gain", 0.045))
        self.wide_range_lat_max = float(cfg.get("wide_range_lat_max", 0.012))
        # 2D jaw-pixel match is not range. After centering, walk along the
        # wide-cam look direction so we don't close on a far object that
        # already lines up between the fingers in the image.
        self.wide_range_step = float(cfg.get("wide_servo_range_step", 0.02))
        self.wide_range_max = float(cfg.get("wide_servo_range_max", 0.12))
        self.wide_close_on_blob = bool(cfg.get("wide_close_on_blob", True))
        self.wide_blob_in_jaws_frames = int(
            cfg.get("wide_blob_in_jaws_frames", 2)
        )
        # Blob in the opening is not range — a far object already fills the
        # jaw camera. Only allow blob-close after this much table-plane walk.
        self.wide_blob_min_range = float(cfg.get("wide_blob_min_range", 0.12))
        self.wide_blob_finger_thr = float(cfg.get("wide_blob_finger_thr", 55.0))
        self.wide_blob_delta = float(cfg.get("wide_blob_delta", 10.0))
        self.wide_blob_min_area = float(cfg.get("wide_blob_min_area", 50.0))
        self.wide_blob_frac = float(cfg.get("wide_blob_frac", 0.08))
        self._blob_dbg_t = 0.0
        self.wide_detect_y0 = float(cfg.get("wide_detect_y0_frac", 0.32))
        self.wide_detect_x0 = float(cfg.get("wide_detect_x0_frac", 0.12))
        self.wide_detect_x1 = float(cfg.get("wide_detect_x1_frac", 0.88))
        self.wide_detect_thr = float(cfg.get("wide_detect_thr", 18))
        self.wide_detect_min_area = float(cfg.get("wide_detect_min_area", 400))
        self.wide_stall_ratio = float(cfg.get("wide_stall_ratio", 0.38))
        self.wide_min_motion = float(cfg.get("wide_min_motion", 2.5))
        self.wide_stall_frames = int(cfg.get("wide_stall_frames", 2))
        self.wide_roi_y0 = float(cfg.get("wide_roi_y0_frac", 0.45))
        self.wide_roi_y1 = float(cfg.get("wide_roi_y1_frac", 0.95))
        self.wide_roi_x0 = float(cfg.get("wide_roi_x0_frac", 0.20))
        self.wide_roi_x1 = float(cfg.get("wide_roi_x1_frac", 0.80))
        self.wide_stale_s = float(cfg.get("wide_stale_s", 1.2))
        # Wait for object appearance in wide ROI before starting close.
        self.wide_object_enter_min = float(cfg.get("wide_object_enter_min", 8.0))
        self.wide_object_enter_frames = int(cfg.get("wide_object_enter_frames", 2))
        self.wide_object_wait_s = float(cfg.get("wide_object_wait_s", 2.5))
        self._hold_gripper_open = False
        # Soft gate for inverted IK (AGX collision). Post-filter only — do NOT
        # put these on MoveIt Constraints (that made every solve fail/hang).
        self.wrist_angle_min = float(cfg.get("wrist_angle_min", -2.0))
        self.wrist_angle_max = float(cfg.get("wrist_angle_max", -0.15))
        self.wrist_rotate_max = float(cfg.get("wrist_rotate_max_abs", 1.57))
        self.shoulder_min_grasp = float(cfg.get("shoulder_min_grasp", -0.55))
        self.shoulder_back_slack = float(cfg.get("shoulder_back_slack_rad", 0.10))
        self.forbid_raw_inverted_ik = bool(cfg.get("forbid_raw_inverted_ik", True))
        self.lock_wrist_rotate = bool(cfg.get("lock_wrist_rotate", True))
        # Hold wrist_rotate from table-plane 3D (pinch short axis).
        # lock_wrist_rotate still blocks image-u chase; this is the held wr.
        self.align_wrist_to_object = bool(cfg.get("align_wrist_to_object", True))
        self.wrist_align_mode = str(cfg.get("wrist_align_mode", "pinch_short"))
        self.wrist_align_sign = float(cfg.get("wrist_align_sign", -1.0))
        self.wrist_align_min_aspect = float(cfg.get("wrist_align_min_aspect", 2.00))
        self.wrist_align_gain = float(cfg.get("wrist_align_gain", 0.85))
        self.wrist_align_max_step = float(cfg.get("wrist_align_max_step", 0.18))
        self.wrist_align_refine_jaws = bool(cfg.get("wrist_align_refine_jaws", False))
        self.wrist_jaw_snap = bool(cfg.get("wrist_jaw_snap", True))
        self.click_depth_snapshot = bool(cfg.get("click_depth_snapshot", True))
        self.floor_object_max_z = float(cfg.get("floor_object_max_z", 0.14))
        self.floor_tcp_lift = float(cfg.get("floor_tcp_lift", 0.02))
        self.floor_z_min = float(cfg.get("floor_z_min", -0.08))
        self.floor_sure_z = float(cfg.get("floor_sure_z", 0.08))
        self.floor_min_ee_reach = float(cfg.get("floor_min_ee_reach", 0.18))
        self.d435_jpeg_url = str(
            cfg.get("d435_jpeg_url", "http://127.0.0.1:8081/d435i.jpg")
        )
        self.ik_prefer_current = bool(cfg.get("ik_prefer_current", True))
        self.ik_budget_s = float(cfg.get("ik_budget_s", 6.0))
        self.align_waist = bool(cfg.get("align_waist", True))
        self.waist_align_min = float(cfg.get("waist_align_min_rad", 0.06))
        self.waist_align_offset = float(cfg.get("waist_align_offset_rad", 0.0))
        self.waist_align_gain = float(cfg.get("waist_align_gain", 1.0))
        self.waist_align_sign = float(cfg.get("waist_align_sign", 1.0))
        self.waist_click_u_bias = float(cfg.get("waist_click_u_bias", -0.04))
        self.min_grasp_reach = float(cfg.get("min_grasp_reach", 0.0))
        # 0 = trust calibrated depth XY at click (no radial pull-in fudge).
        self.reliable_grasp_reach = float(cfg.get("reliable_grasp_reach", 0.0))
        self.waist_lim = (
            float(cfg.get("waist_min", -2.0246)),
            float(cfg.get("waist_max", 2.0944)),
        )
        self._waist_face: Optional[float] = None
        self._cube_cam_R: Optional[np.ndarray] = None
        home_q = cfg.get("home_joints", [0.0, 0.1745, -1.309, 0.0])
        sleep_q = cfg.get("sleep_joints", [0.0, -0.2321, -1.7279, 0.0])
        self.home_joints = [float(v) for v in home_q]
        self.sleep_joints = [float(v) for v in sleep_q]
        self.handoff_wait_s = float(cfg.get("handoff_wait_s", 45.0))
        self.handoff_pulse_s = float(cfg.get("handoff_pulse_s", 1.4))
        self.handoff_hold_open_s = float(cfg.get("handoff_hold_open_s", 0.8))
        self.handoff_search_close_frac = float(
            cfg.get("handoff_search_close_frac", 0.55)
        )
        self.handoff_center_tol = float(cfg.get("handoff_center_tol", 0.22))
        self.handoff_center_frames = int(cfg.get("handoff_center_frames", 3))
        self.handoff_jaw_u0 = float(cfg.get("handoff_jaw_u0", 0.36))
        self.handoff_jaw_u1 = float(cfg.get("handoff_jaw_u1", 0.64))
        self.handoff_jaw_v0 = float(cfg.get("handoff_jaw_v0", 0.55))
        self.handoff_jaw_v1 = float(cfg.get("handoff_jaw_v1", 0.95))
        self.handoff_appear_min = float(cfg.get("handoff_appear_min", 10.0))
        self._handoff_baseline: Optional[np.ndarray] = None
        self._handoff_dbg_t = 0.0
        self._ik_seeds = [
            [0.0, 0.1745, -1.309, 0.0],       # Home (shoulder forward)
            [0.0, 0.4, -1.0, 0.0],
            [0.0, 0.6, -0.8, 0.0],
            [0.4, 0.35, -1.1, 0.0],
            [-0.4, 0.35, -1.1, 0.0],
            [0.0, 0.65, -0.41, 0.0],
            [-0.43, 0.77, -0.26, 0.0],
            [0.43, 0.77, -0.26, 0.0],
            [0.56, 0.64, -1.84, 0.0],  # pan-down table grasp
        ]

        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._depth: Optional[np.ndarray] = None
        self._depth_frame = ""
        self._info: Optional[CameraInfo] = None
        self._joint_pos = {j: 0.0 for j in self.arm_joints}
        self._pan_pos: Optional[float] = None
        self._left_finger: Optional[float] = None
        self._gripper_joint: Optional[float] = None
        self._last_grip_joint: Optional[float] = None
        self._js_ok_stamp = 0.0  # last time arm joints looked live (not -pi)
        self._busy = False
        self._cancel = False
        self._run_gen = 0  # bumped on cancel so in-flight runs become stale
        self._active_goal = None  # FollowJointTrajectory goal handle, if any
        self._pending_click: Optional[dict] = None
        self._last_xyz: Optional[Tuple[float, float, float]] = None
        self._click_anchor: Optional[Tuple[float, float, float]] = None
        self._retreat_q: Optional[List[float]] = None
        self._pending_offset: Optional[Tuple[float, float, float, Optional[float]]] = None
        self._standoff_hold: Optional[float] = None
        self._viz_xyz: Optional[Tuple[float, float, float]] = None
        self._last_click_uv: Optional[Tuple[float, float]] = None
        self._wrist_face: Optional[float] = None
        self._wrist_from_depth = False
        self._object_low = False
        self._object_lying = False
        self._object_top_z: Optional[float] = None
        self._object_centroid: Optional[Tuple[float, float, float]] = None
        self._viz_from_depth = False
        self._cube_roll = 0.0
        self._cube_pitch = 0.0
        self._cube_yaw = 0.0
        self._last_status = "idle"
        self._last_status_msg = ""
        self._last_wide_uv: Optional[Tuple[float, float, bool]] = None
        self._wide_jpg: Optional[bytes] = None
        self._wide_stamp = 0.0
        self._wide_got_logged = False
        self._wide_info: Optional[CameraInfo] = None
        self._wide_sub = None
        self._wide_want_frames = False
        self._wide_ring_uv: Optional[Tuple[float, float]] = None
        self._wide_ring_half = 0.035
        self._wide_ring_stamp = 0.0
        self._wide_ring_logged = False
        # Marker stereo cal (band keypoints), not constant UV bias.
        self._wide_cal_done = (
            float(np.linalg.norm(self.wide_extrinsic_xyz)) > 0.002
            or max(abs(a) for a in self.wide_extrinsic_rpy) > 0.01
        )
        self._wide_hw: Optional[Tuple[int, int]] = None
        self._d435_gray_img: Optional[np.ndarray] = None
        self._marker_pairs: List[dict] = []
        self._d435i_marker: Optional[dict] = None
        self._wide_marker: Optional[dict] = None
        self._wide_marker_reproj: Optional[dict] = None
        self._marker_overlay_stamp = 0.0
        self._overlay_pub_stamp = 0.0
        self._overlay_busy = False
        self._reach_overlay_stamp = 0.0
        self._reach_mask_pan: Optional[float] = None
        self._cached_reach_mask: Optional[dict] = None
        self._depth_snap_lock = threading.Lock()
        self._reach_warn_stamp = 0.0
        self._reach_depth_stamp = 0.0
        self._reach_depth_fail_streak = 0
        self._clicking = False
        self._click_gen = 0
        self._click_id: Optional[int] = None
        self._tf_cache: dict = {}
        self._d435i_pair_xyz: Optional[Tuple[float, float, float]] = None
        self._d435i_pair_stamp = 0.0
        self._cached_pole_xy: Optional[Tuple[float, float]] = None

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # Depth is optional — continuous aligned-depth over crossover delays GUI video.
        if self.subscribe_depth:
            self.create_subscription(
                Image, self.depth_topic, self._depth_cb, _DEPTH_QOS
            )
            self.get_logger().info(f"subscribing to depth {self.depth_topic}")
        else:
            self.get_logger().info(
                "depth stream off — one-shot aligned-depth on click for 3D wrap/height"
            )
        self.create_subscription(
            CameraInfo, self.info_topic, self._info_cb, _DEPTH_QOS
        )
        if self.use_wide and cv2 is not None:
            # Subscribe only during a grasp so idle GUI video isn't sharing
            # the Nano JPEG stream with a second Fast-DDS reader.
            self.get_logger().info(
                f"wide-cam servo armed ({self.wide_jpeg_url}, HTTP only — "
                "no ROS sub on compressed)"
            )
        elif self.use_wide:
            self.get_logger().warn("use_wide_servo set but OpenCV missing — visual hone-in off")
        self.create_subscription(
            JointState,
            "/px150_no_elbow/joint_states",
            self._js_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            String, "/grab_it/grasp_click", self._click_cb, 10, callback_group=self._cg
        )
        self.create_subscription(
            String, "/grab_it/grasp_command", self._cmd_cb, 10, callback_group=self._cg
        )

        self._status_pub = self.create_publisher(String, "/grab_it/grasp_status", 10)
        self._pole_pub = self.create_publisher(String, "/grab_it/pole_xy", 10)
        self._il_cmd_pub = self.create_publisher(String, "/grab_it/il_command", 10)
        self._gripper_pub = self.create_publisher(
            JointSingleCommand, "/px150_no_elbow/commands/joint_single", 10
        )
        self._pan_pub = self._gripper_pub
        self._arm_pub = self.create_publisher(
            JointGroupCommand, "/px150_no_elbow/commands/joint_group", 10
        )
        # Prefer xs joint_group (Nano owns DXL). Traj action is optional.
        self.declare_parameter("use_trajectory_action", False)
        self._use_traj = bool(self.get_parameter("use_trajectory_action").value)

        self._ik_cli = self.create_client(
            GetPositionIK, self.ik_service, callback_group=self._cg
        )
        self._traj_cli = ActionClient(
            self, FollowJointTrajectory, self.traj_action, callback_group=self._cg
        )
        self._mode_cli = self.create_client(
            OperatingModes, "/px150_no_elbow/set_operating_modes", callback_group=self._cg
        )
        self._torque_cli = self.create_client(
            TorqueEnable, "/px150_no_elbow/torque_enable", callback_group=self._cg
        )
        self._reg_set_cli = self.create_client(
            RegisterValues,
            "/px150_no_elbow/set_motor_registers",
            callback_group=self._cg,
        )
        self._reg_get_cli = self.create_client(
            RegisterValues,
            "/px150_no_elbow/get_motor_registers",
            callback_group=self._cg,
        )

        if self.wide_overlay or self.show_landing_cube or self.d435i_reach_overlay:
            self.create_timer(0.10, self._wide_overlay_tick, callback_group=self._cg)
        self.get_logger().info(
            f"grasp_from_click ready (depth={self.depth_topic}, frame={self.planning_frame}, "
            f"wide_servo={self.use_wide and cv2 is not None}, "
            f"d435i_cube_servo={self.d435i_cube_servo}, "
            f"act_closes={self._act_closes})"
        )
        self._publish_status("idle", "Ready — click D435i image, then Grasp")

    # ── status ────────────────────────────────────────────────────────────
    def _publish_status(
        self,
        status: str,
        message: str,
        xyz: Optional[Tuple[float, float, float]] = None,
        *,
        skip_overlay: bool = False,
    ):
        payload = {"status": status, "message": message}
        if status == "done":
            payload["oracle"] = True
        self._last_status = status
        self._last_status_msg = message
        if status == "point" and xyz is not None:
            self._click_anchor = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
        if xyz is not None:
            payload["x"], payload["y"], payload["z"] = xyz
            self._last_xyz = xyz
        elif self._last_xyz is not None and status not in ("idle",):
            payload["x"], payload["y"], payload["z"] = self._last_xyz
        if self._click_id is not None:
            payload["click_id"] = int(self._click_id)
        payload["cube_roll"] = float(self._cube_roll)
        payload["cube_pitch"] = float(self._cube_pitch)
        payload["cube_yaw"] = float(self._cube_yaw)
        if status in ("idle", "error", "cancelled"):
            self._viz_xyz = None
            self._viz_from_depth = False
        # Publish core status immediately so the GUI never sticks on
        # "Resolving depth…" while TF/cube projection runs.
        self._status_pub.publish(String(data=json.dumps(payload)))
        if skip_overlay or status in (
            "idle",
            "error",
            "cancelled",
            "planning",
            "unreachable",
        ):
            return
        tgt = self._viz_xyz
        if tgt is None and "x" in payload:
            tgt = (
                float(payload["x"]),
                float(payload["y"]),
                float(payload["z"]),
            )
        if tgt is None:
            return
        rich = dict(payload)
        if self._last_click_uv is not None:
            rich["click"] = {
                "u": float(self._last_click_uv[0]),
                "v": float(self._last_click_uv[1]),
            }
        try:
            if self.show_landing_cube:
                obj = self._landing_cube_uv(tgt, cam="d435i")
                if obj is not None:
                    rich["landing_cube"] = obj
            if self.wide_overlay:
                # Same planning XYZ the gripper approaches (do not undo
                # click_origin_offset — that made the wide cube disagree
                # with the arm by the residual mount offset).
                wproj = self._project_xyz_to_wide(
                    tgt,
                    use_point_z=True,
                    require_in_frame=False,
                )
                if wproj is not None and wproj[2]:
                    rich["wide_click"] = {
                        "u": float(wproj[0]),
                        "v": float(wproj[1]),
                    }
                wobj = self._landing_cube_uv(tgt, cam="wide")
                if wobj is not None:
                    rich["wide_landing_cube"] = wobj
            self._attach_marker_payload(rich)
        except Exception as e:
            self.get_logger().warn(f"status overlay skip: {e}")
            return
        if (
            "landing_cube" in rich
            or "wide_landing_cube" in rich
            or "wide_click" in rich
            or "click" in rich
            or "d435i_marker" in rich
            or "wide_marker" in rich
        ):
            self._status_pub.publish(String(data=json.dumps(rich)))

    def _project_xyz_to_d435i(
        self,
        xyz: Tuple[float, float, float],
        *,
        urdf_cam: bool = False,
    ) -> Optional[Tuple[float, float, bool]]:
        """Project planning-frame XYZ into D435i normalized UV. bool=in front.

        Inverse of `_hit_optical_z` / `_d435i_planning_T`. Do not apply the
        table-plane pan-scale undo — depth clicks do not use it, and at
        pan≈200° that rotation throws the cube to the bottom of the image.
        urdf_cam=True: URDF optical TF only, no click_origin_offset (jaws).
        """
        with self._lock:
            info = self._info
            frame = self._depth_frame or self.camera_frame
        if info is not None and float(info.k[0]) >= 1.0 and float(info.k[4]) >= 1.0:
            fx, fy = float(info.k[0]), float(info.k[4])
            cx, cy = float(info.k[2]), float(info.k[5])
            w = int(info.width) if info.width else 640
            h = int(info.height) if info.height else 480
        else:
            w, h = 640, 480
            if self._pending_click:
                try:
                    w = int(self._pending_click.get("width") or w)
                    h = int(self._pending_click.get("height") or h)
                except Exception:
                    pass
            hfov = 1.204
            fx = w / (2.0 * math.tan(hfov * 0.5))
            fy = fx
            cx, cy = w * 0.5, h * 0.5
        if fx < 1.0 or fy < 1.0 or w < 2 or h < 2:
            return None
        if urdf_cam:
            tf = self._d435i_urdf_T()
        else:
            tf = self._d435i_planning_T(frame or self.camera_frame)
        if tf is None:
            return None
        T, _used = tf
        try:
            cam = np.linalg.inv(T) @ np.array(
                [float(xyz[0]), float(xyz[1]), float(xyz[2]), 1.0],
                dtype=np.float64,
            )
        except np.linalg.LinAlgError:
            return None
        z = float(cam[2])
        if abs(z) < 1e-6:
            return None
        u_px = fx * (float(cam[0]) / z) + cx
        v_px = fy * (float(cam[1]) / z) + cy
        return (u_px / float(w - 1), v_px / float(h - 1), z > 0.01)

    def _object_cube_frame(
        self, xyz: Optional[Tuple[float, float, float]] = None
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Cube origin + rotation. Columns: +X into cube, +Y left, +Z up/blue.

        Approach face is parallel to the D435i lens (cube +X along optical +Z)
        so the solid face is front-on in the click image. Roll is in-image.
        """
        c = xyz or self._viz_xyz or self._object_centroid
        if c is None:
            return None
        cx, cy, cz = float(c[0]), float(c[1]), float(c[2])
        R0 = None
        cam_R = None
        tf = self._d435i_planning_T(self._depth_frame or self.camera_frame)
        if tf is not None:
            cam_R = tf[0][:3, :3].copy()
            self._cube_cam_R = cam_R
        elif self._cube_cam_R is not None:
            cam_R = self._cube_cam_R
        if cam_R is not None:
            cam_y = cam_R[:, 1].copy()
            cam_z = cam_R[:, 2].copy()
            e_r = cam_z
            nr = float(np.linalg.norm(e_r))
            if nr > 1e-4:
                e_r = e_r / nr
                # Image-up = -optical Y (optical Y is down).
                e_z = -cam_y
                e_z = e_z - e_r * float(np.dot(e_z, e_r))
                nz = float(np.linalg.norm(e_z))
                if nz < 1e-4:
                    e_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                    e_z = e_z - e_r * float(np.dot(e_z, e_r))
                    nz = float(np.linalg.norm(e_z))
                if nz > 1e-4:
                    e_z = e_z / nz
                    e_t = np.cross(e_z, e_r)
                    tn = float(np.linalg.norm(e_t))
                    if tn > 1e-4:
                        e_t = e_t / tn
                        e_z = np.cross(e_r, e_t)
                        ez = float(np.linalg.norm(e_z))
                        if ez > 1e-4:
                            e_z = e_z / ez
                            R0 = np.column_stack((e_r, e_t, e_z))
        if R0 is None:
            e_r = np.array([cx, cy, 0.0], dtype=np.float64)
            rn = float(np.linalg.norm(e_r))
            if rn < 1e-4:
                e_r = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            else:
                e_r = e_r / rn
            e_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            e_t = np.cross(e_z, e_r)
            tn = float(np.linalg.norm(e_t))
            if tn < 1e-4:
                e_t = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            else:
                e_t = e_t / tn
            R0 = np.column_stack((e_r, e_t, e_z))
            e_r = R0[:, 0]
            e_t = R0[:, 1]
            e_z = R0[:, 2]
        else:
            e_r, e_t, e_z = R0[:, 0], R0[:, 1], R0[:, 2]
        R = (
            _axis_angle_mat(e_z, float(self._cube_yaw))
            @ _axis_angle_mat(e_t, float(self._cube_pitch))
            @ _axis_angle_mat(e_r, float(self._cube_roll))
            @ R0
        )
        origin = np.array([cx, cy, cz], dtype=np.float64)
        return origin, R

    def _d435i_xyz_to_urdf(
        self, xyz: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """Undo the dual-TF origin patch so URDF cameras (wide) see true XYZ.

        D435i unproject adds click_origin_offset so IK matches the arm tree.
        The wrist cam TF is already in that URDF tree — projecting the patched
        point into wide shifts the cube by ~7 cm.
        """
        o = self.click_origin_offset
        return (
            float(xyz[0]) - float(o[0]),
            float(xyz[1]) - float(o[1]),
            float(xyz[2]) - float(o[2]),
        )

    def _urdf_xyz_to_d435i(
        self, xyz: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """Map a URDF-tree point into the patched D435i planning frame."""
        o = self.click_origin_offset
        return (
            float(xyz[0]) + float(o[0]),
            float(xyz[1]) + float(o[1]),
            float(xyz[2]) + float(o[2]),
        )

    def _object_cube_corners(
        self,
        xyz: Tuple[float, float, float],
        cam: str = "d435i",
    ) -> List[Tuple[float, float, float]]:
        """Same 8 corners in the planning/base frame for every camera.

        Overlay projects these with that camera's TF + K. Do not rebuild the
        cube in the wrist optical frame — that rides with the arm.
        Wide and D435i both use the same grasp XYZ (including any
        click_origin_offset). Undoing the offset for wide used to hide a
        large D435i-only mount patch; with the residual yaml offset it made
        the USB cube sit away from where the gripper actually goes.
        """
        frame = self._object_cube_frame(xyz)
        if frame is None:
            return []
        origin, R = frame
        hx, hy, hz = self.landing_box_half
        corners: List[Tuple[float, float, float]] = []
        for sx in (-hx, hx):
            for sy in (-hy, hy):
                for sz in (-hz, hz):
                    p = origin + R @ np.array([sx, sy, sz], dtype=np.float64)
                    corners.append((float(p[0]), float(p[1]), float(p[2])))
        return corners

    def _approach_face_center(
        self, xyz: Optional[Tuple[float, float, float]] = None
    ) -> Optional[Tuple[float, float, float]]:
        """Center of the solid face the gripper approaches."""
        frame = self._object_cube_frame(xyz)
        if frame is None:
            return None
        origin, R = frame
        hx = float(self.landing_box_half[0])
        p = origin + R @ np.array([-hx, 0.0, 0.0], dtype=np.float64)
        return (float(p[0]), float(p[1]), float(p[2]))

    def _grasp_target_xyz(
        self, xyz: Optional[Tuple[float, float, float]] = None
    ) -> Optional[Tuple[float, float, float]]:
        """EE origin so the gripper opening lands on the object approach face."""
        frame = self._object_cube_frame(xyz)
        if frame is None:
            return None
        origin, R = frame
        hx = float(self.landing_box_half[0])
        ox = float(self.gripper_cube_offset_x)
        # Opening is (ox+hx) along gripper +X ≈ cube +X; face is -hx along cube +X.
        p = origin - (ox + 2.0 * hx) * R[:, 0]
        return (float(p[0]), float(p[1]), float(p[2]))

    def _grasp_deep_xyz(
        self, xyz: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """Final pinch: a short extra along the table-plane heading.

        Cube +X is D435i optical Z (pitched down). Inserting on that axis
        drops into the table instead of reaching the pole.
        """
        d = max(0.0, float(self.grasp_deeper_m))
        if d < 1e-4:
            return xyz
        horiz = math.hypot(float(xyz[0]), float(xyz[1]))
        if horiz < 1e-3:
            return xyz
        s = d / horiz
        nxt = (
            float(xyz[0]) + s * float(xyz[0]),
            float(xyz[1]) + s * float(xyz[1]),
            float(xyz[2]),
        )
        nh = math.hypot(nxt[0], nxt[1])
        if nh > float(self.max_reach) or not self._in_workspace(nxt):
            return xyz
        return nxt

    def _wrist_from_face_blue(self) -> float:
        """wrist_rotate so gripper top matches the solid-face blue arrow (+Z)."""
        return self._clip_wrist_rotate(float(self._cube_roll))

    def _project_corners_uv(
        self,
        corners: List[Tuple[float, float, float]],
        cam: str,
        face: Optional[List[int]] = None,
        urdf_cam: bool = False,
    ) -> Optional[dict]:
        """Project planning-frame corners with live TF + K. No UV pin/lock."""
        uv: List[Optional[List[float]]] = []
        n_front = 0
        for p in corners:
            if cam == "wide":
                proj = self._project_xyz_to_wide(
                    p, use_point_z=True, require_in_frame=False
                )
            else:
                proj = self._project_xyz_to_d435i(p, urdf_cam=urdf_cam)
            if proj is None or not proj[2]:
                uv.append(None)
            else:
                uv.append([float(proj[0]), float(proj[1])])
                n_front += 1
        if n_front < 1:
            return None
        out = {"uv": uv, "edges": [list(e) for e in _CUBE_EDGES]}
        if face:
            out["face"] = [int(i) for i in face]
            kind = (
                "approach"
                if list(face) == list(_CUBE_APPROACH_FACE)
                else "opening"
                if list(face) == list(_CUBE_OPENING_FACE)
                else ""
            )
            arrows = self._face_arrows_uv(corners, cam, kind, urdf_cam=urdf_cam)
            if arrows:
                out["arrows"] = arrows
        return out

    def _project_xyz_uv(
        self,
        p: Tuple[float, float, float],
        cam: str,
        urdf_cam: bool = False,
    ) -> Optional[List[float]]:
        if cam == "wide":
            proj = self._project_xyz_to_wide(
                p, use_point_z=True, require_in_frame=False
            )
        else:
            proj = self._project_xyz_to_d435i(p, urdf_cam=urdf_cam)
        if proj is None or not proj[2]:
            return None
        return [float(proj[0]), float(proj[1])]

    def _face_arrows_uv(
        self,
        corners: List[Tuple[float, float, float]],
        cam: str,
        kind: str,
        urdf_cam: bool = False,
    ) -> Optional[dict]:
        spec = _CUBE_FACE_ARROWS.get(kind)
        if not spec or len(corners) < 8:
            return None
        arrows: dict = {}
        for name, (i0, i1) in spec.items():
            p0 = np.array(corners[int(i0)], dtype=np.float64)
            p1 = np.array(corners[int(i1)], dtype=np.float64)
            a = p0 + 0.14 * (p1 - p0)
            b = p0 + 0.86 * (p1 - p0)
            ua = self._project_xyz_uv(
                (float(a[0]), float(a[1]), float(a[2])), cam, urdf_cam=urdf_cam
            )
            ub = self._project_xyz_uv(
                (float(b[0]), float(b[1]), float(b[2])), cam, urdf_cam=urdf_cam
            )
            if ua is None or ub is None:
                continue
            arrows[name] = [ua, ub]
        return arrows or None

    def _landing_cube_uv(
        self,
        xyz: Tuple[float, float, float],
        cam: str = "d435i",
    ) -> Optional[dict]:
        """Project the world-frame object cube into this camera."""
        corners = self._object_cube_corners(xyz, cam=cam)
        return self._project_corners_uv(
            corners, cam, face=_CUBE_APPROACH_FACE
        )

    def _gripper_cube_corners(self) -> Optional[List[Tuple[float, float, float]]]:
        """Same-size cube in ee_gripper_link, sitting in the jaw opening."""
        parent = self.ee_link
        T = self._tf_matrix(self.planning_frame, parent)
        if T is None:
            prefix = str(self.planning_frame).split("/")[0]
            T = self._tf_matrix(
                self.planning_frame, f"{prefix}/ee_gripper_link"
            )
        if T is None:
            return None
        hx, hy, hz = self.landing_box_half
        ox = float(self.gripper_cube_offset_x)
        corners: List[Tuple[float, float, float]] = []
        for sx in (-hx, hx):
            for sy in (-hy, hy):
                for sz in (-hz, hz):
                    loc = np.array([ox + sx, sy, sz, 1.0], dtype=np.float64)
                    p = T @ loc
                    corners.append((float(p[0]), float(p[1]), float(p[2])))
        return corners

    def _gripper_cube_uv(self, cam: str = "wide") -> Optional[dict]:
        corners = self._gripper_cube_corners()
        if corners is None:
            self._log_grip_miss("no_ee_tf")
            return None
        # Jaw TF is URDF. Project with the URDF D435i optical frame only
        # (no nav camera_* fallback, no click_origin_offset).
        if cam == "d435i":
            out = self._project_corners_uv(
                corners, cam, face=_CUBE_OPENING_FACE, urdf_cam=True
            )
            if out is None:
                self._log_grip_miss("d435i_project")
            return out
        return self._project_corners_uv(
            corners, cam, face=_CUBE_OPENING_FACE
        )

    def _log_grip_miss(self, why: str):
        now = time.monotonic()
        if now - getattr(self, "_grip_miss_t", 0.0) < 2.0:
            return
        self._grip_miss_t = now
        tf = self._d435i_urdf_T()
        extra = "no_urdf_d435i_tf"
        if tf is not None:
            T, used = tf
            corners = self._gripper_cube_corners()
            z0 = None
            if corners:
                try:
                    cam = np.linalg.inv(T) @ np.array(
                        [corners[0][0], corners[0][1], corners[0][2], 1.0]
                    )
                    z0 = float(cam[2])
                except Exception:
                    z0 = None
            extra = f"frame={used} corner0_z={z0}"
        self.get_logger().warn(f"D435i gripper cube miss ({why}) {extra}")

    def _wide_overlay_tick(self):
        """Re-project the same base_link cubes as the wrist camera / jaws move."""
        if self._clicking or self._overlay_busy:
            return
        self._overlay_busy = True
        try:
            self._wide_overlay_tick_body()
        finally:
            self._overlay_busy = False

    def _wide_overlay_tick_body(self):
        self._publish_reach_overlay()
        updated = self._refresh_marker_overlay()
        if self._last_xyz is None and self._viz_xyz is None:
            if updated and (
                self._d435i_marker or self._wide_marker or self._wide_marker_reproj
            ):
                self._publish_status(
                    self._last_status or "idle",
                    self._last_status_msg
                    or "Ready — click D435i, or Calibrate marker",
                )
            return
        if self._last_status in ("idle", "error", "cancelled"):
            return
        now = time.monotonic()
        if now - self._overlay_pub_stamp < 0.20:
            return
        self._overlay_pub_stamp = now
        self._publish_status(self._last_status, self._last_status_msg)

    # ── callbacks ─────────────────────────────────────────────────────────
    def _decode_depth(self, msg: Image) -> Optional[np.ndarray]:
        try:
            if msg.encoding in ("16UC1", "mono16"):
                img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                return img.astype(np.float32) * 0.001  # mm → m
            if msg.encoding == "32FC1":
                depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                return np.asarray(depth, dtype=np.float32)
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            depth = np.asarray(depth, dtype=np.float32)
            if depth.dtype != np.float32 or depth.max() > 20:
                depth = depth.astype(np.float32) * 0.001
            return depth
        except Exception as e:
            self.get_logger().warn(f"depth decode: {e}")
            return None

    def _depth_cb(self, msg: Image):
        depth = self._decode_depth(msg)
        if depth is None:
            return
        with self._lock:
            self._depth = depth
            self._depth_frame = msg.header.frame_id or self.camera_frame

    def _snapshot_depth(self, timeout_s: float = 1.2) -> Optional[np.ndarray]:
        """Grab one aligned-depth frame. Does not leave a live subscriber."""
        if self.subscribe_depth:
            with self._lock:
                return None if self._depth is None else self._depth.copy()
        if not self.click_depth_snapshot:
            return None
        # Serialize one-shots — concurrent overlay ticks destroyed the sub mid-callback.
        if not self._depth_snap_lock.acquire(blocking=False):
            with self._lock:
                return None if self._depth is None else self._depth.copy()
        box: dict = {"img": None, "frame": ""}
        ev = threading.Event()

        def _cb(msg: Image):
            if ev.is_set():
                return
            depth = self._decode_depth(msg)
            if depth is None:
                return
            box["img"] = depth
            box["frame"] = msg.header.frame_id or self.camera_frame
            ev.set()

        sub = self.create_subscription(
            Image,
            self.depth_topic,
            _cb,
            _DEPTH_QOS,
            callback_group=self._cg,
        )
        try:
            ev.wait(max(0.35, float(timeout_s)))
        finally:
            try:
                self.destroy_subscription(sub)
            except Exception:
                pass
            self._depth_snap_lock.release()
        if box["img"] is None:
            self.get_logger().warn(
                "depth snapshot: no aligned-depth frame (keeping table-plane click)"
            )
            return None
        with self._lock:
            self._depth = box["img"]
            if box["frame"]:
                self._depth_frame = box["frame"]
        return box["img"]

    def _info_cb(self, msg: CameraInfo):
        with self._lock:
            self._info = msg

    def _wide_info_cb(self, msg: CameraInfo):
        with self._lock:
            self._wide_info = msg

    def _wide_sub_start(self):
        """Pull wide JPEGs from mjpeg_relay. Do not ROS-subscribe.

        A second Fast-DDS reader on /wide_cam/image_raw/compressed freezes
        the GUI wide pane. Relay is the only allowed consumer.
        """
        if not self.use_wide or cv2 is None:
            return
        self._wide_want_frames = True
        self._fetch_wide_jpeg()
        if self._wide_jpg and not self._wide_got_logged:
            self._wide_got_logged = True
            self.get_logger().info(f"wide cam frames OK via {self.wide_jpeg_url}")

    def _wide_sub_stop(self):
        self._wide_want_frames = False
        sub = self._wide_sub
        self._wide_sub = None
        if sub is not None:
            try:
                self.destroy_subscription(sub)
            except Exception:
                pass
        with self._lock:
            self._wide_jpg = None
            self._wide_stamp = 0.0

    def _wide_cb(self, msg: CompressedImage):
        if not msg.data:
            return
        with self._lock:
            self._wide_jpg = bytes(msg.data)
            self._wide_stamp = time.monotonic()
            first = not self._wide_got_logged
            if first:
                self._wide_got_logged = True
        if first:
            self.get_logger().info(
                f"wide cam frames OK on {self.wide_topic}"
            )

    def _wide_intrinsics(self) -> Tuple[float, float, float, float, int, int]:
        """Return fx, fy, cx, cy, width, height for the wide camera."""
        with self._lock:
            info = self._wide_info
        w = int(info.width) if info and info.width else 0
        h = int(info.height) if info and info.height else 0
        with self._lock:
            hw = self._wide_hw
        if hw is not None:
            if w < 2:
                w, h = int(hw[0]), int(hw[1])
            elif abs(int(hw[0]) - w) > 16:
                # camera_info size doesn't match the JPEG we actually draw on.
                w, h = int(hw[0]), int(hw[1])
        if w < 2 or h < 2:
            w, h = 640, 480
        if info is not None and float(info.k[0]) > 1.0 and float(info.k[4]) > 1.0:
            return (
                float(info.k[0]),
                float(info.k[4]),
                float(info.k[2]),
                float(info.k[5]),
                w,
                h,
            )
        # Uncalibrated usb_cam (K all zeros): approximate pinhole from HFOV.
        if not getattr(self, "_wide_k_fallback_logged", False):
            self._wide_k_fallback_logged = True
            self.get_logger().warn(
                f"wide camera_info K empty — using HFOV="
                f"{math.degrees(self.wide_hfov):.0f} deg fallback ({w}x{h})"
            )
        fx = w / (2.0 * math.tan(max(0.2, self.wide_hfov) * 0.5))
        fy = fx
        return fx, fy, w * 0.5, h * 0.5, w, h

    def _wide_look_rpy(self) -> Tuple[float, float, float]:
        """Optical RPY so gripper +X is in front of the wide cam.

        Published TF with wide_cam_pitch=-90 looks up; +90° optical roll
        aims along the gripper. Re-picked from live TF so a later URDF
        pitch=0 reload does not double-rotate.
        """
        now = time.monotonic()
        cached = self._wide_look_cache
        if cached is not None and now - cached[0] < 0.25:
            return cached[1]
        look = (0.0, 0.0, 0.0)
        T_cam = self._wide_cam_from_planning(apply_extra=False)
        T_ee = self._tf_matrix(self.ee_link, self.planning_frame)
        if T_cam is not None and T_ee is not None:
            try:
                p_plan = np.linalg.inv(T_ee) @ np.array(
                    [0.20, 0.0, 0.0, 1.0], dtype=np.float64
                )
            except np.linalg.LinAlgError:
                p_plan = None
            if p_plan is not None:
                raw = (T_cam @ p_plan)[:3]
                half_pi = 0.5 * math.pi
                cands = [(0.0, 0.0, 0.0)]
                for a in (half_pi, -half_pi, math.pi):
                    cands.extend([(a, 0.0, 0.0), (0.0, a, 0.0), (0.0, 0.0, a)])
                best_z = -1e9
                for rpy in cands:
                    z = float((_rpy_to_mat(*rpy) @ raw)[2])
                    if z > best_z:
                        best_z = z
                        look = rpy
                if not self._wide_look_logged:
                    self._wide_look_logged = True
                    self.get_logger().info(
                        f"wide look rpy=({math.degrees(look[0]):.0f},"
                        f"{math.degrees(look[1]):.0f},"
                        f"{math.degrees(look[2]):.0f}) deg "
                        f"gripper-forward z={best_z:.3f} m"
                    )
        self._wide_look_cache = (now, look)
        return look

    def _wide_cam_from_planning(self, apply_extra: bool = True) -> Optional[np.ndarray]:
        """4x4 optical←planning, with look-fix + yaml residual in optical frame."""
        frames = [self.wide_frame]
        if "/" not in str(self.wide_frame):
            prefix = str(self.planning_frame).split("/")[0]
            frames.append(f"{prefix}/{self.wide_frame}")
        T = None
        for fr in frames:
            T = self._tf_matrix(fr, self.planning_frame)
            if T is not None:
                break
        if T is None:
            return None
        if not apply_extra:
            return T
        extra = np.eye(4, dtype=np.float64)
        extra[:3, :3] = _rpy_to_mat(*self.wide_extrinsic_rpy) @ _rpy_to_mat(
            *self._wide_look_rpy()
        )
        extra[:3, 3] = self.wide_extrinsic_xyz
        return extra @ T

    def _project_xyz_to_wide(
        self,
        xyz: Tuple[float, float, float],
        use_point_z: bool = False,
        require_in_frame: bool = True,
    ) -> Optional[Tuple[float, float, bool]]:
        """Project planning XYZ into wide image (normalized u,v).

        3D comes from D435i depth. Pose comes from live TF (URDF mount +
        joint angles) plus the optical look convention. Cube corners pass
        use_point_z=True so the far face still draws off-image.
        Third tuple value is "in front of camera" (and in-frame if required).
        """
        z_pt = (
            float(xyz[2])
            if (use_point_z or self._viz_from_depth)
            else float(self._click_plane_z())
        )
        T = self._wide_cam_from_planning()
        if T is None:
            # Never block the click/status path on a long TF wait — overlay
            # will refresh on the next tick once the matrix is cached.
            if self._clicking or self._overlay_busy:
                return None
            pt = PointStamped()
            pt.header.frame_id = self.planning_frame
            pt.header.stamp = rclpy.time.Time().to_msg()
            pt.point.x = float(xyz[0])
            pt.point.y = float(xyz[1])
            pt.point.z = z_pt
            try:
                pt_cam = self._tf_buffer.transform(
                    pt,
                    self.wide_frame,
                    timeout=rclpy.duration.Duration(seconds=0.15),
                )
                x = float(pt_cam.point.x)
                y = float(pt_cam.point.y)
                z = float(pt_cam.point.z)
            except Exception as e:
                if not getattr(self, "_wide_tf_fail_logged", False):
                    self._wide_tf_fail_logged = True
                    self.get_logger().warn(f"wide TF for click failed: {e}")
                return None
        else:
            cam = T @ np.array(
                [float(xyz[0]), float(xyz[1]), z_pt, 1.0], dtype=np.float64
            )
            x, y, z = float(cam[0]), float(cam[1]), float(cam[2])
        if abs(z) < 1e-4:
            return None
        fx, fy, cx, cy, w, h = self._wide_intrinsics()
        if w < 2 or h < 2 or fx <= 0.0 or fy <= 0.0:
            return None
        u_px = fx * (x / z) + cx
        v_px = fy * (y / z) + cy
        u = u_px / float(w - 1)
        v = v_px / float(h - 1)
        # Table-plane clicks need the saved UV bias. A real depth pixel already
        # has the correct height; applying v_bias slides every click the same way.
        if not self._viz_from_depth:
            u += self.wide_u_bias
            v += self.wide_v_bias
        in_front = z > 0.05
        in_frame = 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0
        ok = in_front and (in_frame if require_in_frame else True)
        return (u, v, ok)

    def _wide_roi(self, gray: np.ndarray) -> np.ndarray:
        """Lower-center crop where the gripper/table usually sit in the wide FOV."""
        h, w = gray.shape[:2]
        y0 = max(0, min(h - 1, int(self.wide_roi_y0 * h)))
        y1 = max(y0 + 1, min(h, int(self.wide_roi_y1 * h)))
        x0 = max(0, min(w - 1, int(self.wide_roi_x0 * w)))
        x1 = max(x0 + 1, min(w, int(self.wide_roi_x1 * w)))
        return gray[y0:y1, x0:x1]

    def _wide_decode(self) -> Optional[np.ndarray]:
        """Latest wide JPEG → grayscale float32 ROI, or None if missing/stale."""
        if cv2 is None or not self.use_wide:
            return None
        if self._wide_want_frames:
            with self._lock:
                stamp = self._wide_stamp
            if (time.monotonic() - stamp) > 0.12:
                self._fetch_wide_jpeg()
        with self._lock:
            jpg = self._wide_jpg
            stamp = self._wide_stamp
        if jpg is None or (time.monotonic() - stamp) > self.wide_stale_s:
            return None
        try:
            arr = np.frombuffer(jpg, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                return None
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            roi = self._wide_roi(gray)
            if roi.size < 64:
                return None
            blur = cv2.GaussianBlur(roi, (5, 5), 0)
            return blur.astype(np.float32)
        except Exception as e:
            self.get_logger().warn(f"wide decode: {e}")
            return None

    @staticmethod
    def _wide_motion(prev: np.ndarray, cur: np.ndarray) -> float:
        """Mean absolute difference between consecutive ROI frames."""
        if prev.shape != cur.shape:
            return float("inf")
        return float(np.mean(np.abs(cur - prev)))

    def _wide_gray(self) -> Optional[np.ndarray]:
        """Latest wide JPEG as uint8 grayscale, or None if stale."""
        if cv2 is None or not self.use_wide:
            return None
        if self._wide_want_frames:
            with self._lock:
                stamp = self._wide_stamp
            if (time.monotonic() - stamp) > 0.12:
                self._fetch_wide_jpeg()
        with self._lock:
            jpg = self._wide_jpg
            stamp = self._wide_stamp
        if jpg is None or (time.monotonic() - stamp) > self.wide_stale_s:
            return None
        try:
            arr = np.frombuffer(jpg, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                return None
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            self._wide_hw = (int(gray.shape[1]), int(gray.shape[0]))
            return gray
        except Exception:
            return None

    def _fetch_wide_jpeg(self) -> Optional[bytes]:
        url = str(self.wide_jpeg_url or "")
        if not url:
            return None
        try:
            with urllib.request.urlopen(url, timeout=0.25) as resp:
                jpg = resp.read()
            if jpg:
                with self._lock:
                    self._wide_jpg = jpg
                    self._wide_stamp = time.monotonic()
                return jpg
        except Exception:
            return None
        return None

    def _refresh_wide_ring(self):
        if not self.wide_ring_enable or cv2 is None:
            self._wide_ring_uv = None
            return
        now = time.monotonic()
        if self._wide_ring_uv is not None and now - self._wide_ring_stamp < 0.12:
            return
        found = self._detect_black_ring_wide()
        self._wide_ring_stamp = now
        if found is None:
            return
        self._wide_ring_uv = (float(found[0]), float(found[1]))
        self._wide_ring_half = float(found[2])
        if not getattr(self, "_wide_ring_logged", False):
            self._wide_ring_logged = True
            self.get_logger().info(
                f"wide black ring at u={found[0]:.3f} v={found[1]:.3f}"
            )

    def _detect_black_ring_wide(
        self,
    ) -> Optional[Tuple[float, float, float]]:
        """Black band centroid in the wide JPEG. Returns (u, v, half_extent_norm)."""
        band = self._detect_black_band(self._wide_gray(), hint=self._wide_band_hint())
        if band is None or band.get("center") is None:
            return None
        c = band["center"]
        return (float(c[0]), float(c[1]), float(band.get("half", 0.035)))

    def _wide_band_hint(self) -> Optional[Tuple[float, float]]:
        if self._viz_xyz is None and self._last_xyz is None:
            return None
        proj = self._project_xyz_to_wide(
            self._viz_xyz or self._last_xyz,
            use_point_z=True,
            require_in_frame=False,
        )
        if proj is None or not proj[2]:
            return None
        return (float(proj[0]), float(proj[1]))

    def _detect_black_band(
        self,
        gray: Optional[np.ndarray],
        hint: Optional[Tuple[float, float]] = None,
    ) -> Optional[dict]:
        """Black band on a bright stick → center/top/bottom/left/right UV."""
        if gray is None or cv2 is None:
            return None
        h, w = gray.shape[:2]
        if h < 40 or w < 40:
            return None
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thr = max(
            float(self.wide_ring_white_min),
            float(np.percentile(blur, float(self.wide_ring_white_pct))),
        )
        white = (blur >= thr).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 7))
        white = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel)
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(
            white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        best: Optional[Tuple[float, dict]] = None
        for c in cnts:
            x, y, bw, bh = cv2.boundingRect(c)
            area = float(cv2.contourArea(c))
            if bh < 28 or bw < 6 or area < 250.0:
                continue
            if bh < 1.15 * float(bw):
                continue
            if bw > 0.42 * float(w) or bh > 0.95 * float(h):
                continue
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [c], -1, 255, -1)
            ring = self._ring_on_stick_mask(blur, mask, x, y, bw, bh)
            if ring is None:
                continue
            xx, yy, score, half, i0, i1, dark_thr = ring
            u = xx / float(max(1, w - 1))
            v = yy / float(max(1, h - 1))
            if hint is not None:
                score -= 0.35 * math.hypot(u - hint[0], v - hint[1])
            keys = self._band_keypoints(
                blur, mask, x, y, bw, bh, xx, yy, i0, i1, dark_thr, w, h
            )
            if keys is None:
                continue
            keys["half"] = max(0.018, float(half) / float(max(1, w - 1)))
            if best is None or score > best[0]:
                best = (score, keys)
        return None if best is None else best[1]

    def _band_keypoints(
        self,
        gray: np.ndarray,
        mask: np.ndarray,
        x: int,
        y: int,
        bw: int,
        bh: int,
        xx: float,
        yy: float,
        i0: int,
        i1: int,
        dark_thr: float,
        w: int,
        h: int,
    ) -> Optional[dict]:
        def nrm(px: float, py: float) -> Tuple[float, float]:
            return (
                float(px) / float(max(1, w - 1)),
                float(py) / float(max(1, h - 1)),
            )

        top_y = float(y + i0)
        bot_y = float(y + max(i0, i1 - 1))
        row = int(round(yy))
        row = max(0, min(h - 1, row))
        x0 = max(0, x)
        x1 = min(w, x + bw)
        if x1 - x0 < 3:
            return None
        vals = gray[row, x0:x1]
        on = mask[row, x0:x1] > 0
        dark = np.logical_and(on, vals < float(dark_thr))
        if int(np.count_nonzero(dark)) >= 3:
            xs = np.flatnonzero(dark)
            left_x = float(x0 + int(xs[0]))
            right_x = float(x0 + int(xs[-1]))
        else:
            left_x = float(x) + 0.22 * float(bw)
            right_x = float(x) + 0.78 * float(bw)
        return {
            "center": nrm(xx, yy),
            "top": nrm(xx, top_y),
            "bottom": nrm(xx, bot_y),
            "left": nrm(left_x, yy),
            "right": nrm(right_x, yy),
        }

    def _ring_on_stick_mask(
        self,
        gray: np.ndarray,
        mask: np.ndarray,
        x: int,
        y: int,
        bw: int,
        bh: int,
    ) -> Optional[Tuple[float, float, float, float, int, int, float]]:
        """Dark band with bright stick above and some recovery below."""
        x0 = x + max(1, int(0.18 * bw))
        x1 = x + min(bw - 1, int(0.82 * bw))
        if x1 - x0 < 4:
            return None
        profile = np.zeros(bh, dtype=np.float64)
        for i in range(bh):
            profile[i] = float(np.median(gray[y + i, x0:x1]))
        stick_med = float(np.median(profile))
        # Prefer the bright shaft, not floor/cloth tails.
        bright = profile[profile >= 0.85 * stick_med]
        if bright.size >= 8:
            stick_med = float(np.median(bright))
        if stick_med < 90.0:
            return None
        dark_thr = max(40.0, float(self.wide_ring_dark_frac) * stick_med)
        dark = profile < dark_thr
        best_run = None
        i = 0
        while i < bh:
            if not dark[i]:
                i += 1
                continue
            j = i
            while j < bh and dark[j]:
                j += 1
            run = j - i
            if 3 <= run <= max(10, int(0.35 * bh)):
                above = profile[:i]
                bright_above = above[above >= 0.80 * stick_med]
                below = profile[j : min(bh, j + 16)]
                if bright_above.size >= 10 and below.size >= 3:
                    a_m = float(np.mean(bright_above[-24:]))
                    b_m = float(np.max(below[:8]))
                    band = float(np.mean(profile[i:j]))
                    if a_m > 0.85 * stick_med and b_m > 1.12 * band:
                        contrast = a_m - band
                        if best_run is None or contrast > best_run[0]:
                            best_run = (contrast, i, j)
            i = j
        if best_run is None:
            return None
        _, i0, i1 = best_run
        mid = 0.5 * float(i0 + i1 - 1)
        yy = float(y) + mid
        xx = float(x) + 0.5 * float(bw)
        half = 0.55 * float(bw)
        score = float(best_run[0]) + 0.002 * float(bh)
        return (xx, yy, score, half, int(i0), int(i1), float(dark_thr))

    def _wide_object_uv(
        self, xyz: Tuple[float, float, float]
    ) -> Optional[Tuple[float, float, bool]]:
        """TF projection of the click (after ring calibration, if any)."""
        return self._project_xyz_to_wide(
            xyz, use_point_z=True, require_in_frame=False
        )

    def _marker_public(self, band: Optional[dict]) -> Optional[dict]:
        if not band:
            return None
        out: dict = {}
        for k in _MARKER_KEYS:
            p = band.get(k)
            if p is None or len(p) < 2:
                continue
            out[k] = {"u": float(p[0]), "v": float(p[1])}
        if "half" in band:
            out["half"] = float(band["half"])
        return out or None

    def _attach_marker_payload(self, payload: dict) -> None:
        d435 = self._marker_public(self._d435i_marker)
        if d435:
            payload["d435i_marker"] = d435
        wide = self._marker_public(self._wide_marker)
        if wide:
            payload["wide_marker"] = wide
            c = wide.get("center")
            if c:
                payload["wide_ring"] = {
                    "u": float(c["u"]),
                    "v": float(c["v"]),
                    "half": float(wide.get("half", 0.03)),
                }
        reproj = self._marker_public(self._wide_marker_reproj)
        if reproj:
            payload["wide_marker_reproj"] = reproj

    def _refresh_marker_overlay(self) -> bool:
        """Detect cached JPEGs; reproject stored 3D marker points. Throttled."""
        now = time.monotonic()
        if now - self._marker_overlay_stamp < 0.30:
            return False
        self._marker_overlay_stamp = now
        if self._d435_gray_img is not None:
            self._d435i_marker = self._detect_black_band(self._d435_gray_img)
        wgray = self._wide_gray()
        if wgray is not None:
            self._wide_marker = self._detect_black_band(
                wgray, hint=self._wide_band_hint()
            )
            if self._wide_marker and self._wide_marker.get("center"):
                c = self._wide_marker["center"]
                self._wide_ring_uv = (float(c[0]), float(c[1]))
                self._wide_ring_half = float(self._wide_marker.get("half", 0.035))
        self._wide_marker_reproj = self._reproject_marker_pairs()
        return True

    def _reproject_marker_pairs(self) -> Optional[dict]:
        if not self._marker_pairs:
            return None
        out: dict = {}
        for p in self._marker_pairs:
            xyz = self._d435i_xyz_to_urdf(p["xyz"])
            proj = self._project_xyz_to_wide(
                xyz, use_point_z=True, require_in_frame=False
            )
            if proj is None:
                continue
            name = str(p.get("name") or "center")
            out[name] = (float(proj[0]), float(proj[1]))
        return out or None

    def _unproject_d435i_uv(
        self,
        u_norm: float,
        v_norm: float,
        img_w: int,
        img_h: int,
    ) -> Optional[Tuple[float, float, float]]:
        """Planning-frame 3D of a D435i JPEG pixel via aligned depth."""
        with self._lock:
            depth = None if self._depth is None else self._depth.copy()
            info = self._info
            frame = self._depth_frame or self.camera_frame
        if depth is None or info is None:
            return None
        pix = self._click_k_pixels(u_norm, v_norm, img_w, img_h, info, depth)
        if pix is None:
            return None
        u_px, v_px = pix
        tf = self._d435i_planning_T(frame or info.header.frame_id)
        if tf is None:
            return None
        T, _used = tf
        ray = self._optical_ray_unnorm(u_px, v_px, info)
        if ray is None:
            return None
        u_d, v_d = self._depth_sample_uv(u_px, v_px, info, depth)
        z = self._standing_optical_z(depth, u_d, v_d, info, T, ray, v_norm)
        if z is None:
            return None
        xyz = self._hit_optical_z(T, ray, z)
        if not self._viz_plausible(xyz):
            return None
        if float(v_norm) < 0.62 and xyz[2] < 0.09:
            return None
        return xyz

    def _marker_correspondences(self) -> Tuple[List[dict], str]:
        """Detect the band in both cameras and unproject D435i pixels."""
        self._snapshot_depth()
        dgray = self._d435_gray()
        if dgray is not None:
            self._d435_gray_img = dgray
        self._fetch_wide_jpeg()
        wgray = self._wide_gray()
        if dgray is None:
            return [], "no D435i JPEG"
        if wgray is None:
            return [], "no wide JPEG"
        dh, dw = dgray.shape[:2]
        d_band = self._detect_black_band(dgray)
        w_band = self._detect_black_band(wgray, hint=self._wide_band_hint())
        self._d435i_marker = d_band
        self._wide_marker = w_band
        if d_band is None:
            return [], "no black band in D435i"
        if w_band is None:
            return [], "no black band in wide"
        T_base = self._wide_cam_from_planning(apply_extra=False)
        if T_base is None:
            return [], "no wide TF"
        look = self._wide_look_rpy()
        pairs: List[dict] = []
        for name in _MARKER_KEYS:
            duv = d_band.get(name)
            wuv = w_band.get(name)
            if not duv or not wuv:
                continue
            xyz = self._unproject_d435i_uv(
                float(duv[0]), float(duv[1]), dw, dh
            )
            if xyz is None:
                continue
            pairs.append(
                {
                    "xyz": xyz,
                    "u": float(wuv[0]),
                    "v": float(wuv[1]),
                    "name": name,
                    "T_base": T_base.copy(),
                    "look": look,
                }
            )
        if not pairs:
            return [], "band seen but no valid D435i depth on keypoints"
        return pairs, ""

    def _project_xyz_wide_params(
        self,
        xyz: Tuple[float, float, float],
        T_base: np.ndarray,
        look: Tuple[float, float, float],
        fx: float,
        txyz: np.ndarray,
        rpy: Tuple[float, float, float],
        cx: float,
        cy: float,
        w: int,
        h: int,
    ) -> Optional[Tuple[float, float]]:
        extra = np.eye(4, dtype=np.float64)
        extra[:3, :3] = _rpy_to_mat(*rpy) @ _rpy_to_mat(*look)
        extra[:3, 3] = txyz
        ux, uy, uz = self._d435i_xyz_to_urdf(
            (float(xyz[0]), float(xyz[1]), float(xyz[2]))
        )
        cam = (extra @ T_base) @ np.array(
            [ux, uy, uz, 1.0], dtype=np.float64
        )
        z = float(cam[2])
        if z < 0.04:
            return None
        u = (float(fx) * (float(cam[0]) / z) + cx) / float(max(1, w - 1))
        v = (float(fx) * (float(cam[1]) / z) + cy) / float(max(1, h - 1))
        return (u, v)

    def _marker_residuals(
        self,
        x: np.ndarray,
        pairs: List[dict],
        cx: float,
        cy: float,
        w: int,
        h: int,
    ) -> Optional[np.ndarray]:
        fx, tx, ty, tz, rr, pp, yy = [float(v) for v in x]
        txyz = np.array([tx, ty, tz], dtype=np.float64)
        rpy = (rr, pp, yy)
        rows = []
        for p in pairs:
            pred = self._project_xyz_wide_params(
                p["xyz"], p["T_base"], p["look"], fx, txyz, rpy, cx, cy, w, h
            )
            if pred is None:
                return None
            rows.append((pred[0] - float(p["u"])) * float(max(1, w - 1)))
            rows.append((pred[1] - float(p["v"])) * float(max(1, h - 1)))
        return np.array(rows, dtype=np.float64)

    def _clamp_marker_state(self, x: np.ndarray) -> np.ndarray:
        y = x.copy()
        y[0] = min(900.0, max(90.0, float(y[0])))
        for i in (1, 2, 3):
            y[i] = min(0.12, max(-0.12, float(y[i])))
        for i in (4, 5, 6):
            y[i] = min(0.45, max(-0.45, float(y[i])))
        return y

    def _solve_wide_from_marker(
        self, pairs: List[dict]
    ) -> Tuple[bool, str, float]:
        """Gauss-Newton: wide fx only. Pose is live TF (known mount + joints)."""
        if len(pairs) < 3:
            return False, f"need ≥3 correspondences, have {len(pairs)}", 1e9
        _, _, cx, cy, w, h = self._wide_intrinsics()
        if w < 2 or h < 2:
            return False, "wide image size unknown", 1e9
        fx0 = float(w) / (2.0 * math.tan(max(0.2, self.wide_hfov) * 0.5))
        x = np.array(
            [
                fx0,
                float(self.wide_extrinsic_xyz[0]),
                float(self.wide_extrinsic_xyz[1]),
                float(self.wide_extrinsic_xyz[2]),
                float(self.wide_extrinsic_rpy[0]),
                float(self.wide_extrinsic_rpy[1]),
                float(self.wide_extrinsic_rpy[2]),
            ],
            dtype=np.float64,
        )
        x = self._clamp_marker_state(x)
        # Camera pose is the URDF/TF we already trust. Only fx is free.
        idx = [0]
        npar = len(idx)
        lam = 1e-2
        last_cost = 1e18
        eps = np.array(
            [1.0, 1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3], dtype=np.float64
        )
        for _ in range(30):
            r = self._marker_residuals(x, pairs, cx, cy, w, h)
            if r is None:
                return False, "projection behind camera during solve", 1e9
            cost = float(np.dot(r, r))
            J = np.zeros((r.size, npar), dtype=np.float64)
            for k, j in enumerate(idx):
                xp = x.copy()
                xp[j] += eps[j]
                rp = self._marker_residuals(xp, pairs, cx, cy, w, h)
                if rp is None:
                    return False, "projection behind camera during Jacobian", 1e9
                J[:, k] = (rp - r) / eps[j]
            hmat = J.T @ J
            g = J.T @ r
            try:
                d_small = np.linalg.solve(
                    hmat + lam * np.eye(npar, dtype=np.float64), -g
                )
            except np.linalg.LinAlgError:
                return False, "singular normal equations", 1e9
            dx = np.zeros(7, dtype=np.float64)
            dx[idx] = d_small
            x_new = self._clamp_marker_state(x + dx)
            r_new = self._marker_residuals(x_new, pairs, cx, cy, w, h)
            if r_new is None:
                lam = min(1e3, lam * 4.0)
                continue
            cost_new = float(np.dot(r_new, r_new))
            if cost_new < cost:
                x = x_new
                last_cost = cost_new
                lam = max(1e-6, lam * 0.4)
                if float(np.linalg.norm(dx)) < 1e-6:
                    break
            else:
                lam = min(1e3, lam * 3.0)
                last_cost = cost
        rmse = math.sqrt(max(0.0, last_cost) / max(1, 2 * len(pairs)))
        if rmse > _MARKER_RMSE_PX:
            return False, f"reproj {rmse:.1f}px > {_MARKER_RMSE_PX:.0f}px", rmse
        fx = float(x[0])
        self.wide_hfov = float(
            2.0 * math.atan((0.5 * float(w)) / max(1.0, fx))
        )
        self.wide_hfov = max(
            math.radians(50.0), min(math.radians(110.0), self.wide_hfov)
        )
        self.wide_u_bias = 0.0
        self.wide_v_bias = 0.0
        self._wide_k_fallback_logged = False
        self._persist_wide_extrinsics()
        self._wide_cal_done = True
        self._wide_marker_reproj = self._reproject_marker_pairs()
        msg = (
            f"marker-cal n={len(pairs)} rmse={rmse:.1f}px "
            f"hfov={math.degrees(self.wide_hfov):.0f}° "
            f"(TF pose frozen, fx only)"
        )
        self.get_logger().info(msg)
        return True, msg, rmse

    def _run_marker_calibrate(self):
        """GUI Calibrate marker: one snapshot, accumulate, solve."""
        prev = self._last_status
        try:
            self._publish_status("busy", "Calibrating from black marker…")
            pairs, err = self._marker_correspondences()
            if err:
                self._publish_status(
                    "error",
                    f"Marker cal failed: {err} — keep the band in both views",
                )
                return
            self._marker_pairs.extend(pairs)
            names = ",".join(p["name"] for p in pairs)
            ok, msg, rmse = self._solve_wide_from_marker(self._marker_pairs)
            n = len(self._marker_pairs)
            done_st = (
                prev
                if prev not in ("idle", "error", "cancelled", "busy")
                else "idle"
            )
            if ok:
                self._publish_status(done_st, msg)
                return
            hint = (
                "yaw waist a few degrees and Calibrate again"
                if n < 12
                else "check both cameras see the band"
            )
            self._publish_status(
                "error",
                f"Marker cal: {msg} ({n} pts, {names}). {hint}",
            )
        except Exception as e:
            self.get_logger().warn(f"marker cal: {e}")
            self._publish_status("error", f"Marker cal error: {e}")

    def _reset_marker_calibrate(self):
        self._marker_pairs = []
        self._wide_marker_reproj = None
        self._publish_status(
            self._last_status if self._last_status != "busy" else "idle",
            "Marker samples cleared — Calibrate marker again",
        )

    def _persist_wide_extrinsics(self):
        path = DEFAULT_PARAMS
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as e:
            self.get_logger().warn(f"wide-cal: could not read {path}: {e}")
            return
        xyz = self.wide_extrinsic_xyz
        xyz_s = f"[{float(xyz[0]):.4f}, {float(xyz[1]):.4f}, {float(xyz[2]):.4f}]"
        rpy = self.wide_extrinsic_rpy
        rpy_s = f"[{float(rpy[0]):.4f}, {float(rpy[1]):.4f}, {float(rpy[2]):.4f}]"
        text, n1 = re.subn(
            r"wide_extrinsic_xyz:\s*\[[^\]]*\]",
            f"wide_extrinsic_xyz: {xyz_s}",
            text,
            count=1,
        )
        text, n2 = re.subn(
            r"wide_extrinsic_rpy:\s*\[[^\]]*\]",
            f"wide_extrinsic_rpy: {rpy_s}",
            text,
            count=1,
        )
        if "wide_u_bias:" in text:
            text, _ = re.subn(
                r"wide_u_bias:\s*[-0-9.]+",
                f"wide_u_bias: {float(self.wide_u_bias):.4f}",
                text,
                count=1,
            )
        if "wide_v_bias:" in text:
            text, _ = re.subn(
                r"wide_v_bias:\s*[-0-9.]+",
                f"wide_v_bias: {float(self.wide_v_bias):.4f}",
                text,
                count=1,
            )
        if "wide_hfov_rad:" in text:
            text, _ = re.subn(
                r"wide_hfov_rad:\s*[-0-9.]+",
                f"wide_hfov_rad: {float(self.wide_hfov):.4f}",
                text,
                count=1,
            )
        if n1 == 0:
            self.get_logger().warn("wide-cal: no wide_extrinsic_xyz key to update")
            return
        try:
            path.write_text(text, encoding="utf-8")
            self.get_logger().info(f"wide-cal saved to {path}")
        except Exception as e:
            self.get_logger().warn(f"wide-cal: write failed: {e}")

    def _d435_bgr(self) -> Optional[np.ndarray]:
        """Latest D435i color JPEG (MJPEG relay) — no DDS depth required."""
        if cv2 is None:
            return None
        url = str(self.d435_jpeg_url or "")
        if not url:
            return None
        try:
            with urllib.request.urlopen(url, timeout=0.9) as resp:
                jpg = resp.read()
        except Exception as e:
            self.get_logger().warn(f"d435i jpeg fetch failed: {e}")
            return None
        if not jpg:
            return None
        try:
            arr = np.frombuffer(jpg, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                return None
            self._d435_gray_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            return bgr
        except Exception:
            return None

    def _d435_gray(self) -> Optional[np.ndarray]:
        """Latest D435i JPEG from the local MJPEG relay (no extra DDS)."""
        bgr = self._d435_bgr()
        if bgr is None:
            return None
        return self._d435_gray_img

    def _contrast_mask(
        self, gray: np.ndarray, delta: Optional[float] = None
    ) -> Optional[np.ndarray]:
        if gray is None or gray.size < 64:
            return None
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        k = 31 if min(gray.shape[:2]) > 40 else 15
        if k % 2 == 0:
            k += 1
        bg = cv2.medianBlur(blur, k)
        mag = cv2.max(cv2.subtract(bg, blur), cv2.subtract(blur, bg))
        thr = float(delta if delta is not None else max(8.0, self.wide_blob_delta))
        _, mask = cv2.threshold(mag, thr, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _fold_axis_to_half_pi(self, ang: float) -> float:
        a = float(ang)
        while a > 0.5 * math.pi:
            a -= math.pi
        while a < -0.5 * math.pi:
            a += math.pi
        return a

    def _contour_from_vertical(
        self, cnt
    ) -> Optional[Tuple[float, float]]:
        """Long-axis angle from image-vertical (rad) and aspect, or None."""
        if cnt is None or len(cnt) < 5:
            return None
        rect = cv2.minAreaRect(cnt)
        rw, rh = float(rect[1][0]), float(rect[1][1])
        if rw < 1.0 or rh < 1.0:
            return None
        aspect = max(rw, rh) / min(rw, rh)
        if aspect < float(self.wrist_align_min_aspect):
            return None
        angle = float(rect[2])
        long_deg = angle - 90.0 if rw < rh else angle
        long_rad = math.radians(long_deg)
        from_vert = self._fold_axis_to_half_pi(
            math.atan2(math.cos(long_rad), math.sin(long_rad))
        )
        return from_vert, aspect

    def _best_axis_in_mask(
        self,
        mask: np.ndarray,
        click_xy: Optional[Tuple[float, float]] = None,
        min_area: float = 80.0,
    ) -> Optional[Tuple[float, float]]:
        cnts, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        best = None
        best_score = -1.0
        h, w = mask.shape[:2]
        max_area = 0.85 * float(max(1, h * w))
        for c in cnts:
            area = float(cv2.contourArea(c))
            if area < min_area or area > max_area:
                continue
            ax = self._contour_from_vertical(c)
            if ax is None:
                continue
            from_vert, aspect = ax
            score = area * aspect
            if click_xy is not None:
                dist = float(cv2.pointPolygonTest(c, click_xy, True))
                if dist >= 0.0:
                    score *= 4.0
                else:
                    score *= max(0.15, 1.0 / (1.0 + abs(dist)))
            if score > best_score:
                best_score = score
                best = (from_vert, aspect)
        return best

    def _estimate_wrist_from_d435i(
        self, u: float, v: float
    ) -> Optional[float]:
        """wrist_rotate so jaws wrap a leaning object at the D435i click."""
        if not self.align_wrist_to_object or cv2 is None:
            return None
        gray = self._d435_gray()
        if gray is None:
            return None
        h, w = gray.shape[:2]
        cx = int(round(float(u) * (w - 1)))
        cy = int(round(float(v) * (h - 1)))
        rw = max(24, int(0.24 * w))
        rh = max(24, int(0.32 * h))
        x0 = max(0, cx - rw)
        x1 = min(w, cx + rw)
        y0 = max(0, cy - rh)
        y1 = min(h, cy + rh)
        crop = gray[y0:y1, x0:x1]
        mask = self._contrast_mask(
            crop, delta=max(8.0, float(self.wide_blob_delta))
        )
        if mask is None:
            return None
        ax = self._best_axis_in_mask(
            mask,
            click_xy=(float(cx - x0), float(cy - y0)),
            min_area=60.0,
        )
        if ax is None:
            self.get_logger().info(
                "wrist align: no elongated object at D435i click"
            )
            return None
        from_vert, aspect = ax
        wr = self._clip_wrist_rotate(
            float(self.wrist_align_sign) * from_vert
        )
        self.get_logger().info(
            f"wrist align D435i lean={math.degrees(from_vert):.1f}° "
            f"aspect={aspect:.1f} wr={math.degrees(wr):.1f}°"
        )
        return wr

    def _pinch_wrist_from_xy_axis(
        self, axis_xy: np.ndarray, med_xy: np.ndarray
    ) -> float:
        """wrist_rotate so the jaw opening lies along the table-plane long axis.

        wr=0 ≈ opening perpendicular to radial. Pinch-short adds +π/2 so
        fingers close on the thin dimension.
        """
        ax, ay = float(axis_xy[0]), float(axis_xy[1])
        hx, hy = float(med_xy[0]), float(med_xy[1])
        hn = math.hypot(hx, hy)
        if hn < 1e-4:
            long_vs_rad = math.atan2(ay, ax)
        else:
            hx, hy = hx / hn, hy / hn
            cross = hx * ay - hy * ax
            dot = hx * ax + hy * ay
            long_vs_rad = math.atan2(cross, dot)
        pinch = long_vs_rad
        if str(self.wrist_align_mode) != "wrap_long":
            pinch = long_vs_rad + 0.5 * math.pi
        wr = self._fold_axis_to_half_pi(
            float(self.wrist_align_sign) * pinch
        )
        return self._clip_wrist_rotate(wr)

    def _analyze_click_depth(
        self, u_norm: float, v_norm: float
    ) -> Optional[dict]:
        """3D object height + pinch-short wrist from one D435i depth patch."""
        with self._lock:
            depth = None if self._depth is None else self._depth.copy()
            info = self._info
            frame = self._depth_frame or self.camera_frame
        if depth is None or info is None:
            return None
        fx, fy = float(info.k[0]), float(info.k[4])
        cx, cy = float(info.k[2]), float(info.k[5])
        if fx < 1.0 or fy < 1.0:
            return None
        h, w = depth.shape[:2]
        u_px = int(round(float(u_norm) * (w - 1)))
        v_px = int(round(float(v_norm) * (h - 1)))
        rw = max(18, int(0.16 * w))
        rh = max(18, int(0.20 * h))
        x0 = max(0, u_px - rw)
        x1 = min(w, u_px + rw + 1)
        y0 = max(0, v_px - rh)
        y1 = min(h, v_px + rh + 1)
        patch = depth[y0:y1, x0:x1]
        valid = np.isfinite(patch) & (patch > self.min_depth) & (patch < self.max_depth)
        if int(np.count_nonzero(valid)) < 40:
            self.get_logger().info("depth3d: too few valid pixels in click patch")
            return None
        zs = patch[valid].astype(np.float32)
        z_obj = float(np.percentile(zs, 25.0))
        band = np.abs(patch - z_obj) < 0.045
        keep = np.logical_and(valid, band)
        ys, xs = np.where(keep)
        if xs.size < 40:
            keep = valid
            ys, xs = np.where(keep)
        if xs.size < 40:
            return None
        z_vals = patch[ys, xs].astype(np.float64)
        u_s = (xs + x0).astype(np.float64)
        v_s = (ys + y0).astype(np.float64)
        x_cam = (u_s - cx) * z_vals / fx
        y_cam = (v_s - cy) * z_vals / fy
        cam = np.stack([x_cam, y_cam, z_vals, np.ones_like(z_vals)], axis=1)
        T = None
        used_frame = frame
        for fr in self._camera_frame_candidates(frame):
            T = self._tf_matrix(self.planning_frame, fr)
            if T is not None:
                used_frame = fr
                break
        if T is None:
            return None
        T_use = T.copy()
        if "d435i" in str(used_frame):
            T_use[:3, 3] = T_use[:3, 3] + self.click_origin_offset
        pts = (T_use @ cam.T).T[:, :3]
        med = np.median(pts, axis=0)
        dxy = np.hypot(pts[:, 0] - med[0], pts[:, 1] - med[1])
        pts = pts[dxy < 0.10]
        if pts.shape[0] < 30:
            return None
        top_z = float(np.percentile(pts[:, 2], 60.0))
        low = top_z <= float(self.floor_object_max_z)
        # An upper-image patch that only grazes the floor is still the pole.
        # A top below floor_sure_z is the block, even when the camera is tilted.
        if float(v_norm) < 0.58 and top_z >= float(self.floor_sure_z):
            low = False
        cent = np.mean(pts, axis=0)
        # The wide patch's nearest surface is the foreground floor, which sits
        # closer to the base than the block under the cursor. A tight window
        # on the click is the object to grasp.
        half = 5
        tx0, tx1 = max(0, u_px - half), min(w, u_px + half + 1)
        ty0, ty1 = max(0, v_px - half), min(h, v_px + half + 1)
        tpatch = depth[ty0:ty1, tx0:tx1]
        tvalid = (
            np.isfinite(tpatch)
            & (tpatch > self.min_depth)
            & (tpatch < self.max_depth)
        )
        if int(np.count_nonzero(tvalid)) >= 8:
            tys, txs = np.where(tvalid)
            tz = tpatch[tys, txs].astype(np.float64)
            tu = (txs + tx0).astype(np.float64)
            tv = (tys + ty0).astype(np.float64)
            tcam = np.stack(
                [(tu - cx) * tz / fx, (tv - cy) * tz / fy, tz, np.ones_like(tz)],
                axis=1,
            )
            tpts = (T_use @ tcam.T).T[:, :3]
            ttop = float(np.percentile(tpts[:, 2], 60.0))
            tc = np.median(tpts, axis=0)
            if ttop <= float(self.floor_sure_z):
                low = True
            elif ttop > float(self.floor_object_max_z):
                low = False
            if low and ttop <= float(self.floor_object_max_z):
                top_z = ttop
                cent = tc
                self.get_logger().info(
                    f"floor click window top={ttop:.3f} "
                    f"({float(tc[0]):.3f},{float(tc[1]):.3f},{float(tc[2]):.3f}) "
                    f"reach={math.hypot(float(tc[0]), float(tc[1])):.3f}"
                )
        out: dict = {
            "z": top_z,
            "low": low,
            "lying": False,
            "n": int(pts.shape[0]),
            "frame": used_frame,
            "wr": 0.0,
            "centroid": (float(cent[0]), float(cent[1]), float(cent[2])),
        }
        # Classify: upright/compact → wr=0. Lying elongated → pinch short axis.
        # A standing tube viewed at a slant has an elongated XY ellipse and
        # |vz| often < 0.65 (depth streaks along the look ray). Require a
        # clearly horizontal 3D axis AND more table-plane length than height.
        # Tall (non-floor) objects are standing unless the footprint is stick-long.
        centered = pts - np.mean(pts, axis=0)
        vz = 0.0
        if centered.shape[0] >= 30:
            cov3 = np.cov(centered, rowvar=False)
            if cov3.shape == (3, 3) and np.all(np.isfinite(cov3)):
                evals3, evecs3 = np.linalg.eigh(cov3)
                axis3 = evecs3[:, int(np.argmax(evals3))]
                vz = abs(float(axis3[2]))
        z_span = float(
            np.percentile(pts[:, 2], 90.0) - np.percentile(pts[:, 2], 10.0)
        )
        xy = pts[:, :2] - np.mean(pts[:, :2], axis=0)
        if xy.shape[0] < 30:
            return out
        cov = np.cov(xy, rowvar=False)
        if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
            return out
        evals, evecs = np.linalg.eigh(cov)
        if float(evals[1]) < 1e-8:
            return out
        aspect = float(evals[1] / max(float(evals[0]), 1e-8))
        out["aspect"] = aspect
        xy_span = 2.0 * math.sqrt(max(float(evals[1]), 0.0))
        tall_not_stick = (not low) and xy_span < 0.12
        upright = (
            tall_not_stick
            or vz > 0.50
            or z_span >= 0.90 * max(xy_span, 1e-4)
            or aspect < float(self.wrist_align_min_aspect)
        )
        lying = (
            not upright
            and vz < 0.40
            and aspect >= float(self.wrist_align_min_aspect)
            and xy_span > 1.15 * max(z_span, 0.01)
        )
        if not lying:
            self.get_logger().info(
                f"depth3d: z={top_z:.3f} low={low} n={pts.shape[0]} "
                f"|vz|={vz:.2f} zspan={z_span:.3f} xyspan={xy_span:.3f} "
                f"aspect={aspect:.1f} upright — wr=0"
            )
            return out
        out["lying"] = True
        if not self.align_wrist_to_object:
            return out
        axis = evecs[:, 1]
        wr = self._pinch_wrist_from_xy_axis(axis, med[:2])
        out["wr"] = wr
        self.get_logger().info(
            f"depth3d: z={top_z:.3f} low={low} n={pts.shape[0]} "
            f"|vz|={vz:.2f} zspan={z_span:.3f} xyspan={xy_span:.3f} "
            f"aspect={aspect:.1f} lying pinch wr={math.degrees(wr):.1f}°"
        )
        return out

    def _jaw_blob_from_vertical(self) -> Optional[float]:
        """Long-axis tilt from vertical in the jaw camera, or None."""
        if not self.align_wrist_to_object or cv2 is None:
            return None
        gray = self._wide_gray()
        if gray is None:
            return None
        h, w = gray.shape[:2]
        x0 = max(0, min(w - 2, int(self.handoff_jaw_u0 * w)))
        x1 = max(x0 + 2, min(w, int(self.handoff_jaw_u1 * w)))
        y0 = max(0, min(h - 2, int(self.handoff_jaw_v0 * h)))
        y1 = max(y0 + 2, min(h, int(self.handoff_jaw_v1 * h)))
        win = gray[y0:y1, x0:x1]
        if win.size < 64:
            return None
        blur = cv2.GaussianBlur(win, (5, 5), 0)
        finger = blur < float(self.wide_blob_finger_thr)
        valid = np.logical_not(finger)
        if int(np.count_nonzero(valid)) < 80:
            return None
        vals = blur[valid].astype(np.float32)
        med = float(np.median(vals))
        delta = float(self.wide_blob_delta)
        fg = np.logical_and(
            np.logical_or(blur >= (med + delta), blur <= (med - delta)),
            valid,
        )
        fg_u8 = fg.astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_OPEN, kernel)
        fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_CLOSE, kernel)
        ax = self._best_axis_in_mask(
            fg_u8, min_area=float(self.wide_blob_min_area)
        )
        if ax is None:
            return None
        return ax[0]

    def _refine_wrist_from_jaws(self) -> Optional[float]:
        ax = self._jaw_blob_from_vertical()
        if ax is None:
            return None
        cur = (
            float(self._wrist_face)
            if self._wrist_face is not None
            else float(self._joint_pos.get("wrist_rotate", 0.0))
        )
        step = (
            float(self.wrist_align_sign)
            * float(ax)
            * float(self.wrist_align_gain)
        )
        cap = max(0.04, float(self.wrist_align_max_step))
        step = max(-cap, min(cap, step))
        wr = self._clip_wrist_rotate(cur + step)
        self._wrist_face = wr
        return wr

    def _jaw_opening_fg_mask(self) -> Optional[np.ndarray]:
        """Object mask in the jaw opening, fingers ignored. None if no signal."""
        if cv2 is None:
            return None
        gray = self._wide_gray()
        if gray is None:
            return None
        h, w = gray.shape[:2]
        x0 = max(0, min(w - 2, int(self.handoff_jaw_u0 * w)))
        x1 = max(x0 + 2, min(w, int(self.handoff_jaw_u1 * w)))
        y0 = max(0, min(h - 2, int(self.handoff_jaw_v0 * h)))
        y1 = max(y0 + 2, min(h, int(self.handoff_jaw_v1 * h)))
        win = gray[y0:y1, x0:x1]
        if win.size < 64:
            return None
        blur = cv2.GaussianBlur(win, (5, 5), 0)
        finger = blur < float(self.wide_blob_finger_thr)
        valid = np.logical_not(finger)
        if int(np.count_nonzero(valid)) < 80:
            return None
        vals = blur[valid].astype(np.float32)
        med = float(np.median(vals))
        delta = float(self.wide_blob_delta)
        fg = np.logical_and(
            np.logical_or(blur >= (med + delta), blur <= (med - delta)),
            valid,
        )
        fg_u8 = fg.astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_OPEN, kernel)
        fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_CLOSE, kernel)
        return fg_u8

    def _jaw_opening_needs_quarter_turn(self) -> bool:
        """True if the in-jaws silhouette is elongated across the opening."""
        if not self.wrist_jaw_snap or cv2 is None:
            return False
        fg = self._jaw_opening_fg_mask()
        if fg is None:
            return False
        cnts, _ = cv2.findContours(
            fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not cnts:
            return False
        cnt = max(cnts, key=cv2.contourArea)
        if float(cv2.contourArea(cnt)) < float(self.wide_blob_min_area):
            return False
        if len(cnt) < 5:
            return False
        rect = cv2.minAreaRect(cnt)
        rw, rh = float(rect[1][0]), float(rect[1][1])
        if rw < 1.0 or rh < 1.0:
            return False
        aspect = max(rw, rh) / min(rw, rh)
        if aspect < max(1.35, 0.85 * float(self.wrist_align_min_aspect)):
            return False
        angle = float(rect[2])
        # OpenCV: width is the first dim; angle is that side vs +x.
        if rw >= rh:
            long_from_h = angle
        else:
            long_from_h = angle + 90.0
        while long_from_h > 90.0:
            long_from_h -= 180.0
        while long_from_h < -90.0:
            long_from_h += 180.0
        # Closer to horizontal (0°) than vertical (±90°) → object spans the jaws.
        horiz = abs(long_from_h) < 35.0
        if horiz:
            self.get_logger().info(
                f"jaw snap: wide silhouette aspect={aspect:.1f} "
                f"long_from_h={long_from_h:.0f}°"
            )
        return horiz

    def _maybe_snap_wrist_quarter_turn(self, gen: int) -> bool:
        """One ±90° wrist move if the jaw silhouette is the wrong way. Then pinch."""
        # Standing tubes fill the wide frame with a wide white band — that is
        # not a lying stick. Only snap when depth already called it lying.
        if not self._object_lying:
            return True
        if not self._jaw_opening_needs_quarter_turn():
            return True
        ir = self._joint_index("wrist_rotate")
        if ir < 0:
            return True
        cur = (
            float(self._wrist_face)
            if self._wrist_face is not None
            else float(self._joint_pos.get("wrist_rotate", 0.0))
        )
        step = math.copysign(
            0.5 * math.pi,
            float(self.wrist_align_sign) if abs(float(self.wrist_align_sign)) > 1e-6 else 1.0,
        )
        nxt = self._clip_wrist_rotate(cur + step)
        if abs(nxt - cur) < 0.25:
            nxt = self._clip_wrist_rotate(cur - step)
        if abs(nxt - cur) < 0.25:
            self.get_logger().info("jaw snap: quarter-turn would clip — skip")
            return True
        self._wrist_face = nxt
        q = [float(self._joint_pos.get(j, 0.0)) for j in self.arm_joints]
        q[ir] = nxt
        self.get_logger().info(
            f"jaw snap: wrist_rotate {math.degrees(cur):.0f}° → {math.degrees(nxt):.0f}°"
        )
        self._publish_status(
            "grasping",
            f"Wrist quarter-turn to {math.degrees(nxt):.0f}° then pinch…",
        )
        old = self.seg_dur
        self.seg_dur = max(0.45, float(self.wide_servo_nudge_s))
        try:
            ok = self._send_trajectory([q], gen)
        finally:
            self.seg_dur = old
        if ok:
            self._joint_pos["wrist_rotate"] = float(nxt)
        return ok

    def _apply_align_joints(self, q: List[float]) -> List[float]:
        q = list(q)
        ir = self._joint_index("wrist_rotate")
        if ir >= 0:
            if self._wrist_face is not None:
                q[ir] = self._clip_wrist_rotate(self._wrist_face)
            elif self.lock_wrist_rotate:
                q[ir] = 0.0
        return q

    def _jaw_bbox(self) -> dict:
        """Normalized jaw-opening rectangle in the wide image."""
        u0 = float(self.handoff_jaw_u0)
        u1 = float(self.handoff_jaw_u1)
        v0 = float(self.handoff_jaw_v0)
        v1 = float(self.handoff_jaw_v1)
        if u1 < u0:
            u0, u1 = u1, u0
        if v1 < v0:
            v0, v1 = v1, v0
        return {
            "u0": u0,
            "v0": v0,
            "u1": u1,
            "v1": v1,
            "u": 0.5 * (u0 + u1),
            "v": 0.5 * (v0 + v1),
        }

    def _wide_object_bbox(
        self,
        seed_uv: Optional[Tuple[float, float]] = None,
    ) -> Optional[dict]:
        """Blob bbox in the wide image, preferring the one nearest seed UV."""
        if cv2 is None:
            return None
        gray = self._wide_gray()
        if gray is None:
            return None
        h, w = gray.shape[:2]
        y0 = max(0, min(h - 2, int(self.wide_detect_y0 * h)))
        x0 = max(0, min(w - 2, int(self.wide_detect_x0 * w)))
        x1 = max(x0 + 2, min(w, int(self.wide_detect_x1 * w)))
        crop = gray[y0:h, x0:x1]
        if crop.size < 64:
            return None
        blur = cv2.GaussianBlur(crop, (7, 7), 0)
        k = 31 if min(crop.shape[:2]) > 40 else 15
        if k % 2 == 0:
            k += 1
        bg = cv2.medianBlur(blur, k)
        mag = cv2.max(cv2.subtract(bg, blur), cv2.subtract(blur, bg))
        _, mask = cv2.threshold(
            mag, float(self.wide_detect_thr), 255, cv2.THRESH_BINARY
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = -1.0
        min_area = float(self.wide_detect_min_area)
        max_area = 0.35 * float(crop.size)
        su = sv = None
        if seed_uv is not None:
            su, sv = float(seed_uv[0]), float(seed_uv[1])
        for c in cnts:
            area = float(cv2.contourArea(c))
            if area < min_area or area > max_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            cx = float(x) + 0.5 * float(bw) + float(x0)
            cy = float(y) + 0.5 * float(bh) + float(y0)
            u = cx / max(1.0, float(w - 1))
            v = cy / max(1.0, float(h - 1))
            if su is not None:
                dist = math.hypot(u - su, v - sv)
                score = area / (1.0 + 12.0 * dist)
            else:
                score = area * (0.25 + 0.75 * (cy / max(1.0, float(h - 1))))
            if score > best_score:
                best_score = score
                u0 = (float(x) + float(x0)) / max(1.0, float(w - 1))
                v0 = (float(y) + float(y0)) / max(1.0, float(h - 1))
                u1 = (float(x + bw) + float(x0)) / max(1.0, float(w - 1))
                v1 = (float(y + bh) + float(y0)) / max(1.0, float(h - 1))
                best = {
                    "u0": float(u0),
                    "v0": float(v0),
                    "u1": float(u1),
                    "v1": float(v1),
                    "u": float(u),
                    "v": float(v),
                }
        return best

    def _wide_closest_object(
        self,
    ) -> Optional[Tuple[float, float, float]]:
        """Centroid of the preferred wide blob. Returns (u, v, score)."""
        box = self._wide_object_bbox()
        if box is None:
            return None
        return (float(box["u"]), float(box["v"]), 1.0)

    def _log_blob(self, msg: str, force: bool = False):
        now = time.monotonic()
        if not force and now - self._blob_dbg_t < 0.45:
            return
        self._blob_dbg_t = now
        self.get_logger().info(f"jaw blob {msg}")

    def _blob_in_jaws(self) -> Optional[Tuple[float, float]]:
        """Occupancy in the opening between the fingers.

        Does not use the global closest-blob (that locked onto the dark
        gripper jaws). Ignores dark fingers and looks for a light/dark
        patch vs the table in the jaw window — the in-hand object is a
        low-contrast white blob on beige.
        """
        if not self.wide_close_on_blob:
            return None
        if cv2 is None:
            return None
        gray = self._wide_gray()
        if gray is None:
            self._log_blob("no_frame")
            return None
        h, w = gray.shape[:2]
        x0 = max(0, min(w - 2, int(self.handoff_jaw_u0 * w)))
        x1 = max(x0 + 2, min(w, int(self.handoff_jaw_u1 * w)))
        y0 = max(0, min(h - 2, int(self.handoff_jaw_v0 * h)))
        y1 = max(y0 + 2, min(h, int(self.handoff_jaw_v1 * h)))
        win = gray[y0:y1, x0:x1]
        if win.size < 64:
            self._log_blob("no_win")
            return None
        blur = cv2.GaussianBlur(win, (5, 5), 0)
        finger = blur < float(self.wide_blob_finger_thr)
        valid = np.logical_not(finger)
        n_valid = int(np.count_nonzero(valid))
        if n_valid < 80:
            self._log_blob(f"fingers_only valid={n_valid}")
            return None
        vals = blur[valid].astype(np.float32)
        med = float(np.median(vals))
        delta = float(self.wide_blob_delta)
        fg = np.logical_and(
            np.logical_or(blur >= (med + delta), blur <= (med - delta)),
            valid,
        )
        fg_u8 = fg.astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_OPEN, kernel)
        fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_CLOSE, kernel)
        frac = float(np.count_nonzero(fg_u8)) / float(max(1, n_valid))
        cnts, _ = cv2.findContours(
            fg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        min_area = float(self.wide_blob_min_area)
        best = None
        best_a = 0.0
        for c in cnts:
            area = float(cv2.contourArea(c))
            if area < min_area or area > 0.72 * float(n_valid):
                continue
            m = cv2.moments(c)
            if abs(m["m00"]) < 1e-6:
                continue
            if area > best_a:
                best_a = area
                best = (
                    float(m["m10"] / m["m00"]),
                    float(m["m01"] / m["m00"]),
                    area,
                )
        if best is not None:
            u = (best[0] + float(x0)) / max(1.0, float(w - 1))
            v = (best[1] + float(y0)) / max(1.0, float(h - 1))
            self._log_blob(
                f"hit u={u:.2f} v={v:.2f} area={best_a:.0f} "
                f"med={med:.0f} frac={frac:.2f}",
                force=True,
            )
            return (u, v)
        if frac >= float(self.wide_blob_frac):
            u = 0.5 * (self.handoff_jaw_u0 + self.handoff_jaw_u1)
            v = 0.5 * (self.handoff_jaw_v0 + self.handoff_jaw_v1)
            self._log_blob(
                f"frac hit frac={frac:.2f} med={med:.0f}", force=True
            )
            return (u, v)
        ch, cw = blur.shape[:2]
        inner = blur[ch // 4 : max(ch // 4 + 2, 3 * ch // 4),
                     cw // 4 : max(cw // 4 + 2, 3 * cw // 4)]
        inner_valid = valid[ch // 4 : max(ch // 4 + 2, 3 * ch // 4),
                            cw // 4 : max(cw // 4 + 2, 3 * cw // 4)]
        if inner.size >= 40 and int(np.count_nonzero(inner_valid)) > 20:
            inner_med = float(np.median(inner[inner_valid].astype(np.float32)))
            if abs(inner_med - med) >= delta:
                u = 0.5 * (self.handoff_jaw_u0 + self.handoff_jaw_u1)
                v = 0.5 * (self.handoff_jaw_v0 + self.handoff_jaw_v1)
                self._log_blob(
                    f"inner hit inner={inner_med:.0f} med={med:.0f}",
                    force=True,
                )
                return (u, v)
        self._log_blob(
            f"miss med={med:.0f} frac={frac:.2f} n={len(cnts)} valid={n_valid}"
        )
        return None

    def _ee_xyz(self) -> Optional[Tuple[float, float, float]]:
        try:
            tf = self._tf_buffer.lookup_transform(
                self.planning_frame,
                self.ee_link,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.4),
            )
            t = tf.transform.translation
            return (float(t.x), float(t.y), float(t.z))
        except Exception:
            return None

    def _optical_delta_to_planning(self, d_opt: np.ndarray) -> Optional[np.ndarray]:
        """Rotate a wide-cam optical translation into the planning frame."""
        T = self._tf_matrix(self.planning_frame, self.wide_frame)
        if T is None:
            return None
        v = T[:3, :3] @ np.asarray(d_opt, dtype=np.float64).reshape(3)
        return v

    def _range_heading_xy(
        self, target: Optional[Tuple[float, float, float]] = None
    ) -> Optional[np.ndarray]:
        """Unit XY heading to close standoff (table plane, not optical meters).

        Prefer walking toward the D435i click so waist can stay facing it.
        Fallback: EE +X (out the fingers), then radial from the base.
        """
        ee = self._ee_xyz()
        if ee is not None:
            er = math.hypot(float(ee[0]), float(ee[1]))
            if er > 1e-3:
                outward = np.array(
                    [float(ee[0]) / er, float(ee[1]) / er, 0.0], dtype=np.float64
                )
            else:
                outward = None
        else:
            outward = None
        if target is not None:
            tx, ty = float(target[0]), float(target[1])
            if ee is not None:
                dx = tx - float(ee[0])
                dy = ty - float(ee[1])
                n = math.hypot(dx, dy)
                if n > 0.008:
                    heading = np.array([dx / n, dy / n, 0.0], dtype=np.float64)
                    # Closing must not walk back toward the base (shoulder folds back).
                    if outward is not None and float(np.dot(heading[:2], outward[:2])) < 0.0:
                        return outward
                    return heading
            n = math.hypot(tx, ty)
            if n > 1e-6:
                return np.array([tx / n, ty / n, 0.0], dtype=np.float64)
        if outward is not None:
            return outward
        heading = None
        T_ee = self._tf_matrix(self.planning_frame, self.ee_link)
        if T_ee is not None:
            heading = T_ee[:3, 0].copy()  # Interbotix: +X out the gripper
        T_w = self._tf_matrix(self.planning_frame, self.wide_frame)
        if T_w is not None:
            oz = T_w[:3, 2]
            if heading is None or (oz[0] ** 2 + oz[1] ** 2) > (
                heading[0] ** 2 + heading[1] ** 2
            ):
                heading = oz.copy()
        xy = None
        if heading is not None:
            xy = np.array([heading[0], heading[1], 0.0], dtype=np.float64)
            n = float(np.linalg.norm(xy))
            if n < 0.20:
                xy = None
            else:
                xy /= n
                if ee is not None and (xy[0] * ee[0] + xy[1] * ee[1]) < 0.0:
                    xy = -xy
        if xy is None and ee is not None:
            xy = np.array([ee[0], ee[1], 0.0], dtype=np.float64)
            n = float(np.linalg.norm(xy))
            if n < 1e-6:
                return None
            xy /= n
        return xy

    def _click_align_joints(
        self, tgt_xyz: Tuple[float, float, float]
    ) -> Tuple[float, float]:
        """Waist facing the click; wrist_rotate wraps a leaning object if known."""
        waist = self._waist_face_target(tgt_xyz)
        self._waist_face = waist
        if self._wrist_face is not None:
            wr = float(self._wrist_face)
        elif self.lock_wrist_rotate:
            wr = 0.0
        else:
            wr = float(self._joint_pos.get("wrist_rotate", 0.0))
        return waist, self._clip_wrist_rotate(wr)

    def _clip_wrist_rotate(self, wr: float) -> float:
        lim = abs(float(self.wrist_rotate_max))
        return max(-lim, min(lim, float(wr)))

    def _clip_wrist_angle(self, ang: float) -> float:
        return max(
            float(self.wrist_angle_min),
            min(float(self.wrist_angle_max), float(ang)),
        )

    def _wrist_from_image_u(
        self, u: float, max_step: Optional[float] = None
    ) -> Optional[float]:
        """Wrist_rotate that reduces wide-cam left/right error. None = skip."""
        if not self.wide_servo_wrist:
            return None
        if self._joint_index("wrist_rotate") < 0:
            return None
        du = float(u) - float(self.wide_grasp_u)
        if abs(du) <= 0.35 * float(self.wide_servo_tol):
            return float(self._joint_pos.get("wrist_rotate", 0.0))
        d_wr = (
            float(self.wide_servo_wr_sign)
            * float(self.wide_servo_gain_wr)
            * du
        )
        cap = max(0.04, float(self.wide_servo_max_wr if max_step is None else max_step))
        d_wr = max(-cap, min(cap, d_wr))
        cur = float(self._joint_pos.get("wrist_rotate", 0.0))
        return self._clip_wrist_rotate(cur + d_wr)

    def _clip_waist(self, w: float) -> float:
        lo, hi = self.waist_lim
        return max(float(lo), min(float(hi), float(w)))

    def _clamp_shoulder_forward(self, q: List[float]) -> List[float]:
        """On this 4-DOF, decreasing shoulder lifts the arm (back). Forbid that."""
        if not self._shoulder_forward_only:
            return q
        ish = self._joint_index("shoulder")
        if ish < 0:
            return q
        cur = float(self._joint_pos.get("shoulder", q[ish]))
        floor = max(
            float(self.shoulder_min_grasp),
            cur - float(self.shoulder_back_slack),
        )
        if q[ish] < floor:
            q = list(q)
            q[ish] = floor
        return q

    def _waist_from_image_u(
        self, u: float, max_step: Optional[float] = None
    ) -> Optional[float]:
        """Waist yaw that reduces wide-cam left/right error. None = skip."""
        if not self.wide_range_waist:
            return None
        if self._joint_index("waist") < 0:
            return None
        du = float(u) - float(self.wide_grasp_u)
        cur = float(self._joint_pos.get("waist", 0.0))
        if abs(du) <= 0.35 * float(self.wide_servo_tol):
            return cur
        d_w = float(self.wide_range_waist_sign) * float(self.wide_range_gain_waist) * du
        cap = max(0.02, float(self.wide_range_max_waist if max_step is None else max_step))
        d_w = max(-cap, min(cap, d_w))
        return self._clip_waist(cur + d_w)

    def _nudge_to_xyz(
        self,
        xyz: Tuple[float, float, float],
        gen: int,
        dur: float,
        wrist_rotate: Optional[float] = None,
        waist: Optional[float] = None,
        wrist_angle: Optional[float] = None,
    ) -> bool:
        if not self._in_workspace(xyz):
            return False
        if waist is not None:
            self._waist_face = self._clip_waist(waist)
        pose = self._pose_at(xyz, 0.0)
        q = self._solve_ik(pose, gen, nearby=True)
        if q is None or self._is_cancelled(gen):
            return False
        q = self._clamp_shoulder_forward(q)
        ir = self._joint_index("wrist_rotate")
        if ir >= 0:
            if wrist_rotate is None:
                wrist_rotate = float(self._joint_pos.get("wrist_rotate", 0.0))
            q[ir] = self._clip_wrist_rotate(wrist_rotate)
        iw = self._joint_index("waist")
        if iw >= 0 and waist is not None:
            q[iw] = self._clip_waist(waist)
            self._waist_face = q[iw]
        ia = self._joint_index("wrist_angle")
        if ia >= 0 and wrist_angle is not None:
            q[ia] = self._clip_wrist_angle(wrist_angle)
        old = self.seg_dur
        self.seg_dur = max(0.25, float(dur))
        try:
            ok = self._send_trajectory([q], gen)
        finally:
            self.seg_dur = old
        if ok:
            for j, v in zip(self.arm_joints, q):
                self._joint_pos[j] = float(v)
            ish = self._joint_index("shoulder")
            if ish >= 0:
                self._joint_pos["shoulder"] = q[ish]
        return ok

    def _cube_uv_stats(
        self, cube: Optional[dict]
    ) -> Optional[Tuple[float, float, Optional[float], float]]:
        """Centroid / 2D angle / span of the approach (or opening) face."""
        if cube is None or not cube.get("uv"):
            return None
        uv = cube["uv"]
        face = cube.get("face") or []
        pts = []
        for i in face:
            if 0 <= int(i) < len(uv):
                p = uv[int(i)]
                if p is not None and len(p) >= 2:
                    pts.append(p)
        if len(pts) < 2:
            pts = [p for p in uv if p is not None and len(p) >= 2]
        if len(pts) < 2:
            return None
        cu = sum(p[0] for p in pts) / len(pts)
        cv = sum(p[1] for p in pts) / len(pts)
        ang: Optional[float] = None
        blue = (cube.get("arrows") or {}).get("blue")
        if (
            isinstance(blue, (list, tuple))
            and len(blue) >= 2
            and blue[0] is not None
            and blue[1] is not None
            and len(blue[0]) >= 2
            and len(blue[1]) >= 2
        ):
            ang = math.atan2(
                float(blue[1][1]) - float(blue[0][1]),
                float(blue[1][0]) - float(blue[0][0]),
            )
        # Blue edge first (face[0]→face[1] = +Z = gripper top).
        pairs = []
        if len(face) >= 2:
            pairs.append((face[0], face[1]))
        if len(face) >= 4:
            pairs.extend(((face[0], face[3]), (face[1], face[2])))
        pairs.extend(((0, 1), (4, 5), (0, 2), (4, 6)))
        if ang is None:
            for a, b in pairs:
                pa = uv[a] if a < len(uv) else None
                pb = uv[b] if b < len(uv) else None
                if pa is None or pb is None:
                    continue
                ang = math.atan2(pb[1] - pa[1], pb[0] - pa[0])
                break
        span = 0.0
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                span = max(
                    span,
                    math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]),
                )
        return (cu, cv, ang, span)

    def _d435i_uv_delta_to_xy(
        self, du: float, dv: float, z_m: float
    ) -> Optional[Tuple[float, float]]:
        """D435i normalized UV error → table-plane meters (camera fixed)."""
        with self._lock:
            info = self._info
            frame = self._depth_frame or self.camera_frame
        if info is None or float(info.k[0]) < 1.0 or float(info.k[4]) < 1.0:
            return None
        fx, fy = float(info.k[0]), float(info.k[4])
        w = int(info.width) if info.width else 640
        h = int(info.height) if info.height else 480
        z = max(0.10, float(z_m))
        d_opt = np.array(
            [
                float(du) * float(max(1, w - 1)) * z / fx,
                float(dv) * float(max(1, h - 1)) * z / fy,
                0.0,
            ],
            dtype=np.float64,
        )
        tf = self._d435i_planning_T(frame or self.camera_frame)
        if tf is None:
            return None
        T, _used = tf
        d_plan = T[:3, :3] @ d_opt
        return float(d_plan[0]), float(d_plan[1])

    def _d435i_cube_servo(
        self,
        xyz: Tuple[float, float, float],
        gen: int,
        target: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[Tuple[float, float, float], str]:
        """Nudge EE until the magenta jaw cube matches the green object cube in D435i."""
        if not self.d435i_cube_servo:
            return xyz, "off"
        tgt = tuple(target or self._viz_xyz or xyz)
        ee0 = self._ee_xyz()
        if ee0 is not None:
            cur = [float(ee0[0]), float(ee0[1]), float(xyz[2])]
        else:
            cur = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
        tf = self._d435i_planning_T(self._depth_frame or self.camera_frame)
        if tf is not None:
            origin = tf[0][:3, 3]
            z_est = float(
                np.linalg.norm(
                    np.array(tgt, dtype=np.float64) - origin
                )
            )
        else:
            z_est = max(0.12, math.hypot(tgt[0], tgt[1]))
        t0 = time.monotonic()
        timeout = max(4.0, min(12.0, float(self.wide_cube_servo_timeout_s)))
        max_iters = max(4, min(10, int(self.wide_cube_servo_iters)))
        gain = float(self.d435i_cube_servo_gain_uv)
        max_step = 0.012
        for i in range(max_iters):
            if self._is_cancelled(gen):
                return tuple(cur), "cancelled"
            if time.monotonic() - t0 >= timeout:
                break
            obj = self._landing_cube_uv(tgt, cam="d435i")
            grip = self._gripper_cube_uv(cam="d435i")
            os_ = self._cube_uv_stats(obj)
            gs_ = self._cube_uv_stats(grip)
            if os_ is None or gs_ is None:
                self._publish_status(
                    "grasping",
                    "Waiting for D435i jaw + object cubes…",
                    tgt,
                )
                if not self._sleep_interruptible(0.15, gen):
                    return tuple(cur), "cancelled"
                continue
            du = float(os_[0] - gs_[0])
            dv = float(os_[1] - gs_[1])
            droll = 0.0
            if os_[2] is not None and gs_[2] is not None:
                droll = _wrap_pi(float(os_[2] - gs_[2]))
            err = math.hypot(du, dv) + 0.35 * abs(droll)
            self._publish_status(
                "grasping",
                f"D435i cube align {i + 1}/{max_iters}  "
                f"err={err:.3f} (tol {float(self.wide_cube_servo_tol):.3f})  "
                f"droll={math.degrees(droll):.0f}°",
                tgt,
            )
            if err <= float(self.wide_cube_servo_tol):
                return tuple(cur), "aligned"
            if math.hypot(du, dv) > 0.45:
                dx, dy = 0.0, 0.0
            else:
                delta = self._d435i_uv_delta_to_xy(du, dv, z_est)
                if delta is None:
                    return tuple(cur), "no_tf"
                dx, dy = delta[0] * gain, delta[1] * gain
                step = math.hypot(dx, dy)
                if step > max_step and step > 1e-6:
                    dx *= max_step / step
                    dy *= max_step / step
            nxt = (
                float(cur[0] + dx),
                float(cur[1] + dy),
                float(cur[2]),
            )
            if not self._in_workspace(nxt):
                nxt = (
                    min(self.ws_max[0], max(self.ws_min[0], nxt[0])),
                    min(self.ws_max[1], max(self.ws_min[1], nxt[1])),
                    cur[2],
                )
            wr_cur = float(self._joint_pos.get("wrist_rotate", 0.0))
            d_wr = float(self.wide_cube_servo_wr_sign) * droll * float(
                self.wide_cube_servo_gain_roll
            )
            cap_wr = max(0.02, float(self.wide_cube_servo_max_wr))
            d_wr = max(-cap_wr, min(cap_wr, d_wr))
            wr = self._clip_wrist_rotate(wr_cur + d_wr)
            self._wrist_face = wr
            if not self._nudge_to_xyz(
                nxt,
                gen,
                float(self.wide_servo_nudge_s),
                wrist_rotate=wr,
            ):
                return tuple(cur), "nudge_fail"
            cur = [float(nxt[0]), float(nxt[1]), float(nxt[2])]
            if not self._sleep_interruptible(0.08, gen):
                return tuple(cur), "cancelled"
        return tuple(cur), "held"

    def _wide_cube_servo(
        self,
        xyz: Tuple[float, float, float],
        gen: int,
        target: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[Tuple[float, float, float], str]:
        """Nudge until the cyan opening face matches the green approach face.

        Roll uses the blue (+Z) edges so gripper top follows the blue arrow.
        Close is not allowed until this returns 'aligned'.
        """
        if not self.wide_cube_servo:
            return xyz, "off"
        tgt = tuple(target or self._viz_xyz or xyz)
        cur = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
        t0 = time.monotonic()
        timeout = max(1.0, min(4.0, float(self.wide_cube_servo_timeout_s)))
        max_iters = max(2, min(6, int(self.wide_cube_servo_iters)))
        for i in range(max_iters):
            if self._is_cancelled(gen):
                return tuple(cur), "cancelled"
            if time.monotonic() - t0 >= timeout:
                break
            obj = self._landing_cube_uv(tgt, cam="wide")
            grip = self._gripper_cube_uv(cam="wide")
            os_ = self._cube_uv_stats(obj)
            gs_ = self._cube_uv_stats(grip)
            if os_ is None or gs_ is None:
                self._publish_status(
                    "grasping",
                    "Waiting for both wide cubes…",
                    tgt,
                )
                if not self._sleep_interruptible(0.15, gen):
                    return tuple(cur), "cancelled"
                continue
            du = float(os_[0] - gs_[0])
            dv = float(os_[1] - gs_[1])
            droll = 0.0
            if os_[2] is not None and gs_[2] is not None:
                droll = _wrap_pi(float(os_[2] - gs_[2]))
            scale_err = 0.0
            if gs_[3] > 1e-4:
                scale_err = 1.0 - (float(os_[3]) / float(gs_[3]))
            err = math.hypot(du, dv) + 0.35 * abs(droll)
            self._publish_status(
                "grasping",
                f"Aligning cubes {i + 1}/{max_iters}  "
                f"err={err:.3f} (tol {float(self.wide_cube_servo_tol):.3f})  "
                f"droll={math.degrees(droll):.0f}°",
                tgt,
            )
            if err <= float(self.wide_cube_servo_tol) and abs(scale_err) < 0.18:
                return tuple(cur), "aligned"
            wr_cur = float(self._joint_pos.get("wrist_rotate", 0.0))
            d_wr = float(self.wide_cube_servo_wr_sign) * droll * float(
                self.wide_cube_servo_gain_roll
            )
            cap_wr = max(0.02, float(self.wide_cube_servo_max_wr))
            d_wr = max(-cap_wr, min(cap_wr, d_wr))
            wr = self._clip_wrist_rotate(wr_cur + d_wr)
            self._wrist_face = wr
            # Hold the IK grasp XY. Walk-in / pitch / sidestep overshoot the object.
            if not self._nudge_to_xyz(
                tuple(cur),
                gen,
                float(self.wide_servo_nudge_s),
                wrist_rotate=wr,
            ):
                return tuple(cur), "nudge_fail"
            if not self._sleep_interruptible(0.08, gen):
                return tuple(cur), "cancelled"
        return tuple(cur), "held"

    def _wide_visual_servo(
        self,
        xyz: Tuple[float, float, float],
        gen: int,
        target: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[Tuple[float, float, float], str]:
        """Nudge EE until the D435i click projects onto the jaw pixel.

        Blob detection is not a close trigger here — a far object already
        fills the opening. Close happens after range-advance.
        Returns (ee_xyz, reason) centered|no_tf|no_wide|iters|cancelled.
        """
        if cv2 is None or not self.use_wide:
            return xyz, "no_wide"
        tgt_xyz = tuple(target or xyz)
        tgt_u = float(self.wide_grasp_u)
        tgt_v = float(self.wide_grasp_v)
        z_est = max(0.06, float(self.wide_servo_z_est))
        ee0 = self._ee_xyz()
        if ee0 is not None:
            cur = [float(ee0[0]), float(ee0[1]), float(xyz[2])]
        else:
            cur = list(xyz)
        last_uv = None
        misses = 0
        for i in range(max(1, self.wide_servo_max_iters)):
            if self._is_cancelled(gen):
                return tuple(cur), "cancelled"
            proj = self._wide_object_uv(tgt_xyz)
            if proj is None:
                misses += 1
                if misses >= 2 and last_uv is None:
                    self.get_logger().warn("wide servo: click TF into wide cam failed")
                    return tuple(cur), "no_tf"
                if not self._sleep_interruptible(0.12, gen):
                    return tuple(cur), "cancelled"
                continue
            misses = 0
            u, v, visible = proj
            last_uv = (u, v)
            du = u - tgt_u
            dv = v - tgt_v
            err = math.hypot(du, dv)
            cur_wr = float(self._joint_pos.get("wrist_rotate", 0.0))
            vis = "on" if visible else "off-frame"
            self._publish_status(
                "grasping",
                f"Wide servo {i + 1}/{self.wide_servo_max_iters} "
                f"click {vis} err={err:.2f} wr={math.degrees(cur_wr):.0f}°",
                tgt_xyz,
            )
            if err <= self.wide_servo_tol:
                return tuple(cur), "centered"
            d_opt = np.array(
                [
                    self.wide_servo_gain_u * du * z_est,
                    self.wide_servo_gain_v * dv * z_est,
                    0.0,
                ],
                dtype=np.float64,
            )
            waist_cmd, wr_hold = self._click_align_joints(tgt_xyz)
            wr_cmd = wr_hold
            if (
                self.wide_servo_wrist
                and abs(du) > 0.4 * float(self.wide_servo_tol)
            ):
                wr_img = self._wrist_from_image_u(u)
                if wr_img is not None:
                    wr_cmd = wr_img
                    d_opt[0] *= float(self.wide_servo_wrist_u_scale)
            d_plan = self._optical_delta_to_planning(d_opt)
            if d_plan is None:
                d_plan = np.array(
                    [-self.wide_servo_gain_v * dv, -self.wide_servo_gain_u * du, 0.0]
                )
            step = float(np.linalg.norm(d_plan[:2]))
            max_s = float(self.wide_servo_max_step)
            if step > max_s and step > 1e-6:
                d_plan[:2] *= max_s / step
            ee = self._ee_xyz()
            base_xy = (ee[0], ee[1]) if ee is not None else (cur[0], cur[1])
            nxt = (
                float(base_xy[0] + d_plan[0]),
                float(base_xy[1] + d_plan[1]),
                float(cur[2]),
            )
            if not self._in_workspace(nxt):
                nxt = (
                    min(self.ws_max[0], max(self.ws_min[0], nxt[0])),
                    min(self.ws_max[1], max(self.ws_min[1], nxt[1])),
                    cur[2],
                )
            wr_deg = None if wr_cmd is None else math.degrees(wr_cmd)
            self.get_logger().info(
                f"wide servo[{i}] click uv=({u:.3f},{v:.3f}) vis={visible} "
                f"err={err:.3f} wr={math.degrees(cur_wr):.1f}"
                f"{'' if wr_deg is None else f'→{wr_deg:.1f}'}°"
            )
            if not self._nudge_to_xyz(
                nxt,
                gen,
                self.wide_servo_nudge_s,
                wrist_rotate=wr_cmd,
                waist=waist_cmd,
            ):
                self.get_logger().warn(f"wide servo nudge IK/motion failed @ {nxt}")
                break
            cur = list(nxt)
            if not self._sleep_interruptible(0.12, gen):
                return tuple(cur), "cancelled"
        if last_uv is None:
            return tuple(cur), "no_tf"
        return tuple(cur), "iters"

    def _wide_bbox_servo(
        self,
        xyz: Tuple[float, float, float],
        gen: int,
        target: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[Tuple[float, float, float], str]:
        """Nudge EE until the object bbox center sits in the jaw opening."""
        if cv2 is None or not self.use_wide:
            return xyz, "no_wide"
        tgt_xyz = tuple(target or self._viz_xyz or xyz)
        jaw = self._jaw_bbox()
        ju = float(jaw["u"])
        jv = float(jaw["v"])
        z_est = max(0.06, float(self.wide_servo_z_est))
        ee0 = self._ee_xyz()
        if ee0 is not None:
            cur = [float(ee0[0]), float(ee0[1]), float(xyz[2])]
        else:
            cur = list(xyz)
        last_err = None
        misses = 0
        n_iter = max(8, int(self.wide_servo_max_iters))
        tol = float(self.wide_servo_tol)
        for i in range(n_iter):
            if self._is_cancelled(gen):
                return tuple(cur), "cancelled"
            if (
                self._wide_jpg is None
                or (time.monotonic() - float(self._wide_stamp)) > 0.40
            ):
                self._fetch_wide_jpeg()
            seed = None
            proj = self._project_xyz_to_wide(
                tgt_xyz, use_point_z=True, require_in_frame=False
            )
            if proj is not None and proj[2]:
                seed = (float(proj[0]), float(proj[1]))
            obj = self._wide_object_bbox(seed)
            if obj is not None:
                ou, ov = float(obj["u"]), float(obj["v"])
            elif seed is not None:
                ou, ov = float(seed[0]), float(seed[1])
            else:
                misses += 1
                if misses >= 3:
                    self.get_logger().warn("bbox servo: no object blob and no seed UV")
                    return tuple(cur), "no_blob"
                if not self._sleep_interruptible(0.12, gen):
                    return tuple(cur), "cancelled"
                continue
            misses = 0
            du = ou - ju
            dv = ov - jv
            err = math.hypot(du, dv)
            last_err = err
            inside = (
                float(jaw["u0"]) - 0.5 * tol <= ou <= float(jaw["u1"]) + 0.5 * tol
                and float(jaw["v0"]) - 0.5 * tol <= ov <= float(jaw["v1"]) + 0.5 * tol
            )
            self._publish_status(
                "grasping",
                f"Align boxes {i + 1}/{n_iter} err={err:.3f} "
                f"{'in-jaws' if inside else 'outside'}",
                tgt_xyz,
            )
            if inside or err <= tol:
                return tuple(cur), "aligned"
            d_opt = np.array(
                [
                    self.wide_servo_gain_u * du * z_est,
                    self.wide_servo_gain_v * dv * z_est,
                    0.0,
                ],
                dtype=np.float64,
            )
            waist_cmd, wr_hold = self._click_align_joints(tgt_xyz)
            d_plan = self._optical_delta_to_planning(d_opt)
            if d_plan is None:
                d_plan = np.array(
                    [-self.wide_servo_gain_v * dv, -self.wide_servo_gain_u * du, 0.0]
                )
            step = float(np.linalg.norm(d_plan[:2]))
            max_s = float(self.wide_servo_max_step)
            if step > max_s and step > 1e-6:
                d_plan[:2] *= max_s / step
            ee = self._ee_xyz()
            base_xy = (ee[0], ee[1]) if ee is not None else (cur[0], cur[1])
            nxt = (
                float(base_xy[0] + d_plan[0]),
                float(base_xy[1] + d_plan[1]),
                float(cur[2]),
            )
            if not self._in_workspace(nxt):
                nxt = (
                    min(self.ws_max[0], max(self.ws_min[0], nxt[0])),
                    min(self.ws_max[1], max(self.ws_min[1], nxt[1])),
                    cur[2],
                )
            self.get_logger().info(
                f"bbox servo[{i}] obj=({ou:.3f},{ov:.3f}) jaws=({ju:.3f},{jv:.3f}) "
                f"err={err:.3f}"
            )
            if not self._nudge_to_xyz(
                nxt,
                gen,
                self.wide_servo_nudge_s,
                wrist_rotate=wr_hold,
                waist=waist_cmd,
            ):
                self.get_logger().warn(f"bbox servo nudge IK/motion failed @ {nxt}")
                break
            cur = list(nxt)
            if not self._sleep_interruptible(0.12, gen):
                return tuple(cur), "cancelled"
        if last_err is None:
            return tuple(cur), "no_blob"
        return tuple(cur), "iters"

    def _wide_close_range(
        self,
        xyz: Tuple[float, float, float],
        gen: int,
        target: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[Tuple[float, float, float], str]:
        """Advance toward the click with jaws held at the crosshair angle.

        Does not close the gripper. Blob occupancy is not a pinch trigger —
        close waits for the two wide cubes to line up.
        """
        tgt_xyz = tuple(target or xyz)
        step = max(0.01, float(self.wide_range_step))
        max_d = max(step, float(self.wide_range_max))
        if max_d < 0.015:
            return xyz, "skip"
        ee0 = self._ee_xyz()
        if ee0 is not None:
            cur = [float(ee0[0]), float(ee0[1]), float(xyz[2])]
        else:
            cur = list(xyz)
        n = 0
        commanded = 0.0
        range_wr_cap = max(0.04, 0.70 * float(self.wide_servo_max_wr))
        max_steps = max(3, int(math.ceil(max_d / step)) + 1)
        while True:
            if commanded + 1e-6 >= max_d or n >= max_steps:
                break
            if self._is_cancelled(gen):
                return tuple(cur), "cancelled"
            heading = self._range_heading_xy(tgt_xyz)
            if heading is None:
                self.get_logger().warn("range advance: no EE/click heading")
                return tuple(cur), "no_tf"
            if self.align_wrist_to_object and (
                self.wrist_align_refine_jaws and not self._wrist_from_depth
            ):
                self._refine_wrist_from_jaws()
            waist_cmd, wr_cmd = self._click_align_joints(tgt_xyz)
            uv_txt = "click=—"
            du = 0.0
            have_uv = False
            proj = self._project_xyz_to_wide(tgt_xyz)
            if proj is not None:
                u, v, visible = proj
                have_uv = True
                du = float(u) - float(self.wide_grasp_u)
                vis = "on" if visible else "off"
                uv_txt = f"click u={u:.2f} v={v:.2f} {vis}"
                if self.wide_servo_wrist:
                    wr_img = self._wrist_from_image_u(u, max_step=range_wr_cap)
                    if wr_img is not None:
                        wr_cmd = wr_img
                if self.wide_range_waist:
                    wa_img = self._waist_from_image_u(u)
                    if wa_img is not None:
                        waist_cmd = wa_img
            cur_wr = float(self._joint_pos.get("wrist_rotate", 0.0))
            cur_wa = float(self._joint_pos.get("waist", 0.0))
            wr_show = cur_wr if wr_cmd is None else wr_cmd
            wa_show = cur_wa if waist_cmd is None else waist_cmd
            self._publish_status(
                "grasping",
                f"Walking in {commanded * 100:.0f}/{max_d * 100:.0f} cm  "
                f"waist={math.degrees(wa_show):.0f}° wr={math.degrees(wr_show):.0f}° "
                f"{uv_txt}",
                tgt_xyz,
            )
            nxt = (
                float(cur[0] + heading[0] * step),
                float(cur[1] + heading[1] * step),
                float(cur[2]),
            )
            if (
                self.wide_range_waist
                and have_uv
                and abs(du) > 0.35 * float(self.wide_servo_tol)
            ):
                # Optional image-u sidestep (off by default — twists off click).
                lat = (
                    float(self.wide_range_waist_sign)
                    * float(self.wide_range_lat_gain)
                    * du
                )
                cap = max(0.003, float(self.wide_range_lat_max))
                lat = max(-cap, min(cap, lat))
                nxt = (
                    float(nxt[0] + (-heading[1]) * lat),
                    float(nxt[1] + heading[0] * lat),
                    nxt[2],
                )
            if not self._in_workspace(nxt):
                nxt = (
                    min(self.ws_max[0], max(self.ws_min[0], nxt[0])),
                    min(self.ws_max[1], max(self.ws_min[1], nxt[1])),
                    cur[2],
                )
            horiz = math.hypot(nxt[0], nxt[1])
            if horiz > self.max_reach:
                self.get_logger().info(
                    f"range advance stop: reach {horiz:.3f}m"
                )
                return tuple(cur), "reach"
            if not self._nudge_to_xyz(
                nxt,
                gen,
                self.wide_servo_nudge_s,
                wrist_rotate=wr_cmd,
                waist=waist_cmd,
            ):
                self.get_logger().warn(f"range advance IK/motion failed @ {nxt}")
                return tuple(cur), "ik"
            cur = list(nxt)
            commanded += step
            n += 1
            if not self._sleep_interruptible(0.10, gen):
                return tuple(cur), "cancelled"
            self.get_logger().info(
                f"range advance[{n}] cmd={commanded:.3f}m "
                f"waist={math.degrees(wa_show):.1f}° wr={math.degrees(wr_show):.1f}° "
                f"{uv_txt} -> ({cur[0]:.3f},{cur[1]:.3f},{cur[2]:.3f})"
            )
        return tuple(cur), "done"

    def _js_cb(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            p = float(pos)
            # xs_sdk publishes ~-pi when Dynamixel syncRead fails
            if abs(p + math.pi) <= 0.05:
                continue
            if name in self._joint_pos:
                self._joint_pos[name] = p
            if name == "pan":
                self._pan_pos = p
            if name == "left_finger":
                self._left_finger = p
            if name == "gripper":
                self._gripper_joint = p
        # Topic arriving = bus is up, even if this sample was -pi.
        self._js_ok_stamp = time.monotonic()

    def _click_cb(self, msg: String):
        """Parse on the executor, resolve 3D on a worker.

        `_snapshot_depth` waits for a one-shot subscription. If that wait
        runs on an executor thread, overlay TF lookups fill the pool and
        the depth callback never runs — so only the first click works.
        """
        try:
            data = json.loads(msg.data)
        except Exception:
            self._publish_status("error", "Bad grasp_click JSON")
            return
        u = float(data.get("u", -1))
        v = float(data.get("v", -1))
        if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
            self._publish_status("error", "Click outside image")
            return
        try:
            cid = int(data.get("click_id"))
        except (TypeError, ValueError):
            cid = None
        self._click_gen += 1
        gen = self._click_gen
        self._click_id = cid
        self._clicking = True
        threading.Thread(
            target=self._click_cb_worker,
            args=(data, u, v, gen),
            daemon=True,
            name="grasp_click",
        ).start()

    def _click_cb_worker(self, data: dict, u: float, v: float, gen: int):
        try:
            if gen != self._click_gen:
                return
            cam = str(data.get("cam") or data.get("camera") or "d435i").lower()
            if cam in ("wide", "wide_cam", "jaw"):
                self.get_logger().info("ignoring wide-image click — D435i only")
                return
            self._d435i_click_cb(data, u, v)
        except Exception as e:
            self.get_logger().warn(f"click worker: {e}")
            self._publish_status("error", f"Click failed: {e}")
        finally:
            if gen == self._click_gen:
                self._clicking = False

    def _d435i_click_cb(self, data: dict, u: float, v: float):
        self._last_click_uv = (u, v)
        self._wrist_face = None
        self._wrist_from_depth = False
        self._object_low = False
        self._object_lying = False
        self._object_top_z = None
        self._cube_roll = 0.0
        self._cube_pitch = 0.0
        self._cube_yaw = 0.0
        self._object_centroid = None
        self._wide_ring_uv = None
        self._wide_ring_stamp = 0.0
        self._wide_ring_logged = False
        # Clear stale viz so "Resolving depth…" does not re-enter overlay TF.
        self._viz_xyz = None
        self._viz_from_depth = False
        self._publish_status(
            "planning", "Resolving depth…", skip_overlay=True
        )
        self._snapshot_depth()
        feat = self._analyze_click_depth(u, v)
        if feat is not None:
            self._object_top_z = float(feat["z"])
            self._object_low = bool(feat["low"])
            self._object_lying = bool(feat.get("lying", False))
            if feat.get("centroid") is not None:
                c = feat["centroid"]
                self._object_centroid = (float(c[0]), float(c[1]), float(c[2]))
            if feat.get("wr") is not None:
                self._wrist_face = float(feat["wr"])
                self._wrist_from_depth = True
        with self._lock:
            self._pending_click = data
        try:
            img_w = int(data.get("width") or 0)
            img_h = int(data.get("height") or 0)
        except (TypeError, ValueError):
            img_w = img_h = 0
        how = "pix-depth"
        extra = ""
        pixel = self._click_ray_point(u, v, img_w, img_h)
        if pixel is not None:
            patch = self._object_centroid
            self.get_logger().info(
                f"click pixel ({pixel[0]:.3f},{pixel[1]:.3f},{pixel[2]:.3f}) "
                f"reach={math.hypot(pixel[0], pixel[1]):.3f} "
                f"patch={None if patch is None else tuple(round(float(c), 3) for c in patch)}"
            )
            self._object_centroid = pixel
            self._object_top_z = float(pixel[2])
            if float(pixel[2]) <= float(self.floor_object_max_z):
                self._object_low = True
            else:
                self._object_low = False
                self._viz_xyz = pixel
                self._viz_from_depth = True
        if self._object_low and self._object_centroid is not None:
            floored = self._apply_floor_xyz(self._object_centroid)
            if floored is not None:
                self._viz_xyz = floored
                self._viz_from_depth = True
                how = "floor-depth"
                extra += (
                    f" viz=floor ({floored[0]:.2f},{floored[1]:.2f},{floored[2]:.2f})"
                )
                self.get_logger().info(
                    f"floor click ({floored[0]:.3f},{floored[1]:.3f},{floored[2]:.3f}) "
                    f"reach={math.hypot(floored[0], floored[1]):.3f} "
                    f"top={float(self._object_top_z):.3f} v={v:.2f}"
                )
        unp = None if self._viz_xyz is not None else self._unproject_d435i_uv(
            u, v, img_w, img_h
        )
        if unp is not None:
            self._viz_xyz = unp
            self._viz_from_depth = True
            extra += (
                f" viz=pix-depth ({unp[0]:.2f},{unp[1]:.2f},{unp[2]:.2f})"
            )
            self.get_logger().info(
                f"d435i-unproject ({unp[0]:.3f},{unp[1]:.3f},{unp[2]:.3f}) "
                f"click=({u:.3f},{v:.3f})"
            )
        elif self._viz_xyz is None:
            viz = self._viz_click_xyz(u, v, img_w=img_w, img_h=img_h)
            if viz is not None:
                self._viz_xyz = viz[0]
                how = str(viz[1])
                self._viz_from_depth = True
                extra += (
                    f" viz={viz[1]} ({self._viz_xyz[0]:.2f},"
                    f"{self._viz_xyz[1]:.2f},{self._viz_xyz[2]:.2f})"
                )
                self.get_logger().info(
                    f"d435i-viz {viz[1]} "
                    f"({self._viz_xyz[0]:.3f},{self._viz_xyz[1]:.3f},"
                    f"{self._viz_xyz[2]:.3f}) click v={v:.2f}"
                )
        if self._viz_xyz is not None:
            ok, xyz, err = self._apply_grasp_biases(self._viz_xyz, how)
            if not ok:
                self._publish_status("error", err or "Outside workspace", xyz)
                return
        else:
            ok, xyz, err = self._resolve_click(
                u,
                v,
                img_w=data.get("width"),
                img_h=data.get("height"),
            )
            if not ok:
                self._object_centroid = None
                self._publish_status("error", err or "Unproject failed", xyz)
                return
            self._viz_xyz = (
                float(xyz[0]),
                float(xyz[1]),
                float(xyz[2]),
            )
            self._viz_from_depth = False
            extra += " viz=table"
        if self._object_centroid is None:
            self._object_centroid = self._viz_xyz
        if self._object_low:
            extra += " floor"
        if self._wrist_from_depth:
            extra += " depth3d"
        if self._wrist_face is not None:
            extra += f" wr={math.degrees(self._wrist_face):.0f}°"
        self._wide_ring_uv = None
        self._publish_status(
            "point",
            f"Target {xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f} m ({err}){extra} — press Grasp",
            xyz,
        )
        self._d435i_pair_xyz = tuple(self._viz_xyz or xyz)
        self._d435i_pair_stamp = time.monotonic()
        vz = self._viz_xyz or xyz
        if vz is not None and float(vz[2]) > 0.09:
            self._cached_pole_xy = (float(vz[0]), float(vz[1]))

    def _wide_click_cb(self, data: dict, u: float, v: float):
        """Jaw-camera click: recover 3D along the wide ray via D435i depth."""
        self._wrist_face = None
        self._wrist_from_depth = False
        self._object_low = False
        self._object_lying = False
        self._object_top_z = None
        self._cube_roll = 0.0
        self._cube_pitch = 0.0
        self._cube_yaw = 0.0
        self._object_centroid = None
        self._last_click_uv = None
        self._snapshot_depth()
        with self._lock:
            self._pending_click = data
        xyz, how = self._xyz_from_wide_click(
            u, v, img_w=data.get("width"), img_h=data.get("height")
        )
        if xyz is None:
            self._viz_xyz = None
            self._publish_status(
                "error", how or "Wide click: no D435i depth on that ray"
            )
            return
        self._viz_xyz = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
        self._viz_from_depth = True
        if self._last_click_uv is None:
            d435 = self._project_xyz_to_d435i(self._viz_xyz)
            if d435 is not None and d435[2]:
                self._last_click_uv = (float(d435[0]), float(d435[1]))
        if self._last_click_uv is not None:
            lu, lv = self._last_click_uv
            feat = self._analyze_click_depth(lu, lv)
            if feat is not None:
                same_h = abs(float(feat["z"]) - float(self._viz_xyz[2])) < 0.06
                if same_h:
                    self._object_top_z = float(feat["z"])
                    self._object_low = bool(feat["low"])
                    if feat.get("centroid") is not None:
                        c = feat["centroid"]
                        self._object_centroid = (
                            float(c[0]),
                            float(c[1]),
                            float(c[2]),
                        )
                    if feat.get("wr") is not None:
                        self._wrist_face = float(feat["wr"])
                        self._wrist_from_depth = True
        ok, grasp_xyz, err = self._apply_grasp_biases(xyz, how)
        if not ok:
            self._viz_xyz = None
            self._publish_status("error", err or "Wide click outside workspace", grasp_xyz)
            return
        extra = f" viz=wide ({self._viz_xyz[0]:.2f},{self._viz_xyz[1]:.2f},{self._viz_xyz[2]:.2f})"
        cal = self._maybe_pair_calibrate_wide(u, v)
        if cal:
            extra += cal
        if self._object_centroid is None:
            self._object_centroid = self._viz_xyz
        if self._object_low:
            extra += " floor"
        if self._wrist_from_depth:
            extra += " depth3d"
        self._publish_status(
            "point",
            f"Target {grasp_xyz[0]:.3f}, {grasp_xyz[1]:.3f}, {grasp_xyz[2]:.3f} m "
            f"({err}){extra} — press Grasp",
            grasp_xyz,
        )

    def _maybe_pair_calibrate_wide(self, u: float, v: float) -> str:
        """If D435i was clicked recently, treat this wide click as the same 3D point."""
        if self._d435i_pair_xyz is None:
            return ""
        if time.monotonic() - self._d435i_pair_stamp > 45.0:
            return ""
        T_base = self._wide_cam_from_planning(apply_extra=False)
        if T_base is None:
            return ""
        self._marker_pairs.append(
            {
                "xyz": tuple(self._d435i_pair_xyz),
                "u": float(u),
                "v": float(v),
                "name": "click",
                "T_base": T_base.copy(),
                "look": self._wide_look_rpy(),
            }
        )
        ok, msg, _rmse = self._solve_wide_from_marker(self._marker_pairs)
        self._d435i_pair_xyz = None
        self._d435i_pair_stamp = 0.0
        n = len(self._marker_pairs)
        if ok:
            return f" {msg}"
        return f" pair-cal {n} pts ({msg})"

    def _wide_ray_planning(
        self,
        u_norm: float,
        v_norm: float,
        img_w=None,
        img_h=None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Wide click → (origin, unit direction) in the planning frame."""
        T = self._wide_cam_from_planning()
        if T is None:
            return None
        try:
            Tinv = np.linalg.inv(T)
        except np.linalg.LinAlgError:
            return None
        fx, fy, cx, cy, w, h = self._wide_intrinsics()
        try:
            cw = int(img_w) if img_w else 0
            ch = int(img_h) if img_h else 0
        except (TypeError, ValueError):
            cw = ch = 0
        if cw >= 32 and ch >= 32:
            w, h = cw, ch
        if w < 2 or h < 2 or fx <= 0.0 or fy <= 0.0:
            return None
        u_px = float(u_norm) * float(w - 1) - self.wide_u_bias * float(w - 1)
        v_px = float(v_norm) * float(h - 1) - self.wide_v_bias * float(h - 1)
        ray = np.array(
            [(u_px - cx) / fx, (v_px - cy) / fy, 1.0], dtype=np.float64
        )
        origin = Tinv[:3, 3].copy()
        direction = Tinv[:3, :3] @ ray
        n = float(np.linalg.norm(direction))
        if n < 1e-9:
            return None
        return origin, direction / n

    def _wide_ray_table_hit(
        self, origin: np.ndarray, direction: np.ndarray
    ) -> Optional[Tuple[float, float, float]]:
        dz = float(direction[2])
        if abs(dz) < 1e-6:
            return None
        t = (self._click_plane_z() - float(origin[2])) / dz
        if t < 0.05 or t > 1.4:
            return None
        p = origin + t * direction
        xyz = (float(p[0]), float(p[1]), float(p[2]))
        if float(xyz[0]) < 0.02 or math.hypot(xyz[0], xyz[1]) > 0.55:
            return None
        return xyz

    def _xyz_from_wide_click(
        self,
        u_norm: float,
        v_norm: float,
        img_w=None,
        img_h=None,
    ) -> Tuple[Optional[Tuple[float, float, float]], str]:
        """Wide click → stay on the wide ray; recover z from D435i optical Z.

        Walk the ray in base_link, project each sample into D435i, accept the
        sample whose predicted D435i optical Z matches aligned depth. Keep the
        ray sample — do not unproject the D435i pixel (that jumps to the floor)
        and do not snap to the black band.
        """
        rd = self._wide_ray_planning(u_norm, v_norm, img_w, img_h)
        if rd is None:
            return None, "no wide TF/intrinsics"
        origin, direction = rd
        with self._lock:
            depth = None if self._depth is None else self._depth.copy()
            info = self._info
            frame = self._depth_frame or self.camera_frame
        if depth is None or info is None:
            return None, "no D435i aligned-depth"
        tf = self._d435i_planning_T(frame or info.header.frame_id)
        if tf is None:
            return None, "no D435i TF"
        T_d, _used = tf
        try:
            T_d_inv = np.linalg.inv(T_d)
        except np.linalg.LinAlgError:
            return None, "bad D435i TF"
        best = None  # (score, xyz, du, dv, z_err, t)
        n_proj = 0
        n_depth = 0
        for t in np.linspace(0.06, 0.55, 140):
            p = origin + float(t) * direction
            cand = (float(p[0]), float(p[1]), float(p[2]))
            if not self._viz_plausible(cand):
                continue
            cam = T_d_inv @ np.array(
                [cand[0], cand[1], cand[2], 1.0], dtype=np.float64
            )
            z_pred = float(cam[2])
            if z_pred < 0.08:
                continue
            proj = self._project_xyz_to_d435i(cand)
            if proj is None or not proj[2]:
                continue
            du, dv = float(proj[0]), float(proj[1])
            if not (0.0 <= du <= 1.0 and 0.0 <= dv <= 1.0):
                continue
            n_proj += 1
            pix = self._click_k_pixels(
                du, dv, info.width, info.height, info, depth
            )
            if pix is None:
                continue
            u_d, v_d = self._depth_sample_uv(pix[0], pix[1], info, depth)
            z_meas = self._near_depth(depth, u_d, v_d)
            if z_meas is None:
                continue
            n_depth += 1
            z_err = abs(z_meas - z_pred)
            if z_err > 0.05:
                continue
            score = z_err
            if best is None or score < best[0]:
                best = (score, cand, du, dv, z_err, float(t))
        if best is None:
            self.get_logger().warn(
                f"wide-epi: no D435i match click=({u_norm:.3f},{v_norm:.3f}) "
                f"proj={n_proj} depth={n_depth}"
            )
            return None, f"no D435i depth on wide ray (proj={n_proj} depth={n_depth})"
        _score, xyz, du, dv, z_err, t = best
        self._last_click_uv = (float(du), float(dv))
        self.get_logger().info(
            f"wide-epi ray ({xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f}) "
            f"t={t:.3f} zerr={z_err*100:.1f}cm "
            f"d435i=({du:.3f},{dv:.3f}) click=({u_norm:.3f},{v_norm:.3f}) "
            f"proj={n_proj} depth={n_depth}"
        )
        return xyz, f"wide-epi zerr={z_err*100:.0f}mm"

    def _apply_grasp_biases(
        self,
        xyz: Tuple[float, float, float],
        method: str,
    ) -> Tuple[bool, Optional[Tuple[float, float, float]], str]:
        horiz = math.hypot(xyz[0], xyz[1])
        bias = float(self.grasp_farther_m)
        if bias > 1e-4 and horiz > 0.04:
            new_r = min(horiz + bias, float(self.max_reach) - 0.01)
            if new_r > horiz:
                s = new_r / horiz
                xyz = (float(xyz[0]) * s, float(xyz[1]) * s, float(xyz[2]))
                method = f"{method}+farther{bias * 100:.0f}cm"
                horiz = new_r
        floor_xyz = self._apply_floor_xyz(xyz)
        if floor_xyz is not None:
            xyz = floor_xyz
            method = f"{method}+floorZ{xyz[2]:.2f}"
            horiz = math.hypot(xyz[0], xyz[1])
        if horiz > self.max_reach:
            return (
                False,
                xyz,
                f"Too far for short arm ({horiz:.2f}m > {self.max_reach:.2f}m) "
                f"via {method}",
            )
        if not self._in_workspace(xyz):
            return (
                False,
                xyz,
                f"Outside workspace via {method}: "
                f"({xyz[0]:.2f},{xyz[1]:.2f},{xyz[2]:.2f})",
            )
        return True, xyz, method

    def _gripper_is_open(self) -> bool:
        """True if jaws look open enough to receive an object."""
        # left_finger ~0.0115 closed … ~0.0375 open (GUI constants).
        if self._left_finger is not None:
            return float(self._left_finger) >= 0.028
        # gripper joint ~-0.95 closed … ~1.65 open.
        if self._gripper_joint is not None:
            return float(self._gripper_joint) >= 0.40
        return False

    def _open_gripper_on_point_click(self):
        """If the gripper is closed when a grasp point is clicked, open it."""
        if self._busy:
            return
        if self._gripper_is_open():
            return
        threading.Thread(
            target=self._open_gripper_click_worker, daemon=True
        ).start()

    def _open_gripper_click_worker(self):
        try:
            before = self._left_finger
            gj = self._gripper_joint
            self.get_logger().info(
                f"point-click: gripper closed "
                f"(finger={before}, joint={gj}) — opening"
            )
            self._ensure_gripper_mode()
            self._set_gripper(self.grip_open, settle_s=0.7, hold=True)
            after = self._left_finger
            if before is not None and after is not None and abs(after - before) < 1e-4:
                self.get_logger().warn(
                    f"point-click open: left_finger unchanged ({after:.4f})"
                )
        except Exception as e:
            self.get_logger().warn(f"point-click gripper open: {e}")

    def _is_cancelled(self, gen: int) -> bool:
        return self._cancel or gen != self._run_gen

    def _cancel_motion(self):
        gh = self._active_goal
        if gh is None:
            return
        try:
            gh.cancel_goal_async()
        except Exception:
            pass

    def _sleep_interruptible(self, seconds: float, gen: int) -> bool:
        """Sleep up to `seconds`; return False if this run was cancelled."""
        deadline = time.time() + max(0.0, seconds)
        while time.time() < deadline:
            if self._is_cancelled(gen):
                return False
            time.sleep(min(0.05, deadline - time.time()))
        return not self._is_cancelled(gen)

    def _cmd_cb(self, msg: String):
        raw = (msg.data or "").strip()
        data: Optional[dict] = None
        if raw.startswith("{"):
            try:
                data = json.loads(raw)
            except Exception:
                self._publish_status("error", "Bad grasp_command JSON")
                return
            cmd = str(data.get("cmd", "")).strip().lower()
            if cmd == "cube_orient":
                self._cube_roll = float(data.get("roll", 0.0))
                self._cube_pitch = float(data.get("pitch", 0.0))
                self._cube_yaw = float(data.get("yaw", 0.0))
                if self._viz_xyz is not None or self._last_xyz is not None:
                    st = (
                        self._last_status
                        if self._last_status not in ("idle", "error", "cancelled")
                        else "point"
                    )
                    self._publish_status(
                        st,
                        self._last_status_msg or "Cube orientation updated",
                    )
                return
        else:
            cmd = raw.lower()
        if cmd == "cancel":
            # Free the slot immediately so a new Grasp can start; invalidate
            # the in-flight thread via _run_gen so it exits without actuating.
            with self._lock:
                self._cancel = True
                self._run_gen += 1
                self._busy = False
                handed = self._act_handed
                self._act_handed = False
            self._cancel_motion()
            if handed:
                self._il_cmd_pub.publish(String(data=json.dumps({"cmd": "stop"})))
            self._publish_status("cancelled", "Grasp cancelled — click / Grasp again")
            return
        if cmd == "lock_pan":
            self._lock_pan()
            self._publish_status("idle", "Pan locked for grasp")
            return
        if cmd == "anchor":
            held = self._click_anchor or self._last_xyz
            if held is None:
                self._publish_status(
                    "error", "No target — click the D435i image first"
                )
                return
            self._publish_status("point", "Click held for auto collect", held)
            return
        if cmd == "retreat":
            return_q: Optional[List[float]] = None
            if data is not None and "return_joints" in data:
                raw_q = data.get("return_joints")
                if (
                    not isinstance(raw_q, list)
                    or len(raw_q) != len(self.arm_joints)
                ):
                    self._publish_status("error", "Invalid take-start joints")
                    return
                try:
                    return_q = [float(v) for v in raw_q]
                except (TypeError, ValueError):
                    self._publish_status("error", "Invalid take-start joints")
                    return
                if not all(math.isfinite(v) for v in return_q):
                    self._publish_status("error", "Invalid take-start joints")
                    return
                ish = self._joint_index("shoulder")
                if (
                    ish >= 0
                    and return_q[ish] < float(self.shoulder_min_grasp)
                ):
                    self._publish_status(
                        "error", "Take start has an unsafe shoulder angle"
                    )
                    return
            gen = self._try_start("Retreat already running")
            if gen is None:
                return
            threading.Thread(
                target=self._run_retreat, args=(gen, return_q), daemon=True
            ).start()
            return
        if cmd == "release":
            gen = self._try_start("Release already running")
            if gen is None:
                return
            threading.Thread(
                target=self._run_release, args=(gen,), daemon=True
            ).start()
            return
        if cmd == "set_down":
            gen = self._try_start("Set-down already running")
            if gen is None:
                return
            threading.Thread(
                target=self._run_set_down, args=(gen,), daemon=True
            ).start()
            return
        if cmd in ("execute", "grasp"):
            with self._lock:
                if self._busy:
                    self._publish_status("busy", "Grasp already running")
                    return
                if self._last_xyz is None and self._click_anchor is None:
                    self._publish_status(
                        "error", "No target — click the D435i image first"
                    )
                    return
                self._pending_offset = self._offset_from_cmd(data)
                self._busy = True
                self._cancel = False
                self._run_gen += 1
                gen = self._run_gen
            threading.Thread(
                target=self._run_grasp, args=(gen,), daemon=True
            ).start()
            return
        if cmd in ("handoff_grab", "grab_object"):
            gen = self._try_start("Grab already running")
            if gen is None:
                return
            threading.Thread(
                target=self._run_handoff_grab, args=(gen,), daemon=True
            ).start()
            return
        if cmd in ("handoff_release", "release_object"):
            gen = self._try_start("Release already running")
            if gen is None:
                return
            threading.Thread(
                target=self._run_handoff_release, args=(gen,), daemon=True
            ).start()
            return
        if cmd in ("calibrate_marker", "calibrate_from_marker"):
            if self._busy:
                self._publish_status("busy", "Busy — cancel grasp first")
                return
            threading.Thread(
                target=self._run_marker_calibrate, daemon=True
            ).start()
            return
        if cmd in ("calibrate_marker_reset", "reset_marker_cal"):
            self._reset_marker_calibrate()
            return
        self._publish_status("error", f"Unknown command: {cmd}")

    def _offset_from_cmd(
        self, data: Optional[dict]
    ) -> Optional[Tuple[float, float, float, Optional[float]]]:
        """Clamp a requested nudge to the safe box around the click.

        Lateral and depth stay within 1 cm. Height stays within 3 cm.
        Standoff, when set, stays between 4 and 7 cm. The approach side
        is not a parameter.
        """
        if not isinstance(data, dict):
            return None
        if not any(k in data for k in ("lateral", "depth", "height", "standoff")):
            return None
        lateral = max(-0.01, min(0.01, float(data.get("lateral", 0.0))))
        depth = max(-0.01, min(0.01, float(data.get("depth", 0.0))))
        height = max(-0.03, min(0.03, float(data.get("height", 0.0))))
        standoff = data.get("standoff", None)
        if standoff is not None:
            standoff = max(0.04, min(0.07, float(standoff)))
        return lateral, depth, height, standoff

    def _apply_auto_offset(
        self, xyz: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        off = self._pending_offset
        self._pending_offset = None
        if off is None:
            return xyz
        lateral, depth, height, standoff = off
        anchor = self._click_anchor or xyz
        hx, hy = self._approach_xy_hat(anchor)
        lx, ly = -hy, hx
        varied = (
            float(anchor[0]) + depth * hx + lateral * lx,
            float(anchor[1]) + depth * hy + lateral * ly,
            float(anchor[2]) + height,
        )
        self._standoff_hold = float(self.grasp_standoff_m)
        if standoff is not None:
            self.grasp_standoff_m = float(standoff)
        self._last_xyz = varied
        self._viz_xyz = varied
        self.get_logger().info(
            f"grasp offset lateral={lateral:+.3f} depth={depth:+.3f} "
            f"height={height:+.3f} standoff={self.grasp_standoff_m:.3f} "
            f"-> ({varied[0]:.3f}, {varied[1]:.3f}, {varied[2]:.3f})"
        )
        return varied

    def _restore_click_anchor(self):
        if self._standoff_hold is not None:
            self.grasp_standoff_m = float(self._standoff_hold)
            self._standoff_hold = None
        if self._click_anchor is not None:
            self._last_xyz = self._click_anchor
            self._viz_xyz = self._click_anchor

    def _try_start(self, busy_msg: str) -> Optional[int]:
        with self._lock:
            if self._busy:
                self._publish_status("busy", busy_msg)
                return None
            self._busy = True
            self._cancel = False
            self._run_gen += 1
            return self._run_gen

    # ── geometry ──────────────────────────────────────────────────────────
    def _near_depth(self, depth: np.ndarray, u_px: int, v_px: int) -> Optional[float]:
        """Prefer nearer depths — white cylinders often produce far flying pixels."""
        h, w = depth.shape[:2]
        half = max(1, self.depth_window // 2)
        y0, y1 = max(0, v_px - half), min(h, v_px + half + 1)
        x0, x1 = max(0, u_px - half), min(w, u_px + half + 1)
        patch = depth[y0:y1, x0:x1].reshape(-1)
        valid = patch[
            np.isfinite(patch) & (patch > self.min_depth) & (patch < self.max_depth)
        ]
        if valid.size == 0:
            return None
        return float(np.percentile(valid, self.depth_near_pct))

    def _standing_optical_z(
        self,
        depth: np.ndarray,
        u_d: int,
        v_d: int,
        info: CameraInfo,
        T: np.ndarray,
        ray: np.ndarray,
        v_norm: float,
    ) -> Optional[float]:
        """Optical Z on the click ray that is a standing object, not floor/bag.

        `_near_depth` takes the closest pixel in a window (20th percentile).
        A pole behind a bag then unprojects onto the bag/floor.
        """
        h, w = depth.shape[:2]
        need_standing = float(v_norm) < 0.62
        zs: List[float] = []

        def _consider(uu: int, vv: int) -> None:
            z = float(depth[vv, uu])
            if not np.isfinite(z) or z <= self.min_depth or z >= self.max_depth:
                return
            xyz = self._hit_optical_z(T, ray, z)
            if not self._viz_plausible(xyz):
                return
            if need_standing and xyz[2] < 0.10:
                return
            zs.append(z)

        half = max(2, self.depth_window // 2)
        for vv in range(max(0, v_d - half), min(h, v_d + half + 1)):
            for uu in range(max(0, u_d - half), min(w, u_d + half + 1)):
                _consider(uu, vv)
        if len(zs) >= 3:
            return float(np.median(zs))
        # White PVC often has depth only on the tape: search the column.
        hu = 6
        for vv in range(0, h, 2):
            for uu in range(max(0, u_d - hu), min(w, u_d + hu + 1)):
                _consider(uu, vv)
        if zs:
            return float(np.median(zs))
        return None

    def _click_k_pixels(
        self,
        u_norm: float,
        v_norm: float,
        img_w,
        img_h,
        info: CameraInfo,
        depth: Optional[np.ndarray],
    ) -> Optional[Tuple[int, int]]:
        """Map a GUI click into camera_info / K pixel coordinates."""
        try:
            click_w = int(img_w) if img_w else 0
            click_h = int(img_h) if img_h else 0
        except (TypeError, ValueError):
            click_w = click_h = 0
        if click_w >= 32 and click_h >= 32:
            w, h = click_w, click_h
        elif depth is not None:
            h, w = depth.shape[:2]
        else:
            w = int(info.width) if info.width else 848
            h = int(info.height) if info.height else 480
        u_px = int(round(float(u_norm) * (w - 1)))
        v_px = int(round(float(v_norm) * (h - 1)))
        info_w = int(info.width) if info.width else w
        info_h = int(info.height) if info.height else h
        if info_w > 1 and info_h > 1 and (info_w != w or info_h != h):
            u_px = int(round(u_px * (info_w - 1) / max(1, w - 1)))
            v_px = int(round(v_px * (info_h - 1) / max(1, h - 1)))
        return u_px, v_px

    def _depth_sample_uv(
        self,
        u_px: int,
        v_px: int,
        info: CameraInfo,
        depth: np.ndarray,
    ) -> Tuple[int, int]:
        dh, dw = depth.shape[:2]
        iw = int(info.width) if info.width else dw
        ih = int(info.height) if info.height else dh
        if iw > 1 and ih > 1 and (iw != dw or ih != dh):
            u_d = int(round(u_px * (dw - 1) / max(1, iw - 1)))
            v_d = int(round(v_px * (dh - 1) / max(1, ih - 1)))
        else:
            u_d, v_d = int(u_px), int(v_px)
        return (
            max(0, min(dw - 1, u_d)),
            max(0, min(dh - 1, v_d)),
        )

    def _d435i_planning_T(
        self, frame: str
    ) -> Optional[Tuple[np.ndarray, str]]:
        T = None
        used = frame
        for fr in self._camera_frame_candidates(frame):
            T = self._tf_matrix(self.planning_frame, fr)
            if T is not None:
                used = fr
                break
        if T is None:
            return None
        T_use = T.copy()
        if "d435i" in str(used):
            T_use[:3, 3] = T_use[:3, 3] + self.click_origin_offset
        return T_use, used

    def _d435i_urdf_T(self) -> Optional[Tuple[np.ndarray, str]]:
        """URDF D435i optical pose. No nav camera_* and no origin patch."""
        for fr in (
            "d435i_color_optical_frame",
            "d435i_depth_optical_frame",
            "px150_no_elbow/d435i_color_optical_frame",
            "px150_no_elbow/d435i_depth_optical_frame",
        ):
            T = self._tf_matrix(self.planning_frame, fr)
            if T is not None:
                return T.copy(), fr
        return None

    def _depth_at_click_pixel(
        self, depth: np.ndarray, u: int, v: int
    ) -> Optional[float]:
        """Depth of the clicked pixel. A hole uses the nearest valid pixel.

        Nearest in the image, not the nearest range. A range percentile pulls
        in the floor in front of the block.
        """
        h, w = depth.shape[:2]

        def _ok(z: float) -> bool:
            return bool(
                np.isfinite(z) and self.min_depth < z < self.max_depth
            )

        if 0 <= v < h and 0 <= u < w and _ok(float(depth[v, u])):
            return float(depth[v, u])
        for rad in range(1, 6):
            found: List[float] = []
            for dv in range(-rad, rad + 1):
                for du in range(-rad, rad + 1):
                    if max(abs(du), abs(dv)) != rad:
                        continue
                    uu, vv = u + du, v + dv
                    if uu < 0 or vv < 0 or uu >= w or vv >= h:
                        continue
                    z = float(depth[vv, uu])
                    if _ok(z):
                        found.append(z)
            if found:
                return float(np.median(found))
        return None

    def _click_ray_point(
        self,
        u_norm: float,
        v_norm: float,
        img_w,
        img_h,
    ) -> Optional[Tuple[float, float, float]]:
        """Planning-frame point of the clicked pixel along its own ray."""
        with self._lock:
            info = self._info
            frame = self._depth_frame or self.camera_frame
            depth = None if self._depth is None else self._depth.copy()
        if info is None or depth is None:
            return None
        pix = self._click_k_pixels(u_norm, v_norm, img_w, img_h, info, depth)
        if pix is None:
            return None
        tf = self._d435i_planning_T(frame or info.header.frame_id)
        if tf is None:
            return None
        ray = self._optical_ray_unnorm(float(pix[0]), float(pix[1]), info)
        if ray is None:
            return None
        u_d, v_d = self._depth_sample_uv(int(pix[0]), int(pix[1]), info, depth)
        z_opt = self._depth_at_click_pixel(depth, u_d, v_d)
        if z_opt is None:
            return None
        T, _used = tf
        origin = np.asarray(T[:3, 3], dtype=np.float64)
        direction = T[:3, :3] @ np.asarray(ray, dtype=np.float64)
        hit = origin + float(z_opt) * direction
        if not np.all(np.isfinite(hit)):
            return None
        # The center pixel is the near face. The jaws have to reach the
        # floor under that pixel, which the surrounding depth still sees.
        floor_z = self._floor_z_around_click(
            depth, u_d, v_d, info, T, (float(hit[0]), float(hit[1]))
        )
        if floor_z is not None:
            grasp_z = float(floor_z) + float(self.floor_tcp_lift)
            if float(hit[2]) > grasp_z + 0.015:
                dz = float(direction[2])
                if abs(dz) > 1e-6:
                    t_hit = (grasp_z - float(origin[2])) / dz
                    if 0.05 < t_hit < 1.30:
                        lowered = origin + t_hit * direction
                        if np.all(np.isfinite(lowered)):
                            self.get_logger().info(
                                f"click ray to floor {float(hit[2]):.3f}"
                                f"→{float(lowered[2]):.3f} "
                                f"({float(lowered[0]):.3f},{float(lowered[1]):.3f})"
                            )
                            hit = lowered
        return (float(hit[0]), float(hit[1]), float(hit[2]))

    def _floor_z_around_click(
        self,
        depth: np.ndarray,
        u_d: int,
        v_d: int,
        info: CameraInfo,
        T: np.ndarray,
        near_xy: Tuple[float, float],
    ) -> Optional[float]:
        """Height of the surface under the click, not the near face."""
        fx, fy = float(info.k[0]), float(info.k[4])
        cx, cy = float(info.k[2]), float(info.k[5])
        if fx < 1.0 or fy < 1.0:
            return None
        h, w = depth.shape[:2]
        iw = int(info.width) if info.width else w
        ih = int(info.height) if info.height else h
        R = T[:3, :3]
        origin = np.asarray(T[:3, 3], dtype=np.float64)
        zs: List[float] = []
        rad = 48
        for vv in range(max(0, v_d - rad), min(h, v_d + rad + 1), 4):
            for uu in range(max(0, u_d - rad), min(w, u_d + rad + 1), 4):
                z = float(depth[vv, uu])
                if not (np.isfinite(z) and self.min_depth < z < self.max_depth):
                    continue
                if iw != w or ih != h:
                    ku = uu * (iw - 1) / max(1, w - 1)
                    kv = vv * (ih - 1) / max(1, h - 1)
                else:
                    ku, kv = float(uu), float(vv)
                ray = np.array(
                    [(ku - cx) / fx, (kv - cy) / fy, 1.0], dtype=np.float64
                )
                p = origin + z * (R @ ray)
                if abs(float(p[0]) - near_xy[0]) > 0.18:
                    continue
                if abs(float(p[1]) - near_xy[1]) > 0.18:
                    continue
                zs.append(float(p[2]))
        if len(zs) < 12:
            return None
        return float(np.percentile(zs, 20.0))

    def _optical_ray_unnorm(
        self, u_px: float, v_px: float, info: CameraInfo
    ) -> Optional[np.ndarray]:
        fx, fy = float(info.k[0]), float(info.k[4])
        cx, cy = float(info.k[2]), float(info.k[5])
        if fx < 1.0 or fy < 1.0:
            return None
        return np.array(
            [(float(u_px) - cx) / fx, (float(v_px) - cy) / fy, 1.0],
            dtype=np.float64,
        )

    def _hit_optical_z(
        self,
        T: np.ndarray,
        ray_unnorm: np.ndarray,
        z_opt: float,
    ) -> Tuple[float, float, float]:
        """Planning XYZ of optical point (x,y,z)=(z*ray_unnorm). ray_z is 1."""
        origin = T[:3, 3]
        direction = T[:3, :3] @ ray_unnorm
        p = origin + float(z_opt) * direction
        return (float(p[0]), float(p[1]), float(p[2]))

    def _intersect_ray_vertical_xy(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        xy: Tuple[float, float],
    ) -> Optional[Tuple[float, float, float]]:
        """Planning-frame ray ∩ vertical plane through xy (pole column)."""
        n = np.array(
            [float(xy[0]) - float(origin[0]), float(xy[1]) - float(origin[1]), 0.0],
            dtype=np.float64,
        )
        nn = float(math.hypot(n[0], n[1]))
        if nn < 1e-4:
            return None
        n /= nn
        denom = float(n[0] * direction[0] + n[1] * direction[1])
        if abs(denom) < 1e-8:
            return None
        t = float(
            n[0] * (float(xy[0]) - float(origin[0]))
            + n[1] * (float(xy[1]) - float(origin[1]))
        ) / denom
        if t < 0.05 or t > 1.4:
            return None
        p = origin + t * direction
        xyz = (float(p[0]), float(p[1]), float(p[2]))
        if not self._viz_plausible(xyz):
            return None
        return xyz

    def _pole_xy_from_d435i_band(self) -> Optional[Tuple[float, float]]:
        """Pole XY from the black band (the only reliable depth on the white stick)."""
        cached = self._cached_pole_xy
        dgray = self._d435_gray()
        if dgray is None:
            return cached
        self._d435_gray_img = dgray
        band = self._detect_black_band(dgray)
        if band is None or not band.get("center"):
            return cached
        c = band["center"]
        h, w = dgray.shape[:2]
        xyz = self._unproject_d435i_uv(float(c[0]), float(c[1]), w, h)
        if xyz is not None and float(xyz[2]) > 0.09:
            self._cached_pole_xy = (float(xyz[0]), float(xyz[1]))
            return self._cached_pole_xy
        with self._lock:
            depth = None if self._depth is None else self._depth.copy()
            info = self._info
            frame = self._depth_frame or self.camera_frame
        if depth is None or info is None:
            return cached
        pix = self._click_k_pixels(float(c[0]), float(c[1]), w, h, info, depth)
        if pix is None:
            return cached
        tf = self._d435i_planning_T(frame or info.header.frame_id)
        if tf is None:
            return cached
        T, _used = tf
        xy = self._pole_xy_from_depth_column(pix[0], pix[1], depth, info, T)
        if xy is None:
            return cached
        self._cached_pole_xy = (float(xy[0]), float(xy[1]))
        return self._cached_pole_xy

    def _wide_click_on_pole(self, u_norm: float, v_norm: float) -> bool:
        """True if this wide click is on the stick (any height), not a side object."""
        del v_norm
        self._fetch_wide_jpeg()
        gray = self._wide_gray()
        if gray is None:
            return True
        band = self._detect_black_band(gray)
        if band is None or not band.get("center"):
            return True
        cu = float(band["center"][0])
        half = float(band.get("half", 0.04))
        return abs(float(u_norm) - cu) <= max(0.12, 3.0 * half)

    def _intersect_vertical_xy(
        self,
        T: np.ndarray,
        ray_unnorm: np.ndarray,
        xy: Tuple[float, float],
    ) -> Optional[Tuple[float, float, float]]:
        """D435i ray ∩ vertical plane through xy (pole column, not table)."""
        origin = T[:3, 3]
        direction = T[:3, :3] @ ray_unnorm
        return self._intersect_ray_vertical_xy(origin, direction, xy)

    def _pole_xy_from_depth_column(
        self,
        u_px: int,
        v_px: int,
        depth: np.ndarray,
        info: CameraInfo,
        T: np.ndarray,
    ) -> Optional[Tuple[float, float]]:
        """Nearest surface XY in the click column (white poles often miss at the pixel)."""
        u_d, v_d = self._depth_sample_uv(u_px, v_px, info, depth)
        dh, dw = depth.shape[:2]
        half_u = 5
        half_v = max(12, dh // 5)
        x0 = max(0, u_d - half_u)
        x1 = min(dw, u_d + half_u + 1)
        y0 = max(0, v_d - half_v)
        y1 = min(dh, v_d + half_v + 1)
        patch = depth[y0:y1, x0:x1]
        valid = np.isfinite(patch) & (patch > self.min_depth) & (patch < self.max_depth)
        if int(np.count_nonzero(valid)) < 8:
            return None
        z = float(np.percentile(patch[valid], self.depth_near_pct))
        ray = self._optical_ray_unnorm(u_px, v_px, info)
        if ray is None:
            return None
        standing: List[Tuple[float, float]] = []
        ys, xs = np.where(valid)
        for yi, xi in zip(ys.tolist(), xs.tolist()):
            zz = float(patch[yi, xi])
            xyz = self._hit_optical_z(T, ray, zz)
            if xyz[2] >= 0.10 and self._viz_plausible(xyz):
                standing.append((xyz[0], xyz[1]))
        if standing:
            arr = np.array(standing, dtype=np.float64)
            return (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])))
        p = self._hit_optical_z(T, ray, z)
        if p[2] < 0.10:
            return None
        return (p[0], p[1])

    def _mid_v_table_xy(
        self, u_px: int, info: CameraInfo, T: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """Table-plane XY of this image column at the principal row (V-independent)."""
        cy = float(info.k[5]) if float(info.k[5]) > 1.0 else 0.5 * float(
            info.height or 480
        )
        ray = self._optical_ray_unnorm(u_px, cy, info)
        if ray is None:
            return None
        origin = T[:3, 3]
        direction = T[:3, :3] @ ray
        dz = float(direction[2])
        if abs(dz) < 1e-6:
            return None
        plane_z = self._click_plane_z()
        t = (plane_z - float(origin[2])) / dz
        if t < 0.08 or t > 1.4:
            return None
        p = origin + t * direction
        if float(p[0]) < 0.02 or math.hypot(float(p[0]), float(p[1])) > 0.55:
            return None
        return (float(p[0]), float(p[1]))

    def _viz_plausible(self, xyz: Tuple[float, float, float]) -> bool:
        return 0.03 <= xyz[2] <= 0.50 and math.hypot(xyz[0], xyz[1]) < 0.55

    def _click_pixel_xyz(
        self,
        u_norm: float,
        v_norm: float,
        img_w=None,
        img_h=None,
    ) -> Optional[Tuple[float, float, float]]:
        """Planning-frame 3D of the D435i click pixel (depth, not table plane)."""
        got = self._viz_click_xyz(u_norm, v_norm, img_w, img_h)
        return None if got is None else got[0]

    def _viz_click_xyz(
        self,
        u_norm: float,
        v_norm: float,
        img_w=None,
        img_h=None,
    ) -> Optional[Tuple[Tuple[float, float, float], str]]:
        """3D overlay point on the D435i click ray.

        Table-plane XY changes with V (floor behind / in front of a pole).
        Wide overlay must sit on the ray at the pole's range so clicking
        above/below the ring moves the cube up/down, not along the floor.
        """
        with self._lock:
            depth = None if self._depth is None else self._depth.copy()
            info = self._info
            frame = self._depth_frame or self.camera_frame
        if info is None:
            return None
        pix = self._click_k_pixels(u_norm, v_norm, img_w, img_h, info, depth)
        if pix is None:
            return None
        u_px, v_px = pix
        tf = self._d435i_planning_T(frame or info.header.frame_id)
        if tf is None:
            return None
        T, _used = tf
        ray = self._optical_ray_unnorm(u_px, v_px, info)
        if ray is None:
            return None

        def _ok_depth_hit(xyz: Tuple[float, float, float]) -> bool:
            if not self._viz_plausible(xyz):
                return False
            # Upper-image clicks on a standing pole cannot be table height;
            # that is the floor behind the pipe along the same ray.
            if float(v_norm) < 0.62 and xyz[2] < 0.09:
                return False
            return True

        if depth is not None:
            u_d, v_d = self._depth_sample_uv(u_px, v_px, info, depth)
            z = self._standing_optical_z(depth, u_d, v_d, info, T, ray, u_norm * 0.0 + v_norm)
            if z is not None:
                xyz = self._hit_optical_z(T, ray, z)
                if _ok_depth_hit(xyz):
                    return xyz, "pix-depth"
            pole_xy = self._pole_xy_from_depth_column(u_px, v_px, depth, info, T)
            if pole_xy is not None:
                hit = self._intersect_vertical_xy(T, ray, pole_xy)
                if hit is not None and _ok_depth_hit(hit):
                    return hit, "col-depth"
            band_xy = self._pole_xy_from_d435i_band()
            if band_xy is not None:
                hit = self._intersect_vertical_xy(T, ray, band_xy)
                if hit is not None and _ok_depth_hit(hit):
                    return hit, "band"

        # Stay on the click ray at standing height. Table-plane XY at this
        # column is the floor in front of the pole (cube jumps to the bottom).
        origin = T[:3, 3]
        direction = T[:3, :3] @ ray
        dz = float(direction[2])
        if abs(dz) > 1e-6:
            z_aim = 0.16 if float(v_norm) < 0.70 else float(self._click_plane_z())
            t = (z_aim - float(origin[2])) / dz
            if 0.08 < t < 1.2:
                p = origin + t * direction
                xyz = (float(p[0]), float(p[1]), float(p[2]))
                if self._viz_plausible(xyz):
                    return xyz, "ray-z"
        return None

    def _tf_matrix(self, target: str, source: str) -> Optional[np.ndarray]:
        # DDS discovery + dual TF trees (nav static vs URDF) often need >1s.
        # Overlay ticks must not block the executor for those waits — cache hits
        # and a 0s probe first so cube projection cannot starve later clicks.
        key = (str(target), str(source))
        now = time.monotonic()
        hit = self._tf_cache.get(key)
        if hit is not None and now - hit[0] < 0.20:
            return None if hit[1] is None else hit[1].copy()
        timeouts = (0.0,) if (self._overlay_busy or self._clicking) else (0.0, 0.25)
        tf = None
        for timeout_s in timeouts:
            try:
                tf = self._tf_buffer.lookup_transform(
                    target,
                    source,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=timeout_s),
                )
                break
            except Exception:
                tf = None
        if tf is None:
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        # quaternion (x,y,z,w) → rotation matrix
        x, y, z, w = q.x, q.y, q.z, q.w
        R = np.array(
            [
                [
                    1 - 2 * (y * y + z * z),
                    2 * (x * y - z * w),
                    2 * (x * z + y * w),
                ],
                [
                    2 * (x * y + z * w),
                    1 - 2 * (x * x + z * z),
                    2 * (y * z - x * w),
                ],
                [
                    2 * (x * z - y * w),
                    2 * (y * z + x * w),
                    1 - 2 * (x * x + y * y),
                ],
            ],
            dtype=np.float64,
        )
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = [t.x, t.y, t.z]
        self._tf_cache[key] = (now, T)
        return T

    def _click_plane_z(self) -> float:
        """Horizontal plane the click ray should hit (object mid-height, not floor)."""
        return float(self.table_z + self.object_height)

    def _floor_target_z(self, top_z: float) -> float:
        """Measured top plus a small clearance, allowed down to the floor minimum."""
        z = float(top_z) + float(self.floor_tcp_lift)
        return max(
            float(self.floor_z_min),
            min(float(self.floor_object_max_z), z),
        )

    def _apply_floor_xyz(
        self, xyz: Tuple[float, float, float]
    ) -> Optional[Tuple[float, float, float]]:
        """Keep the measured click. Do not move it onto another plane."""
        if not self._object_low or self._object_centroid is None:
            return None
        src = self._object_centroid
        return (float(src[0]), float(src[1]), float(src[2]))

    def _floor_ray_xyz(
        self,
        u_norm: float,
        v_norm: float,
        img_w,
        img_h,
    ) -> Optional[Tuple[float, float, float]]:
        """Click ray at the measured range, pitched onto the floor.

        Pitch is about the camera's horizontal axis, so the point stays on the
        clicked column. Scaling the base-frame azimuth shoves it sideways.
        """
        with self._lock:
            info = self._info
            frame = self._depth_frame or self.camera_frame
            depth = None if self._depth is None else self._depth.copy()
        if info is None:
            return None
        pix = self._click_k_pixels(u_norm, v_norm, img_w, img_h, info, depth)
        if pix is None:
            return None
        tf = self._d435i_planning_T(frame or info.header.frame_id)
        if tf is None:
            return None
        T, _used = tf
        ray = self._optical_ray_unnorm(float(pix[0]), float(pix[1]), info)
        if ray is None:
            return None
        origin = np.array(T[:3, 3], dtype=np.float64)
        direction = self._pan_undo_dir(T[:3, :3] @ np.asarray(ray, dtype=np.float64))
        dz = float(direction[2])
        if abs(dz) < 1e-6:
            return None
        if self._object_top_z is not None:
            plane_z = self._floor_target_z(float(self._object_top_z))
        else:
            plane_z = float(self.floor_z_min)
        t_hit = (plane_z - float(origin[2])) / dz
        hit = None
        z_opt = self._floor_optical_z(depth, pix, info)
        if z_opt is not None:
            measured = origin + float(z_opt) * direction
            if float(measured[2]) >= 0.02:
                hit = measured
            else:
                hit = self._pitch_onto_floor(
                    origin, measured, plane_z, T[:3, :3][:, 0]
                )
        if hit is None:
            if t_hit <= 0.05 or t_hit > 1.30:
                return None
            hit = origin + t_hit * direction
        if float(hit[0]) < 0.02:
            return None
        horiz = float(math.hypot(float(hit[0]), float(hit[1])))
        if horiz > min(2.5, float(self.max_reach)):
            return None
        return (float(hit[0]), float(hit[1]), plane_z)

    def _floor_optical_z(
        self,
        depth: Optional[np.ndarray],
        pix: Tuple[int, int],
        info: CameraInfo,
    ) -> Optional[float]:
        """Median range at the click. No standing-object reject."""
        if depth is None:
            return None
        u_d, v_d = self._depth_sample_uv(int(pix[0]), int(pix[1]), info, depth)
        half = 2
        h, w = depth.shape[:2]
        patch = depth[
            max(0, v_d - half) : min(h, v_d + half + 1),
            max(0, u_d - half) : min(w, u_d + half + 1),
        ]
        valid = patch[
            np.isfinite(patch) & (patch > self.min_depth) & (patch < self.max_depth)
        ]
        if valid.size < 5:
            return None
        z_opt = float(np.median(valid))
        if z_opt <= 0.05 or z_opt > 1.30:
            return None
        return z_opt

    def _pitch_onto_floor(
        self,
        origin: np.ndarray,
        measured: np.ndarray,
        plane_z: float,
        cam_x: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Rotate the measured point about image-horizontal onto the floor.

        Image-horizontal is the camera X axis. That changes height in the
        image and leaves the clicked column. A base-frame azimuth scale does
        not, and it walks the point off to the side.
        """
        axis = np.asarray(cam_x, dtype=np.float64)
        an = float(np.linalg.norm(axis))
        if an < 1e-6:
            return None
        axis = axis / an
        delta = np.asarray(measured, dtype=np.float64) - np.asarray(
            origin, dtype=np.float64
        )
        par = float(np.dot(delta, axis)) * axis
        perp = delta - par
        length = float(np.linalg.norm(perp))
        if length < 1e-4:
            return None
        z_perp = (float(plane_z) - float(origin[2])) - float(par[2])
        if abs(z_perp) > length + 1e-6:
            return None
        e1 = perp / length
        e2 = np.cross(axis, e1)
        e2n = float(np.linalg.norm(e2))
        if e2n < 1e-6:
            return None
        e2 = e2 / e2n
        a_c = float(e1[2])
        b_c = float(e2[2])
        scale = math.hypot(a_c, b_c)
        if scale < 1e-6:
            return None
        rhs = max(-1.0, min(1.0, (z_perp / length) / scale))
        phi = math.atan2(b_c, a_c)
        turn = math.acos(rhs)
        best = None
        best_abs = 1e9
        for theta in (phi + turn, phi - turn):
            theta = (theta + math.pi) % (2.0 * math.pi) - math.pi
            if abs(theta) < best_abs:
                best_abs = abs(theta)
                best = theta
        if best is None:
            return None
        perp2 = length * (math.cos(best) * e1 + math.sin(best) * e2)
        hit = np.asarray(origin, dtype=np.float64) + par + perp2
        hit[2] = float(plane_z)
        self.get_logger().info(
            f"floor pitch {math.degrees(best):+.1f}° about camera X"
        )
        return hit

    def _pan_undo_dir(self, direction: np.ndarray) -> np.ndarray:
        """Pitch the look ray up — URDF pan+cam_pitch is steeper than the real D435i."""
        pan = float(self._pan_pos) if self._pan_pos is not None else 0.0
        undo = (1.0 - self.click_pan_scale) * pan
        if abs(undo) < 1e-4:
            return np.asarray(direction, dtype=np.float64)
        c_u, s_u = math.cos(undo), math.sin(undo)
        dx, dy, dz_d = (
            float(direction[0]),
            float(direction[1]),
            float(direction[2]),
        )
        return np.array(
            [c_u * dx - s_u * dz_d, dy, s_u * dx + c_u * dz_d],
            dtype=np.float64,
        )

    def _table_xyz_from_click_uv(
        self,
    ) -> Optional[Tuple[float, float, float]]:
        """Pan-corrected table-plane hit of the last D435i click."""
        if self._last_click_uv is None:
            return None
        with self._lock:
            info = self._info
            frame = self._depth_frame or self.camera_frame
        if info is None:
            return None
        w = int(info.width) if info.width else 640
        h = int(info.height) if info.height else 480
        u_n, v_n = self._last_click_uv
        u_px = int(round(float(u_n) * max(1, w - 1)))
        v_px = int(round(float(v_n) * max(1, h - 1)))
        ok, xyz, _msg = self._resolve_via_table_plane(
            u_px, v_px, info, frame or getattr(info.header, "frame_id", "")
        )
        if ok and xyz is not None:
            return (float(xyz[0]), float(xyz[1]), float(xyz[2]))
        return None

    def _pan_depth_xyz(
        self, viz: Tuple[float, float, float]
    ) -> Optional[Tuple[float, float, float]]:
        """Click depth along the pan-corrected look ray (not the table plane).

        Table-plane XY is the floor in front of a standing pole, so IK stops
        short and the gripper closes before the object. D435i Z is the range
        to the click; pan-undo aims that range forward instead of into the base.
        """
        if self._last_click_uv is None:
            return None
        with self._lock:
            info = self._info
            frame = self._depth_frame or self.camera_frame
        if info is None:
            return None
        tf = self._d435i_planning_T(frame or info.header.frame_id)
        if tf is None:
            return None
        T, _used = tf
        origin = T[:3, 3]
        w = int(info.width) if info.width else 640
        h = int(info.height) if info.height else 480
        u_n, v_n = self._last_click_uv
        pix = self._click_k_pixels(u_n, v_n, w, h, info, None)
        if pix is None:
            return None
        ray = self._optical_ray_unnorm(float(pix[0]), float(pix[1]), info)
        if ray is None:
            return None
        try:
            cam = np.linalg.inv(T) @ np.array(
                [float(viz[0]), float(viz[1]), float(viz[2]), 1.0],
                dtype=np.float64,
            )
            z_opt = float(cam[2])
        except np.linalg.LinAlgError:
            z_opt = float(
                np.linalg.norm(
                    np.array(
                        [float(viz[0]), float(viz[1]), float(viz[2])],
                        dtype=np.float64,
                    )
                    - origin
                )
            )
        if z_opt < 0.10 or z_opt > 1.30:
            return None
        d = self._pan_undo_dir(T[:3, :3] @ ray)
        p = origin + z_opt * d
        return (float(p[0]), float(p[1]), float(p[2]))

    def _approach_xy_hat(
        self, xyz: Tuple[float, float, float]
    ) -> Tuple[float, float]:
        n = math.hypot(float(xyz[0]), float(xyz[1]))
        if n < 1e-4:
            return (1.0, 0.0)
        return (float(xyz[0]) / n, float(xyz[1]) / n)

    def _ee_for_bar_contact(
        self, contact: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """EE origin so the gripper-bar center sits on `contact`."""
        if self._object_low:
            # Floor IK solves the gripper link, so this point is the jaws.
            return (float(contact[0]), float(contact[1]), float(contact[2]))
        b = float(self.gripper_bar_offset_x)
        hx, hy = self._approach_xy_hat(contact)
        return (
            float(contact[0]) - b * hx,
            float(contact[1]) - b * hy,
            float(contact[2]),
        )

    def _standoff_xyz(
        self, xyz: Tuple[float, float, float], back_m: float
    ) -> Tuple[float, float, float]:
        hx, hy = self._approach_xy_hat(xyz)
        d = max(0.0, float(back_m))
        return (
            float(xyz[0]) - d * hx,
            float(xyz[1]) - d * hy,
            float(xyz[2]),
        )

    def _grasp_facing_quat(self, xyz: Tuple[float, float, float]) -> Quaternion:
        """Gripper +X along the table heading, no camera pitch."""
        yaw = math.atan2(float(xyz[1]), float(xyz[0]))
        return _rpy_to_quat(0.0, 0.0, yaw)

    def _push_xy_farther(
        self, xyz: Tuple[float, float, float], extra: float
    ) -> Tuple[float, float, float]:
        horiz = math.hypot(float(xyz[0]), float(xyz[1]))
        if horiz < 1e-3 or extra <= 1e-4:
            return xyz
        new_r = min(horiz + extra, float(self.max_reach) - 0.01)
        if new_r <= horiz:
            return xyz
        s = new_r / horiz
        return (float(xyz[0]) * s, float(xyz[1]) * s, float(xyz[2]))

    def _pull_xy_to_reach(
        self, xyz: Tuple[float, float, float], reach: float
    ) -> Tuple[float, float, float]:
        horiz = math.hypot(float(xyz[0]), float(xyz[1]))
        r = max(0.12, float(reach))
        if horiz < 1e-3 or horiz <= r + 1e-4:
            return xyz
        s = r / horiz
        return (float(xyz[0]) * s, float(xyz[1]) * s, float(xyz[2]))

    def _reliable_reach_for_z(self, z: float) -> float:
        """Horizontal envelope shrinks as TCP rises (PX150 4-DOF)."""
        base = float(self.reliable_grasp_reach)
        zz = float(z)
        if zz > 0.15:
            base -= 0.45 * (zz - 0.15)
        return max(0.22, min(float(self.max_reach) - 0.02, base))

    def _reachable_grasp_xyz(
        self, xyz: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """EE for gripper-bar contact at the click. Trust measured XYZ.

        Soft-cap horizontal reach to the 4-DOF envelope — calibrated depth can
        place the bar past where MoveIt can solve (was succeeding at ~0.30 m,
        failing at ~0.37 m with -31).
        """
        contact = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
        how = "viz"
        # Pole clicks re-aim along the look ray. A floor click is already the
        # ray at floor height; that re-aim slides the block sideways.
        depth_p = None if self._object_low else self._pan_depth_xyz(xyz)
        if depth_p is not None:
            dh = math.hypot(depth_p[0], depth_p[1])
            ch = math.hypot(contact[0], contact[1])
            if (
                depth_p[0] >= 0.05
                and 0.08 <= dh <= 0.55
                and (ch < 1e-3 or abs(dh - ch) < 0.12)
            ):
                contact = (float(depth_p[0]), float(depth_p[1]), float(xyz[2]))
                how = "depth-xy"
        if float(self.grasp_farther_m) > 1e-4:
            contact = self._push_xy_farther(contact, float(self.grasp_farther_m))
        horiz = math.hypot(contact[0], contact[1])
        min_r = float(self.min_grasp_reach)
        if min_r > 1e-4 and horiz < min_r and horiz > 1e-3:
            s = min_r / horiz
            contact = (contact[0] * s, contact[1] * s, contact[2])
            horiz = min_r
        soft = self._reliable_reach_for_z(contact[2])
        if self.reliable_grasp_reach > 1e-4 and horiz > soft + 1e-4:
            self.get_logger().info(
                f"pull-in reach {horiz:.3f}→{soft:.3f} m (z={contact[2]:.3f})"
            )
            contact = self._pull_xy_to_reach(contact, soft)
            horiz = soft
            how = f"{how}+pull"
        if horiz > float(self.max_reach) and horiz > 1e-3:
            s = (float(self.max_reach) - 0.01) / horiz
            contact = (contact[0] * s, contact[1] * s, contact[2])
        if not self._in_workspace(contact):
            contact = (
                min(self.ws_max[0], max(self.ws_min[0], contact[0])),
                min(self.ws_max[1], max(self.ws_min[1], contact[1])),
                min(self.ws_max[2], max(self.ws_min[2] + 0.02, contact[2])),
            )
        ee = self._ee_for_bar_contact(contact)
        self.get_logger().info(
            f"bar contact {how} ({contact[0]:.3f},{contact[1]:.3f},{contact[2]:.3f}) "
            f"ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) "
            f"viz=({xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f})"
        )
        return ee

    def _solve_pregrasp_ik(
        self,
        pre_xyz: Tuple[float, float, float],
        ik_xyz: Tuple[float, float, float],
        gen: int,
    ) -> Tuple[Optional[List[float]], Tuple[float, float, float], Tuple[float, float, float]]:
        """Solve pre-grasp; on failure walk the target inward along heading."""
        def _try(pre: Tuple[float, float, float], contact: Tuple[float, float, float]):
            # Negative offsets drop a floor click under the block.
            zoffs = (0.00, 0.02) if self._object_low else (0.00, -0.02, 0.02, -0.04)
            for zoff in zoffs:
                if self._is_cancelled(gen):
                    return None
                q = self._solve_ik(self._pose_at(pre, zoff), gen)
                if q is not None:
                    return q
            return self._solve_ik(self._pose_at(contact, 0.00), gen)

        q = _try(pre_xyz, ik_xyz)
        if q is not None or self._is_cancelled(gen):
            return q, pre_xyz, ik_xyz
        # A floor target that is already too close cannot be fixed by walking in.
        if self._object_low:
            return None, pre_xyz, ik_xyz

        hx, hy = self._approach_xy_hat(ik_xyz)
        # Walk contact toward the base until MoveIt accepts a pose.
        for back in (0.04, 0.08, 0.12, 0.16, 0.20):
            if self._is_cancelled(gen):
                return None, pre_xyz, ik_xyz
            c = (
                float(ik_xyz[0]) - back * hx,
                float(ik_xyz[1]) - back * hy,
                float(ik_xyz[2]),
            )
            if math.hypot(c[0], c[1]) < 0.16:
                break
            pre = self._standoff_xyz(c, min(self.grasp_standoff_m, 0.04))
            # No extra hover Z on long-reach recovery — height kills envelope.
            self.get_logger().info(
                f"pre-grasp pull-back {back:.2f}m → "
                f"({c[0]:.3f},{c[1]:.3f},{c[2]:.3f})"
            )
            q = _try(pre, c)
            if q is not None:
                return q, pre, c
        return None, pre_xyz, ik_xyz

    def _camera_frame_candidates(self, reported: str) -> List[str]:
        # Grasp must follow the pan joint. Static camera_* (nav calibration at
        # pan≈0) and link-frame nearest-hit picking mapped clicks too close /
        # too far. Prefer URDF optical frames; static optical is fallback only.
        frames = []
        for f in (
            "d435i_color_optical_frame",
            "d435i_depth_optical_frame",
            reported,
            self.camera_frame,
            "camera_color_optical_frame",
            "camera_depth_optical_frame",
        ):
            if f and f not in frames:
                frames.append(f)
        return frames

    def _table_hit_for_frame(
        self, ray_optical: np.ndarray, frame: str
    ) -> Optional[Tuple[float, Tuple[float, float, float], str]]:
        """Return (horiz_reach, xyz, frame) or None if ray misses / behind."""
        T = self._tf_matrix(self.planning_frame, frame)
        if T is None:
            return None
        ray = ray_optical
        link_like = (
            frame == "camera_link"
            or frame.endswith("/camera_link")
            or frame == "d435i_link"
            or frame.endswith("/d435i_link")
        )
        if link_like:
            x_o, y_o, z_o = ray
            ray = np.array([z_o, -x_o, -y_o], dtype=np.float64)

        origin = np.array(T[:3, 3], dtype=np.float64)
        # Nav camera_link is ~20cm forward of URDF d435i. Keep pan-aware
        # rotation; only shift d435i origins (do not double-apply on camera_*).
        if "d435i" in frame:
            origin = origin + self.click_origin_offset
        direction = T[:3, :3] @ ray
        direction = self._pan_undo_dir(direction)
        dz = float(direction[2])
        if abs(dz) < 1e-6:
            return None
        # Intersect the object-height plane (not the floor).
        plane_z = self._click_plane_z()
        t_hit = (plane_z - float(origin[2])) / dz
        if t_hit <= 0.05:
            return None
        hit = origin + t_hit * direction
        # Must be in front of the arm base (planning X forward).
        if float(hit[0]) < 0.02:
            return None
        z_high = (
            float(self.grasp_tcp_z)
            if self.grasp_tcp_z > 0.0
            else float(plane_z + self.grasp_z_bias)
        )
        z_low = float(plane_z + self.grasp_z_bias)
        pan = float(self._pan_pos) if self._pan_pos is not None else 0.0
        pan_down = max(0.0, pan)
        blend = min(1.0, pan_down / max(0.15, self.click_pan_z_ref))
        z_tcp = z_high * (1.0 - blend) + z_low * blend
        xyz = (
            float(hit[0]),
            float(hit[1]),
            z_tcp,
        )
        horiz = float(math.hypot(xyz[0], xyz[1]))
        # Discard absurd intersections (level camera → tens of meters).
        if horiz > 2.5:
            return None
        return horiz, xyz, frame

    def _pick_table_frame(self, info: CameraInfo, cam_frame: str) -> Optional[str]:
        """Best camera frame for table-plane rays (same ranking as click resolve)."""
        w = int(info.width) if info.width else 848
        h = int(info.height) if info.height else 480
        u_px = w // 2
        v_px = h // 2
        fx, fy = float(info.k[0]), float(info.k[4])
        cx, cy = float(info.k[2]), float(info.k[5])
        if fx == 0.0 or fy == 0.0:
            return None
        v_adj = float(v_px) + self.click_v_bias * float(h)
        ray_cam = np.array(
            [(u_px - cx) / fx, (v_adj - cy) / fy, 1.0], dtype=np.float64
        )
        ray_cam /= np.linalg.norm(ray_cam)
        hits = []
        for fr in self._camera_frame_candidates(cam_frame):
            hhit = self._table_hit_for_frame(ray_cam, fr)
            if hhit is not None:
                hits.append(hhit)
        if not hits:
            return None

        def _rank(item):
            _horiz, _xyz, fr = item
            d435i = 0 if "d435i" in fr else 1
            optical = 0 if "optical" in fr else 1
            color = 0 if "color" in fr else 1
            return (d435i, optical, color)

        hits.sort(key=_rank)
        return hits[0][2]

    def _table_plane_xyz_at_px(
        self, u_px: int, v_px: int, info: CameraInfo, cam_frame: str, *, frame: str
    ) -> Optional[Tuple[float, float, float]]:
        fx, fy = float(info.k[0]), float(info.k[4])
        cx, cy = float(info.k[2]), float(info.k[5])
        if fx == 0.0 or fy == 0.0:
            return None
        v_adj = float(v_px) + self.click_v_bias * float(info.height or 480)
        ray_cam = np.array(
            [(u_px - cx) / fx, (v_adj - cy) / fy, 1.0], dtype=np.float64
        )
        ray_cam /= np.linalg.norm(ray_cam)
        hit = self._table_hit_for_frame(ray_cam, frame)
        if hit is None:
            return None
        return hit[1]

    def _xyz_grasp_reachable(self, xyz: Tuple[float, float, float]) -> bool:
        horiz = math.hypot(float(xyz[0]), float(xyz[1]))
        if horiz > float(self.max_reach):
            return False
        if self.min_grasp_reach > 1e-4 and horiz < float(self.min_grasp_reach):
            return False
        return self._in_workspace(xyz)

    def _xyz_is_grasp_object(self, xyz: Tuple[float, float, float]) -> bool:
        """True for a floor block or a point in the tabletop object volume."""
        z = float(xyz[2])
        if 0.02 <= z <= float(self.floor_object_max_z):
            return True
        lo = float(self.table_z) + float(self.d435i_reach_min_above_table)
        hi = float(self.table_z) + float(self.d435i_reach_max_above_table)
        return lo <= z <= hi

    def _depth_frame_for_reach(self, reported: str) -> Optional[str]:
        """Prefer D435i optical frames that follow pan (same family as clicks)."""
        for fr in self._camera_frame_candidates(reported):
            if self._tf_matrix(self.planning_frame, fr) is not None:
                return fr
        return None

    def _compute_d435i_reach_mask(self) -> Optional[dict]:
        """Blue = in-reach object pixels (prefer depth; JPEG contrast fallback).

        Depth path: floor-height and tabletop points inside workspace.
        Fallback: MJPEG contrast blobs whose table-plane hit is graspable —
        used when aligned-depth is not bridged to the AGX.
        """
        with self._lock:
            info = self._info
            depth = None if self._depth is None else self._depth.copy()
            cam_frame = self._depth_frame or self.camera_frame
            depth_age = (
                time.monotonic() - self._reach_depth_stamp
                if self._depth is not None
                else 1e9
            )
        if info is None:
            return None

        use_depth = False
        if self.subscribe_depth and depth is not None:
            use_depth = True
        elif self._reach_depth_fail_streak < 3:
            need_snap = depth is None or (
                not self.subscribe_depth
                and depth_age > max(2.5, float(self.d435i_reach_period_s) * 2.0)
            )
            if need_snap:
                was_busy = self._overlay_busy
                self._overlay_busy = False
                try:
                    depth = self._snapshot_depth(timeout_s=1.2)
                finally:
                    self._overlay_busy = was_busy
                if depth is not None:
                    self._reach_depth_stamp = time.monotonic()
                    self._reach_depth_fail_streak = 0
                else:
                    self._reach_depth_fail_streak += 1
                with self._lock:
                    cam_frame = self._depth_frame or cam_frame
                    if depth is None and self._depth is not None:
                        depth = self._depth.copy()
            if depth is not None:
                use_depth = True

        if use_depth and depth is not None:
            mask = self._reach_mask_from_depth(depth, info, cam_frame)
            if mask is not None:
                return mask

        return self._reach_mask_from_jpeg_objects(info, cam_frame)

    def _reach_mask_from_depth(
        self,
        depth: np.ndarray,
        info: CameraInfo,
        cam_frame: str,
    ) -> Optional[dict]:
        frame = self._depth_frame_for_reach(cam_frame or info.header.frame_id)
        if frame is None:
            return None
        was_busy = self._overlay_busy
        self._overlay_busy = False
        try:
            T = self._tf_matrix(self.planning_frame, frame)
        finally:
            self._overlay_busy = was_busy
        if T is None:
            return None
        fx, fy = float(info.k[0]), float(info.k[4])
        cx, cy = float(info.k[2]), float(info.k[5])
        if fx == 0.0 or fy == 0.0:
            return None
        nu = int(self.d435i_reach_grid_u)
        nv = int(self.d435i_reach_grid_v)
        dh, dw = depth.shape[:2]
        info_w = int(info.width) if info.width else dw
        info_h = int(info.height) if info.height else dh
        R = T[:3, :3]
        tvec = T[:3, 3]
        cells: List[int] = []
        n_obj = 0
        for j in range(nv):
            v_norm = (j + 0.5) / float(nv)
            v_d = int(round(v_norm * (dh - 1)))
            for i in range(nu):
                u_norm = (i + 0.5) / float(nu)
                u_d = int(round(u_norm * (dw - 1)))
                z = self._near_depth(depth, u_d, v_d)
                if z is None:
                    cells.append(0)
                    continue
                if info_w > 1 and info_h > 1 and (info_w != dw or info_h != dh):
                    u_k = u_d * (info_w - 1) / max(1, dw - 1)
                    v_k = v_d * (info_h - 1) / max(1, dh - 1)
                else:
                    u_k, v_k = float(u_d), float(v_d)
                p_cam = np.array(
                    [(u_k - cx) * z / fx, (v_k - cy) * z / fy, z],
                    dtype=np.float64,
                )
                p_arm = R @ p_cam + tvec
                xyz = (float(p_arm[0]), float(p_arm[1]), float(p_arm[2]))
                ok = self._xyz_grasp_reachable(xyz) and self._xyz_is_grasp_object(xyz)
                if ok:
                    n_obj += 1
                cells.append(1 if ok else 0)
        self._log_reach_overlay_stats(n_obj, nu * nv, "depth")
        return {"nu": nu, "nv": nv, "cells": cells}

    def _reach_mask_from_jpeg_objects(
        self, info: CameraInfo, cam_frame: str
    ) -> Optional[dict]:
        """Contrast blobs in the live JPEG ∩ table-plane grasp envelope."""
        if not self.use_table_plane or cv2 is None:
            return None
        bgr = self._d435_bgr()
        if bgr is None:
            return None
        gray = self._d435_gray_img
        if gray is None:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        obj = self._contrast_mask(gray, delta=max(10.0, float(self.wide_blob_delta)))
        if obj is None:
            return None
        # Drop thin noise; keep solid object patches (cup, cans, …).
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        obj = cv2.morphologyEx(obj, cv2.MORPH_OPEN, kernel, iterations=1)
        obj = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, kernel, iterations=2)
        table_frame = self._pick_table_frame(info, cam_frame or info.header.frame_id)
        if table_frame is None:
            return None
        nu = int(self.d435i_reach_grid_u)
        nv = int(self.d435i_reach_grid_v)
        h, w = gray.shape[:2]
        cells: List[int] = []
        n_obj = 0
        for j in range(nv):
            v_norm = (j + 0.5) / float(nv)
            v_px = int(round(v_norm * (h - 1)))
            for i in range(nu):
                u_norm = (i + 0.5) / float(nu)
                u_px = int(round(u_norm * (w - 1)))
                # Require a small neighborhood of contrast (object, not speck).
                y0, y1 = max(0, v_px - 2), min(h, v_px + 3)
                x0, x1 = max(0, u_px - 2), min(w, u_px + 3)
                patch = obj[y0:y1, x0:x1]
                if patch.size == 0 or int(np.count_nonzero(patch)) < max(3, patch.size // 4):
                    cells.append(0)
                    continue
                # Scale JPEG pixels into camera_info / K if sizes differ.
                info_w = int(info.width) if info.width else w
                info_h = int(info.height) if info.height else h
                if info_w > 1 and info_h > 1 and (info_w != w or info_h != h):
                    u_k = int(round(u_px * (info_w - 1) / max(1, w - 1)))
                    v_k = int(round(v_px * (info_h - 1) / max(1, h - 1)))
                else:
                    u_k, v_k = u_px, v_px
                xyz = self._table_plane_xyz_at_px(
                    u_k, v_k, info, cam_frame or info.header.frame_id, frame=table_frame
                )
                ok = xyz is not None and self._xyz_grasp_reachable(xyz)
                if ok:
                    n_obj += 1
                cells.append(1 if ok else 0)
        self._log_reach_overlay_stats(n_obj, nu * nv, "jpeg")
        return {"nu": nu, "nv": nv, "cells": cells}

    def _log_reach_overlay_stats(self, n_obj: int, total: int, how: str) -> None:
        now = time.monotonic()
        if now - self._reach_warn_stamp < 5.0:
            return
        self._reach_warn_stamp = now
        self.get_logger().info(
            f"reach overlay[{how}]: {n_obj}/{total} object cells "
            f"(floor 0.02–{self.floor_object_max_z:.2f}m or "
            f"above table≥{self.d435i_reach_min_above_table:.2f}m, "
            f"reach≤{self.max_reach:.2f}m)"
        )

    def _publish_reach_overlay(self):
        if not self.d435i_reach_overlay or self._clicking:
            return
        now = time.monotonic()
        pan = float(self._pan_pos) if self._pan_pos is not None else 0.0
        if self._reach_mask_pan is not None and abs(pan - self._reach_mask_pan) > 0.03:
            self._reach_overlay_stamp = 0.0
            self._reach_depth_fail_streak = 0
        period = max(0.75, float(self.d435i_reach_period_s))
        if now - self._reach_overlay_stamp < period:
            return
        if self._depth_snap_lock.locked() and self._depth is None:
            return
        mask = self._compute_d435i_reach_mask()
        if mask is None:
            self._reach_overlay_stamp = now
            return
        if mask == self._cached_reach_mask:
            self._reach_overlay_stamp = now
            return
        self._reach_overlay_stamp = now
        self._reach_mask_pan = pan
        self._cached_reach_mask = mask
        payload = {
            "status": self._last_status or "idle",
            "message": self._last_status_msg or "",
            "d435i_reach": mask,
        }
        if self._click_id is not None:
            payload["click_id"] = int(self._click_id)
        self._status_pub.publish(String(data=json.dumps(payload)))

    def _resolve_via_table_plane(
        self, u_px: int, v_px: int, info: CameraInfo, cam_frame: str
    ) -> Tuple[bool, Optional[Tuple[float, float, float]], str]:
        """Pixel ray ∩ horizontal table plane — immune to bad depth magnitude."""
        table_frame = self._pick_table_frame(info, cam_frame)
        if table_frame is None:
            return (
                False,
                None,
                f"No TF/table hit {self.planning_frame} <- camera "
                "(check static camera TF + robot_state_publisher)",
            )
        xyz = self._table_plane_xyz_at_px(
            u_px, v_px, info, cam_frame, frame=table_frame
        )
        if xyz is None:
            return (
                False,
                None,
                f"No table hit via {table_frame}",
            )
        horiz = float(math.hypot(xyz[0], xyz[1]))
        self.get_logger().info(
            f"click uv=({u_px},{v_px}) origin_off={self.click_origin_offset.tolist()} "
            f"v_bias={self.click_v_bias:.3f} "
            f"-> ({xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f}) via {table_frame}"
        )
        return True, xyz, f"table_plane via {table_frame} ({horiz:.2f}m)"

    def _resolve_via_depth(
        self,
        u_px: int,
        v_px: int,
        depth: np.ndarray,
        info: CameraInfo,
        frame: str,
    ) -> Tuple[bool, Optional[Tuple[float, float, float]], str]:
        z = self._near_depth(depth, u_px, v_px)
        if z is None:
            return False, None, "No valid depth at click"
        fx, fy = float(info.k[0]), float(info.k[4])
        cx, cy = float(info.k[2]), float(info.k[5])
        if fx == 0.0 or fy == 0.0:
            return False, None, "Invalid camera intrinsics"
        x = (u_px - cx) * z / fx
        y = (v_px - cy) * z / fy
        pose_cam = PoseStamped()
        pose_cam.header.frame_id = frame or self.camera_frame
        pose_cam.header.stamp = self.get_clock().now().to_msg()
        pose_cam.pose.position.x = float(x)
        pose_cam.pose.position.y = float(y)
        pose_cam.pose.position.z = float(z)
        pose_cam.pose.orientation.w = 1.0
        try:
            pose_arm = self._tf_buffer.transform(
                pose_cam,
                self.planning_frame,
                timeout=rclpy.duration.Duration(seconds=1.0),
            )
        except Exception as e:
            return False, None, f"TF depth→arm: {e}"
        xyz = (
            float(pose_arm.pose.position.x),
            float(pose_arm.pose.position.y),
            float(pose_arm.pose.position.z),
        )
        return True, xyz, f"depth={z:.3f}m"

    def _resolve_click(
        self,
        u_norm: float,
        v_norm: float,
        img_w=None,
        img_h=None,
    ) -> Tuple[bool, Optional[Tuple[float, float, float]], str]:
        with self._lock:
            depth = None if self._depth is None else self._depth.copy()
            info = self._info
            frame = self._depth_frame or self.camera_frame
        if info is None:
            return False, None, "Waiting for camera_info"
        # Pixel mapping: prefer the JPEG size the GUI clicked in, else camera_info.
        try:
            click_w = int(img_w) if img_w else 0
            click_h = int(img_h) if img_h else 0
        except (TypeError, ValueError):
            click_w = click_h = 0
        if click_w >= 32 and click_h >= 32:
            w, h = click_w, click_h
        elif depth is not None:
            h, w = depth.shape[:2]
        else:
            w = int(info.width) if info.width else 848
            h = int(info.height) if info.height else 480
        u_px = int(round(u_norm * (w - 1)))
        v_px = int(round(v_norm * (h - 1)))
        # If click size != camera_info size, scale pixels into K's coordinate system.
        info_w = int(info.width) if info.width else w
        info_h = int(info.height) if info.height else h
        if info_w > 1 and info_h > 1 and (info_w != w or info_h != h):
            u_px = int(round(u_px * (info_w - 1) / max(1, w - 1)))
            v_px = int(round(v_px * (info_h - 1) / max(1, h - 1)))
            w, h = info_w, info_h

        plane_xyz = None
        plane_msg = ""
        if self.use_table_plane:
            ok, plane_xyz, plane_msg = self._resolve_via_table_plane(
                u_px, v_px, info, frame or info.header.frame_id
            )
            if not ok:
                # still try depth fallback
                self.get_logger().warn(f"table plane: {plane_msg}")
            else:
                self.get_logger().info(
                    f"table plane hit ({plane_xyz[0]:.3f},{plane_xyz[1]:.3f},"
                    f"{plane_xyz[2]:.3f}) {plane_msg}"
                )

        depth_xyz = None
        if depth is not None:
            ok_d, depth_xyz, dmsg = self._resolve_via_depth(
                u_px, v_px, depth, info, frame or info.header.frame_id
            )
            if ok_d and depth_xyz is not None:
                self.get_logger().info(
                    f"depth hit ({depth_xyz[0]:.3f},{depth_xyz[1]:.3f},"
                    f"{depth_xyz[2]:.3f}) {dmsg}"
                )

        # Prefer table-plane XY (stable on white objects). Use nearer of the two
        # when depth is clearly closer and still in front of the arm.
        xyz = None
        method = ""
        if plane_xyz is not None:
            xyz = plane_xyz
            method = "table"
            if depth_xyz is not None:
                hp = math.hypot(plane_xyz[0], plane_xyz[1])
                hd = math.hypot(depth_xyz[0], depth_xyz[1])
                # If depth is nearer and plausible, trust depth XY/Z (good texture).
                if hd + 0.03 < hp and hd >= 0.08:
                    xyz = depth_xyz
                    method = "depth(nearer)"
        elif depth_xyz is not None:
            xyz = depth_xyz
            method = "depth"
        else:
            return False, None, plane_msg or "Could not resolve click (need camera TF)"

        horiz = math.hypot(xyz[0], xyz[1])
        bias = float(self.grasp_farther_m)
        if bias > 1e-4 and horiz > 0.04:
            new_r = min(horiz + bias, float(self.max_reach) - 0.01)
            if new_r > horiz:
                s = new_r / horiz
                xyz = (float(xyz[0]) * s, float(xyz[1]) * s, float(xyz[2]))
                method = f"{method}+farther{bias * 100:.0f}cm"
                horiz = new_r
        floor_xyz = self._apply_floor_xyz(xyz)
        if floor_xyz is not None:
            xyz = floor_xyz
            method = f"{method}+floorZ{xyz[2]:.2f}"
            horiz = math.hypot(xyz[0], xyz[1])
        if horiz > self.max_reach:
            return (
                False,
                xyz,
                f"Too far for short arm ({horiz:.2f}m > {self.max_reach:.2f}m) "
                f"via {method} [{xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f}] m — "
                f"drive base closer (~{self.max_reach:.2f}m reach) or click "
                f"nearer / lower in the image",
            )
        if not self._in_workspace(xyz):
            return (
                False,
                xyz,
                f"Outside workspace via {method}: "
                f"({xyz[0]:.2f},{xyz[1]:.2f},{xyz[2]:.2f})",
            )
        return True, xyz, method

    def _in_workspace(self, xyz: Tuple[float, float, float]) -> bool:
        for i in range(3):
            if xyz[i] < self.ws_min[i] or xyz[i] > self.ws_max[i]:
                return False
        return True

    def _ee_quat(self) -> Quaternion:
        """Prefer live EE orientation; fall back to configured RPY."""
        if self.use_current_ee_orient:
            try:
                tf = self._tf_buffer.lookup_transform(
                    self.planning_frame,
                    self.ee_link,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.5),
                )
                q = Quaternion()
                q.x = tf.transform.rotation.x
                q.y = tf.transform.rotation.y
                q.z = tf.transform.rotation.z
                q.w = tf.transform.rotation.w
                return q
            except Exception:
                pass
        return self.grasp_quat

    def _pose_at(self, xyz: Tuple[float, float, float], z_off: float) -> PoseStamped:
        ps = PoseStamped()
        ps.header.frame_id = self.planning_frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = xyz[0]
        ps.pose.position.y = xyz[1]
        ps.pose.position.z = xyz[2] + z_off
        ps.pose.orientation = self._grasp_facing_quat(
            (float(xyz[0]), float(xyz[1]), float(xyz[2]))
        )
        return ps

    # ── actuation helpers ─────────────────────────────────────────────────
    def _lock_pan(self):
        """Hold current tilt/pan (do not snap to zero — URDF TF follows the joint)."""
        cmd = (
            float(self._pan_pos)
            if self._pan_pos is not None
            else float(self.pan_lock)
        )
        msg = JointSingleCommand()
        msg.name = "pan"
        msg.cmd = cmd
        self._pan_pub.publish(msg)

    def _bus_live(self) -> bool:
        return (time.monotonic() - self._js_ok_stamp) < 6.0

    def _call_srv(self, cli, request, timeout: float = 3.0) -> bool:
        if not cli.service_is_ready():
            if not cli.wait_for_service(timeout_sec=timeout):
                return False
        fut = cli.call_async(request)
        t0 = time.time()
        while not fut.done() and time.time() - t0 < timeout:
            time.sleep(0.05)
        return fut.done() and fut.result() is not None

    @staticmethod
    def _reg_first_int(values) -> Optional[int]:
        if values is None:
            return None
        if isinstance(values, (list, tuple)):
            if not values:
                return None
            return int(values[0])
        try:
            if hasattr(values, "__len__") and not isinstance(values, (str, bytes)):
                if len(values) == 0:
                    return None
                return int(values[0])
            return int(values)
        except (TypeError, ValueError, IndexError):
            return None

    @staticmethod
    def _load_mag(raw: Optional[int]) -> Optional[int]:
        """XL430 Present_Load → unsigned magnitude 0–1023."""
        if raw is None:
            return None
        v = int(raw)
        if v > 1023:
            v = v - 1024
        return abs(v)

    def _read_gripper_load(self, timeout: float = 0.28) -> Optional[int]:
        """Present_Load magnitude, or None if the Nano register read is busy."""
        if not self._reg_get_cli.service_is_ready():
            return None
        req = RegisterValues.Request()
        req.cmd_type = "single"
        req.name = "gripper"
        req.reg = "Present_Load"
        fut = self._reg_get_cli.call_async(req)
        t0 = time.time()
        while not fut.done() and time.time() - t0 < timeout:
            time.sleep(0.02)
        if not fut.done():
            return None
        try:
            return self._load_mag(self._reg_first_int(fut.result().values))
        except Exception:
            return None

    def _load_contact_thr(self) -> int:
        return max(int(self.grip_load_contact), int(self._grip_load_idle) + 70)

    def _load_grab_thr(self) -> int:
        return max(int(self.grip_load_grab), int(self._grip_load_idle) + 110)

    def _grab_feels_object(self, load: Optional[int] = None) -> bool:
        """True if the pinch stopped short of an empty hard-close with load."""
        gj = self._gripper_joint
        if load is None:
            load = self._read_gripper_load()
        if gj is None:
            return load is not None and load >= self._load_grab_thr()
        empty = float(self.grip_joint_closed) + float(self.grip_empty_closed)
        if float(gj) <= empty:
            return False
        return load is not None and load >= self._load_contact_thr()

    def _ensure_gripper_mode(self, force: bool = False) -> bool:
        """Position mode + restored Current_Limit + torque.

        current_based_position with Current_Limit=280 left the jaws open
        (joint stuck ~1.53) while the oval stayed green. Overheat is handled
        by holding the stall pose after a real close, not by starving current.
        """
        if self._gripper_mode_ok and not force:
            return True
        if self._torque_cli.service_is_ready():
            tq = TorqueEnable.Request()
            tq.cmd_type = "single"
            tq.name = "gripper"
            tq.enable = False
            self._torque_cli.call_async(tq)
            time.sleep(0.15)
        if self._reg_set_cli.service_is_ready():
            try:
                rr = RegisterValues.Request()
                rr.cmd_type = "single"
                rr.name = "gripper"
                rr.reg = "Current_Limit"
                rr.value = int(max(400, min(1193, self.grip_current_limit)))
                self._reg_set_cli.call_async(rr)
                time.sleep(0.15)
            except Exception as e:
                self.get_logger().warn(f"Current_Limit write skipped: {e}")
        if self._mode_cli.service_is_ready():
            req = OperatingModes.Request()
            req.cmd_type = "single"
            req.name = "gripper"
            req.mode = "position"
            req.profile_type = "velocity"
            req.profile_velocity = int(max(1, self.grip_profile_velocity))
            req.profile_acceleration = int(max(1, self.grip_profile_acceleration))
            self._mode_cli.call_async(req)
            time.sleep(0.25)
        if self._torque_cli.service_is_ready():
            tq = TorqueEnable.Request()
            tq.cmd_type = "single"
            tq.name = "gripper"
            tq.enable = True
            self._torque_cli.call_async(tq)
            time.sleep(0.1)
        self._gripper_mode_ok = True
        return True

    def _apply_gripper_profile(self) -> None:
        """Re-assert velocity profile without a full torque/mode cycle.

        GUI / xs_sdk can leave a different Profile_Velocity than grasp expects;
        `_gripper_mode_ok` then skipped the write and closes felt inconsistent.
        """
        if not self._mode_cli.service_is_ready():
            return
        req = OperatingModes.Request()
        req.cmd_type = "single"
        req.name = "gripper"
        req.mode = "position"
        req.profile_type = "velocity"
        req.profile_velocity = int(max(1, self.grip_profile_velocity))
        req.profile_acceleration = int(max(1, self.grip_profile_acceleration))
        try:
            self._mode_cli.call_async(req)
        except Exception as e:
            self.get_logger().warn(f"gripper profile refresh: {e}")

    def _hold_gripper_present(self, extra: float = None):
        """Stop fighting: hold the stall pose, not the fully-closed goal.

        Only do this after the jaws have actually moved toward closed.
        """
        gj = self._gripper_joint
        if gj is None:
            return
        if extra is None:
            extra = -abs(float(self.grip_hold_extra))
        hold = float(gj) + float(extra)
        hold = max(float(self.grip_joint_closed), min(float(self.grip_joint_open), hold))
        self._last_grip_joint = hold
        self._publish_gripper_joint(hold, n=2)

    def _gripper_target_joint(self, cmd: float) -> float:
        # Historic normalized cmd: positive = open, negative = close.
        if float(cmd) >= 0.0:
            return float(self.grip_joint_open)
        return float(self.grip_joint_closed)

    def _publish_gripper_joint(self, joint: float, n: int = 3):
        msg = JointSingleCommand()
        msg.name = "gripper"
        msg.cmd = float(joint)
        for _ in range(max(1, int(n))):
            self._gripper_pub.publish(msg)
            time.sleep(0.03)

    def _set_gripper(
        self,
        cmd: float,
        settle_s: float = 0.0,
        gen: Optional[int] = None,
        hold: bool = False,
    ):
        """Drive gripper in position mode (same command space as the GUI slider)."""
        self._ensure_gripper_mode()
        joint = self._gripper_target_joint(cmd)
        self._last_grip_joint = joint
        for _ in range(3):
            if gen is not None and self._is_cancelled(gen):
                return False
            self._publish_gripper_joint(joint, n=1)
            time.sleep(0.03)
        if hold:
            self._last_grip_joint = joint
        return True

    def _nudge_pinch_posture(self, gen: int) -> bool:
        """Shoulder down + wrist_angle up one step. No XY walk (that backed up)."""
        q = [float(self._joint_pos.get(j, 0.0)) for j in self.arm_joints]
        ish = self._joint_index("shoulder")
        ia = self._joint_index("wrist_angle")
        ir = self._joint_index("wrist_rotate")
        if ish >= 0:
            lo, hi = self.shoulder_lim
            cur = float(self._joint_pos.get("shoulder", q[ish]))
            # Increase shoulder = down/forward. Never command up/back.
            nxt = cur + abs(float(self.grasp_retry_d_shoulder))
            q[ish] = max(float(lo), min(float(hi), nxt))
        if ia >= 0:
            d_w = float(self.grasp_retry_d_wrist)
            if self._object_low:
                d_w = float(self.grasp_retry_d_wrist_floor)
            q[ia] = max(
                float(self.wrist_angle_min),
                min(
                    float(self.wrist_angle_max),
                    q[ia] + d_w,
                ),
            )
        if ir >= 0:
            if self._wrist_face is not None:
                q[ir] = self._clip_wrist_rotate(self._wrist_face)
            elif self.lock_wrist_rotate:
                q[ir] = 0.0
        sh = q[ish] if ish >= 0 else 0.0
        wa = q[ia] if ia >= 0 else 0.0
        how = "wrist down" if self._object_low else "wrist up"
        self.get_logger().info(
            f"pinch posture shoulder={math.degrees(sh):.1f}° "
            f"wrist_angle={math.degrees(wa):.1f}° ({how})"
        )
        old = self.seg_dur
        self.seg_dur = max(0.45, float(self.wide_servo_nudge_s))
        try:
            ok = self._send_trajectory([q], gen)
        finally:
            self.seg_dur = old
        if ok:
            for j, v in zip(self.arm_joints, q):
                self._joint_pos[j] = float(v)
        return ok

    def _gripper_reopen(self, gen: int, timeout: float = 2.2) -> bool:
        """Open until the joint actually leaves the closed stop.

        A 0.45s burst after an empty slam leaves the XL430 shut, so the
        next pinch never happens.
        """
        self._gripper_mode_ok = False
        self._ensure_gripper_mode()
        self._last_grip_joint = float(self.grip_joint_open)
        t0 = time.monotonic()
        while time.monotonic() - t0 < max(0.8, float(timeout)):
            if self._is_cancelled(gen):
                return False
            self._publish_gripper_joint(self.grip_joint_open, n=2)
            gj = self._gripper_joint
            if gj is not None and float(gj) >= 0.40:
                return True
            time.sleep(0.08)
        self.get_logger().warn(
            f"reopen failed joint={self._gripper_joint} "
            f"(need >= 0.40 to pinch again)"
        )
        return False

    def _gripper_actuate(
        self,
        cmd: float,
        settle_s: float,
        gen: int,
        hold: bool = False,
    ) -> bool:
        """Mode + burst command. Returns False if cancelled."""
        before = self._gripper_joint
        if not self._set_gripper(cmd, settle_s=settle_s, gen=gen, hold=hold):
            return False
        if not self._sleep_interruptible(max(0.0, settle_s - 0.6), gen):
            return False
        after = self._gripper_joint
        moved = (
            before is not None
            and after is not None
            and float(cmd) < 0.0
            and (before - after) > 0.10
        )
        if moved:
            self._hold_gripper_present()
        if before is not None and after is not None and abs(after - before) < 0.05:
            self.get_logger().warn(
                f"gripper cmd={cmd:.2f} joint unchanged ({after:.4f})"
            )
        return True

    def _gripper_coast(self):
        # Position mode: cmd=0 is a mid-open pose, not "stop". Hold last goal.
        joint = self._last_grip_joint
        if joint is None:
            joint = self._gripper_joint
        if joint is None:
            return
        self._publish_gripper_joint(joint, n=2)

    def _wait_object_in_gripper(
        self, gen: int, baseline: Optional[np.ndarray] = None
    ) -> Tuple[bool, str]:
        """Stay open until a blob sits in the jaw opening, then close.

        Returns (ok, reason): 'seen' | 'timeout' | 'no_wide' | 'cancelled'.
        """
        if cv2 is None or not self.use_wide or not self.wide_close_on_blob:
            return True, "no_wide"
        hit = 0
        need = max(1, int(self.wide_blob_in_jaws_frames))
        t0 = time.monotonic()
        wait = max(0.3, float(self.wide_object_wait_s))
        while time.monotonic() - t0 < wait:
            if self._is_cancelled(gen):
                return False, "cancelled"
            self._gripper_pulse_open()
            blob = self._blob_in_jaws()
            if blob is None and baseline is not None:
                cur = self._wide_decode()
                if cur is not None:
                    score = self._wide_motion(baseline, cur)
                    if score >= self.wide_object_enter_min:
                        blob = (self.wide_grasp_u, self.wide_grasp_v)
            if blob is not None:
                hit += 1
                if hit >= need:
                    self.get_logger().info(
                        f"object in jaws blob=({blob[0]:.2f},{blob[1]:.2f})"
                    )
                    return True, "seen"
            else:
                hit = 0
            time.sleep(0.05)
        self.get_logger().warn(
            "object-in-jaws wait timed out — closing anyway"
        )
        return True, "timeout"

    def _gripper_close_until_stall(
        self, gen: int
    ) -> Tuple[bool, str]:
        """Close until pinch load, foam stall, empty hard-stop, or timeout.

        Vision motion is not a grasp by itself. Success is Present_Load
        (reason 'load'), or jaws that moved past halfway and stopped short
        of the empty stop (reason 'foam'). Empty slam is 'empty'.

        Commands advance toward fully-closed in small steps (same class of
        motion as the handoff search pulse). Publishing the hard-closed goal
        every tick let the XL430 velocity profile slam after a slow search.
        """
        self._ensure_gripper_mode()
        self._apply_gripper_profile()
        close_j = float(self.grip_joint_closed)
        open_j = float(self.grip_joint_open)
        step = max(0.015, float(self.grip_close_step_rad))
        gj0 = self._gripper_joint
        cmd_j = float(gj0) if gj0 is not None else open_j
        cmd_j = max(close_j, min(open_j, cmd_j))
        self._last_grip_joint = cmd_j
        idle = self._read_gripper_load()
        # Refresh idle each pinch — a stale first-grab idle skews later thresholds.
        if idle is not None:
            self._grip_load_idle = int(idle)
        prev = self._wide_decode()
        peak = 0.0
        stall = 0
        saw_motion = False
        still = 0
        prev_gj = self._gripper_joint
        start_gj = prev_gj
        t0 = time.monotonic()
        settle = max(2.5, float(self.grip_close_s), float(self.grip_settle))
        warmup_s = 0.45
        next_load_t = t0 + warmup_s
        load_thr = self._load_grab_thr()
        empty_j = float(self.grip_joint_closed) + float(self.grip_empty_closed)
        last_load: Optional[int] = idle
        require_load = bool(self.grasp_require_load)

        def _pinch_by_load() -> bool:
            nonlocal last_load
            gj = self._gripper_joint
            moved = start_gj is None or (
                gj is not None and (start_gj - gj) > 0.12
            )
            if not moved:
                return False
            load = self._read_gripper_load()
            if load is not None:
                last_load = load
            if load is None or load < load_thr:
                return False
            # Reject "load" while jaws are still mostly open (false pinch →
            # used to retreat after a miss). Need a real close stroke.
            if gj is not None:
                if float(gj) <= empty_j:
                    return False
                mid = 0.45 * (
                    float(self.grip_joint_open) + float(self.grip_joint_closed)
                )
                if float(gj) > mid:
                    return False
            self.get_logger().info(
                f"gripper load pinch load={load} thr={load_thr} "
                f"joint={gj} after {time.monotonic() - t0:.2f}s"
            )
            self._hold_gripper_present()
            return True

        def _foam_stall() -> bool:
            """Jaws stopped on something thicker than an empty close."""
            gj = self._gripper_joint
            if gj is None or start_gj is None or still < 8:
                return False
            if (float(start_gj) - float(gj)) < 0.12:
                return False
            mid = 0.45 * (
                float(self.grip_joint_open) + float(self.grip_joint_closed)
            )
            if not (empty_j < float(gj) <= mid):
                return False
            self.get_logger().info(
                f"gripper foam stall joint={float(gj):.3f} load={last_load} "
                f"empty_stop={empty_j:.3f} mid={mid:.3f} "
                f"after {time.monotonic() - t0:.2f}s"
            )
            self._hold_gripper_present()
            return True

        def _empty_slam() -> bool:
            gj = self._gripper_joint
            if gj is None or start_gj is None:
                return False
            if float(gj) > empty_j:
                return False
            if (start_gj - gj) < 0.25:
                return False
            load = last_load
            self.get_logger().info(
                f"gripper empty slam joint={gj:.3f} load={load} "
                f"after {time.monotonic() - t0:.2f}s"
            )
            return True

        while time.monotonic() - t0 < settle:
            if self._is_cancelled(gen):
                return False, "cancelled"
            # Step the goal toward closed — do not jump to close_j in one shot.
            cmd_j = max(close_j, cmd_j - step)
            self._last_grip_joint = cmd_j
            self._publish_gripper_joint(cmd_j, n=1)
            time.sleep(0.05)
            elapsed = time.monotonic() - t0
            if elapsed >= warmup_s and time.monotonic() >= next_load_t:
                next_load_t = time.monotonic() + 0.15
                if _pinch_by_load():
                    return True, "load"
                if _empty_slam():
                    return True, "empty"
                gj = self._gripper_joint
                self._publish_status(
                    "grasping",
                    f"Pinching… load={last_load if last_load is not None else '—'} "
                    f"thr={load_thr}"
                    f"{'' if gj is None else f' j={gj:.2f}'}",
                )
            cur = self._wide_decode()
            if cur is not None and prev is not None:
                motion = self._wide_motion(prev, cur)
                prev = cur
                if elapsed >= warmup_s:
                    if motion > peak:
                        peak = motion
                    if motion >= self.wide_min_motion:
                        saw_motion = True
                    thr = max(
                        self.wide_min_motion,
                        self.wide_stall_ratio * peak if peak > 0.0 else self.wide_min_motion,
                    )
                    if saw_motion and motion < thr:
                        stall += 1
                    else:
                        stall = 0
            elif cur is not None:
                prev = cur
            gj = self._gripper_joint
            if elapsed >= warmup_s and gj is not None:
                if prev_gj is not None and abs(gj - prev_gj) < 0.015:
                    still += 1
                else:
                    still = 0
                prev_gj = gj
            # Vision/joint stall: only freeze if load already confirms a pinch.
            if not require_load and stall >= self.wide_stall_frames:
                if _pinch_by_load():
                    return True, "load"
                self._hold_gripper_present()
                return True, "stall"
            if still >= 8 and _pinch_by_load():
                return True, "load"
            if still >= 8 and _foam_stall():
                return True, "foam"
            if _empty_slam():
                return True, "empty"

        if _pinch_by_load() or self._grab_feels_object(last_load):
            self._hold_gripper_present()
            return True, "load"
        if _foam_stall():
            return True, "foam"
        if _empty_slam():
            return True, "empty"
        self.get_logger().warn(
            f"gripper close timeout load={last_load} joint={self._gripper_joint}"
        )
        return True, "timeout"

    def _wait_ik(self, gen: int) -> bool:
        deadline = time.time() + 15.0
        while time.time() < deadline and rclpy.ok():
            if self._is_cancelled(gen):
                return False
            if self._ik_cli.service_is_ready():
                return True
            time.sleep(0.2)
        return False

    def _rotate_xy_about_z(
        self, p: Tuple[float, float, float], dw: float
    ) -> Tuple[float, float, float]:
        c, s = math.cos(dw), math.sin(dw)
        return (c * p[0] - s * p[1], s * p[0] + c * p[1], p[2])

    def _waist_delta_match_d435i_u(
        self, ee: Tuple[float, float, float], u_tgt: float
    ) -> Optional[Tuple[float, float]]:
        """Waist delta that puts EE in the same D435i column as the click.

        D435i is on the base, so atan2(click XY) does not line the gripper up
        in the JPEG — especially when the click is near the origin.
        """
        p0 = (float(ee[0]), float(ee[1]), float(ee[2]))
        if math.hypot(p0[0], p0[1]) < 0.06:
            return None
        best_dw: Optional[float] = None
        best_err = 1e9
        for i in range(-180, 181):
            dw = math.radians(float(i) * 0.5)
            p = self._rotate_xy_about_z(p0, dw)
            proj = self._project_xyz_to_d435i(p, urdf_cam=True)
            if proj is None or not proj[2]:
                pp = self._urdf_xyz_to_d435i(p)
                proj = self._project_xyz_to_d435i(pp, urdf_cam=False)
            if proj is None or not proj[2]:
                continue
            err = abs(float(proj[0]) - float(u_tgt))
            if err < best_err:
                best_err = err
                best_dw = dw
        if best_dw is None or best_err > 0.40:
            return None
        return best_dw, best_err

    def _waist_aim_xy(
        self, xyz: Tuple[float, float, float]
    ) -> Tuple[float, float]:
        """Click-ray point at a graspable XY radius so atan2 is stable."""
        p = np.array(
            [float(xyz[0]), float(xyz[1]), float(xyz[2])], dtype=np.float64
        )
        if math.hypot(float(p[0]), float(p[1])) >= 0.12 and float(p[0]) > 0.08:
            return float(p[0]), float(p[1])
        ee = self._ee_xyz()
        r_min = 0.14
        if ee is not None:
            r_min = max(r_min, 0.70 * math.hypot(float(ee[0]), float(ee[1])))
        tf = self._d435i_planning_T(self._depth_frame or self.camera_frame)
        if tf is None:
            return float(p[0]), float(p[1])
        origin = tf[0][:3, 3].copy()
        cam_z = tf[0][:3, 2].copy()
        d = p - origin
        if float(np.dot(d, cam_z)) < 0.0:
            d = cam_z
        dn = float(np.linalg.norm(d))
        if dn < 1e-4:
            return float(p[0]), float(p[1])
        d = d / dn
        q = p.copy()
        for _ in range(48):
            if math.hypot(float(q[0]), float(q[1])) >= r_min and float(q[0]) > 0.04:
                return float(q[0]), float(q[1])
            q = q + 0.02 * d
        if math.hypot(float(q[0]), float(q[1])) > math.hypot(float(p[0]), float(p[1])):
            return float(q[0]), float(q[1])
        return float(p[0]), float(p[1])

    def _waist_face_target(self, xyz: Tuple[float, float, float]) -> float:
        """Waist that faces the click XY (kinematic). Small visual trim only.

        Visual column match used to keep waist near the current joint when
        that was already wrong (e.g. 0° vs kin −10°), which seeded IK badly.
        """
        ax, ay = self._waist_aim_xy(xyz)
        kinematic = math.atan2(float(ay), float(ax))
        kinematic = float(self.waist_align_sign) * kinematic + float(
            self.waist_align_offset
        )
        kinematic = self._clip_waist(kinematic)
        ee = self._ee_xyz()
        u_tgt = None
        if self._last_click_uv is not None:
            u_tgt = float(self._last_click_uv[0]) + float(self.waist_click_u_bias)
        if u_tgt is not None and ee is not None:
            matched = self._waist_delta_match_d435i_u(ee, u_tgt)
            if matched is not None:
                dw, err = matched
                cur = float(self._joint_pos.get("waist", 0.0))
                visual = self._clip_waist(cur + dw)
                if abs(_wrap_pi(visual - kinematic)) <= math.radians(6.0):
                    self.get_logger().info(
                        f"waist visual trim {math.degrees(kinematic):.1f}→"
                        f"{math.degrees(visual):.1f} deg (du={err:.3f})"
                    )
                    return visual
        return kinematic

    def _align_waist_to_target(
        self,
        xyz: Tuple[float, float, float],
        gen: int,
    ) -> bool:
        """Yaw waist until the gripper shares the D435i click column."""
        iw = self._joint_index("waist")
        if iw < 0 or not self.align_waist:
            return True
        face = self._waist_face_target(xyz)
        cur = float(self._joint_pos.get("waist", 0.0))
        delta = _wrap_pi(face - cur)
        cmd = self._clip_waist(cur + float(self.waist_align_gain) * delta)
        self._waist_face = face
        if abs(cmd - cur) < self.waist_align_min:
            self._waist_face = face
            self.get_logger().info(
                f"waist already facing target ({math.degrees(cur):.1f} deg)"
            )
            return True
        q = [float(self._joint_pos[j]) for j in self.arm_joints]
        q[iw] = cmd
        self._publish_status(
            "planning",
            f"Aligning waist {math.degrees(cur):.0f}→{math.degrees(cmd):.0f}° "
            f"(face {math.degrees(face):.0f}°)…",
            xyz,
        )
        old = self.seg_dur
        self.seg_dur = max(1.2, min(3.0, abs(cmd - cur) / 0.45))
        try:
            ok = self._send_trajectory([q], gen)
        finally:
            self.seg_dur = old
        if ok:
            self._joint_pos["waist"] = cmd
            self._waist_face = face
            if not self._sleep_interruptible(0.20, gen):
                return False
            self.get_logger().info(
                f"waist aligned to {math.degrees(cmd):.1f} deg "
                f"(face {math.degrees(face):.1f}° "
                f"xy={xyz[0]:.3f},{xyz[1]:.3f})"
            )
            # Second look: EE has moved, trim leftover column error.
            face2 = self._waist_face_target(xyz)
            d2 = _wrap_pi(face2 - cmd)
            if abs(d2) >= 0.035:
                cmd2 = self._clip_waist(cmd + d2)
                q2 = [float(self._joint_pos[j]) for j in self.arm_joints]
                q2[iw] = cmd2
                self._publish_status(
                    "planning",
                    f"Waist trim {math.degrees(cmd):.0f}→{math.degrees(cmd2):.0f}°…",
                    xyz,
                )
                old2 = self.seg_dur
                self.seg_dur = max(0.6, min(1.6, abs(cmd2 - cmd) / 0.50))
                try:
                    ok2 = self._send_trajectory([q2], gen)
                finally:
                    self.seg_dur = old2
                if ok2:
                    self._joint_pos["waist"] = cmd2
                    self._waist_face = face2
                    self.get_logger().info(
                        f"waist trimmed to {math.degrees(cmd2):.1f} deg"
                    )
        return ok

    def _joint_index(self, name: str) -> int:
        try:
            return self.arm_joints.index(name)
        except ValueError:
            return -1

    def _ik_solution_ok(self, q: List[float]) -> bool:
        """Reject inverted / folded-back poses that swing into the AGX."""
        ia = self._joint_index("wrist_angle")
        ir = self._joint_index("wrist_rotate")
        ish = self._joint_index("shoulder")
        if ia >= 0:
            if q[ia] < self.wrist_angle_min or q[ia] > self.wrist_angle_max:
                return False
        if ir >= 0:
            if abs(q[ir]) > self.wrist_rotate_max:
                return False
        if ish >= 0:
            sh = float(q[ish])
            if sh < float(self.shoulder_min_grasp):
                return False
            # Relative gate: decreasing shoulder lifts/folds toward the stack.
            if self._shoulder_forward_only:
                cur = float(self._joint_pos.get("shoulder", sh))
                if sh < cur - float(self.shoulder_back_slack):
                    return False
        return True

    def _ik_solution_cost(self, q: List[float], seed: List[float]) -> float:
        """Lower is better: stay near seed/current, prefer wrist_rotate≈0."""
        cost = 0.0
        for i, (a, b) in enumerate(zip(q, seed)):
            w = 1.0
            name = self.arm_joints[i] if i < len(self.arm_joints) else ""
            if name == "wrist_rotate":
                w = 4.0
            elif name == "waist" and self._waist_face is not None:
                w = 3.0
            elif name == "wrist_angle":
                w = 2.0
            elif name == "shoulder":
                w = 2.5
            cost += w * (a - b) * (a - b)
        ish = self._joint_index("shoulder")
        if ish >= 0:
            cur_sh = float(self._joint_pos.get("shoulder", q[ish]))
            # Always discourage lift/back toward AGX; stronger when gated.
            if q[ish] < cur_sh:
                ds = cur_sh - q[ish]
                w = 40.0 if self._shoulder_forward_only else 18.0
                cost += w * ds * ds
            if q[ish] < float(self.shoulder_min_grasp):
                cost += 80.0
        iw = self._joint_index("waist")
        if iw >= 0 and self._waist_face is not None:
            dw = q[iw] - self._waist_face
            cost += 4.0 * dw * dw
        ir = self._joint_index("wrist_rotate")
        if ir >= 0:
            target_wr = (
                0.0 if self._wrist_face is None else float(self._wrist_face)
            )
            dw = q[ir] - target_wr
            cost += 6.0 * dw * dw
        ia = self._joint_index("wrist_angle")
        if ia >= 0:
            # Prefer near-level bar for tabletop; hard -1.2 / -0.9 parked the
            # wrist straight down (~-100°) instead of reaching the click.
            prefer = -1.20 if self._object_low else -0.50
            cost += 1.2 * (q[ia] - prefer) * (q[ia] - prefer)
        return cost

    def _solve_ik(
        self,
        pose: PoseStamped,
        gen: int,
        nearby: bool = False,
        prefer_seed: Optional[List[float]] = None,
        fast: bool = False,
    ) -> Optional[List[float]]:
        """Try MoveIt IK; prefer non-inverted wrists without stalling."""
        if not self._wait_ik(gen):
            if not self._is_cancelled(gen):
                self.get_logger().error("compute_ik not available")
            return None

        current = [self._joint_pos[j] for j in self.arm_joints]
        if prefer_seed is not None and len(prefer_seed) == len(self.arm_joints):
            seed_list = [list(prefer_seed), current]
            if not fast:
                seed_list += list(self._ik_seeds)[:3]
        elif nearby or fast:
            seed_list = [current]
        else:
            seed_list = [current] + list(self._ik_seeds)[:4]
        seeded = []
        for s in seed_list:
            s2 = list(s)
            ir = self._joint_index("wrist_rotate")
            if ir >= 0:
                if self._wrist_face is not None:
                    s2[ir] = self._clip_wrist_rotate(self._wrist_face)
                elif self.lock_wrist_rotate:
                    s2[ir] = 0.0
            iw = self._joint_index("waist")
            if (
                iw >= 0
                and self._waist_face is not None
                and prefer_seed is None
            ):
                s2[iw] = self._waist_face
            seeded.append(s2)
        seeds = seeded
        if self.ee_link.endswith("ee_gripper_link"):
            arm = self.ee_link.replace("ee_gripper_link", "ee_arm_link")
            if self._object_low:
                # The jaws are the tool on a floor pick. Solving the arm
                # link puts the wrist on the click and leaves the jaws short.
                links = [self.ee_link] if fast else [self.ee_link, arm]
            else:
                # Arm link first — fewer -31s; skip gripper unless needed.
                links = [arm] if fast else [arm, self.ee_link]
        elif self.ee_link.endswith("ee_arm_link"):
            links = [self.ee_link]
            if not fast:
                links.append(
                    self.ee_link.replace("ee_arm_link", "ee_gripper_link")
                )
        else:
            links = [self.ee_link]

        yaw = math.atan2(pose.pose.position.y, pose.pose.position.x)
        if fast:
            orients: List[Quaternion] = [
                _rpy_to_quat(0.0, 0.0, yaw),
                pose.pose.orientation,
            ]
        else:
            orients = [
                _rpy_to_quat(0.0, 0.0, yaw),
                pose.pose.orientation,
                self.grasp_quat,
                _rpy_to_quat(0.0, 0.20, yaw),
            ]
            if self.use_current_ee_orient:
                orients.append(self._ee_quat())

        def _uniq(qs: List[Quaternion]) -> List[Quaternion]:
            out: List[Quaternion] = []
            for q in qs:
                key = (round(q.x, 3), round(q.y, 3), round(q.z, 3), round(q.w, 3))
                if all(
                    (round(u.x, 3), round(u.y, 3), round(u.z, 3), round(u.w, 3)) != key
                    for u in out
                ):
                    out.append(q)
            return out

        uniq_orients = _uniq(orients)
        if nearby:
            uniq_orients = _uniq([self._ee_quat(), pose.pose.orientation])
        # take_first_gated exits on first ok solution — keep budget modest.
        if fast:
            budget = 1.2
        elif prefer_seed is not None:
            budget = 2.0
        elif nearby:
            budget = 1.5
        else:
            budget = min(3.0, max(2.0, float(self.ik_budget_s)))
        deadline = time.time() + budget
        best_gated: Optional[List[float]] = None
        best_gated_cost = float("inf")
        best_raw: Optional[List[float]] = None
        best_raw_cost = float("inf")
        rejected = 0
        last_code = None
        tries = 0
        # First gated wins. Old cost<0.35 never fired (costs ~4–6) and burned
        # the full budget after a good solve. Do NOT use fast= for pre-grasp:
        # one seed×one orient → tries=1 and perpetual -31.
        take_first_gated = True

        def _consider(sol: List[float], seed: List[float]) -> Optional[List[float]]:
            nonlocal best_gated, best_gated_cost, best_raw, best_raw_cost, rejected
            if nearby:
                ish = self._joint_index("shoulder")
                ia = self._joint_index("wrist_angle")
                if ish >= 0 and abs(float(sol[ish]) - float(current[ish])) > 0.30:
                    rejected += 1
                    return None
                if ia >= 0 and abs(float(sol[ia]) - float(current[ia])) > 0.40:
                    rejected += 1
                    return None
            ref = current if self.ik_prefer_current else seed
            cost = self._ik_solution_cost(sol, ref)
            if cost < best_raw_cost:
                best_raw_cost = cost
                best_raw = sol
            if self._ik_solution_ok(sol):
                if cost < best_gated_cost:
                    best_gated_cost = cost
                    best_gated = sol
                if take_first_gated or cost < 0.80:
                    return sol
            else:
                rejected += 1
                if rejected <= 3:
                    reach = math.hypot(
                        float(pose.pose.position.x), float(pose.pose.position.y)
                    )
                    self.get_logger().info(
                        f"IK reject inverted reach={reach:.3f} "
                        f"q={[round(v, 3) for v in sol]}"
                    )
            return None

        for link in links:
            for seed in seeds:
                for q in uniq_orients:
                    if self._is_cancelled(gen):
                        return None
                    if time.time() > deadline:
                        break
                    trial = PoseStamped()
                    trial.header = pose.header
                    trial.pose.position = pose.pose.position
                    trial.pose.orientation = q
                    sol, code = self._ik_once(trial, link, seed, gen)
                    last_code = code
                    tries += 1
                    if sol is None:
                        continue
                    early = _consider(sol, seed)
                    if early is not None:
                        self.get_logger().info(
                            f"IK ok cost={best_gated_cost:.3f} "
                            f"q={[round(v, 3) for v in early]} tries={tries}"
                        )
                        return early
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break

        if best_gated is not None:
            self.get_logger().info(
                f"IK ok cost={best_gated_cost:.3f} "
                f"q={[round(v, 3) for v in best_gated]} "
                f"tries={tries} rejected_inverted={rejected}"
            )
            return best_gated

        # Never command a pose that failed the AGX/shoulder gate — high/low
        # clicks used to fall through to best_raw and whip the arm back.
        if (
            best_raw is not None
            and rejected > 0
            and not self.forbid_raw_inverted_ik
            and self._ik_solution_ok(best_raw)
        ):
            self.get_logger().warn(
                f"IK using raw (gated rejected) q={[round(v, 3) for v in best_raw]} "
                f"rejected_inverted={rejected}"
            )
            return best_raw
        if best_raw is not None and rejected > 0 and self.forbid_raw_inverted_ik:
            self.get_logger().warn(
                f"IK blocked raw inverted q={[round(v, 3) for v in best_raw]} "
                f"rejected_inverted={rejected} (shoulder/wrist gate)"
            )

        if self._is_cancelled(gen):
            return None
        self.get_logger().warn(
            f"IK failed (last_code={last_code}, tries={tries}, "
            f"rejected_inverted={rejected}) "
            f"at ({pose.pose.position.x:.3f},{pose.pose.position.y:.3f},"
            f"{pose.pose.position.z:.3f})"
        )
        return None

    def _solve_grasp_contact_ik(
        self,
        contact: Tuple[float, float, float],
        pre_xyz: Tuple[float, float, float],
        gen: int,
        prefer_seed: Optional[List[float]] = None,
    ) -> Tuple[Optional[List[float]], Tuple[float, float, float]]:
        """Prefer exact contact; short walk-back toward pre; else caller uses pre."""
        # Seeded from pre — usually enough without the old 16×2s walk.
        q = self._solve_ik(
            self._pose_at(contact, 0.0),
            gen,
            prefer_seed=prefer_seed,
        )
        if q is not None:
            return q, contact
        if self._is_cancelled(gen):
            return None, contact
        # Exact contact first; walk toward pre only as fallback.
        # Too-tight blends (0.15/0.30 only) left most grasps on folded
        # pre-grasp joints — wrist parked ~-100° short of the click.
        cx, cy, cz = contact
        px, py, pz = pre_xyz
        for a in (0.12, 0.28, 0.45, 0.60, 0.75):
            if self._is_cancelled(gen):
                return None, contact
            xyz = (
                cx + a * (px - cx),
                cy + a * (py - cy),
                cz + a * (pz - cz),
            )
            q = self._solve_ik(
                self._pose_at(xyz, 0.0),
                gen,
                prefer_seed=prefer_seed,
            )
            if q is not None:
                self.get_logger().info(
                    f"grasp contact walk-back a={a:.2f} xyz="
                    f"({xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f})"
                )
                return q, xyz
        self.get_logger().info(
            "grasp contact IK skipped — using pre-grasp joints"
        )
        return None, contact

    def _ik_once(
        self, pose: PoseStamped, link: str, seed: List[float], gen: int
    ) -> Tuple[Optional[List[float]], Optional[int]]:
        req = GetPositionIK.Request()
        req.ik_request = PositionIKRequest()
        req.ik_request.group_name = self.planning_group
        req.ik_request.pose_stamped = pose
        req.ik_request.ik_link_name = link
        # Keep individual solves short so a full search stays interactive.
        # ik_timeout_s in yaml is the overall budget, not per-call.
        to = 0.40
        req.ik_request.timeout.sec = int(to)
        req.ik_request.timeout.nanosec = int((to % 1.0) * 1e9)
        req.ik_request.avoid_collisions = self.avoid_collisions
        # Do not set constraints — breaks/hangs position_only_ik on this setup.
        rs = RobotState()
        rs.joint_state.name = list(self.arm_joints)
        rs.joint_state.position = [float(v) for v in seed]
        req.ik_request.robot_state = rs

        fut = self._ik_cli.call_async(req)
        t0 = time.time()
        while not fut.done() and time.time() - t0 < to + 1.0:
            if self._is_cancelled(gen):
                return None, None
            time.sleep(0.03)
        if not fut.done():
            return None, None
        res = fut.result()
        if res is None:
            return None, None
        code = int(res.error_code.val)
        if code != res.error_code.SUCCESS:
            return None, code
        name_to_pos = dict(
            zip(res.solution.joint_state.name, res.solution.joint_state.position)
        )
        try:
            return [float(name_to_pos[j]) for j in self.arm_joints], code
        except KeyError:
            stripped = {n.split("/")[-1]: p for n, p in name_to_pos.items()}
            try:
                return [float(stripped[j]) for j in self.arm_joints], code
            except KeyError:
                return None, code

    def _send_trajectory(self, waypoints: List[List[float]], gen: int) -> bool:
        if not waypoints:
            return False
        # Last-line defense: never command a shoulder lift/back toward AGX.
        safe: List[List[float]] = []
        for q in waypoints:
            q2 = self._clamp_shoulder_forward(list(q))
            ish = self._joint_index("shoulder")
            if ish >= 0 and float(q2[ish]) < float(self.shoulder_min_grasp):
                self.get_logger().warn(
                    f"blocked traj shoulder={q2[ish]:.3f} < "
                    f"min {self.shoulder_min_grasp:.3f}"
                )
                return False
            safe.append(q2)
        waypoints = safe
        if self._use_traj and self._traj_cli.wait_for_server(timeout_sec=2.0):
            if self._send_traj_action(waypoints, gen):
                return True
            if self._is_cancelled(gen):
                return False
            self.get_logger().warn("Trajectory action failed — falling back to joint_group")
        return self._send_joint_group_path(waypoints, gen)

    def _send_traj_action(
        self, waypoints: List[List[float]], gen: int
    ) -> bool:
        traj = JointTrajectory()
        traj.joint_names = list(self.arm_joints)
        t = 0.0
        pts = [list(self._joint_pos[j] for j in self.arm_joints)] + waypoints
        for i, q in enumerate(pts):
            t = 0.0 if i == 0 else t + self.seg_dur
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in q]
            pt.velocities = [0.0] * len(q)
            pt.time_from_start = Duration(sec=int(t), nanosec=int((t % 1.0) * 1e9))
            traj.points.append(pt)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        send_fut = self._traj_cli.send_goal_async(goal)
        t0 = time.time()
        while not send_fut.done() and time.time() - t0 < 5.0:
            if self._is_cancelled(gen):
                return False
            time.sleep(0.05)
        if not send_fut.done():
            return False
        goal_handle = send_fut.result()
        if goal_handle is None or not goal_handle.accepted:
            return False
        self._active_goal = goal_handle
        result_fut = goal_handle.get_result_async()
        timeout = self.seg_dur * max(1, len(waypoints)) + 10.0
        t0 = time.time()
        while not result_fut.done() and time.time() - t0 < timeout:
            if self._is_cancelled(gen):
                try:
                    goal_handle.cancel_goal_async()
                except Exception:
                    pass
                return False
            time.sleep(0.05)
        if not result_fut.done():
            return False
        result = result_fut.result().result
        return int(getattr(result, "error_code", 0)) == 0

    def _send_joint_group_path(
        self, waypoints: List[List[float]], gen: int
    ) -> bool:
        """Smoothly interpolate to each waypoint via xs_sdk joint_group (Nano bus)."""
        rate_hz = 20.0
        dt = 1.0 / rate_hz
        q_cur = [float(self._joint_pos[j]) for j in self.arm_joints]
        for q_goal in waypoints:
            if self._is_cancelled(gen):
                return False
            steps = max(1, int(self.seg_dur * rate_hz))
            for s in range(1, steps + 1):
                if self._is_cancelled(gen):
                    return False
                a = s / float(steps)
                q = [c + a * (g - c) for c, g in zip(q_cur, q_goal)]
                msg = JointGroupCommand()
                msg.name = "arm"
                msg.cmd = [float(v) for v in q]
                self._arm_pub.publish(msg)
                time.sleep(dt)
            q_cur = list(q_goal)
            # Hold final
            msg = JointGroupCommand()
            msg.name = "arm"
            msg.cmd = [float(v) for v in q_cur]
            self._arm_pub.publish(msg)
            if not self._sleep_interruptible(0.15, gen):
                return False
        return True

    def _named_joints(self, name: str) -> List[float]:
        if name == "sleep":
            q = list(self.sleep_joints)
        else:
            q = list(self.home_joints)
        if len(q) < len(self.arm_joints):
            q.extend([0.0] * (len(self.arm_joints) - len(q)))
        return q[: len(self.arm_joints)]

    def _joints_err(self, q: List[float]) -> float:
        err = 0.0
        for j, v in zip(self.arm_joints, q):
            err = max(err, abs(float(self._joint_pos.get(j, 0.0)) - float(v)))
        return err

    def _goto_named(self, name: str, gen: int) -> bool:
        q = self._named_joints(name)
        if self._joints_err(q) < 0.06:
            self.get_logger().info(f"already at {name}")
            return True
        old = self.seg_dur
        self.seg_dur = max(1.6, min(3.2, 0.8 + 1.4 * self._joints_err(q)))
        try:
            if not self._send_trajectory([q], gen):
                return False
        finally:
            self.seg_dur = old
        if self._last_grip_joint is not None:
            self._publish_gripper_joint(self._last_grip_joint, n=2)
        t0 = time.monotonic()
        while time.monotonic() - t0 < 6.0:
            if self._is_cancelled(gen):
                return False
            if self._joints_err(q) < 0.08:
                break
            time.sleep(0.05)
        for j, v in zip(self.arm_joints, q):
            self._joint_pos[j] = float(v)
        return True

    def _search_close_joint(self) -> float:
        lo = float(self.grip_joint_closed)
        hi = float(self.grip_joint_open)
        frac = max(0.05, min(0.85, float(self.handoff_search_close_frac)))
        return hi + frac * (lo - hi)

    def _jaw_window(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """Crop the opening between the fingers in the wide image."""
        h, w = gray.shape[:2]
        x0 = max(0, min(w - 2, int(self.handoff_jaw_u0 * w)))
        x1 = max(x0 + 2, min(w, int(self.handoff_jaw_u1 * w)))
        y0 = max(0, min(h - 2, int(self.handoff_jaw_v0 * h)))
        y1 = max(y0 + 2, min(h, int(self.handoff_jaw_v1 * h)))
        win = gray[y0:y1, x0:x1]
        if win.size < 64:
            return None
        return win

    def _capture_jaw_baseline(self) -> bool:
        gray = self._wide_gray()
        if gray is None:
            self._handoff_baseline = None
            return False
        win = self._jaw_window(gray)
        if win is None:
            self._handoff_baseline = None
            return False
        self._handoff_baseline = win.astype(np.float32)
        self.get_logger().info(
            f"handoff jaw baseline {win.shape[1]}x{win.shape[0]} "
            f"mean={float(np.mean(win)):.1f}"
        )
        return True

    def _jaws_see_object(self) -> bool:
        """True if the jaw opening looks occupied (appearance and/or blob).

        Not contact sensing. The wide USB cam looks out through the fingers.
        Click-to-grasp blob-hunting wanted the nearest table object; handoff
        instead compares the jaw window to the empty-open baseline.
        """
        gray = self._wide_gray()
        if gray is None:
            now = time.monotonic()
            if now - self._handoff_dbg_t > 1.5:
                self._handoff_dbg_t = now
                self.get_logger().warn("handoff: no wide frame (stale/missing)")
            return False
        win = self._jaw_window(gray)
        appear = 0.0
        if win is not None and self._handoff_baseline is not None:
            a = win.astype(np.float32)
            b = self._handoff_baseline
            if a.shape == b.shape:
                appear = float(np.mean(np.abs(a - b)))
        hit = self._wide_closest_object()
        u = v = None
        in_box = False
        if hit is not None:
            u, v, _score = hit
            in_box = (
                self.handoff_jaw_u0 <= float(u) <= self.handoff_jaw_u1
                and self.handoff_jaw_v0 <= float(v) <= self.handoff_jaw_v1
            )
        seen = appear >= float(self.handoff_appear_min) or in_box
        now = time.monotonic()
        if now - self._handoff_dbg_t > 1.0:
            self._handoff_dbg_t = now
            uv = (
                f"blob=({u:.2f},{v:.2f})" if u is not None else "blob=none"
            )
            self.get_logger().info(
                f"handoff detect appear={appear:.1f} "
                f"(min={self.handoff_appear_min:.1f}) {uv} in_jaws={in_box} seen={seen}"
            )
        return seen

    def _sweep_gripper(
        self,
        start_j: float,
        end_j: float,
        dur: float,
        gen: int,
        detect: bool,
    ) -> Tuple[bool, str]:
        """Lerp gripper joint. Returns (not_cancelled, reason).

        reason is '' | 'vision' | 'load'.
        Step size is capped by gripper_close_step_rad so search pulses and
        the final pinch share the same max close rate.
        """
        span = abs(float(end_j) - float(start_j))
        step = max(0.015, float(self.grip_close_step_rad))
        # Prefer configured duration, but never exceed the shared max step rate.
        min_dur = span / max(step / 0.05, 1e-3)
        dur = max(float(dur), min_dur)
        steps = max(4, int(max(0.2, dur) / 0.05))
        hits = 0
        need = max(2, int(self.handoff_center_frames))
        contact = self._load_contact_thr()
        for s in range(1, steps + 1):
            if self._is_cancelled(gen):
                return False, "cancelled"
            a = s / float(steps)
            j = float(start_j) + a * (float(end_j) - float(start_j))
            self._last_grip_joint = j
            self._publish_gripper_joint(j, n=1)
            if detect and self._jaws_see_object():
                hits += 1
                if hits >= need:
                    return True, "vision"
            else:
                hits = 0
            if detect and s % 3 == 0:
                load = self._read_gripper_load()
                if load is not None and load >= contact:
                    gj = self._gripper_joint
                    empty = float(self.grip_joint_closed) + float(
                        self.grip_empty_closed
                    )
                    if gj is None or float(gj) > empty:
                        self.get_logger().info(
                            f"handoff squeeze load={load} thr={contact} joint={gj}"
                        )
                        return True, "load"
            time.sleep(0.05)
        return True, ""

    def _wait_object_centered(
        self, gen: int, wait_s: Optional[float] = None
    ) -> Tuple[bool, str]:
        """Pulse jaws until vision or gripper load says something is in them."""
        self._apply_gripper_profile()
        open_j = float(self.grip_joint_open)
        mid_j = self._search_close_joint()
        t0 = time.monotonic()
        wait = max(5.0, float(self.handoff_wait_s if wait_s is None else wait_s))
        pulse = max(0.6, float(self.handoff_pulse_s))
        hold = max(0.3, float(self.handoff_hold_open_s))
        while time.monotonic() - t0 < wait:
            if self._is_cancelled(gen):
                return False, "cancelled"
            elapsed = int(time.monotonic() - t0)
            self._publish_status(
                "planning",
                f"Waiting for object (cam or pinch)… {elapsed}s / {int(wait)}s",
            )
            ok, why = self._sweep_gripper(open_j, mid_j, pulse, gen, True)
            if not ok:
                return False, "cancelled"
            if why:
                return True, why
            ok, why = self._sweep_gripper(mid_j, open_j, pulse, gen, True)
            if not ok:
                return False, "cancelled"
            if why:
                return True, why
            hold_t0 = time.monotonic()
            hits = 0
            while time.monotonic() - hold_t0 < hold:
                if self._is_cancelled(gen):
                    return False, "cancelled"
                self._publish_gripper_joint(open_j, n=1)
                if self._jaws_see_object():
                    hits += 1
                    if hits >= max(2, int(self.handoff_center_frames)):
                        return True, "vision"
                else:
                    hits = 0
                time.sleep(0.05)
        return False, "timeout"

    def _hand_off_act_close(self, xyz: Tuple[float, float, float]) -> None:
        """Arm is at the click. ACT closes; it does not move to the demo park."""
        xy = [float(xyz[0]), float(xyz[1])]
        self._cached_pole_xy = (xy[0], xy[1])
        self._pole_pub.publish(
            String(
                data=json.dumps(
                    {"x": xy[0], "y": xy[1], "z": float(xyz[2]), "source": "click"}
                )
            )
        )
        time.sleep(0.05)
        self._act_handed = True
        self._il_cmd_pub.publish(
            String(
                data=json.dumps(
                    {"cmd": "play", "pole_xy": xy, "from": "click"}
                )
            )
        )
        self.get_logger().info(
            f"click handoff → ACT close pole_xy=({xy[0]:.3f}, {xy[1]:.3f})"
        )
        self._publish_status(
            "grasping",
            f"At click — ACT closing @ ({xy[0]:.3f}, {xy[1]:.3f})",
            xyz,
        )

    def _finish_run(self, gen: int):
        self._hold_gripper_open = False
        self._wide_sub_stop()
        with self._lock:
            if gen == self._run_gen:
                self._busy = False
            self._active_goal = None

    def _run_handoff_grab(self, gen: int):
        try:
            if self._is_cancelled(gen):
                return
            if not self._bus_live():
                self._publish_status(
                    "error",
                    "Arm bus not live — check 12V + U2D2, restart Nano xsarm_control",
                )
                return
            self._wide_sub_start()
            self._publish_status(
                "planning", "Grab: holding pose, opening gripper…"
            )
            if not self._gripper_actuate(
                self.grip_open, max(0.6, self.grip_settle * 0.6), gen, hold=True
            ):
                self._publish_status("cancelled", "Cancelled")
                return
            if not self._sleep_interruptible(0.35, gen):
                self._publish_status("cancelled", "Cancelled")
                return
            if not self._capture_jaw_baseline():
                # One more beat for the first JPEG from the relay.
                if not self._sleep_interruptible(0.4, gen):
                    self._publish_status("cancelled", "Cancelled")
                    return
                self._capture_jaw_baseline()
            idle = self._read_gripper_load()
            self._grip_load_idle = int(idle or 0)
            self.get_logger().info(f"handoff idle gripper load={self._grip_load_idle}")

            deadline = time.monotonic() + max(8.0, float(self.handoff_wait_s))
            grabbed = False
            why = ""
            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                ok, reason = self._wait_object_centered(gen, wait_s=remaining)
                if not ok:
                    if reason == "cancelled":
                        self._publish_status("cancelled", "Cancelled")
                        return
                    break
                label = "pinch" if reason == "load" else "camera"
                self._publish_status("grasping", f"Object sensed ({label}) — closing…")
                closed, why = self._gripper_close_until_stall(gen)
                if self._is_cancelled(gen):
                    if closed:
                        self._hold_gripper_present()
                    self._publish_status("cancelled", "Cancelled")
                    return
                if not closed:
                    self._publish_status("error", "Gripper did not close on the object")
                    return
                load = self._read_gripper_load()
                if why == "load" or self._grab_feels_object(load):
                    grabbed = True
                    break
                self._publish_status(
                    "planning",
                    "Close felt empty — opening, still looking…",
                )
                if not self._gripper_actuate(
                    self.grip_open, 0.5, gen, hold=True
                ):
                    self._publish_status("cancelled", "Cancelled")
                    return
                self._capture_jaw_baseline()

            if not grabbed:
                self._publish_status(
                    "error",
                    "No grab — hold the object in the jaws until it pinches or the wide cam sees it",
                )
                return
            self._hold_gripper_present()
            self._publish_status(
                "done", f"Object grabbed ({why or 'load'}) — arm held"
            )
        except Exception as e:
            self.get_logger().error(f"handoff grab exception: {e}")
            self._publish_status("error", f"Grab exception: {e}")
        finally:
            self._finish_run(gen)

    def _run_handoff_release(self, gen: int):
        try:
            if self._is_cancelled(gen):
                return
            if not self._bus_live():
                self._publish_status(
                    "error",
                    "Arm bus not live — check 12V + U2D2, restart Nano xsarm_control",
                )
                return
            if self._gripper_joint is not None:
                self._last_grip_joint = float(self._gripper_joint)
                self._hold_gripper_present(extra=0.0)
            self._publish_status("planning", "Release: moving to Home…")
            if not self._goto_named("home", gen):
                self._publish_status("cancelled", "Cancelled during Home")
                return
            if self._last_grip_joint is not None:
                self._publish_gripper_joint(self._last_grip_joint, n=2)
            self._publish_status("planning", "Release: opening gripper…")
            if not self._gripper_actuate(
                self.grip_open, max(0.7, self.grip_settle * 0.7), gen, hold=True
            ):
                self._publish_status("cancelled", "Cancelled")
                return
            self._publish_status("done", "Object released — gripper open at Home")
        except Exception as e:
            self.get_logger().error(f"handoff release exception: {e}")
            self._publish_status("error", f"Release exception: {e}")
        finally:
            self._finish_run(gen)

    def _jaws_in_foam_band(self) -> bool:
        """Past halfway toward closed, and short of the empty hard-stop."""
        gj = self._gripper_joint
        if gj is None:
            return False
        empty = float(self.grip_joint_closed) + float(self.grip_empty_closed)
        mid = 0.45 * (
            float(self.grip_joint_open) + float(self.grip_joint_closed)
        )
        return empty < float(gj) <= mid

    def _still_holding(self) -> Tuple[bool, str]:
        """Recheck after a short lift. Load or a foam-width stall both count."""
        load = self._read_gripper_load()
        gj = self._gripper_joint
        self.get_logger().info(f"lift recheck load={load} joint={gj}")
        if self._grab_feels_object(load):
            return True, "load"
        if self._jaws_in_foam_band():
            return True, "foam"
        return False, "lost"

    def _retry_forward(self, gen: int, toward_xyz, q_seed) -> Optional[List[float]]:
        """Open, step about 1 cm toward the click, return the new joints."""
        step_m = float(self.grasp_retry_forward_m)
        if step_m < 1e-4:
            return None
        ee = self._ee_xyz()
        if ee is None:
            self.get_logger().warn("1 cm retry skipped — no EE pose")
            return None
        dx = float(toward_xyz[0]) - float(ee[0])
        dy = float(toward_xyz[1]) - float(ee[1])
        h = math.hypot(dx, dy)
        if h < 1e-3:
            hx, hy = self._approach_xy_hat(toward_xyz)
            dx, dy, h = hx, hy, 1.0
        step = min(step_m, h)
        nxt = (
            float(ee[0]) + (dx / h) * step,
            float(ee[1]) + (dy / h) * step,
            float(ee[2]),
        )
        self.get_logger().info(
            f"grasp retry +{step * 100:.1f} cm toward pole "
            f"ee={tuple(round(float(v), 3) for v in ee)} "
            f"-> {tuple(round(v, 3) for v in nxt)}"
        )
        self._publish_status(
            "grasping", f"Retry +{step * 100:.0f} cm toward the pole…", nxt
        )
        if not self._gripper_reopen(gen):
            return None
        pose = self._pose_at(nxt, 0.0)
        q = self._solve_ik(pose, gen, nearby=True, prefer_seed=q_seed, fast=True)
        if q is None or self._is_cancelled(gen):
            self.get_logger().warn("1 cm retry IK failed — staying at the reached pose")
            return None
        q = self._apply_align_joints(q)
        if not self._send_trajectory([q], gen):
            return None
        if not self._sleep_interruptible(0.25, gen):
            return None
        return q

    def _staged_retreat(self, gen: int, hold_xyz, q_hold) -> str:
        """Lift a few cm, recheck, then finish the retreat or lower and open.

        Returns held, dropped, cancelled, or lift_failed.
        """
        check_z = max(0.01, float(self.grasp_lift_check_z))
        self._publish_status(
            "grasping", f"Lift check ({check_z * 100:.0f} cm)…", hold_xyz
        )
        q_check = self._solve_ik(
            self._pose_at(hold_xyz, check_z),
            gen,
            nearby=True,
            prefer_seed=q_hold,
            fast=True,
        )
        if q_check is None or self._is_cancelled(gen):
            self.get_logger().warn("lift-check IK failed")
            return "cancelled" if self._is_cancelled(gen) else "lift_failed"
        q_check = self._apply_align_joints(q_check)
        if not self._send_trajectory([q_check], gen):
            return "cancelled" if self._is_cancelled(gen) else "lift_failed"
        if not self._sleep_interruptible(0.30, gen):
            return "cancelled"
        held, why = self._still_holding()
        if not held:
            self.get_logger().info(
                f"lift recheck lost the object ({why}) — lowering and opening"
            )
            self._publish_status(
                "grasping", "Object slipped — lowering and opening", hold_xyz
            )
            self._send_trajectory([q_hold], gen)
            self._gripper_reopen(gen)
            return "dropped"
        self._publish_status(
            "grasping", f"Still holding ({why}) — finishing lift", hold_xyz
        )
        q_retreat = self._solve_ik(
            self._pose_at(hold_xyz, self.retreat_z),
            gen,
            nearby=True,
            prefer_seed=q_check,
            fast=True,
        )
        if self._is_cancelled(gen):
            return "cancelled"
        if q_retreat is None:
            self.get_logger().warn("retreat IK failed — holding the lift-check pose")
            return "held"
        q_retreat = self._apply_align_joints(q_retreat)
        if not self._send_trajectory([q_retreat], gen):
            return "cancelled" if self._is_cancelled(gen) else "lift_failed"
        return "held"

    def _scripted_pinch_and_lift(
        self, gen: int, q_hold, hold_xyz, toward_xyz, click_xyz
    ):
        """Close at the pose already reached. One 1 cm retry on a miss."""
        self._hold_gripper_open = False
        self._publish_status(
            "grasping", "Closing at the reached pose…", click_xyz
        )
        ok_close, close_why = self._gripper_close_until_stall(gen)
        if not ok_close:
            self._publish_status("cancelled", "Cancelled after close", hold_xyz)
            return
        if close_why not in ("load", "foam"):
            q2 = self._retry_forward(gen, toward_xyz, q_hold)
            if q2 is None:
                if self._is_cancelled(gen):
                    self._publish_status("cancelled", "Cancelled", hold_xyz)
                    return
            else:
                q_hold = q2
                ee2 = self._ee_xyz()
                if ee2 is not None:
                    hold_xyz = ee2
                ok_close, close_why = self._gripper_close_until_stall(gen)
                if not ok_close:
                    self._publish_status(
                        "cancelled", "Cancelled after close", hold_xyz
                    )
                    return
        if close_why in ("load", "foam"):
            # A held pole is the sample. Do not lift, and do not fail if a
            # later lift would be blocked by the shoulder limit.
            self._publish_status(
                "done",
                f"Grasp complete ({close_why}) — held @ "
                f"({hold_xyz[0]:.3f}, {hold_xyz[1]:.3f}, {hold_xyz[2]:.3f})",
                hold_xyz,
            )
            return
        self.get_logger().info(
            f"grasp miss ({close_why}) — holding commanded grasp "
            f"q={[round(float(v), 3) for v in q_hold]} (no retreat)"
        )
        self._publish_status(
            "grasping",
            f"No gripper resistance ({close_why}) — holding pose",
            hold_xyz,
        )
        self._gripper_actuate(self.grip_open, 0.35, gen, hold=True)
        try:
            msg = JointGroupCommand()
            msg.name = "arm"
            msg.cmd = [float(v) for v in q_hold]
            for _ in range(4):
                self._arm_pub.publish(msg)
                time.sleep(0.05)
        except Exception:
            pass
        if self._is_cancelled(gen):
            self._publish_status("cancelled", "Cancelled", hold_xyz)
            return
        self._publish_status(
            "error",
            "Miss — arm left at grasp pose (reposition or click again)",
            hold_xyz,
        )

    def _backup_joints(self) -> List[float]:
        """Joints that pull the reach back without folding toward the stack.

        A forward grasp sits at a positive shoulder. Backing away decreases
        that shoulder toward the pre-grasp, and at least ~0.35 rad, but not
        below 0. That is still far from the -1.9 fold.
        """
        cur = [float(self._joint_pos.get(j, 0.0)) for j in self.arm_joints]
        saved = (
            [float(v) for v in self._retreat_q]
            if self._retreat_q is not None and len(self._retreat_q) == len(cur)
            else list(cur)
        )
        target = list(saved)
        ish = self._joint_index("shoulder")
        if ish < 0:
            return target
        cur_sh = float(cur[ish])
        if cur_sh <= 0.05:
            target[ish] = cur_sh
            return target
        back_sh = min(float(saved[ish]), cur_sh - 0.35)
        back_sh = max(0.0, float(self.shoulder_min_grasp), back_sh)
        back_sh = min(cur_sh, back_sh)
        back_sh = max(cur_sh - 0.50, back_sh)
        target[ish] = back_sh
        if self._retreat_q is None:
            for i in range(len(target)):
                if i != ish:
                    target[i] = cur[i]
        return target

    def _run_retreat(
        self, gen: int, return_q: Optional[List[float]] = None
    ):
        """Open, then back away or return to the auto-take starting joints."""
        try:
            if self._is_cancelled(gen):
                return
            self._shoulder_forward_only = False
            self._lock_pan()
            self._publish_status("grasping", "Opening gripper…")
            if not self._gripper_reopen(gen):
                self._publish_status(
                    "cancelled" if self._is_cancelled(gen) else "error",
                    "Could not open the gripper",
                )
                return
            target = (
                [float(v) for v in return_q]
                if return_q is not None
                else self._backup_joints()
            )
            cur = [float(self._joint_pos.get(j, 0.0)) for j in self.arm_joints]
            ish = self._joint_index("shoulder")
            delta = (
                float(target[ish]) - float(cur[ish]) if ish >= 0 else 0.0
            )
            self.get_logger().info(
                f"retreat shoulder {math.degrees(cur[ish]) if ish >= 0 else 0:.1f}° "
                f"-> {math.degrees(target[ish]) if ish >= 0 else 0:.1f}° "
                f"({math.degrees(delta):.1f}°)"
            )
            max_delta = max(
                (abs(float(a) - float(b)) for a, b in zip(target, cur)),
                default=0.0,
            )
            already_back = (
                max_delta < 0.04
                if return_q is not None
                else abs(delta) < 0.08
            )
            if already_back:
                self._publish_status(
                    "retreated",
                    (
                        "Gripper open — already at take start"
                        if return_q is not None
                        else "Gripper open — shoulder already back"
                    ),
                )
                return
            self._publish_status(
                "grasping",
                (
                    "Returning to take start…"
                    if return_q is not None
                    else "Backing away…"
                ),
            )
            old = self.seg_dur
            self.seg_dur = 3.0 if return_q is not None else 2.0
            try:
                ok = self._send_trajectory([target], gen)
            finally:
                self.seg_dur = old
            if self._is_cancelled(gen):
                self._publish_status("cancelled", "Cancelled")
                return
            if not ok:
                self._publish_status("error", "Backup motion failed")
                return
            self._publish_status(
                "retreated",
                "Returned to take start" if return_q is not None else "Backed away",
            )
        except Exception as e:
            self.get_logger().error(f"retreat exception: {e}")
            self._publish_status("error", f"Retreat exception: {e}")
        finally:
            self._shoulder_forward_only = False
            self._hold_gripper_open = False
            self._restore_click_anchor()
            with self._lock:
                if gen == self._run_gen:
                    self._busy = False
                self._active_goal = None

    def _run_release(self, gen: int):
        """Open the jaws where they are. The arm is not moved."""
        try:
            if self._is_cancelled(gen):
                return
            self._publish_status("grasping", "Opening gripper…")
            if not self._gripper_reopen(gen):
                self._publish_status(
                    "cancelled" if self._is_cancelled(gen) else "error",
                    "Could not open the gripper",
                )
                return
            try:
                msg = JointGroupCommand()
                msg.name = "arm"
                msg.cmd = [float(self._joint_pos.get(j, 0.0)) for j in self.arm_joints]
                for _ in range(3):
                    self._arm_pub.publish(msg)
                    time.sleep(0.04)
            except Exception:
                pass
            self._publish_status("released", "Gripper open")
        except Exception as e:
            self.get_logger().error(f"release exception: {e}")
            self._publish_status("error", f"Release exception: {e}")
        finally:
            self._hold_gripper_open = False
            with self._lock:
                if gen == self._run_gen:
                    self._busy = False
                self._active_goal = None

    def _run_set_down(self, gen: int):
        """Lower onto the original click and open. Not part of the recording.

        Opens only if the arm actually reached the click height. A shoulder
        clamp that blocks the lower leaves the pole held.
        """
        try:
            if self._is_cancelled(gen):
                return
            anchor = self._click_anchor or self._last_xyz
            if anchor is None:
                self._publish_status("error", "No click to set the pole down on")
                return
            if not self._bus_live():
                self._publish_status(
                    "error",
                    "Arm bus not live — pole still held",
                    anchor,
                )
                return
            self._shoulder_forward_only = True
            self._lock_pan()
            target = self._ee_for_bar_contact(tuple(anchor))
            self._publish_status("grasping", "Setting the pole down…", anchor)
            q = self._solve_ik(
                self._pose_at(target, 0.0),
                gen,
                nearby=True,
                fast=True,
            )
            if q is None or self._is_cancelled(gen):
                self._publish_status(
                    "cancelled" if self._is_cancelled(gen) else "error",
                    "Set-down IK failed — pole still held",
                    anchor,
                )
                return
            q = self._apply_align_joints(q)
            if not self._send_trajectory([q], gen):
                self._publish_status(
                    "cancelled" if self._is_cancelled(gen) else "error",
                    "Set-down motion failed — pole still held",
                    anchor,
                )
                return
            if not self._sleep_interruptible(0.25, gen):
                self._publish_status("cancelled", "Cancelled", anchor)
                return
            ee = self._ee_xyz()
            if ee is None or abs(float(ee[2]) - float(anchor[2])) > 0.03:
                self.get_logger().warn(
                    f"set-down stopped short ee={ee} anchor_z={float(anchor[2]):.3f}"
                )
                self._publish_status(
                    "error",
                    "Set-down stopped short — pole still held. Shoulder was not folded.",
                    anchor,
                )
                return
            if not self._gripper_reopen(gen):
                self._publish_status(
                    "cancelled" if self._is_cancelled(gen) else "error",
                    "Set-down could not open — pole may still be held",
                    anchor,
                )
                return
            self._publish_status("placed", "Pole set down at the click", anchor)
        except Exception as e:
            self.get_logger().error(f"set-down exception: {e}")
            self._publish_status("error", f"Set-down exception: {e}")
        finally:
            self._shoulder_forward_only = False
            self._hold_gripper_open = False
            self._restore_click_anchor()
            with self._lock:
                if gen == self._run_gen:
                    self._busy = False
                self._active_goal = None

    # ── state machine ─────────────────────────────────────────────────────
    def _run_grasp(self, gen: int):
        try:
            if self._is_cancelled(gen):
                return
            xyz = self._click_anchor or self._last_xyz
            if xyz is None:
                self._publish_status("error", "No target point")
                return
            xyz = self._apply_auto_offset(tuple(xyz))

            self._lock_pan()
            # Block shoulder lift/back toward AGX + pan servo for this grasp.
            self._shoulder_forward_only = True
            if self._act_handed:
                self._act_handed = False
                self._il_cmd_pub.publish(String(data=json.dumps({"cmd": "stop"})))
                time.sleep(0.15)
            if not self._bus_live():
                self._publish_status(
                    "error",
                    "Arm bus not live (joint_states are -pi) — check 12V + U2D2, "
                    "restart Nano xsarm_control",
                    xyz,
                )
                return
            self._publish_status("planning", "Locking pan + opening gripper…", xyz)
            # Hold open PWM through approach — coasting to 0 lets jaws shut early.
            self._hold_gripper_open = True
            if not self._gripper_actuate(
                self.grip_open, self.grip_settle, gen, hold=True
            ):
                self._hold_gripper_open = False
                self._publish_status("cancelled", "Cancelled", xyz)
                return

            self._waist_face = None
            click_xyz = tuple(self._viz_xyz or xyz)
            # Overlay stays on the depth cube. IK puts the gripper bar on
            # the click at click height, approaching along the table heading.
            ik_xyz = self._reachable_grasp_xyz(click_xyz)
            horiz0 = math.hypot(ik_xyz[0], ik_xyz[1])
            contact_h = math.hypot(float(click_xyz[0]), float(click_xyz[1]))
            standoff = float(self.grasp_standoff_m)
            if self._object_low:
                # The bar offset already sets the end effector behind the block.
                # Only the block itself, not that offset, is "too close".
                if contact_h < float(self.floor_min_ee_reach):
                    self._hold_gripper_open = False
                    self._publish_status(
                        "unreachable",
                        f"Block is too close to the base (reach {contact_h:.2f}m)",
                        xyz,
                    )
                    return
                room = horiz0 - float(self.floor_min_ee_reach)
                standoff = min(standoff, max(0.0, room))
            pre_xyz = self._standoff_xyz(ik_xyz, standoff)
            # Extra hover Z shrinks the 4-DOF envelope — only use it up close.
            if (
                not self._object_low
                and self.approach_z > 1e-4
                and horiz0 < 0.30
            ):
                pre_xyz = (
                    pre_xyz[0],
                    pre_xyz[1],
                    pre_xyz[2] + float(self.approach_z),
                )
            deep_xyz = ik_xyz
            self._wrist_face = self._wrist_from_face_blue()
            if not self._align_waist_to_target(ik_xyz, gen):
                self._hold_gripper_open = False
                self._waist_face = None
                self._publish_status("cancelled", "Cancelled during waist align", xyz)
                return

            horiz = math.hypot(ik_xyz[0], ik_xyz[1])

            self._publish_status(
                "planning",
                f"Solving IK, bar→click (pre-grasp) reach={horiz:.2f}m…",
                ik_xyz,
            )
            q_pre, pre_xyz, ik_xyz = self._solve_pregrasp_ik(pre_xyz, ik_xyz, gen)
            deep_xyz = ik_xyz
            if self._is_cancelled(gen):
                self._hold_gripper_open = False
                self._publish_status("cancelled", "Cancelled", xyz)
                return
            if q_pre is None:
                self._hold_gripper_open = False
                obj_h = math.hypot(float(click_xyz[0]), float(click_xyz[1]))
                if obj_h < float(self.floor_min_ee_reach):
                    why = "block is too close to the base"
                else:
                    why = "drive closer or click lower in the image"
                self._publish_status(
                    "unreachable",
                    f"IK failed at pre-grasp (reach {horiz:.2f}m) — {why}",
                    xyz,
                )
                return
            q_pre = self._apply_align_joints(q_pre)
            iw = self._joint_index("waist")
            if iw >= 0:
                self._waist_face = float(q_pre[iw])

            self._publish_status(
                "planning", "Solving IK (gripper bar on click)…", ik_xyz
            )
            q_grasp, grasp_xyz = self._solve_grasp_contact_ik(
                deep_xyz, pre_xyz, gen, prefer_seed=q_pre
            )
            if self._is_cancelled(gen):
                self._hold_gripper_open = False
                self._publish_status("cancelled", "Cancelled", xyz)
                return
            contact_goal = tuple(deep_xyz)
            short_of_contact = False
            if q_grasp is None:
                # Contact IK can fail after the jaws are already on the pole.
                # Close at the approach pose instead of pushing farther.
                self.get_logger().warn(
                    "grasp contact IK failed — closing from the approach pose"
                )
                self._publish_status(
                    "grasping", "Moving to pre-grasp (open)…", xyz
                )
                if not self._send_trajectory([q_pre], gen):
                    self._hold_gripper_open = False
                    self._publish_status(
                        "error" if not self._is_cancelled(gen) else "cancelled",
                        "Motion to pre-grasp failed",
                        xyz,
                    )
                    return
                self._retreat_q = [float(v) for v in q_pre]
                if not self._sleep_interruptible(0.35, gen):
                    self._hold_gripper_open = False
                    self._publish_status("cancelled", "Cancelled", xyz)
                    return
                if self._is_cancelled(gen):
                    self._hold_gripper_open = False
                    self._publish_status("cancelled", "Cancelled", click_xyz)
                    return
                if self._act_closes:
                    self._hold_gripper_open = False
                    self._hand_off_act_close(tuple(click_xyz))
                    return
                self._scripted_pinch_and_lift(
                    gen,
                    q_pre,
                    tuple(pre_xyz),
                    tuple(deep_xyz),
                    tuple(click_xyz),
                )
                return
            q_grasp = self._apply_align_joints(q_grasp)
            # How far short of the intended bar-on-click point?
            short_m = math.sqrt(
                (float(grasp_xyz[0]) - float(contact_goal[0])) ** 2
                + (float(grasp_xyz[1]) - float(contact_goal[1])) ** 2
                + (float(grasp_xyz[2]) - float(contact_goal[2])) ** 2
            )
            short_of_contact = short_m > 0.035
            deep_xyz = grasp_xyz
            ik_xyz = grasp_xyz

            wr_txt = ""
            if self._wrist_face is not None:
                wr_txt = f" wr={math.degrees(self._wrist_face):.0f}°"
            self._publish_status(
                "grasping", f"Moving to pre-grasp (open){wr_txt}…", xyz
            )
            if not self._send_trajectory([q_pre], gen):
                self._hold_gripper_open = False
                self._publish_status(
                    "error" if not self._is_cancelled(gen) else "cancelled",
                    "Motion to pre-grasp failed",
                    xyz,
                )
                return
            self._retreat_q = [float(v) for v in q_pre]

            self._publish_status("grasping", "Approaching grasp (open)…", xyz)
            if not self._send_trajectory([q_grasp], gen):
                self._hold_gripper_open = False
                self._publish_status(
                    "error" if not self._is_cancelled(gen) else "cancelled",
                    "Approach failed",
                    xyz,
                )
                return

            # Let the arm settle before pinch — closing mid-arrival looks early.
            if not self._sleep_interruptible(0.35, gen):
                self._hold_gripper_open = False
                self._publish_status("cancelled", "Cancelled", xyz)
                return

            if short_of_contact:
                who = "ACT" if self._act_closes else "scripted close"
                self.get_logger().info(
                    f"grasp {short_m * 100:.1f} cm short of the click point — "
                    f"{who} closes from the reached pose"
                )

            if self._is_cancelled(gen):
                self._hold_gripper_open = False
                self._publish_status("cancelled", "Cancelled", click_xyz)
                return
            if self._act_closes:
                self._hold_gripper_open = False
                self._hand_off_act_close(tuple(click_xyz))
                return
            self._scripted_pinch_and_lift(
                gen,
                q_grasp,
                tuple(ik_xyz),
                tuple(contact_goal),
                tuple(click_xyz),
            )
        except Exception as e:
            self.get_logger().error(f"grasp exception: {e}")
            self._publish_status("error", f"Grasp exception: {e}")
        finally:
            self._shoulder_forward_only = False
            self._hold_gripper_open = False
            self._wide_sub_stop()
            self._restore_click_anchor()
            with self._lock:
                if gen == self._run_gen:
                    self._busy = False
                self._active_goal = None


def main(args=None):
    rclpy.init(args=args)
    node = GraspFromClick()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
