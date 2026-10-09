#!/usr/bin/env python3
"""Teach-tip calibration: free arm → save contact pose → sleep→taught → pan sweep.

No typed distance. Truth = saved taught tip (not live TF — servos sag under load).
Camera estimate = project taught tip into D435i, then depth (preferred) or table-plane.

Commands on /grab_it/cal_command (JSON string):
  {"cmd":"save_pose"}              # torque must already be ON (Hold arm)
  {"cmd":"goto_taught"}            # sleep then taught joints
  {"cmd":"goto_taught_only"}
  {"cmd":"run_sweep","n":8}        # N pan samples between pan_start..pan_end
  {"cmd":"abort"}

Status on /grab_it/cal_status (JSON).

Usage (Isaac container, bridge profile):
  export FASTRTPS_DEFAULT_PROFILES_FILE=/workspaces/isaac_ros-dev/bridge_both.xml
  python3 /workspaces/isaac_ros-dev/scripts/calibration/teach_cal_node.py
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rclpy
import yaml
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

try:
    from interbotix_xs_msgs.msg import JointGroupCommand, JointSingleCommand
except ImportError as e:
    raise SystemExit(
        "interbotix_xs_msgs missing — source the workspace first"
    ) from e

_DEPTH_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
)

DEFAULT_OPTICAL = [
    "d435i_color_optical_frame",
    "px150_no_elbow/d435i_color_optical_frame",
    "camera_color_optical_frame",
    "d435i_depth_optical_frame",
]

WS = Path(os.environ.get("GRAB_IT_WS", "/workspaces/isaac_ros-dev"))
DEFAULT_POSE = WS / "config" / "calibration" / "taught_contact.yaml"
DEFAULT_OUT = WS / "config" / "calibration" / "teach_cal.jsonl"
DEFAULT_GRASP = WS / "config" / "grasp_params.yaml"


def _quat_to_mat(x: float, y: float, z: float, w: float) -> np.ndarray:
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class TeachCalNode(Node):
    def __init__(self):
        super().__init__("teach_cal")
        self._cg = ReentrantCallbackGroup()
        self._lock = threading.Lock()
        self._busy = False
        self._cancel = False
        self._run_gen = 0

        cfg = self._load_grasp_cfg()
        cal = cfg.get("teach_cal", {}) if isinstance(cfg.get("teach_cal"), dict) else {}
        g = cfg.get("grasp", cfg)

        self.planning_frame = str(
            cal.get("planning_frame", g.get("planning_frame", "px150_no_elbow/base_link"))
        )
        self.ee_link = str(
            cal.get("ee_link", g.get("ee_link", "px150_no_elbow/ee_gripper_link"))
        )
        self.arm_joints = list(
            cal.get("arm_joints", g.get("arm_joints", ["waist", "shoulder", "wrist_angle", "wrist_rotate"]))
        )
        self.sleep_joints = [float(v) for v in (
            cal.get("sleep_joints") or g.get("sleep_joints") or [0.0, -0.2321, -1.7279, 0.0]
        )]
        self.bar_offset_x = float(
            cal.get("bar_offset_x", g.get("gripper_bar_offset_x", 0.045))
        )
        if abs(self.bar_offset_x) < 1e-6:
            # Match harness default when yaml still has 0.0 for EE-at-bar.
            self.bar_offset_x = float(cal.get("truth_offset_x", 0.045))
        self.table_z = float(cal.get("table_z", g.get("table_z", 0.04)))
        self.object_height = float(
            cal.get("object_height", g.get("object_height", 0.07))
        )
        self.depth_topic = str(
            cal.get(
                "depth_topic",
                g.get("depth_topic", "/d435i/d435i/aligned_depth_to_color/image_raw"),
            )
        )
        self.info_topic = str(
            cal.get(
                "camera_info_topic",
                g.get("camera_info_topic", "/d435i/d435i/color/camera_info"),
            )
        )
        self.js_topic = str(
            cal.get("joint_states_topic", "/px150_no_elbow/joint_states")
        )
        self.arm_cmd_topic = str(
            cal.get("arm_command_topic", "/px150_no_elbow/commands/joint_group")
        )
        self.pan_cmd_topic = str(
            cal.get("pan_command_topic", "/px150_no_elbow/commands/joint_single")
        )
        self.pan_start = float(cal.get("pan_start_rad", 0.0))
        self.pan_end = float(cal.get("pan_end_rad", 0.70))
        self.settle_s = float(cal.get("settle_s", 1.5))
        self.seg_dur = float(cal.get("segment_duration_s", 2.5))
        self.min_depth = float(cal.get("min_depth_m", g.get("min_depth_m", 0.12)))
        self.max_depth = float(cal.get("max_depth_m", g.get("max_depth_m", 1.20)))
        self.depth_window = int(cal.get("depth_window", g.get("depth_window", 7)))
        self.n_default = int(cal.get("n_default", 8))
        self.pose_path = Path(str(cal.get("pose_path", DEFAULT_POSE)))
        self.out_path = Path(str(cal.get("out_jsonl", DEFAULT_OUT)))
        # Truth + UV projection use the saved taught tip, not live TF (servos sag).
        self.truth_use_taught = bool(cal.get("truth_use_taught", True))
        # New pan sweep replaces jsonl (old file → .bak) so runs never append.
        self.fresh_jsonl_each_sweep = bool(cal.get("fresh_jsonl_each_sweep", True))
        # Re-command taught joints before each pan sample to limit sag drift.
        self.regoto_taught_each_sample = bool(
            cal.get("regoto_taught_each_sample", False)
        )
        self.tip_sag_warn_m = float(cal.get("tip_sag_warn_m", 0.015))
        self.tip_height_warn_m = float(cal.get("tip_height_warn_m", 0.05))
        self.optical_cal_path = Path(
            str(cal.get("optical_cal_path", WS / "config/calibration/d435i_optical_cal.yaml"))
        )
        self.optical_cal_trial_path = Path(
            str(cal.get(
                "optical_cal_trial_path",
                WS / "config/calibration/d435i_optical_cal_trial.yaml",
            ))
        )

        self._joint_pos = {j: 0.0 for j in self.arm_joints}
        self._pan: Optional[float] = None
        self._info: Optional[CameraInfo] = None
        self._taught: Optional[dict] = None
        self._cal_frame_ok: Optional[bool] = None

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.create_subscription(
            JointState, self.js_topic, self._js_cb, qos_profile_sensor_data,
            callback_group=self._cg,
        )
        self.create_subscription(
            CameraInfo, self.info_topic, self._info_cb, _DEPTH_QOS,
            callback_group=self._cg,
        )
        self.create_subscription(
            String, "/grab_it/cal_command", self._cmd_cb, 10,
            callback_group=self._cg,
        )
        self._arm_pub = self.create_publisher(JointGroupCommand, self.arm_cmd_topic, 10)
        self._pan_pub = self.create_publisher(JointSingleCommand, self.pan_cmd_topic, 10)
        self._status_pub = self.create_publisher(String, "/grab_it/cal_status", 10)

        self._load_taught_file()
        self._publish_status(
            "idle",
            "Teach-cal ready — Free arm, touch tip, Hold, then Save tip pose",
        )
        self.get_logger().info(
            f"teach_cal ready pose={self.pose_path} out={self.out_path} "
            f"pan=[{self.pan_start:.2f},{self.pan_end:.2f}] "
            f"bar_offset_x={self.bar_offset_x:.3f}"
        )

    def _load_grasp_cfg(self) -> dict:
        path = Path(os.environ.get("GRAB_IT_GRASP_PARAMS", str(DEFAULT_GRASP)))
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

    def _js_cb(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            if abs(float(pos) + math.pi) <= 0.05:
                continue
            if name in self._joint_pos:
                self._joint_pos[name] = float(pos)
            if name == "pan":
                self._pan = float(pos)

    def _info_cb(self, msg: CameraInfo):
        with self._lock:
            self._info = msg

    def _publish_status(self, status: str, message: str, **extra):
        payload = {"status": status, "message": message, **extra}
        if self._taught is not None:
            payload["has_taught"] = True
            payload["taught_name"] = self._taught.get("name", "taught_contact")
        else:
            payload["has_taught"] = False
        self._status_pub.publish(String(data=json.dumps(payload)))

    def _cmd_cb(self, msg: String):
        try:
            data = json.loads(msg.data)
        except Exception:
            data = {"cmd": str(msg.data).strip()}
        cmd = str(data.get("cmd") or data.get("command") or "").strip().lower()
        if cmd in ("abort", "cancel"):
            self._cancel = True
            self._run_gen += 1
            self._publish_status("cancelled", "Cal aborted")
            return
        if self._busy:
            self._publish_status("busy", "Teach-cal busy — Abort first")
            return
        if cmd == "save_pose":
            threading.Thread(target=self._save_pose, daemon=True).start()
        elif cmd in ("goto_taught", "goto"):
            n = data.get("n")
            threading.Thread(
                target=self._run_goto_taught, args=(True,), daemon=True
            ).start()
        elif cmd == "goto_taught_only":
            threading.Thread(
                target=self._run_goto_taught, args=(False,), daemon=True
            ).start()
        elif cmd in ("run_sweep", "sweep", "run"):
            try:
                n = int(data.get("n", self.n_default))
            except (TypeError, ValueError):
                n = self.n_default
            n = max(2, min(40, n))
            threading.Thread(target=self._run_sweep, args=(n,), daemon=True).start()
        else:
            self._publish_status("error", f"Unknown cal cmd: {cmd}")

    # ── TF / depth helpers ────────────────────────────────────────────────
    def _tf_matrix(self, target: str, source: str, timeout_s: float = 0.8
                   ) -> Optional[np.ndarray]:
        try:
            tf = self._tf_buffer.lookup_transform(
                target, source, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=timeout_s),
            )
        except Exception:
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = _quat_to_mat(q.x, q.y, q.z, q.w)
        T[:3, 3] = [t.x, t.y, t.z]
        return T

    def _ee_xyz(self) -> Optional[Tuple[float, float, float]]:
        T = self._tf_matrix(self.planning_frame, self.ee_link)
        if T is None:
            return None
        return float(T[0, 3]), float(T[1, 3]), float(T[2, 3])

    def _tip_xyz(self) -> Optional[Tuple[float, float, float]]:
        """Contact point = EE origin + bar_offset along gripper +X in base."""
        T = self._tf_matrix(self.planning_frame, self.ee_link)
        if T is None:
            return None
        origin = T[:3, 3]
        x_axis = T[:3, 0]
        tip = origin + float(self.bar_offset_x) * x_axis
        return float(tip[0]), float(tip[1]), float(tip[2])

    def _load_optical_cal(self) -> Optional[dict]:
        for p in (self.optical_cal_trial_path, self.optical_cal_path):
            if not p.is_file():
                continue
            try:
                with open(p, "r", encoding="utf-8") as f:
                    d = yaml.safe_load(f) or {}
                if d.get("parent_frame") and d.get("child_frame"):
                    return d
            except Exception as e:
                self.get_logger().warn(f"optical cal load {p}: {e}")
        return None

    def _pick_optical(self) -> Optional[Tuple[str, np.ndarray]]:
        cal = self._load_optical_cal()
        if cal is not None:
            cf = str(cal["child_frame"])
            prefix = ""
            if "/" in self.planning_frame:
                prefix = self.planning_frame.rsplit("/", 1)[0] + "/"
            cf_candidates = [cf]
            if prefix and not cf.startswith(prefix):
                cf_candidates.insert(0, prefix + cf)
            if self._cal_frame_ok is None:
                self._cal_frame_ok = False
                for cf_try in cf_candidates:
                    if self._tf_matrix(self.planning_frame, cf_try, timeout_s=0.8) is not None:
                        self._cal_frame_ok = cf_try
                        break
                if not self._cal_frame_ok:
                    self.get_logger().warn(
                        f"cal frame {cf} not in TF — using uncorrected optical frame. "
                        f"Run: bash scripts/calibration/grasp_cal_static_tf.sh"
                    )
            if self._cal_frame_ok:
                T = self._tf_matrix(self.planning_frame, self._cal_frame_ok)
                if T is not None:
                    return str(self._cal_frame_ok), T
                self._cal_frame_ok = False
        for fr in DEFAULT_OPTICAL:
            T = self._tf_matrix(self.planning_frame, fr)
            if T is not None:
                return fr, T
        return None

    def _decode_depth(self, msg: Image) -> Optional[np.ndarray]:
        try:
            h, w = int(msg.height), int(msg.width)
            if msg.encoding in ("16UC1", "mono16"):
                arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w)
                return arr.astype(np.float32) * 0.001
            if msg.encoding == "32FC1":
                return np.frombuffer(msg.data, dtype=np.float32).reshape(h, w)
            return None
        except Exception:
            return None

    def _snapshot_depth(self, timeout_s: float = 2.5) -> Optional[np.ndarray]:
        box = {"img": None}
        ev = threading.Event()

        def _cb(msg: Image):
            if ev.is_set():
                return
            d = self._decode_depth(msg)
            if d is None:
                return
            box["img"] = d
            ev.set()

        sub = self.create_subscription(
            Image, self.depth_topic, _cb, _DEPTH_QOS, callback_group=self._cg
        )
        try:
            time.sleep(0.2)
            ev.wait(max(0.8, float(timeout_s)))
        finally:
            try:
                self.destroy_subscription(sub)
            except Exception:
                pass
        return box["img"]

    def _robust_depth(self, depth: np.ndarray, u: int, v: int) -> Optional[float]:
        h, w = depth.shape[:2]
        half = max(1, self.depth_window // 2)
        y0, y1 = max(0, v - half), min(h, v + half + 1)
        x0, x1 = max(0, u - half), min(w, u + half + 1)
        patch = depth[y0:y1, x0:x1].reshape(-1)
        valid = patch[
            np.isfinite(patch) & (patch > self.min_depth) & (patch < self.max_depth)
        ]
        if valid.size < 3:
            return None
        m0 = float(np.median(valid))
        near = valid[np.abs(valid - m0) < 0.02]
        return float(np.median(near)) if near.size >= 3 else m0

    def _project_tip_uv(
        self, tip: Tuple[float, float, float], T_bo: np.ndarray, info: CameraInfo
    ) -> Optional[Tuple[float, float, float, float, float]]:
        """Return (u_norm, v_norm, u_px, v_px, z_opt) or None if behind camera."""
        try:
            T_ob = np.linalg.inv(T_bo)
        except np.linalg.LinAlgError:
            return None
        p = (T_ob @ np.array([tip[0], tip[1], tip[2], 1.0]))[:3]
        z = float(p[2])
        if z <= 0.05:
            return None
        fx, fy = float(info.k[0]), float(info.k[4])
        cx, cy = float(info.k[2]), float(info.k[5])
        if fx < 1.0 or fy < 1.0:
            return None
        u_px = fx * float(p[0]) / z + cx
        v_px = fy * float(p[1]) / z + cy
        w = int(info.width) if info.width else 848
        h = int(info.height) if info.height else 480
        u_n = u_px / max(1.0, w - 1)
        v_n = v_px / max(1.0, h - 1)
        if not ( -0.05 <= u_n <= 1.05 and -0.05 <= v_n <= 1.05):
            return None
        return (
            float(max(0.0, min(1.0, u_n))),
            float(max(0.0, min(1.0, v_n))),
            float(u_px),
            float(v_px),
            z,
        )

    def _estimate_at_uv(
        self,
        u_px: float,
        v_px: float,
        info: CameraInfo,
        T_bo: np.ndarray,
        depth: Optional[np.ndarray],
    ) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float], str, Optional[float]]]:
        """Return ((est_xyz), (cam_xyz), method, depth_m)."""
        fx, fy = float(info.k[0]), float(info.k[4])
        cx, cy = float(info.k[2]), float(info.k[5])
        iw = int(info.width) if info.width else 848
        ih = int(info.height) if info.height else 480
        ui = int(round(max(0, min(iw - 1, u_px))))
        vi = int(round(max(0, min(ih - 1, v_px))))

        if depth is not None:
            dh, dw = depth.shape[:2]
            ud = int(round(ui * (dw - 1) / max(1, iw - 1))) if iw != dw else ui
            vd = int(round(vi * (dh - 1) / max(1, ih - 1))) if ih != dh else vi
            z = self._robust_depth(depth, ud, vd)
            if z is not None:
                p_cam = np.array(
                    [(ui - cx) * z / fx, (vi - cy) * z / fy, z], dtype=np.float64
                )
                p_est = (T_bo @ np.append(p_cam, 1.0))[:3]
                return (
                    (float(p_est[0]), float(p_est[1]), float(p_est[2])),
                    (float(p_cam[0]), float(p_cam[1]), float(p_cam[2])),
                    "depth",
                    float(z),
                )

        # Ray-plane fallback when aligned depth is unavailable.
        ray = np.array([(ui - cx) / fx, (vi - cy) / fy, 1.0], dtype=np.float64)
        ray /= np.linalg.norm(ray)
        origin = T_bo[:3, 3]
        direction = T_bo[:3, :3] @ ray
        dz = float(direction[2])
        if abs(dz) < 1e-6:
            return None
        hit_z = float(self._contact_plane_z())
        t_hit = (hit_z - float(origin[2])) / dz
        if t_hit <= 0.05:
            return None
        hit = origin + t_hit * direction
        # cam point along ray at optical range ≈ |hit-origin|
        z_opt = float(np.linalg.norm(hit - origin))
        p_cam = ray * z_opt
        return (
            (float(hit[0]), float(hit[1]), float(hit[2])),
            (float(p_cam[0]), float(p_cam[1]), float(p_cam[2])),
            "table_plane",
            None,
        )

    # ── teach / motion ────────────────────────────────────────────────────
    def _contact_plane_z(self) -> float:
        return float(self.table_z + self.object_height)

    def _tip_height_delta(self, tip_z: float) -> float:
        return float(tip_z) - self._contact_plane_z()

    def _tip_height_warning(self, tip_z: float) -> Optional[str]:
        dz = self._tip_height_delta(tip_z)
        if dz <= self.tip_height_warn_m:
            return None
        return (
            f"Tip Z={tip_z:.3f}m is {dz * 100:.0f} cm above contact plane "
            f"(z={self._contact_plane_z():.3f}m). Touch the table/object — "
            f"high tips break table-plane fallback when depth is missing."
        )

    def _current_joints(self) -> List[float]:
        return [float(self._joint_pos.get(j, 0.0)) for j in self.arm_joints]

    def _load_taught_file(self):
        if not self.pose_path.is_file():
            return
        try:
            with open(self.pose_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if data.get("joints"):
                self._taught = data
                self.get_logger().info(f"loaded taught pose from {self.pose_path}")
        except Exception as e:
            self.get_logger().warn(f"taught pose load: {e}")

    def _save_pose(self):
        self._busy = True
        try:
            tip = self._tip_xyz()
            ee = self._ee_xyz()
            if tip is None or ee is None:
                self._publish_status(
                    "error",
                    "No EE TF — is robot_state_publisher up?",
                )
                return
            q = self._current_joints()
            pan = float(self._pan) if self._pan is not None else 0.0
            rec = {
                "name": "taught_contact",
                "planning_frame": self.planning_frame,
                "ee_link": self.ee_link,
                "arm_joints": list(self.arm_joints),
                "joints": [round(float(v), 6) for v in q],
                "ee_xyz": [round(float(v), 5) for v in ee],
                "tip_xyz": [round(float(v), 5) for v in tip],
                "bar_offset_x": round(float(self.bar_offset_x), 5),
                "pan_at_teach": round(pan, 6),
                "stamp": time.time(),
            }
            self.pose_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.pose_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(rec, f, default_flow_style=False, sort_keys=False)
            self._taught = rec
            height_warn = self._tip_height_warning(float(tip[2]))
            msg = (
                f"Saved tip pose XYZ=({tip[0]:.3f},{tip[1]:.3f},{tip[2]:.3f}) "
                f"pan={math.degrees(pan):.0f}° → {self.pose_path.name}"
            )
            if height_warn:
                msg += f"  WARNING: {height_warn}"
                self.get_logger().warn(height_warn)
            self._publish_status(
                "saved",
                msg,
                tip_xyz=rec["tip_xyz"],
                joints=rec["joints"],
            )
            self.get_logger().info(f"taught pose saved: {rec}")
        finally:
            self._busy = False

    def _send_arm_path(self, waypoints: List[List[float]]) -> bool:
        rate_hz = 20.0
        dt = 1.0 / rate_hz
        q_cur = self._current_joints()
        for q_goal in waypoints:
            if self._cancel:
                return False
            err = max(abs(a - b) for a, b in zip(q_cur, q_goal))
            dur = max(1.2, min(3.5, 0.6 + 1.6 * err))
            steps = max(1, int(dur * rate_hz))
            for s in range(1, steps + 1):
                if self._cancel:
                    return False
                a = s / float(steps)
                q = [c + a * (g - c) for c, g in zip(q_cur, q_goal)]
                msg = JointGroupCommand()
                msg.name = "arm"
                msg.cmd = [float(v) for v in q]
                self._arm_pub.publish(msg)
                time.sleep(dt)
            q_cur = list(q_goal)
            msg = JointGroupCommand()
            msg.name = "arm"
            msg.cmd = [float(v) for v in q_cur]
            self._arm_pub.publish(msg)
            time.sleep(0.15)
        return True

    def _set_pan(self, rad: float, settle: Optional[float] = None) -> bool:
        cmd = float(rad)
        msg = JointSingleCommand()
        msg.name = "pan"
        msg.cmd = cmd
        for _ in range(4):
            if self._cancel:
                return False
            self._pan_pub.publish(msg)
            time.sleep(0.05)
        t0 = time.monotonic()
        wait = float(self.settle_s if settle is None else settle)
        while time.monotonic() - t0 < wait:
            if self._cancel:
                return False
            if self._pan is not None and abs(float(self._pan) - cmd) < 0.04:
                time.sleep(0.25)
                return True
            # keep asserting goal (DDS drop)
            self._pan_pub.publish(msg)
            time.sleep(0.1)
        return True

    def _run_goto_taught(self, via_sleep: bool):
        self._busy = True
        self._cancel = False
        gen = self._run_gen
        try:
            if self._taught is None or not self._taught.get("joints"):
                self._publish_status("error", "No taught pose — Save tip pose first")
                return
            q_tgt = [float(v) for v in self._taught["joints"]]
            if len(q_tgt) < len(self.arm_joints):
                q_tgt.extend([0.0] * (len(self.arm_joints) - len(q_tgt)))
            q_tgt = q_tgt[: len(self.arm_joints)]
            if via_sleep:
                self._publish_status("planning", "Moving to Sleep…")
                q_sleep = list(self.sleep_joints)
                if len(q_sleep) < len(self.arm_joints):
                    q_sleep.extend([0.0] * (len(self.arm_joints) - len(q_sleep)))
                if not self._send_arm_path([q_sleep[: len(self.arm_joints)]]):
                    self._publish_status("cancelled", "Cancelled during Sleep")
                    return
            self._publish_status("planning", "Moving to taught tip pose…")
            if not self._send_arm_path([q_tgt]):
                self._publish_status("cancelled", "Cancelled during goto taught")
                return
            tip = self._tip_xyz()
            tip_txt = (
                f"({tip[0]:.3f},{tip[1]:.3f},{tip[2]:.3f})"
                if tip
                else "(no TF)"
            )
            self._publish_status("done", f"At taught pose tip={tip_txt}")
        except Exception as e:
            self.get_logger().error(f"goto_taught: {e}")
            self._publish_status("error", f"goto_taught: {e}")
        finally:
            if gen == self._run_gen:
                self._busy = False

    def _rotate_out_jsonl(self):
        """Start a fresh jsonl for this sweep; keep one backup of the previous file."""
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.out_path.is_file():
            return
        bak = self.out_path.with_suffix(self.out_path.suffix + ".bak")
        try:
            if bak.is_file():
                bak.unlink()
            self.out_path.replace(bak)
            self.get_logger().info(f"previous cal log moved to {bak.name}")
        except OSError as e:
            self.get_logger().warn(f"could not rotate {self.out_path}: {e}")

    def _write_session_header(self, n: int, truth_fixed: List[float]):
        hdr = {
            "source": "teach_cal_session",
            "stamp": time.time(),
            "n_planned": int(n),
            "truth_x": round(float(truth_fixed[0]), 5),
            "truth_y": round(float(truth_fixed[1]), 5),
            "truth_z": round(float(truth_fixed[2]), 5),
            "pose_path": str(self.pose_path),
            "note": "session header — not a calibration sample",
        }
        with open(self.out_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(hdr) + "\n")

    def _append_jsonl(self, rec: dict):
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def _run_sweep(self, n: int):
        self._busy = True
        self._cancel = False
        gen = self._run_gen
        try:
            if self._taught is None or not self._taught.get("tip_xyz"):
                self._publish_status("error", "No taught tip — Save tip pose first")
                return
            # Ensure arm at taught before sweeping.
            self._publish_status("planning", "Going to taught pose before sweep…")
            q_tgt = [float(v) for v in self._taught["joints"]][: len(self.arm_joints)]
            if not self._send_arm_path([q_tgt]):
                self._publish_status("cancelled", "Cancelled before sweep")
                return

            pans = np.linspace(self.pan_start, self.pan_end, int(n))
            truth_fixed = [float(v) for v in self._taught["tip_xyz"]]
            height_warn = self._tip_height_warning(truth_fixed[2])
            if height_warn:
                self.get_logger().warn(height_warn)
                self._publish_status("planning", f"WARNING: {height_warn}")
            self._cal_frame_ok = None
            if self.fresh_jsonl_each_sweep:
                self._rotate_out_jsonl()
            self._write_session_header(n, truth_fixed)
            ok_n = 0
            for i, pan_cmd in enumerate(pans):
                if self._cancel or gen != self._run_gen:
                    self._publish_status("cancelled", f"Sweep aborted after {ok_n} samples")
                    return
                self._publish_status(
                    "planning",
                    f"Pan sample {i + 1}/{n}: {math.degrees(pan_cmd):.0f}°…",
                    sample=i + 1,
                    n=n,
                )
                if self.regoto_taught_each_sample:
                    if not self._send_arm_path([q_tgt]):
                        self._publish_status("cancelled", "Cancelled during re-goto")
                        return
                    time.sleep(0.25)
                if not self._set_pan(float(pan_cmd)):
                    self._publish_status("cancelled", "Cancelled during pan")
                    return
                time.sleep(0.2)
                rec = self._capture_sample(i + 1, truth_fixed)
                if rec is None:
                    self.get_logger().warn(f"sample {i + 1} failed — continuing")
                    fail_reason = "depth/TF"
                    if height_warn:
                        fail_reason = "depth missing + tip too high"
                    self._publish_status(
                        "planning",
                        f"Sample {i + 1}/{n} failed ({fail_reason}) — continuing",
                        sample=i + 1,
                        n=n,
                    )
                    continue
                self._append_jsonl(rec)
                ok_n += 1
                err = math.sqrt(
                    rec["err_x"] ** 2 + rec["err_y"] ** 2 + rec["err_z"] ** 2
                )
                self._publish_status(
                    "planning",
                    f"Sample {ok_n}: pan={rec['pan_deg']:.0f}° "
                    f"|e|={err * 100:.1f}cm via {rec['method']}",
                    sample=ok_n,
                    n=n,
                    last_error_m=round(err, 4),
                )
            if ok_n == 0:
                self._publish_status(
                    "error",
                    f"Sweep finished with 0/{n} samples. "
                    f"Tip must be on the table (z≈{self._contact_plane_z():.2f}m) when depth "
                    f"is missing, or fix aligned depth on {self.depth_topic}.",
                    samples=0,
                )
                return
            self._publish_status(
                "done",
                f"Sweep complete — {ok_n}/{n} samples → {self.out_path.name}. "
                f"Analyze: python3 scripts/calibration/grasp_cal_analyze.py "
                f"{self.out_path}",
                samples=ok_n,
            )
        except Exception as e:
            self.get_logger().error(f"sweep: {e}")
            self._publish_status("error", f"sweep: {e}")
        finally:
            if gen == self._run_gen:
                self._busy = False

    def _capture_sample(self, sample_id: int, truth_fixed: List[float]) -> Optional[dict]:
        with self._lock:
            info = self._info
        if info is None:
            return None
        got = self._pick_optical()
        if got is None:
            return None
        optical, T_bo = got
        tx, ty, tz = (
            float(truth_fixed[0]),
            float(truth_fixed[1]),
            float(truth_fixed[2]),
        )
        tip_proj = (tx, ty, tz)
        live = self._tip_xyz()
        tip_sag_m: Optional[float] = None
        if live is not None:
            tip_sag_m = float(
                math.sqrt(
                    (live[0] - tx) ** 2
                    + (live[1] - ty) ** 2
                    + (live[2] - tz) ** 2
                )
            )
            if tip_sag_m > self.tip_sag_warn_m:
                self.get_logger().warn(
                    f"tip sag {tip_sag_m * 100:.1f} cm vs taught "
                    f"(live TF ignored for truth)"
                )
        proj = self._project_tip_uv(tip_proj, T_bo, info)
        if proj is None:
            self.get_logger().warn("tip not in camera FOV at this pan")
            return None
        u_n, v_n, u_px, v_px, _z_opt = proj
        depth = self._snapshot_depth()
        est = self._estimate_at_uv(u_px, v_px, info, T_bo, depth)
        if est is None:
            return None
        (ex, ey, ez), (cx, cy, cz), method, depth_m = est
        tip_dz = self._tip_height_delta(tz)
        if method == "table_plane" and tip_dz > self.tip_height_warn_m:
            self.get_logger().warn(
                f"sample {sample_id}: table_plane fallback; tip {tip_dz * 100:.0f} cm "
                f"above contact plane — error will look inflated (teach on table if possible)"
            )
        pan = float(self._pan) if self._pan is not None else float("nan")
        rec = {
            "sample_id": int(sample_id),
            "stamp": time.time(),
            "source": "teach_cal_sweep",
            "truth_source": "taught" if self.truth_use_taught else "live",
            "click_u": round(u_n, 5),
            "click_v": round(v_n, 5),
            "pixel": [int(round(u_px)), int(round(v_px))],
            "pan_rad": None if math.isnan(pan) else round(pan, 6),
            "pan_deg": None if math.isnan(pan) else round(math.degrees(pan), 3),
            "depth_m": None if depth_m is None else round(float(depth_m), 5),
            "method": method,
            "tip_height_delta_m": round(tip_dz, 5),
            "cam_x": round(cx, 5),
            "cam_y": round(cy, 5),
            "cam_z": round(cz, 5),
            "est_x": round(ex, 5),
            "est_y": round(ey, 5),
            "est_z": round(ez, 5),
            "truth_x": round(tx, 5),
            "truth_y": round(ty, 5),
            "truth_z": round(tz, 5),
            "err_x": round(ex - tx, 5),
            "err_y": round(ey - ty, 5),
            "err_z": round(ez - tz, 5),
            "err_norm": round(
                math.sqrt((ex - tx) ** 2 + (ey - ty) ** 2 + (ez - tz) ** 2), 5
            ),
            "err_xy": round(math.hypot(ex - tx, ey - ty), 5),
            "truth_reach": round(math.hypot(tx, ty), 5),
            "optical_frame": optical,
            "T_base_optical": [round(float(x), 8) for x in T_bo.reshape(-1)],
            "K": [float(info.k[0]), float(info.k[4]), float(info.k[2]), float(info.k[5])],
            "joints": {k: round(float(v), 6) for k, v in self._joint_pos.items()},
            "note": "auto teach_cal",
        }
        if live is not None:
            rec["live_tip_x"] = round(float(live[0]), 5)
            rec["live_tip_y"] = round(float(live[1]), 5)
            rec["live_tip_z"] = round(float(live[2]), 5)
        if tip_sag_m is not None:
            rec["tip_sag_m"] = round(tip_sag_m, 5)
        return rec


def main(args=None):
    rclpy.init(args=args)
    node = TeachCalNode()
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
