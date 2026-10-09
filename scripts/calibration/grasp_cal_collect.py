#!/usr/bin/env python3
"""Click-to-grasp accuracy harness: collect (camera-said, truth-was) pairs.

Measures how wrong the D435i -> base_link chain actually is, independently of
grasp_from_click.py.  It does its OWN clean unprojection (pixel -> K -> depth ->
optical frame) so none of the fudge factors in the grasp node (click_origin_offset,
click_pan_scale, click_v_bias, grasp_farther_m ...) can hide inside the numbers.

Workflow per sample
-------------------
  1. Put a POINTY probe on the table.  A pen or chopstick taped upright works;
     the tip is unambiguous from both the camera's view and the gripper's.
  2. Click the probe TIP in the GUI's D435i pane.  This harness snapshots depth,
     unprojects, and records the live TF chain + pan angle.  It prints the point.
  3. Jog the arm until the gripper reference point touches that exact same tip.
  4. Type  t  <ENTER> here.  Truth is read from TF.  Row written.

Alternative if you cannot jog the arm: type the measured coordinates directly,
e.g.   0.240 -0.035 0.115   <ENTER>

Commands at the prompt
----------------------
  t                capture truth from TF (--truth-frame, + --truth-offset-x)
  <x> <y> <z>      capture truth as typed metres in the planning frame
  s                skip / discard the pending sample
  n <text>         attach a note to the pending sample
  p                print a running summary
  q                quit

Outputs
-------
  <out>.jsonl   full records incl. 4x4 TF matrices  (feed to grasp_cal_analyze.py)
  <out>.csv     flat table for eyeballing in a spreadsheet

Run it alongside the normal stack; it only listens.  Depth is subscribed
one-shot per click (same trick grasp_from_click uses) so it does not add a
continuous reader to the aligned-depth topic.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import threading
import time
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

# Match Nano D435i pubs (RELIABLE). BEST_EFFORT sensor_data often never
# connects across the AGX Fast-DDS bridge, so camera_info/depth look "missing".
_DEPTH_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=5,
    durability=QoSDurabilityPolicy.VOLATILE,
)


# Optical frames tried in order.  Aligned depth is aligned to COLOR, so the
# color optical frame is the correct one to unproject through.
DEFAULT_OPTICAL_FRAMES = [
    "d435i_color_optical_frame",
    "px150_no_elbow/d435i_color_optical_frame",
    "camera_color_optical_frame",
    "d435i_depth_optical_frame",
]


def quat_to_mat(x: float, y: float, z: float, w: float) -> np.ndarray:
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class CalCollector(Node):
    def __init__(self, args):
        super().__init__("grasp_cal_collect")
        self.args = args
        self._cg = ReentrantCallbackGroup()

        self.planning_frame = args.planning_frame
        self.truth_frame = args.truth_frame
        self.truth_offset_x = float(args.truth_offset_x)
        self.depth_topic = args.depth_topic
        self.info_topic = args.info_topic
        self.click_topic = args.click_topic

        self._lock = threading.Lock()
        self._info: Optional[CameraInfo] = None
        self._pan: Optional[float] = None
        self._joints: dict = {}

        self._pending: Optional[dict] = None
        self._rows: List[dict] = []
        self._sample_id = 0

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.create_subscription(
            CameraInfo, self.info_topic, self._info_cb, _DEPTH_QOS
        )
        self.create_subscription(
            JointState, args.joint_states_topic, self._js_cb, qos_profile_sensor_data
        )
        self.create_subscription(
            String, self.click_topic, self._click_cb, 10, callback_group=self._cg
        )

        self.jsonl_path = args.out + ".jsonl"
        self.csv_path = args.out + ".csv"
        self._init_outputs()

        self.get_logger().info(
            f"collector ready — depth={self.depth_topic} info={self.info_topic} "
            f"click={self.click_topic} truth_frame={self.truth_frame}"
        )

    # ── outputs ───────────────────────────────────────────────────────────
    CSV_FIELDS = [
        "sample_id", "stamp", "click_u", "click_v", "pan_deg",
        "cam_x", "cam_y", "cam_z", "depth_m", "depth_n_valid",
        "est_x", "est_y", "est_z",
        "truth_x", "truth_y", "truth_z", "truth_mode",
        "err_x", "err_y", "err_z", "err_norm", "err_xy",
        "truth_reach", "optical_frame", "note",
    ]

    def _init_outputs(self):
        if self.args.append and os.path.exists(self.csv_path):
            return
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.CSV_FIELDS).writeheader()
        open(self.jsonl_path, "w", encoding="utf-8").close()

    def _write_row(self, rec: dict):
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        flat = {k: rec.get(k, "") for k in self.CSV_FIELDS}
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.CSV_FIELDS).writerow(flat)

    # ── subscriptions ─────────────────────────────────────────────────────
    def _info_cb(self, msg: CameraInfo):
        with self._lock:
            self._info = msg

    def _js_cb(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            p = float(pos)
            if abs(p + math.pi) <= 0.05:   # xs_sdk publishes ~-pi on syncRead fail
                continue
            self._joints[name] = p
            if name == "pan":
                self._pan = p

    # ── depth snapshot (one-shot, like the grasp node) ────────────────────
    def _decode_depth(self, msg: Image) -> Optional[np.ndarray]:
        try:
            h, w = int(msg.height), int(msg.width)
            enc = msg.encoding
            if enc in ("16UC1", "mono16"):
                arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w)
                return arr.astype(np.float32) * 0.001          # mm -> m
            if enc == "32FC1":
                arr = np.frombuffer(msg.data, dtype=np.float32).reshape(h, w)
                return np.asarray(arr, dtype=np.float32)
            self.get_logger().warn(f"unhandled depth encoding {enc}")
            return None
        except Exception as e:
            self.get_logger().warn(f"depth decode: {e}")
            return None

    def _snapshot_depth(self, timeout_s: float = 4.0) -> Optional[np.ndarray]:
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
            Image, self.depth_topic, _cb, _DEPTH_QOS,
            callback_group=self._cg,
        )
        try:
            # Give DDS matching time across Nano↔AGX bridge before waiting.
            time.sleep(0.25)
            ev.wait(max(1.0, float(timeout_s)))
        finally:
            try:
                self.destroy_subscription(sub)
            except Exception:
                pass
        return box["img"]

    # ── geometry ──────────────────────────────────────────────────────────
    def _tf_matrix(self, target: str, source: str,
                   timeout_s: float = 0.6) -> Optional[np.ndarray]:
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
        T[:3, :3] = quat_to_mat(q.x, q.y, q.z, q.w)
        T[:3, 3] = [t.x, t.y, t.z]
        return T

    def _pick_optical_frame(self) -> Optional[Tuple[str, np.ndarray]]:
        cands = ([self.args.optical_frame] if self.args.optical_frame
                 else DEFAULT_OPTICAL_FRAMES)
        for fr in cands:
            T = self._tf_matrix(self.planning_frame, fr)
            if T is not None:
                return fr, T
        return None

    def _robust_depth(self, depth: np.ndarray, u_px: int, v_px: int
                      ) -> Tuple[Optional[float], int]:
        """Median depth in a window, re-medianed over the dominant surface."""
        h, w = depth.shape[:2]
        half = max(1, int(self.args.depth_window) // 2)
        y0, y1 = max(0, v_px - half), min(h, v_px + half + 1)
        x0, x1 = max(0, u_px - half), min(w, u_px + half + 1)
        patch = depth[y0:y1, x0:x1].reshape(-1)
        valid = patch[np.isfinite(patch)
                      & (patch > self.args.min_depth)
                      & (patch < self.args.max_depth)]
        if valid.size < 3:
            return None, int(valid.size)
        m0 = float(np.median(valid))
        near = valid[np.abs(valid - m0) < 0.02]      # stay on one surface
        if near.size >= 3:
            return float(np.median(near)), int(near.size)
        return m0, int(valid.size)

    # ── click handling ────────────────────────────────────────────────────
    def _click_cb(self, msg: String):
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        cam = str(data.get("cam") or data.get("camera") or "d435i").lower()
        if cam in ("wide", "wide_cam", "jaw"):
            return
        u = float(data.get("u", -1))
        v = float(data.get("v", -1))
        if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
            return
        threading.Thread(
            target=self._click_worker, args=(data, u, v), daemon=True
        ).start()

    def _click_worker(self, data: dict, u: float, v: float):
        try:
            with self._lock:
                info = self._info
            if info is None:
                self._say("!! no camera_info yet — is the D435i publishing?")
                return

            depth = self._snapshot_depth()
            if depth is None:
                self._say("!! no aligned-depth frame arrived within timeout")
                return

            fx, fy = float(info.k[0]), float(info.k[4])
            cx, cy = float(info.k[2]), float(info.k[5])
            if fx < 1.0 or fy < 1.0:
                self._say("!! camera_info K is empty — cannot unproject")
                return

            # Map the normalized GUI click into K's pixel grid.
            try:
                cw = int(data.get("width") or 0)
                ch = int(data.get("height") or 0)
            except (TypeError, ValueError):
                cw = ch = 0
            iw = int(info.width) if info.width else depth.shape[1]
            ih = int(info.height) if info.height else depth.shape[0]
            src_w, src_h = (cw, ch) if (cw >= 32 and ch >= 32) else (iw, ih)
            u_px = int(round(u * (src_w - 1)))
            v_px = int(round(v * (src_h - 1)))
            if (src_w, src_h) != (iw, ih):
                u_px = int(round(u_px * (iw - 1) / max(1, src_w - 1)))
                v_px = int(round(v_px * (ih - 1) / max(1, src_h - 1)))

            # Depth image may differ in size from camera_info.
            dh, dw = depth.shape[:2]
            u_d = int(round(u_px * (dw - 1) / max(1, iw - 1))) if iw != dw else u_px
            v_d = int(round(v_px * (dh - 1) / max(1, ih - 1))) if ih != dh else v_px
            u_d = max(0, min(dw - 1, u_d))
            v_d = max(0, min(dh - 1, v_d))

            z, n_valid = self._robust_depth(depth, u_d, v_d)
            if z is None:
                self._say(f"!! no valid depth at pixel ({u_d},{v_d}) — "
                          f"aim at a textured spot, or fix the IR emitter/preset")
                return

            # Clean pinhole unprojection in the OPTICAL frame.
            p_cam = np.array(
                [(u_px - cx) * z / fx, (v_px - cy) * z / fy, z], dtype=np.float64
            )

            got = self._pick_optical_frame()
            if got is None:
                self._say("!! no TF from planning frame to any known optical frame")
                return
            optical_frame, T_bc = got
            p_est = (T_bc @ np.append(p_cam, 1.0))[:3]

            self._sample_id += 1
            pan = self._pan
            rec = {
                "sample_id": self._sample_id,
                "stamp": time.time(),
                "click_u": round(u, 5),
                "click_v": round(v, 5),
                "pixel": [u_px, v_px],
                "pan_rad": None if pan is None else round(float(pan), 6),
                "pan_deg": None if pan is None else round(math.degrees(pan), 3),
                "depth_m": round(float(z), 5),
                "depth_n_valid": n_valid,
                "cam_x": round(float(p_cam[0]), 5),
                "cam_y": round(float(p_cam[1]), 5),
                "cam_z": round(float(p_cam[2]), 5),
                "est_x": round(float(p_est[0]), 5),
                "est_y": round(float(p_est[1]), 5),
                "est_z": round(float(p_est[2]), 5),
                "optical_frame": optical_frame,
                "T_base_optical": [round(float(x), 8) for x in T_bc.reshape(-1)],
                "K": [float(info.k[0]), float(info.k[4]),
                      float(info.k[2]), float(info.k[5])],
                "joints": {k: round(float(v2), 6) for k, v2 in self._joints.items()},
                "note": "",
            }
            self._pending = rec
            pan_txt = "pan=?" if pan is None else f"pan={math.degrees(pan):+.1f}deg"
            self._say(
                f"\n--- sample {self._sample_id} pending ---\n"
                f"  click   u={u:.4f} v={v:.4f}   px=({u_px},{v_px})   {pan_txt}\n"
                f"  depth   {z:.4f} m  ({n_valid} valid px in window)\n"
                f"  optical {optical_frame}\n"
                f"  camera says base XYZ = "
                f"({p_est[0]:+.4f}, {p_est[1]:+.4f}, {p_est[2]:+.4f})\n"
                f"  -> touch that same point with the gripper, then type 't'"
            )
        except Exception as e:
            self._say(f"!! click worker failed: {e}")

    # ── truth capture ─────────────────────────────────────────────────────
    def _truth_from_tf(self) -> Optional[Tuple[np.ndarray, str]]:
        T = self._tf_matrix(self.planning_frame, self.truth_frame, timeout_s=1.0)
        if T is None:
            return None
        p = T[:3, 3].copy()
        if abs(self.truth_offset_x) > 1e-9:
            p = p + T[:3, 0] * self.truth_offset_x      # along gripper +X
        return p, f"tf:{self.truth_frame}+{self.truth_offset_x:.4f}x"

    def _commit(self, truth: np.ndarray, mode: str):
        rec = self._pending
        if rec is None:
            self._say("(nothing pending — click a point first)")
            return
        est = np.array([rec["est_x"], rec["est_y"], rec["est_z"]], dtype=np.float64)
        err = est - truth
        rec.update({
            "truth_x": round(float(truth[0]), 5),
            "truth_y": round(float(truth[1]), 5),
            "truth_z": round(float(truth[2]), 5),
            "truth_mode": mode,
            "err_x": round(float(err[0]), 5),
            "err_y": round(float(err[1]), 5),
            "err_z": round(float(err[2]), 5),
            "err_norm": round(float(np.linalg.norm(err)), 5),
            "err_xy": round(float(math.hypot(err[0], err[1])), 5),
            "truth_reach": round(float(math.hypot(truth[0], truth[1])), 5),
        })
        self._write_row(rec)
        self._rows.append(rec)
        self._pending = None
        self._say(
            f"  truth   ({truth[0]:+.4f}, {truth[1]:+.4f}, {truth[2]:+.4f})  [{mode}]\n"
            f"  ERROR   dx={err[0]:+.4f} dy={err[1]:+.4f} dz={err[2]:+.4f}  "
            f"|e|={np.linalg.norm(err) * 100:.1f} cm   "
            f"(xy {math.hypot(err[0], err[1]) * 100:.1f} cm)\n"
            f"  saved — {len(self._rows)} samples total\n"
        )

    def summary(self):
        if not self._rows:
            self._say("(no samples yet)")
            return
        e = np.array([[r["err_x"], r["err_y"], r["err_z"]] for r in self._rows])
        n = np.linalg.norm(e, axis=1)
        self._say(
            f"\n=== {len(self._rows)} samples ===\n"
            f"  mean error vector : ({e[:, 0].mean():+.4f}, "
            f"{e[:, 1].mean():+.4f}, {e[:, 2].mean():+.4f}) m\n"
            f"   (a large constant here = camera mount TRANSLATION is wrong)\n"
            f"  spread about mean : ({e[:, 0].std():.4f}, "
            f"{e[:, 1].std():.4f}, {e[:, 2].std():.4f}) m\n"
            f"   (large spread = mount ROTATION / pan axis is wrong too)\n"
            f"  |error|  mean {n.mean() * 100:.2f} cm   "
            f"median {np.median(n) * 100:.2f} cm   max {n.max() * 100:.2f} cm\n"
            f"  -> run: python3 grasp_cal_analyze.py {self.jsonl_path}\n"
        )

    # ── operator console ──────────────────────────────────────────────────
    def _say(self, text: str):
        sys.stdout.write(text + "\n")
        sys.stdout.flush()

    def console(self):
        self._say(__doc__)
        while rclpy.ok():
            try:
                line = input("cal> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            low = line.lower()
            if low in ("q", "quit", "exit"):
                break
            if low in ("p", "print", "summary"):
                self.summary()
                continue
            if low in ("s", "skip"):
                self._pending = None
                self._say("(pending sample discarded)")
                continue
            if low.startswith("n "):
                if self._pending is not None:
                    self._pending["note"] = line[2:].strip()
                    self._say("(note attached)")
                else:
                    self._say("(nothing pending)")
                continue
            if low in ("t", "tf", "touch"):
                if self._pending is None:
                    self._say("(nothing pending — click a point first)")
                    continue
                got = self._truth_from_tf()
                if got is None:
                    self._say(
                        f"!! no TF {self.planning_frame} <- {self.truth_frame}\n"
                        "   (export FASTRTPS_DEFAULT_PROFILES_FILE=…/bridge_both.xml "
                        "and wait a few seconds)"
                    )
                    continue
                self._commit(got[0], got[1])
                continue
            parts = line.replace(",", " ").split()
            if len(parts) == 3:
                try:
                    xyz = np.array([float(p) for p in parts], dtype=np.float64)
                except ValueError:
                    self._say("?? expected: t | <x> <y> <z> | s | n <text> | p | q")
                    continue
                self._commit(xyz, "typed")
                continue
            self._say("?? expected: t | <x> <y> <z> | s | n <text> | p | q")
        self.summary()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="grasp_cal",
                    help="output basename (.jsonl/.csv appended)")
    ap.add_argument("--append", action="store_true",
                    help="append to existing files instead of truncating")
    ap.add_argument("--planning-frame", default="px150_no_elbow/base_link")
    ap.add_argument("--truth-frame", default="px150_no_elbow/ee_gripper_link",
                    help="TF frame used as the physical probe when you type 't'")
    ap.add_argument("--truth-offset-x", type=float, default=0.045,
                    help="metres along the truth frame's +X to reach the actual "
                         "bar/palm contact (default 0.045 m; measure once with calipers)")
    ap.add_argument("--depth-topic",
                    default="/d435i/d435i/aligned_depth_to_color/image_raw")
    ap.add_argument("--info-topic", default="/d435i/d435i/color/camera_info")
    ap.add_argument("--click-topic", default="/grab_it/grasp_click")
    ap.add_argument("--joint-states-topic", default="/px150_no_elbow/joint_states")
    ap.add_argument("--optical-frame", default=None,
                    help="force one optical frame instead of autodetecting")
    ap.add_argument("--depth-window", type=int, default=7)
    ap.add_argument("--min-depth", type=float, default=0.12)
    ap.add_argument("--max-depth", type=float, default=1.50)
    args = ap.parse_args()

    rclpy.init()
    node = CalCollector(args)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        node.console()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
