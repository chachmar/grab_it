#!/usr/bin/env python3
"""Measure the D435i's REAL tilt using the floor as a reference plane.

cam_pitch is set to 0.35 rad (20 deg) in both grab_it.urdf.xacro and
camera_tf.env, and in both places it looks assumed rather than measured.
This measures it.

Method: the floor is a big flat plane.  Fit it in the depth image, take its
normal in the camera's own optical frame, and the angle between that normal
and the optical axis IS the camera's tilt below horizontal.  No phone, no
IMU, no restart — just geometry, good to a fraction of a degree, and it uses
the same depth data the grasp pipeline already relies on.

It then asks TF what the model believes at the same instant and prints the
difference, which is the correction to apply to cam_pitch.

Sweeping several pan angles also re-confirms (independently of the phone)
that camera motion tracks the joint 1:1.

Usage, inside the container with the stack up:
    export FASTRTPS_DEFAULT_PROFILES_FILE=/workspaces/isaac_ros-dev/rtabmap_profile.xml
    python3 cam_pitch_from_floor.py                 # sweep 0.0 .. 0.6
    python3 cam_pitch_from_floor.py --angles 0.0    # just the static offset
    python3 cam_pitch_from_floor.py --no-move       # measure wherever pan is now

Point the camera at a clear patch of floor (or any large flat horizontal
surface) with no clutter filling the view.  The robot must be on that same
flat surface, level.
"""
from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, JointState
from tf2_ros import Buffer, TransformListener

try:
    from interbotix_xs_msgs.msg import JointSingleCommand
except ImportError:
    JointSingleCommand = None

OPTICAL_FRAMES = [
    "d435i_color_optical_frame",
    "px150_no_elbow/d435i_color_optical_frame",
    "camera_color_optical_frame",
]


def fit_plane_ransac(pts: np.ndarray, iters: int = 200, thresh: float = 0.012,
                     rng=None) -> Optional[Tuple[np.ndarray, float, int]]:
    """Robust plane fit. Returns (unit normal, offset d, inlier count)."""
    if rng is None:
        rng = np.random.default_rng(0)
    n_pts = pts.shape[0]
    if n_pts < 50:
        return None
    best = None
    for _ in range(iters):
        idx = rng.choice(n_pts, 3, replace=False)
        p0, p1, p2 = pts[idx]
        nrm = np.cross(p1 - p0, p2 - p0)
        ln = np.linalg.norm(nrm)
        if ln < 1e-9:
            continue
        nrm = nrm / ln
        d = -float(nrm @ p0)
        dist = np.abs(pts @ nrm + d)
        n_in = int((dist < thresh).sum())
        if best is None or n_in > best[2]:
            best = (nrm, d, n_in)
    if best is None:
        return None
    # refine on the inliers with a least-squares fit
    nrm, d, _ = best
    inl = pts[np.abs(pts @ nrm + d) < thresh]
    if inl.shape[0] >= 50:
        c = inl.mean(axis=0)
        _, _, vt = np.linalg.svd(inl - c)
        nrm = vt[2] / np.linalg.norm(vt[2])
        d = -float(nrm @ c)
        inl = pts[np.abs(pts @ nrm + d) < thresh]
    return nrm, d, int(inl.shape[0])


class FloorPitch(Node):
    def __init__(self, args):
        super().__init__("cam_pitch_from_floor")
        self.args = args
        self._cg = ReentrantCallbackGroup()
        self._lock = threading.Lock()
        self._info: Optional[CameraInfo] = None
        self._pan: Optional[float] = None
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self.create_subscription(CameraInfo, args.info_topic, self._info_cb,
                                 qos_profile_sensor_data)
        self.create_subscription(JointState, args.joint_states_topic, self._js_cb,
                                 qos_profile_sensor_data)
        self._pub = None
        if JointSingleCommand is not None:
            self._pub = self.create_publisher(JointSingleCommand,
                                              args.command_topic, 10)

    def _info_cb(self, msg):
        with self._lock:
            self._info = msg

    def _js_cb(self, msg):
        for n, p in zip(msg.name, msg.position):
            if n == "pan" and abs(float(p) + math.pi) > 0.05:
                self._pan = float(p)

    def set_pan(self, value: float) -> None:
        if self._pub is None:
            print("   (interbotix_xs_msgs missing — move pan yourself)")
            return
        m = JointSingleCommand()
        m.name = "pan"
        m.cmd = float(value)
        for _ in range(3):
            self._pub.publish(m)
            time.sleep(0.05)

    def grab_depth(self, timeout_s: float = 2.0) -> Optional[np.ndarray]:
        box = {"img": None}
        ev = threading.Event()

        def _cb(msg: Image):
            if ev.is_set():
                return
            try:
                h, w = int(msg.height), int(msg.width)
                if msg.encoding in ("16UC1", "mono16"):
                    a = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w)
                    box["img"] = a.astype(np.float32) * 0.001
                elif msg.encoding == "32FC1":
                    box["img"] = np.frombuffer(msg.data,
                                               dtype=np.float32).reshape(h, w).copy()
                else:
                    return
                ev.set()
            except Exception:
                return

        sub = self.create_subscription(Image, self.args.depth_topic, _cb,
                                       qos_profile_sensor_data,
                                       callback_group=self._cg)
        try:
            ev.wait(timeout_s)
        finally:
            try:
                self.destroy_subscription(sub)
            except Exception:
                pass
        return box["img"]

    def measure(self) -> Optional[dict]:
        """One floor-plane measurement of the camera's tilt below horizontal."""
        with self._lock:
            info = self._info
        if info is None:
            print("   !! no camera_info")
            return None
        depth = self.grab_depth()
        if depth is None:
            print("   !! no depth frame")
            return None
        fx, fy = float(info.k[0]), float(info.k[4])
        cx, cy = float(info.k[2]), float(info.k[5])
        if fx < 1.0:
            print("   !! camera_info K is empty")
            return None

        h, w = depth.shape[:2]
        # Use the lower part of the image, subsampled — that is where floor lives.
        y0 = int(self.args.roi_v0 * h)
        step = max(1, self.args.step)
        vs, us = np.mgrid[y0:h:step, 0:w:step]
        zs = depth[y0:h:step, 0:w:step]
        m = np.isfinite(zs) & (zs > self.args.min_depth) & (zs < self.args.max_depth)
        if int(m.sum()) < 200:
            print(f"   !! only {int(m.sum())} valid depth points — aim at more floor")
            return None
        z = zs[m].astype(np.float64)
        # scale image coords if depth resolution differs from camera_info
        iw = int(info.width) if info.width else w
        ih = int(info.height) if info.height else h
        su, sv = (iw - 1) / max(1, w - 1), (ih - 1) / max(1, h - 1)
        u = us[m].astype(np.float64) * su
        v = vs[m].astype(np.float64) * sv
        pts = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=1)

        got = fit_plane_ransac(pts, iters=self.args.ransac_iters,
                               thresh=self.args.inlier_m)
        if got is None:
            print("   !! plane fit failed")
            return None
        nrm, d, n_in = got
        frac = n_in / float(pts.shape[0])
        # orient the normal so it points from the floor up toward the camera
        if float(nrm @ pts.mean(axis=0)) > 0:
            nrm = -nrm
        opt_z = np.array([0.0, 0.0, 1.0])
        ang = math.degrees(math.acos(max(-1.0, min(1.0, float(opt_z @ nrm)))))
        pitch_down = ang - 90.0

        # what does TF think, right now?
        tf_pitch = None
        used = None
        for fr in OPTICAL_FRAMES:
            try:
                t = self._tf_buffer.lookup_transform(
                    self.args.planning_frame, fr, rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.5))
            except Exception:
                continue
            q = t.transform.rotation
            x, y, zq, wq = q.x, q.y, q.z, q.w
            R = np.array([
                [1-2*(y*y+zq*zq), 2*(x*y-zq*wq), 2*(x*zq+y*wq)],
                [2*(x*y+zq*wq), 1-2*(x*x+zq*zq), 2*(y*zq-x*wq)],
                [2*(x*zq-y*wq), 2*(y*zq+x*wq), 1-2*(x*x+y*y)]])
            look = R[:, 2]
            tf_pitch = math.degrees(math.atan2(-look[2], math.hypot(look[0], look[1])))
            used = fr
            break
        return {"measured": pitch_down, "tf": tf_pitch, "frame": used,
                "pan": self._pan, "inliers": n_in, "frac": frac,
                "n": int(pts.shape[0])}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--angles", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6])
    ap.add_argument("--no-move", action="store_true",
                    help="do not command pan; just measure where it is now")
    ap.add_argument("--settle", type=float, default=3.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--planning-frame", default="px150_no_elbow/base_link")
    ap.add_argument("--depth-topic",
                    default="/d435i/d435i/aligned_depth_to_color/image_raw")
    ap.add_argument("--info-topic", default="/d435i/d435i/color/camera_info")
    ap.add_argument("--joint-states-topic", default="/px150_no_elbow/joint_states")
    ap.add_argument("--command-topic", default="/px150_no_elbow/commands/joint_single")
    ap.add_argument("--roi-v0", type=float, default=0.45,
                    help="use image rows below this fraction (floor lives low)")
    ap.add_argument("--step", type=int, default=4)
    ap.add_argument("--min-depth", type=float, default=0.20)
    ap.add_argument("--max-depth", type=float, default=3.0)
    ap.add_argument("--inlier-m", type=float, default=0.012)
    ap.add_argument("--ransac-iters", type=int, default=200)
    args = ap.parse_args()

    rclpy.init()
    node = FloorPitch(args)
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    time.sleep(1.5)

    print(__doc__)
    seq = [None] if args.no_move else list(args.angles)
    rows = []
    for cmd in seq:
        if cmd is not None:
            print(f"\n--- pan -> {cmd:+.2f} rad ({math.degrees(cmd):+.1f} deg) ---")
            node.set_pan(cmd)
            time.sleep(args.settle)
        else:
            print("\n--- measuring at current pan ---")
        got = []
        for _ in range(max(1, args.repeats)):
            r = node.measure()
            if r:
                got.append(r)
            time.sleep(0.3)
        if not got:
            continue
        meas = float(np.median([g["measured"] for g in got]))
        spread = float(np.max([g["measured"] for g in got])
                       - np.min([g["measured"] for g in got]))
        tfp = got[-1]["tf"]
        pan = got[-1]["pan"]
        print(f"   floor plane: {got[-1]['inliers']}/{got[-1]['n']} inliers "
              f"({got[-1]['frac']*100:.0f}%)   repeat spread {spread:.2f} deg")
        print(f"   MEASURED tilt below horizontal : {meas:7.2f} deg")
        if tfp is not None:
            print(f"   TF ({got[-1]['frame']}) says   : {tfp:7.2f} deg")
            print(f"   error (model - reality)        : {tfp - meas:+7.2f} deg")
        rows.append({"pan": pan, "meas": meas, "tf": tfp})

    print("\n" + "=" * 66)
    if rows:
        print("  pan(deg)   measured   TF model   error")
        for r in rows:
            p = "  --  " if r["pan"] is None else f"{math.degrees(r['pan']):+7.2f}"
            t = "  --  " if r["tf"] is None else f"{r['tf']:+8.2f}"
            e = "  --  " if r["tf"] is None else f"{r['tf']-r['meas']:+7.2f}"
            print(f"  {p}   {r['meas']:+8.2f}   {t}  {e}")
        errs = [r["tf"] - r["meas"] for r in rows if r["tf"] is not None]
        if errs:
            e = np.array(errs)
            print(f"\n  model error: mean {e.mean():+.2f} deg, "
                  f"spread {e.max()-e.min():.2f} deg")
            if e.max() - e.min() < 2.5:
                new = 0.35 - math.radians(e.mean())
                print(f"  => CONSTANT offset. This is a static cam_pitch error.")
                print(f"     cam_pitch is 0.35 rad; it should be about "
                      f"{new:.4f} rad ({math.degrees(new):.2f} deg).")
                print(f"     Set that in grab_it.urdf.xacro AND camera_tf.env,")
                print(f"     then click_pan_scale can go back to 1.0.")
            else:
                print("  => error VARIES with pan — not a static offset after all.")
    print("=" * 66)
    ex.shutdown()
    rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
