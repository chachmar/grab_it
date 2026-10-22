#!/usr/bin/env python3
"""Closed-loop IK grasp servo: pool → ik_eef → wide-cam jaw center → foam stall.

Learning does not move the arm. This script is the baseline the IL ranker and
RL timing policy must beat.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from ik_grasp_lib import (  # noqa: E402
    CLOSED_GRIP,
    GRIP_OPEN,
    JAW_V,
    JawView,
    ServoTrial,
    append_trial,
    blend_toward_ik,
    pick_grasp,
    project_reachable_tip,
    tip_from_twin,
    wrist_max_of,
)
from rl_goal_lib import eef_of, load_goal  # noqa: E402

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState, Image, CompressedImage
    from interbotix_xs_msgs.msg import JointGroupCommand, JointSingleCommand
except ImportError:  # pragma: no cover
    rclpy = None

ARM_JOINTS = ["waist", "shoulder", "wrist_angle", "wrist_rotate", "gripper"]


def _http_json(url: str, timeout: float = 2.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _http_bytes(url: str, timeout: float = 2.0) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read()
    except Exception:
        return None


def wide_shaft_from_jpeg(jpeg: bytes | None) -> dict | None:
    """Reuse the foam-shaft detector from the wrist-finish helper when present."""
    if not jpeg:
        return None
    try:
        from _il_wrist_finish import shaft_pose  # type: ignore

        return shaft_pose(jpeg)
    except Exception:
        pass
    if cv2 is None:
        return None
    img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    # Fallback: assume shaft near jaw center if we cannot import the helper.
    return {"u": 0.5, "v": 0.75, "from_vert": 0.0, "area": float(h * w) * 0.01}


class IkGraspServo(Node if rclpy is not None else object):
    def __init__(self, args: argparse.Namespace):
        if rclpy is None:
            raise SystemExit("rclpy required — run inside grab_it_live / ROS")
        super().__init__("ik_grasp_servo")
        self.args = args
        self.goal = load_goal(args.goal)
        self.wmax = wrist_max_of(self.goal)
        self._js: dict[str, float] = {}
        self._js_t = 0.0
        self._scorer = None
        if args.ranker and Path(args.ranker).is_file():
            from il_grasp_ranker import load_scorer

            self._scorer = load_scorer(Path(args.ranker))
            self.get_logger().info(f"IL ranker {args.ranker}")
        self.create_subscription(
            JointState,
            "/px150_no_elbow/joint_states",
            self._on_js,
            qos_profile_sensor_data,
        )
        self._arm = self.create_publisher(
            JointGroupCommand, "/px150_no_elbow/commands/joint_group", 10
        )
        self._grip = self.create_publisher(
            JointSingleCommand, "/px150_no_elbow/commands/joint_single", 10
        )

    def _on_js(self, msg: JointState) -> None:
        for n, p in zip(msg.name, msg.position):
            self._js[str(n)] = float(p)
        self._js_t = time.time()

    def _wait_js(self, timeout: float = 15.0) -> dict[str, float]:
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if all(j in self._js for j in ARM_JOINTS) and (time.time() - self._js_t) < 1.0:
                return {j: float(self._js[j]) for j in ARM_JOINTS}
        raise SystemExit("no live joint_states")

    def _pub(self, q: dict[str, float]) -> None:
        arm = JointGroupCommand()
        arm.name = "arm"
        arm.cmd = [float(q[n]) for n in ARM_JOINTS[:4]]
        self._arm.publish(arm)
        grip = JointSingleCommand()
        grip.name = "gripper"
        grip.cmd = float(q["gripper"])
        self._grip.publish(grip)

    def _twin(self) -> dict:
        hub = (self.args.hub or "").rstrip("/")
        return _http_json(f"{hub}/isaac/twin") or {}

    def _wide_view(self) -> JawView:
        # REAL mode: MJPEG owns :8081; hub tip-mirror is :8082 (no camera).
        cam = (self.args.cam or self.args.hub or "").rstrip("/")
        jpeg = _http_bytes(f"{cam}/wide.jpg")
        return JawView.from_shaft(wide_shaft_from_jpeg(jpeg))

    def run(self) -> int:
        q = self._wait_js()
        twin = {} if self.args.yaml_aim else self._twin()
        tip_raw, yaw_raw = tip_from_twin(twin, self.goal)
        tip, yaw, aim_note = project_reachable_tip(
            tip_raw, yaw_raw, wrist_max=self.wmax, goal=self.goal
        )
        if self.args.yaml_aim:
            tip = [float(self.goal["x"]), float(self.goal["y"]), float(self.goal["z"])]
            yaw = float(self.goal.get("wrist_rotate", 0.0))
            aim_note = "yaml_aim"
        self.get_logger().info(
            f"servo tip=({tip[0]:.3f},{tip[1]:.3f},{tip[2]:.3f}) yaw={yaw:.3f} "
            f"wmax={self.wmax:.3f} aim={aim_note} "
            f"raw=({tip_raw[0]:.3f},{tip_raw[1]:.3f},{tip_raw[2]:.3f})"
        )

        trial = ServoTrial(
            shaft_xy=[tip[0] - 0.06, tip[1]],
            shaft_z=tip[2] - 0.096,
            shaft_yaw=yaw,
            note=f"{self.args.note or ''}|aim={aim_note}".strip("|"),
        )
        dt = 1.0 / max(float(self.args.hz), 1.0)
        alpha = float(self.args.blend)
        # From Home: shoulder down, wrist_angle up, wrist_rotate servo → jaws, then close.
        phase = "approach"
        close_started = False
        stall_hits = 0
        freeze_q: dict | None = None
        close_t0: float | None = None
        prev_grip: float | None = None
        view_ok_once = False
        jaw_ok_hits = 0
        t0 = time.time()
        last_cmd = dict(q)
        # Taught / nominal approach setpoints (still respect wrist floor).
        jr = self.goal.get("joints_real") or {}
        shoulder_down = float(jr.get("shoulder", 0.70))
        # Tip wrist up off the taught floor so pads can wrap the pole.
        wrist_up = max(float(self.wmax) + 0.50, -0.95)
        # Open the jaw before anything else.
        open_cmd = dict(q)
        open_cmd["gripper"] = GRIP_OPEN
        for _ in range(8):
            self._pub(open_cmd)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(dt)

        while time.time() - t0 < float(self.args.timeout_s):
            started = time.perf_counter()
            rclpy.spin_once(self, timeout_sec=0.0)
            if all(j in self._js for j in ARM_JOINTS):
                q = {j: float(self._js[j]) for j in ARM_JOINTS}

            if self.args.yaml_aim:
                tip = [float(self.goal["x"]), float(self.goal["y"]), float(self.goal["z"])]
                yaw = float(self.goal.get("wrist_rotate", 0.0))
            else:
                twin = self._twin()
                tip_raw, yaw_raw = tip_from_twin(twin, self.goal)
                tip, yaw, _aim = project_reachable_tip(
                    tip_raw, yaw_raw, wrist_max=self.wmax, goal=self.goal
                )
            view = self._wide_view()
            if view.in_window:
                view_ok_once = True
                trial.wide_in_view = True
            else:
                trial.wide_lost_any = True

            # Keep a grasp-pool record for IL labels; motion during approach is joint-space.
            chosen, pool = pick_grasp(
                tip,
                yaw,
                wrist_max=self.wmax,
                seed=q,
                current_wrist_rotate=q["wrist_rotate"],
                scorer=self._scorer,
                prefer_phase="grasp",
            )
            trial.pool_size = max(trial.pool_size, len(pool))
            trial.reachable = len(pool)
            if chosen is not None:
                trial.chosen = {
                    "phase": chosen.phase,
                    "tip": chosen.tip,
                    "yaw": chosen.yaw,
                    "height_off": chosen.height_off,
                    "geo_score": chosen.geo_score,
                    "il_score": chosen.il_score,
                    "ik_err": chosen.ik_err,
                }
            else:
                trial.ik_live = False

            tip_now = eef_of(
                [q["waist"], q["shoulder"], q["wrist_angle"], q["wrist_rotate"], q["gripper"]]
            )
            dist = math.dist(tip_now, tip)

            if phase == "grasp":
                if freeze_q is None:
                    freeze_q = dict(q)
                # Freeze shoulder / wrist_angle / depth; keep rotate+waist jaw servo
                # while the pads close so the shaft does not walk out.
                target = dict(freeze_q)
                target["gripper"] = CLOSED_GRIP
                if view.detected:
                    target["wrist_rotate"] = max(
                        -1.2, min(1.2, -float(view.from_vert))
                    )
                    # Small capped waist trim only — large u spikes from a
                    # flickering detector yanked the arm off the foam.
                    du = max(-0.15, min(0.15, 0.50 - float(view.u)))
                    target["waist"] = max(
                        -1.0,
                        min(1.0, float(freeze_q["waist"]) + 0.20 * du),
                    )
                    freeze_q["wrist_rotate"] = target["wrist_rotate"]
                    freeze_q["waist"] = target["waist"]
                step_alpha = 0.7
            else:
                # Approach from Home — no IK tip chase.
                target = dict(q)
                target["gripper"] = GRIP_OPEN
                target["shoulder"] = shoulder_down
                target["wrist_angle"] = wrist_up
                if view.detected:
                    # Wrist rotate servos pad alignment with the shaft (tilt).
                    # Wide-cam does not spin with the rotate stator — waist owns u.
                    target["wrist_rotate"] = max(
                        -1.2, min(1.2, -float(view.from_vert))
                    )
                    target["waist"] = max(
                        -1.8,
                        min(
                            1.8,
                            float(q["waist"]) + 0.45 * (0.50 - float(view.u)),
                        ),
                    )
                    # Shaft low in image: ease tip-up a little so it settles in jaws.
                    if view.v > JAW_V[1]:
                        target["wrist_angle"] = max(
                            float(self.wmax), wrist_up - 0.12
                        )
                else:
                    # Blind: rotate ~0, waist toward yaml aim.
                    target["wrist_rotate"] = 0.0
                    aim_waist = math.atan2(float(tip[1]), max(float(tip[0]), 1e-3))
                    target["waist"] = max(-1.0, min(1.0, aim_waist))
                # Never dive past taught floor.
                target["wrist_angle"] = max(
                    float(self.wmax), float(target["wrist_angle"])
                )
                step_alpha = max(0.35, alpha)

            cmd = blend_toward_ik(q, target, step_alpha)
            if trial.steps % 24 == 1:
                self.get_logger().info(
                    f"step {trial.steps} phase={phase} "
                    f"view u={view.u:.2f} v={view.v:.2f} in={view.in_window} "
                    f"det={view.detected} sh={cmd['shoulder']:.2f} "
                    f"wr={cmd['wrist_angle']:.2f} rot={cmd['wrist_rotate']:.2f} "
                    f"wa={cmd['waist']:.2f}"
                )
            self._pub(cmd)
            last_cmd = cmd
            trial.steps += 1

            # Tip-up for pad wrap keeps v a bit high in-frame; require u tight
            # in the jaw column and v not below the pads.
            jaw_ready = view.detected and abs(view.u - 0.5) < 0.10 and (
                view.in_window
                or ((JAW_V[0] - 0.22) <= view.v <= JAW_V[1])
            )
            if phase == "approach" and jaw_ready:
                jaw_ok_hits += 1
            else:
                jaw_ok_hits = max(0, jaw_ok_hits - 1)

            sh_ok = abs(float(q["shoulder"]) - shoulder_down) < 0.12
            wr_ok = float(q["wrist_angle"]) > (float(self.wmax) + 0.18)
            if (
                phase == "approach"
                and jaw_ok_hits >= 8
                and sh_ok
                and wr_ok
            ):
                phase = "grasp"
                close_started = True
                view_ok_once = True
                trial.wide_in_view = True
                self.get_logger().info(
                    f"jaws ready → grasp u={view.u:.2f} v={view.v:.2f} "
                    f"sh={q['shoulder']:.2f} wr={q['wrist_angle']:.2f} "
                    f"rot={q['wrist_rotate']:.2f}"
                )
            elif (
                phase == "approach"
                and self.args.allow_blind_close
                and dist < 0.035
                and trial.steps > int(5.0 * self.args.hz)
            ):
                phase = "grasp"
                close_started = True
                self.get_logger().warn(
                    f"blind close (allow_blind_close) dist={dist:.3f} — not a plan success"
                )

            if close_started:
                if close_t0 is None:
                    close_t0 = time.time()
                # Foam stall only after the jaw has had time to travel, and only
                # when measured grip has settled open — not mid-swing from OPEN.
                closing_s = time.time() - close_t0
                grip_now = float(q["gripper"])
                settled = (
                    prev_grip is not None and abs(grip_now - prev_grip) < 0.04
                )
                # Real foam keeps the jaw clearly open; ~CLOSED+0.25 is air-close.
                foam_gap = grip_now > (CLOSED_GRIP + 0.40)
                if (
                    closing_s >= 0.9
                    and cmd["gripper"] < -0.4
                    and foam_gap
                    and settled
                ):
                    stall_hits += 1
                else:
                    stall_hits = max(0, stall_hits - 1)
                prev_grip = grip_now
                if stall_hits >= 10:
                    trial.stalled = True
                    self.get_logger().info(
                        f"foam stall joint={grip_now:.3f} — lifting"
                    )
                    break

            remaining = dt - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)

        # Squeeze dwell, then slow lift: decrease shoulder (raises tip on this 4-DOF).
        if trial.stalled:
            squeeze = dict(last_cmd)
            squeeze["gripper"] = CLOSED_GRIP
            for _ in range(int(0.8 * self.args.hz)):
                self._pub(squeeze)
                rclpy.spin_once(self, timeout_sec=0.0)
                time.sleep(dt)
            lift = dict(squeeze)
            shoulder0 = float(lift["shoulder"])
            n_lift = max(int(3.0 * self.args.hz), 16)
            for i in range(n_lift):
                frac = (i + 1) / float(n_lift)
                lift["shoulder"] = shoulder0 - 0.12 * frac
                lift["gripper"] = CLOSED_GRIP
                self._pub(lift)
                rclpy.spin_once(self, timeout_sec=0.0)
                time.sleep(dt)
            trial.lifted = True
            # Held if jaw still not at hard-close air gap after lift (median of samples).
            grips: list[float] = []
            for _ in range(8):
                rclpy.spin_once(self, timeout_sec=0.05)
                if "gripper" in self._js:
                    grips.append(float(self._js["gripper"]))
                time.sleep(0.05)
            grip_med = float(np.median(grips)) if grips else CLOSED_GRIP
            if grip_med > (CLOSED_GRIP + 0.12):
                trial.held = True
            self.get_logger().info(
                f"post-lift grip_med={grip_med:.3f} held={trial.held} "
                f"thr={CLOSED_GRIP + 0.12:.3f}"
            )

        path = append_trial(trial)
        self.get_logger().info(
            f"trial logged {path} stalled={trial.stalled} held={trial.held} "
            f"wide_ok={trial.wide_in_view} lost_any={trial.wide_lost_any} "
            f"steps={trial.steps} reachable={trial.reachable}"
        )
        return 0 if trial.held else (1 if trial.stalled else 2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--goal", type=Path, default=None)
    ap.add_argument("--hub", default="http://127.0.0.1:8082", help="twin / pole aim (REAL tip-mirror)")
    ap.add_argument(
        "--cam",
        default="http://127.0.0.1:8081",
        help="wide.jpg host (REAL MJPEG :8081; empty = use --hub)",
    )
    ap.add_argument("--hz", type=float, default=12.0)
    ap.add_argument("--timeout-s", type=float, default=20.0)
    ap.add_argument("--blend", type=float, default=0.35, help="IK blend per tick")
    ap.add_argument("--ranker", type=Path, default=None, help="optional IL scorer .pt/.json")
    ap.add_argument(
        "--note",
        default="static",
        help="Logged on the trial. Use 'nudge' / 'nudged' for mid-approach pole moves.",
    )
    ap.add_argument(
        "--yaml-aim",
        action="store_true",
        help="Ignore live twin XY; use taught rl_goal.yaml tip (reachable under wrist cap).",
    )
    ap.add_argument(
        "--allow-blind-close",
        action="store_true",
        help="Close by tip distance alone if wide never sees the shaft (smoke only).",
    )
    args = ap.parse_args()
    if args.cam == "":
        args.cam = args.hub
    if args.goal is None:
        from rl_goal_lib import goal_yaml_path

        args.goal = goal_yaml_path()
    if rclpy is None:
        print("rclpy missing", file=sys.stderr)
        return 3
    rclpy.init()
    node = IkGraspServo(args)
    try:
        return node.run()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
