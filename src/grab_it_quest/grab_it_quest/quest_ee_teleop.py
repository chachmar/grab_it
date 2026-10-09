#!/usr/bin/env python3
"""Quest 3 → PX150 4-DOF teleop (same joint topics as the GUI / Isaac twin)."""
from __future__ import annotations

import json
import math
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped, Quaternion, TransformStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState, Joy
from std_msgs.msg import String

from grab_it_quest.px150_ik import JOINT_NAMES, clamp, fk_xyz, ik_position

try:
    from interbotix_xs_msgs.msg import JointGroupCommand, JointSingleCommand
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "interbotix_xs_msgs is required (same messages as the GUI / xs_sdk)."
    ) from exc

try:
    import tf2_ros
    from tf2_ros import TransformException
except ImportError:  # pragma: no cover
    tf2_ros = None
    TransformException = Exception


def _clamp3(p, lo, hi):
    return [
        max(lo[i], min(hi[i], float(p[i])))
        for i in range(3)
    ]


def _quat_inv(q: Quaternion) -> tuple[float, float, float, float]:
    return (-q.x, -q.y, -q.z, q.w)


def _quat_mul(a, b) -> tuple[float, float, float, float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _twist_about(q_rel, axis: str) -> float:
    if axis == "x":
        return 2.0 * math.atan2(q_rel[0], q_rel[3])
    if axis == "y":
        return 2.0 * math.atan2(q_rel[1], q_rel[3])
    return 2.0 * math.atan2(q_rel[2], q_rel[3])


def _tf_xyz(t: TransformStamped):
    p = t.transform.translation
    return float(p.x), float(p.y), float(p.z)


def _tf_quat(t: TransformStamped):
    q = t.transform.rotation
    return (float(q.x), float(q.y), float(q.z), float(q.w))


def _rotate_vec(q, v):
    qx, qy, qz, qw = q
    vx, vy, vz = v
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


def _apply_tf(txyz, tq, p):
    r = _rotate_vec(tq, p)
    return (txyz[0] + r[0], txyz[1] + r[1], txyz[2] + r[2])


class QuestEeTeleop(Node):
    def __init__(self) -> None:
        super().__init__("quest_ee_teleop")
        self.declare_parameter("target", "sim")
        self.declare_parameter("allow_both", False)
        self.declare_parameter("rate_hz", 40.0)
        self.declare_parameter("scale", 1.0)
        self.declare_parameter("real_scale", 0.4)
        self.declare_parameter("clutch_mode", "grip")
        self.declare_parameter("trigger_mode", "toggle")
        self.declare_parameter("twist_axis", "z")
        self.declare_parameter("twist_sign", 1.0)
        self.declare_parameter("waist_offset", 0.0)
        self.declare_parameter("planning_frame", "px150_no_elbow/base_link")
        self.declare_parameter("pose_fallback_frame", "base_link")
        self.declare_parameter("joint_stale_s", 0.5)
        self.declare_parameter("workspace_min", [0.05, -0.35, 0.00])
        self.declare_parameter("workspace_max", [0.42, 0.35, 0.45])
        self.declare_parameter("gripper_open", 1.65)
        self.declare_parameter("gripper_closed", -0.95)
        self.declare_parameter("speed_waist", 1.2)
        self.declare_parameter("speed_shoulder", 0.8)
        self.declare_parameter("speed_wrist_angle", 1.0)
        self.declare_parameter("speed_wrist_rotate", 1.5)
        self.declare_parameter("squeeze_threshold", 0.50)
        self.declare_parameter("trigger_threshold", 0.55)
        self.declare_parameter("right_pose_topic", "/grab_it/quest/right_pose")
        self.declare_parameter("joy_topic", "/grab_it/quest/joy")
        self.declare_parameter("ee_target_topic", "/grab_it/quest/ee_target")
        self.declare_parameter("status_topic", "/grab_it/quest/status")
        self.declare_parameter("il_status_topic", "/grab_it/il_status")
        self.declare_parameter("joint_states_topic", "/px150_no_elbow/joint_states")
        self.declare_parameter("joint_group_topic", "/px150_no_elbow/commands/joint_group")
        self.declare_parameter("joint_single_topic", "/px150_no_elbow/commands/joint_single")
        self.declare_parameter(
            "joint_targets_topic", "/px150_no_elbow/commands/joint_targets"
        )

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._arm_pub = self.create_publisher(
            JointGroupCommand, self._s("joint_group_topic"), qos
        )
        self._grip_pub = self.create_publisher(
            JointSingleCommand, self._s("joint_single_topic"), qos
        )
        self._js_cmd_pub = self.create_publisher(
            JointState, self._s("joint_targets_topic"), qos
        )
        self._ee_pub = self.create_publisher(
            PoseStamped, self._s("ee_target_topic"), qos
        )
        self._status_pub = self.create_publisher(String, self._s("status_topic"), qos)
        self.create_subscription(
            PoseStamped, self._s("right_pose_topic"), self._on_pose, qos
        )
        self.create_subscription(Joy, self._s("joy_topic"), self._on_joy, qos)
        self.create_subscription(
            JointState, self._s("joint_states_topic"), self._on_js, qos
        )
        self.create_subscription(String, self._s("il_status_topic"), self._on_il, 10)

        if tf2_ros is not None:
            self._tf_buf = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(self._tf_buf, self)
        else:
            self._tf_buf = None
            self._tf_listener = None

        self._pose: Optional[PoseStamped] = None
        self._joy: Optional[Joy] = None
        self._joy_seen = False
        self._pos: dict[str, float] = {}
        self._js_stamp = 0.0
        self._cmd = {n: 0.0 for n in JOINT_NAMES}
        self._have_cmd = False
        self._gripper_open = True
        self._clutched = False
        self._clutch_ctrl = None
        self._clutch_ee = None
        self._clutch_wr = 0.0
        self._last_trig = False
        self._il_playing = False
        self._last_status = ""
        self._hb = 0

        hz = max(5.0, float(self.get_parameter("rate_hz").value))
        self._dt = 1.0 / hz
        self.create_timer(self._dt, self._tick)
        self.get_logger().info(
            f"Quest EE teleop target={self._target()} clutch={self._s('clutch_mode')} "
            f"hz={hz:.0f}"
        )

    def _s(self, name: str) -> str:
        return str(self.get_parameter(name).value)

    def _b(self, name: str) -> bool:
        v = self.get_parameter(name).value
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes", "on")
        return bool(v)

    def _f(self, name: str) -> float:
        return float(self.get_parameter(name).value)

    def _target(self) -> str:
        return str(self.get_parameter("target").value or "sim").strip().lower()

    def _uses_real(self) -> bool:
        return self._target() in ("real", "both")

    def _on_pose(self, msg: PoseStamped) -> None:
        self._pose = msg

    def _on_joy(self, msg: Joy) -> None:
        self._joy = msg
        self._joy_seen = True

    def _on_js(self, msg: JointState) -> None:
        for name, pos in zip(msg.name, msg.position):
            key = name.split("/")[-1]
            if key in JOINT_NAMES or key == "gripper":
                self._pos[key] = float(pos)
        self._js_stamp = self.get_clock().now().nanoseconds * 1e-9
        if not self._have_cmd and all(n in self._pos for n in JOINT_NAMES):
            self._cmd = {n: float(self._pos[n]) for n in JOINT_NAMES}
            self._have_cmd = True

    def _on_il(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except Exception:
            data = {}
        if isinstance(data, dict) and "playing" in data:
            self._il_playing = bool(data.get("playing"))

    def _js_fresh(self) -> bool:
        now = self.get_clock().now().nanoseconds * 1e-9
        return (now - self._js_stamp) <= self._f("joint_stale_s")

    def _squeeze(self) -> float:
        if self._joy is None or not self._joy.axes:
            return 0.0
        if len(self._joy.axes) > 1:
            return float(self._joy.axes[1])
        return float(self._joy.axes[0])

    def _trigger(self) -> float:
        if self._joy is None or not self._joy.axes:
            return 0.0
        return float(self._joy.axes[0])

    def _want_clutch(self) -> bool:
        mode = self._s("clutch_mode").strip().lower()
        if mode == "always":
            if self._uses_real():
                return False
            return self._pose is not None
        return self._squeeze() >= self._f("squeeze_threshold")

    def _scale(self) -> float:
        if self._uses_real():
            return self._f("real_scale")
        return self._f("scale")

    def _speed(self) -> dict[str, float]:
        s = {
            "waist": self._f("speed_waist"),
            "shoulder": self._f("speed_shoulder"),
            "wrist_angle": self._f("speed_wrist_angle"),
            "wrist_rotate": self._f("speed_wrist_rotate"),
        }
        if self._uses_real():
            return {k: v * 0.5 for k, v in s.items()}
        return s

    def _pose_in_base(self, msg: PoseStamped) -> Optional[tuple[tuple[float, float, float], Quaternion]]:
        src = (msg.header.frame_id or "odom").lstrip("/")
        dests = [
            self._s("planning_frame"),
            self._s("pose_fallback_frame"),
            "base_link",
        ]
        p = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
        if self._tf_buf is None or src in dests:
            return p, msg.pose.orientation
        for dest in dests:
            dest = dest.lstrip("/")
            try:
                tf = self._tf_buf.lookup_transform(dest, src, rclpy.time.Time())
            except Exception:
                continue
            t = _tf_xyz(tf)
            q = _tf_quat(tf)
            return _apply_tf(t, q, p), msg.pose.orientation
        return p, msg.pose.orientation

    def _publish_status(self, **extra) -> None:
        payload = {
            "clutched": self._clutched,
            "target": self._target(),
            "il_playing": self._il_playing,
            "js_fresh": self._js_fresh(),
            "joy": self._joy_seen,
        }
        payload.update(extra)
        raw = json.dumps(payload)
        if raw != self._last_status:
            self._last_status = raw
        self._status_pub.publish(String(data=raw))

    def _can_command(self) -> tuple[bool, str]:
        if self._il_playing:
            return False, "il_playing"
        tgt = self._target()
        if tgt not in ("sim", "real", "both"):
            return False, "bad_target"
        if tgt == "both" and not self._b("allow_both"):
            return False, "both_not_allowed"
        if self._uses_real() and not self._js_fresh():
            return False, "stale_joint_states"
        if self._uses_real() and self._s("clutch_mode").strip().lower() == "always":
            return False, "always_blocked_on_real"
        if not self._want_clutch():
            return False, "no_clutch"
        if self._pose is None:
            return False, "no_pose"
        if not self._have_cmd and not all(n in self._pos for n in JOINT_NAMES):
            if tgt == "sim":
                self._cmd = {
                    "waist": 0.0,
                    "shoulder": 0.1745,
                    "wrist_angle": -1.309,
                    "wrist_rotate": 0.0,
                }
                self._have_cmd = True
            else:
                return False, "no_joints"
        return True, "ok"

    def _engage_clutch(self) -> None:
        pose = self._pose_in_base(self._pose)
        if pose is None:
            return
        xyz, ori = pose
        if all(n in self._pos for n in JOINT_NAMES):
            self._cmd = {n: float(self._pos[n]) for n in JOINT_NAMES}
        ee = fk_xyz(self._cmd["waist"], self._cmd["shoulder"], self._cmd["wrist_angle"])
        self._clutch_ctrl = (xyz, ori)
        self._clutch_ee = ee
        self._clutch_wr = self._cmd["wrist_rotate"]
        self._clutched = True
        self.get_logger().info("Quest clutch on")

    def _release_clutch(self) -> None:
        if self._clutched:
            self.get_logger().info("Quest clutch off")
        self._clutched = False
        self._clutch_ctrl = None
        self._clutch_ee = None

    def _handle_trigger(self) -> None:
        pressed = self._trigger() >= self._f("trigger_threshold")
        mode = self._s("trigger_mode").strip().lower()
        rising = pressed and not self._last_trig
        self._last_trig = pressed
        if mode == "hold":
            want_closed = pressed
            if want_closed == (not self._gripper_open):
                return
            self._gripper_open = not want_closed
        else:
            if not rising:
                return
            self._gripper_open = not self._gripper_open
        msg = JointSingleCommand()
        msg.name = "gripper"
        msg.cmd = self._f("gripper_open") if self._gripper_open else self._f("gripper_closed")
        self._grip_pub.publish(msg)
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = ["gripper"]
        js.position = [float(msg.cmd)]
        self._js_cmd_pub.publish(js)

    def _tick(self) -> None:
        try:
            self._tick_inner()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"quest tick failed: {exc}")
            self._publish_status(reason="tick_error", error=str(exc))

    def _tick_inner(self) -> None:
        ok, reason = self._can_command()
        self._hb += 1
        if self._hb == 1 or self._hb % 80 == 0:
            self.get_logger().info(
                f"quest tick reason={reason} clutched={self._clutched} "
                f"pose={'yes' if self._pose else 'no'} "
                f"js_fresh={self._js_fresh()}"
            )
        if not ok:
            if self._clutched and reason != "ok":
                self._release_clutch()
            self._publish_status(reason=reason)
            return

        if not self._clutched:
            self._engage_clutch()
            if not self._clutched:
                self._publish_status(reason="clutch_failed")
                return

        self._handle_trigger()
        now = self._pose_in_base(self._pose)
        if now is None or self._clutch_ctrl is None or self._clutch_ee is None:
            self._publish_status(reason="pose_tf")
            return
        xyz, ori = now
        c_xyz, c_ori = self._clutch_ctrl
        scale = self._scale()
        delta = (
            (xyz[0] - c_xyz[0]) * scale,
            (xyz[1] - c_xyz[1]) * scale,
            (xyz[2] - c_xyz[2]) * scale,
        )
        target = [
            self._clutch_ee[0] + delta[0],
            self._clutch_ee[1] + delta[1],
            self._clutch_ee[2] + delta[2],
        ]
        lo = list(self.get_parameter("workspace_min").value)
        hi = list(self.get_parameter("workspace_max").value)
        target = _clamp3(target, lo, hi)

        q_rel = _quat_mul(_quat_inv(c_ori), (ori.x, ori.y, ori.z, ori.w))
        roll = _twist_about(q_rel, self._s("twist_axis")) * self._f("twist_sign")
        wr = clamp("wrist_rotate", self._clutch_wr + roll)

        sol = ik_position(
            target[0],
            target[1],
            target[2],
            wr,
            seed=self._cmd,
            waist_offset=self._f("waist_offset"),
        )
        reachable = sol is not None
        if sol is not None:
            spd = self._speed()
            for name in JOINT_NAMES:
                step = spd[name] * self._dt
                cur = self._cmd[name]
                nxt = max(cur - step, min(cur + step, sol[name]))
                self._cmd[name] = clamp(name, nxt)
            msg = JointGroupCommand()
            msg.name = "arm"
            msg.cmd = [self._cmd[n] for n in JOINT_NAMES]
            self._arm_pub.publish(msg)
            js = JointState()
            js.header.stamp = self.get_clock().now().to_msg()
            js.name = list(JOINT_NAMES)
            js.position = [self._cmd[n] for n in JOINT_NAMES]
            self._js_cmd_pub.publish(js)

        ee_msg = PoseStamped()
        ee_msg.header.stamp = self.get_clock().now().to_msg()
        ee_msg.header.frame_id = self._s("planning_frame")
        ee_msg.pose.position.x = float(target[0])
        ee_msg.pose.position.y = float(target[1])
        ee_msg.pose.position.z = float(target[2])
        ee_msg.pose.orientation.w = 1.0
        self._ee_pub.publish(ee_msg)
        self._publish_status(
            reason="ok" if reachable else "ik_fail",
            reachable=reachable,
            ee=[round(float(v), 3) for v in target],
            q=[round(self._cmd[n], 3) for n in JOINT_NAMES],
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = QuestEeTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
