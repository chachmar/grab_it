#!/usr/bin/env python3
"""Keep grab_it robot_state_publisher fed with arm joint_states.

Nano xs_sdk publishes /px150_no_elbow/joint_states. When the U2D2 is unplugged
or DDS drops briefly, snapping to Home zeros makes RViz arm links "pulse".

This bridge always publishes /px150_no_elbow/joint_states_rsp:
  - live Nano states when fresh
  - last good live states while stale (no snap)
  - Home defaults only if we have never received live data
"""
from __future__ import annotations

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import JointState

_PI = math.pi

# robot_state_publisher subscribes Reliable; sensor_data (Best Effort) never
# matches and the pan/camera TF chain stays disconnected.
_RSP_QOS = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
)

DEFAULT_NAMES = [
    "waist",
    "shoulder",
    "wrist_angle",
    "wrist_rotate",
    "gripper",
    "pan",
    "left_finger",
    "right_finger",
]
# Hardware-safe sleep-ish pose for boom-removed arm (visible, not stock zeros).
DEFAULT_POS = [0.0, -0.2321, -1.7279, 0.0, 0.0, 0.0, 0.037, -0.037]

# How long without live data before we *warn* (still hold last good).
STALE_WARN_SEC = 1.0


class ArmJointStatesBridge(Node):
    def __init__(self) -> None:
        super().__init__("arm_joint_states_bridge")
        self._last_msg: JointState | None = None
        self._last_t = 0.0
        self._warned_stale = False
        self._pub = self.create_publisher(
            JointState, "/px150_no_elbow/joint_states_rsp", _RSP_QOS
        )
        self.create_subscription(
            JointState,
            "/px150_no_elbow/joint_states",
            self._cb,
            qos_profile_sensor_data,
        )
        self.create_timer(0.05, self._tick)  # 20 Hz
        self.get_logger().info(
            "Bridging /px150_no_elbow/joint_states -> "
            "/px150_no_elbow/joint_states_rsp (hold last-good if stale)"
        )

    @staticmethod
    def _is_live(pos: float) -> bool:
        # xs_sdk publishes ~-pi when Dynamixel syncRead fails.
        return abs(float(pos) + _PI) > 0.05

    def _cb(self, msg: JointState) -> None:
        # Ignore empty / all-NaN bursts
        if not msg.name or not msg.position:
            return
        # Merge live readings onto last-good / defaults. Dead -pi joints (bus
        # blips) must not drop pan/arm out of the message — RSP then omits those
        # links and click-to-grasp loses camera TF.
        merged: dict[str, float] = dict(zip(DEFAULT_NAMES, DEFAULT_POS))
        if self._last_msg is not None:
            for n, p in zip(self._last_msg.name, self._last_msg.position):
                merged[n] = float(p)
        got_live = False
        for name, pos in zip(msg.name, msg.position):
            p = float(pos)
            if self._is_live(p):
                merged[name] = p
                got_live = True
        if not got_live:
            return
        # Prefer stable joint order for RSP.
        names = [n for n in DEFAULT_NAMES if n in merged]
        for n in merged:
            if n not in names:
                names.append(n)
        out = JointState()
        out.header = msg.header
        out.name = names
        out.position = [merged[n] for n in names]
        self._last_msg = out
        self._last_t = time.monotonic()
        if self._warned_stale:
            self.get_logger().info("Live /px150_no_elbow/joint_states restored")
            self._warned_stale = False

    def _tick(self) -> None:
        now = time.monotonic()
        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()

        if self._last_msg is not None:
            age = now - self._last_t
            if age > STALE_WARN_SEC and not self._warned_stale:
                self.get_logger().warn(
                    f"joint_states stale ({age:.1f}s) — holding last pose "
                    "(check Nano xs_sdk / U2D2 / 12V)"
                )
                self._warned_stale = True
            # Republish last good with fresh stamp (avoid TF extrapolation issues)
            out.name = list(self._last_msg.name)
            out.position = list(self._last_msg.position)
            if self._last_msg.velocity:
                out.velocity = list(self._last_msg.velocity)
            if self._last_msg.effort:
                out.effort = list(self._last_msg.effort)
            self._pub.publish(out)
            return

        # Never seen live data — publish a static default so links still appear.
        out.name = list(DEFAULT_NAMES)
        out.position = list(DEFAULT_POS)
        self._pub.publish(out)


def main() -> None:
    rclpy.init()
    node = ArmJointStatesBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
