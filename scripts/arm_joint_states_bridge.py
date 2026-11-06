#!/usr/bin/env python3
"""Keep grab_it robot_state_publisher fed with arm joint_states.

Nano xs_sdk publishes /px150_no_elbow/joint_states. When the U2D2 is unplugged
or DDS drops briefly, snapping to Home zeros makes RViz arm links "pulse".

This bridge always publishes /px150_no_elbow/joint_states_rsp:
  - live Nano / GUI states when fresh
  - last good live states while stale (no snap)
  - Home defaults only if we have never received live data

For NexArm (GRAB_IT_ARM=nexarm), the authoritative topic is
/grab_it/nexarm_joint_states (GUI USB). Follower radians are converted to URDF
model radians so RSP TF matches the physical pose. The PX150 topic is ignored
for NexArm names while the NexArm topic is fresh.
"""
from __future__ import annotations

import math
import os
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

NEXARM_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
    "pan",
]
# Park-ish model-space defaults (real park ≈ lift +1.11 / elbow -1.00 / wrist +0.16).
NEXARM_DEFAULT_POS = [
    0.0,
    -1.11,
    -math.pi / 2.0 + 1.00,
    0.16 + math.pi / 2.0,
    0.0,
    0.025,
    0.0,
]

# How long without live data before we *warn* (still hold last good).
STALE_WARN_SEC = 1.0
# Prefer /grab_it/nexarm_joint_states over the PX150 topic while fresher than this.
NEXARM_FRESH_SEC = 0.5

# Physical follower → URDF model (matches nexarm_ik.real_to_model_q).
_NEXARM_SIGN = {
    "shoulder_lift": -1.0,
    "elbow_flex": -1.0,
    "wrist_flex": 1.0,
}
_NEXARM_OFFSET = {
    "shoulder_lift": 0.0,
    "elbow_flex": -math.pi / 2.0,
    "wrist_flex": math.pi / 2.0,
}


class ArmJointStatesBridge(Node):
    def __init__(self) -> None:
        super().__init__("arm_joint_states_bridge")
        self._last_msg: JointState | None = None
        self._last_t = 0.0
        self._nexarm_t = 0.0
        self._warned_stale = False
        self._nexarm = (
            os.environ.get("GRAB_IT_ARM", "").strip().lower() == "nexarm"
        )
        self._pub = self.create_publisher(
            JointState, "/px150_no_elbow/joint_states_rsp", _RSP_QOS
        )
        self.create_subscription(
            JointState,
            "/px150_no_elbow/joint_states",
            self._cb_px150,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            JointState,
            "/grab_it/nexarm_joint_states",
            self._cb_nexarm,
            qos_profile_sensor_data,
        )
        self.create_timer(0.05, self._tick)  # 20 Hz
        mode = (
            "nexarm from /grab_it/nexarm_joint_states"
            if self._nexarm
            else "px150 passthrough (+nexarm if seen)"
        )
        self.get_logger().info(
            "Bridging arm joints -> /px150_no_elbow/joint_states_rsp "
            f"({mode}; hold last-good if stale)"
        )

    @staticmethod
    def _is_live(pos: float) -> bool:
        # xs_sdk publishes ~-pi when Dynamixel syncRead fails.
        return abs(float(pos) + _PI) > 0.05

    def _merge_nexarm_real(self, live_real: dict[str, float]) -> None:
        """Map real follower radians into model space and store as last-good."""
        defaults = dict(zip(NEXARM_NAMES, NEXARM_DEFAULT_POS))
        merged: dict[str, float] = dict(defaults)
        if self._last_msg is not None:
            for n, p in zip(self._last_msg.name, self._last_msg.position):
                merged[n] = float(p)
        for name, p in live_real.items():
            if name in _NEXARM_SIGN:
                merged[name] = (
                    float(_NEXARM_SIGN[name]) * float(p)
                    + float(_NEXARM_OFFSET[name])
                )
            else:
                # pan / gripper / shoulder_pan: real == model.
                merged[name] = float(p)
        names = [n for n in NEXARM_NAMES if n in merged]
        for n in merged:
            if n not in names:
                names.append(n)
        out = JointState()
        out.name = names
        out.position = [merged[n] for n in names]
        self._last_msg = out
        self._last_t = time.monotonic()
        self._nexarm = True
        if self._warned_stale:
            self.get_logger().info("Live arm joint_states restored")
            self._warned_stale = False

    def _cb_nexarm(self, msg: JointState) -> None:
        if not msg.name or not msg.position:
            return
        live_real: dict[str, float] = {}
        for name, pos in zip(msg.name, msg.position):
            p = float(pos)
            if self._is_live(p):
                live_real[name] = p
        if not live_real:
            return
        self._nexarm_t = time.monotonic()
        self._merge_nexarm_real(live_real)

    def _cb_px150(self, msg: JointState) -> None:
        # Ignore empty / all-NaN bursts
        if not msg.name or not msg.position:
            return
        # Prefer the NexArm GUI topic while it is fresh — PX150 may carry
        # stale NexArm names from an old tree and freeze the camera TF.
        if self._nexarm and (time.monotonic() - self._nexarm_t) < NEXARM_FRESH_SEC:
            return
        is_nexarm = self._nexarm or ("shoulder_pan" in msg.name)
        if is_nexarm:
            live_real: dict[str, float] = {}
            for name, pos in zip(msg.name, msg.position):
                p = float(pos)
                if self._is_live(p):
                    live_real[name] = p
            if not live_real:
                return
            self._merge_nexarm_real(live_real)
            return

        merged = dict(zip(DEFAULT_NAMES, DEFAULT_POS))
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
                    "(check /grab_it/nexarm_joint_states or Nano xs_sdk)"
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
        if self._nexarm:
            out.name = list(NEXARM_NAMES)
            out.position = list(NEXARM_DEFAULT_POS)
        else:
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
