#!/usr/bin/env python3
"""Relay IL joint_targets → agx_isaac_hub /isaac/cmd so the Windows twin moves.

Humble play_policy publishes ROS joint commands; the Isaac twin primarily follows
HTTP hub arm snapshots (Jazzy↔Humble custom msgs are unreliable). This node
bridges the gap for ACT verify on sim.
"""
from __future__ import annotations

import json
import os
import urllib.request

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState

HUB = os.environ.get("GRAB_IT_AGX_HUB", "http://127.0.0.1:8081/isaac/cmd")
NAMES = ["waist", "shoulder", "wrist_angle", "wrist_rotate", "gripper", "pan"]
QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)


class TwinCmdRelay(Node):
    def __init__(self) -> None:
        super().__init__("il_twin_cmd_relay")
        self._last = 0.0
        self._posts = 0
        self.create_subscription(
            JointState, "/px150_no_elbow/commands/joint_targets", self._cb, QOS
        )
        self.get_logger().info(f"relaying joint_targets → {HUB}")

    def _cb(self, msg: JointState) -> None:
        import time

        now = time.monotonic()
        if now - self._last < 0.04:  # ~25 Hz max
            return
        self._last = now
        by_name = {
            str(n): float(msg.position[i])
            for i, n in enumerate(msg.name)
            if i < len(msg.position)
        }
        pos = [by_name.get(n, 0.0) for n in NAMES[:-1]]
        # Keep pan from last hub value if absent
        pos.append(by_name.get("pan", 0.0))
        body = json.dumps({"t": "arm", "n": NAMES, "p": pos}).encode()
        try:
            req = urllib.request.Request(
                HUB, data=body, headers={"Content-Type": "application/json"}, method="POST"
            )
            urllib.request.urlopen(req, timeout=0.5).read()
            self._posts += 1
            if self._posts == 1 or self._posts % 50 == 0:
                self.get_logger().info(f"hub POST ok n={self._posts} waist={pos[0]:.3f}")
            self._err = False
        except Exception as exc:
            if not getattr(self, "_err", False):
                self.get_logger().warn(f"hub POST failed: {exc}")
                self._err = True


def main() -> None:
    rclpy.init()
    node = TwinCmdRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
