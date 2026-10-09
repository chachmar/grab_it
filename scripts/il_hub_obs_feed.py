#!/usr/bin/env python3
"""Feed hub JPEG + arm snapshot into ROS topics for host twin ACT play.

Isaac (Jazzy) ↔ AGX (Humble) DDS is unreliable, so cameras/joint_states come
from agx_isaac_hub instead of direct ROS from the twin.
"""
from __future__ import annotations

import json
import os
import urllib.request

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, JointState

HUB = os.environ.get("GRAB_IT_AGX_HUB_BASE", "http://127.0.0.1:8081").rstrip("/")
NAMES = ["waist", "shoulder", "wrist_angle", "wrist_rotate", "gripper", "pan"]
QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class HubObsFeed(Node):
    def __init__(self) -> None:
        super().__init__("il_hub_obs_feed")
        self._js_pub = self.create_publisher(
            JointState, "/px150_no_elbow/joint_states", QOS
        )
        self._d435i_pub = self.create_publisher(
            CompressedImage, "/d435i/d435i/color/image_raw/compressed", QOS
        )
        self._wrist_pub = self.create_publisher(
            CompressedImage, "/wide_cam/image_raw/compressed", QOS
        )
        self._frames = 0
        self.create_timer(1.0 / 20.0, self._tick)
        self.get_logger().info(f"hub obs feed from {HUB}")

    def _get(self, path: str) -> bytes | None:
        try:
            with urllib.request.urlopen(f"{HUB}{path}", timeout=0.4) as resp:
                if resp.status != 200:
                    return None
                return resp.read()
        except Exception:
            return None

    def _tick(self) -> None:
        # Prefer measured twin joints when available (hub /isaac/cmd is commanded
        # and diverges when keep-out rejects through-body ACT actions).
        raw_twin = self._get("/isaac/twin")
        used_measured = False
        if raw_twin:
            try:
                twin = json.loads(raw_twin.decode())
                names = list(twin.get("arm_n") or [])
                pos = [float(x) for x in (twin.get("arm_p") or [])]
                if names and pos and len(pos) >= 4:
                    msg = JointState()
                    msg.header.stamp = self.get_clock().now().to_msg()
                    msg.name = names
                    msg.position = pos[: len(names)]
                    self._js_pub.publish(msg)
                    used_measured = True
            except Exception:
                pass
        if not used_measured:
            raw = self._get("/isaac/cmd")
            if raw:
                try:
                    snap = json.loads(raw.decode())
                    names = list(snap.get("n") or NAMES)
                    pos = [float(x) for x in (snap.get("p") or [])]
                    if names and pos:
                        msg = JointState()
                        msg.header.stamp = self.get_clock().now().to_msg()
                        msg.name = names
                        msg.position = pos[: len(names)]
                        self._js_pub.publish(msg)
                except Exception:
                    pass

        for path, pub, fmt in (
            ("/d435i.jpg", self._d435i_pub, "jpeg"),
            ("/wide.jpg", self._wrist_pub, "jpeg"),
        ):
            blob = self._get(path)
            if not blob:
                continue
            img = CompressedImage()
            img.header.stamp = self.get_clock().now().to_msg()
            img.format = fmt
            img.data = blob
            pub.publish(img)

        self._frames += 1
        if self._frames == 1 or self._frames % 100 == 0:
            self.get_logger().info(f"hub obs frames={self._frames}")


def main() -> None:
    rclpy.init()
    node = HubObsFeed()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
