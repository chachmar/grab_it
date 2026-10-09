#!/usr/bin/env python3
"""Save current map->base_link pose into named_locations.yaml."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import rclpy
import yaml
from rclpy.duration import Duration
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener, TransformException


DEFAULT_YAML = Path("/workspaces/isaac_ros-dev/config/named_locations.yaml")


def quat_to_yaw(x, y, z, w) -> float:
    # yaw (Z) from quaternion
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class SavePose(Node):
    def __init__(self):
        super().__init__("save_named_location")
        self.buf = Buffer(cache_time=Duration(seconds=30.0))
        self.listener = TransformListener(self.buf, self)

    def get_pose(self, timeout_sec: float = 10.0):
        deadline = self.get_clock().now() + Duration(seconds=timeout_sec)
        while rclpy.ok() and self.get_clock().now() < deadline:
            try:
                tf = self.buf.lookup_transform("map", "base_link", rclpy.time.Time())
                t = tf.transform.translation
                q = tf.transform.rotation
                yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
                return {
                    "frame_id": "map",
                    "x": float(t.x),
                    "y": float(t.y),
                    "z": 0.0,
                    "yaw": float(yaw),
                }
            except TransformException:
                rclpy.spin_once(self, timeout_sec=0.2)
        raise RuntimeError("Timed out waiting for map -> base_link (is nav/RTAB up?)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="Name to store (e.g. kitchen)")
    parser.add_argument("--yaml", type=Path, default=DEFAULT_YAML)
    args = parser.parse_args()

    rclpy.init()
    node = SavePose()
    try:
        pose = node.get_pose()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

    path = args.yaml
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if path.is_file():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    locs = data.setdefault("locations", {})
    locs[args.name] = pose
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(
        f"Saved '{args.name}' -> {path}\n"
        f"  x={pose['x']:.3f} y={pose['y']:.3f} yaw={pose['yaw']:.3f}"
    )


if __name__ == "__main__":
    main()
