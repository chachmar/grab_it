#!/usr/bin/env python3
"""Send Nav2 NavigateToPose for a named location from named_locations.yaml."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node


DEFAULT_YAML = Path("/workspaces/isaac_ros-dev/config/named_locations.yaml")


def yaw_to_quat(yaw: float):
    return (0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5))


def load_location(path: Path, name: str) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    locs = data.get("locations") or {}
    if name not in locs:
        known = ", ".join(sorted(locs)) or "(none)"
        raise SystemExit(f"Unknown location '{name}'. Known: {known}")
    return locs[name]


class GoToNamed(Node):
    def __init__(self, loc: dict, name: str):
        super().__init__("go_to_named")
        self._name = name
        self._loc = loc
        self._done = False
        self._ok = False
        self._client = ActionClient(self, NavigateToPose, "navigate_to_pose")

    def send(self):
        if not self._client.wait_for_server(timeout_sec=20.0):
            self.get_logger().error("navigate_to_pose action server not available")
            self._done = True
            return
        goal = NavigateToPose.Goal()
        pose = PoseStamped()
        pose.header.frame_id = self._loc.get("frame_id", "map")
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(self._loc["x"])
        pose.pose.position.y = float(self._loc["y"])
        pose.pose.position.z = float(self._loc.get("z", 0.0))
        qx, qy, qz, qw = yaw_to_quat(float(self._loc["yaw"]))
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        goal.pose = pose
        self.get_logger().info(
            f"Navigating to '{self._name}' "
            f"({pose.pose.position.x:.3f}, {pose.pose.position.y:.3f}, yaw={self._loc['yaw']:.3f})"
        )
        fut = self._client.send_goal_async(goal)
        fut.add_done_callback(self._on_goal)

    def _on_goal(self, fut):
        goal_handle = fut.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error("Goal rejected")
            self._done = True
            return
        self.get_logger().info("Goal accepted")
        res_fut = goal_handle.get_result_async()
        res_fut.add_done_callback(self._on_result)

    def _on_result(self, fut):
        result = fut.result()
        status = result.status
        # 4 = SUCCEEDED in action_msgs
        self._ok = status == 4
        self.get_logger().info(f"Finished '{self._name}' status={status} ok={self._ok}")
        self._done = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="Location name in YAML (e.g. kitchen)")
    parser.add_argument(
        "--yaml",
        type=Path,
        default=DEFAULT_YAML,
        help=f"Path to named_locations.yaml (default: {DEFAULT_YAML})",
    )
    args = parser.parse_args()
    if not args.yaml.is_file():
        raise SystemExit(f"YAML not found: {args.yaml}")
    loc = load_location(args.yaml, args.name)

    rclpy.init()
    node = GoToNamed(loc, args.name)
    node.send()
    try:
        while rclpy.ok() and not node._done:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    sys.exit(0 if node._ok else 1)


if __name__ == "__main__":
    main()
