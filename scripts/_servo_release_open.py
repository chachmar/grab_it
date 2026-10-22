#!/usr/bin/env python3
"""Open the gripper at the current arm pose (drop held foam before Home)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import rclpy
from interbotix_xs_msgs.msg import JointGroupCommand, JointSingleCommand
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

ARM = ["waist", "shoulder", "wrist_angle", "wrist_rotate"]


class Release(Node):
    def __init__(self) -> None:
        super().__init__("servo_release_open")
        self._js: dict[str, float] = {}
        self.create_subscription(
            JointState, "/px150_no_elbow/joint_states", self._on, qos_profile_sensor_data
        )
        self._arm = self.create_publisher(
            JointGroupCommand, "/px150_no_elbow/commands/joint_group", 10
        )
        self._grip = self.create_publisher(
            JointSingleCommand, "/px150_no_elbow/commands/joint_single", 10
        )

    def _on(self, msg: JointState) -> None:
        for n, p in zip(msg.name, msg.position):
            self._js[str(n)] = float(p)

    def run(self) -> int:
        t0 = time.time()
        while time.time() - t0 < 8.0:
            rclpy.spin_once(self, timeout_sec=0.1)
            if all(j in self._js for j in ARM):
                break
        else:
            print("no joint_states", flush=True)
            return 2
        for _ in range(40):
            arm = JointGroupCommand()
            arm.name = "arm"
            arm.cmd = [float(self._js[j]) for j in ARM]
            self._arm.publish(arm)
            grip = JointSingleCommand()
            grip.name = "gripper"
            grip.cmd = 1.65
            self._grip.publish(grip)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(0.05)
        g = float(self._js.get("gripper", 9.0))
        print(f"released gripper={g:.3f}", flush=True)
        return 0 if g > 0.8 else 1


def main() -> int:
    rclpy.init()
    node = Release()
    try:
        return node.run()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
