#!/usr/bin/env python3
"""Slew the real PX150 to GUI Home, gripper open — safe start before servo."""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import rclpy
from interbotix_xs_msgs.msg import JointGroupCommand, JointSingleCommand
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

# Real Home from gui_node.HOME_JOINTS (not SIM_HOME).
HOME = {
    "waist": 0.0,
    "shoulder": 0.1745,
    "wrist_angle": -1.309,
    "wrist_rotate": 0.0,
    "gripper": 1.65,
}
ARM = ["waist", "shoulder", "wrist_angle", "wrist_rotate", "gripper"]
# Slow slew — shoulder is the danger.
MAX_RATE = {"waist": 0.45, "shoulder": 0.22, "wrist_angle": 0.35, "wrist_rotate": 0.5, "gripper": 1.2}


class GoHome(Node):
    def __init__(self):
        super().__init__("servo_go_home")
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

    def _pub(self, q: dict[str, float]) -> None:
        arm = JointGroupCommand()
        arm.name = "arm"
        arm.cmd = [float(q[n]) for n in ARM[:4]]
        self._arm.publish(arm)
        grip = JointSingleCommand()
        grip.name = "gripper"
        grip.cmd = float(q["gripper"])
        self._grip.publish(grip)

    def run(self) -> int:
        t0 = time.time()
        while time.time() - t0 < 12:
            rclpy.spin_once(self, timeout_sec=0.1)
            if all(j in self._js for j in ARM):
                break
        else:
            print("no joint_states", flush=True)
            return 2
        cur = {j: float(self._js[j]) for j in ARM}
        print(f"from {[round(cur[j], 3) for j in ARM]}", flush=True)
        hz = 20.0
        dt = 1.0 / hz
        # Time needed from slowest joint (+ cushion).
        dur = 1.5
        for j in ARM:
            dur = max(dur, abs(HOME[j] - cur[j]) / max(MAX_RATE[j], 1e-3) + 1.0)
        steps = max(1, int(dur * hz))
        print(f"slewing home over {dur:.1f}s ({steps} steps)", flush=True)
        for i in range(1, steps + 1):
            a = i / float(steps)
            cmd = {j: (1.0 - a) * cur[j] + a * HOME[j] for j in ARM}
            self._pub(cmd)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(dt)
        # Hold until close or timeout.
        t_hold = time.time()
        err = 9.0
        while time.time() - t_hold < 6.0:
            self._pub(HOME)
            rclpy.spin_once(self, timeout_sec=0.05)
            err = math.sqrt(
                sum((float(self._js.get(j, HOME[j])) - HOME[j]) ** 2 for j in ARM[:4])
            )
            if err < 0.12:
                break
            time.sleep(dt)
        print(f"home err={err:.3f} q={[round(float(self._js.get(j, 0)), 3) for j in ARM]}", flush=True)
        return 0 if err < 0.35 else 1


def main() -> int:
    rclpy.init()
    node = GoHome()
    try:
        return node.run()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
