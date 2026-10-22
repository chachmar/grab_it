#!/usr/bin/env python3
"""Nudge the arm toward the taught yaml IK pose before a servo trial."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import rclpy
from interbotix_xs_msgs.msg import JointGroupCommand, JointSingleCommand
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rl_goal_lib import ik_eef, load_goal  # noqa: E402

ARM = ["waist", "shoulder", "wrist_angle", "wrist_rotate", "gripper"]


class Reset(Node):
    def __init__(self):
        super().__init__("servo_reset_yaml")
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
        while time.time() - t0 < 10:
            rclpy.spin_once(self, timeout_sec=0.1)
            if all(j in self._js for j in ARM):
                break
        else:
            print("no joint_states", flush=True)
            return 2
        g = load_goal()
        seed = {j: self._js[j] for j in ARM}
        sol = ik_eef(
            [g["x"], g["y"], g["z"]],
            float(g["wrist_rotate"]),
            1.65,
            float(g["wrist_angle_max"]),
            seed=seed,
        )
        if sol is None:
            print("ik failed for yaml tip", flush=True)
            return 3
        wmax = float(g["wrist_angle_max"])
        # Taught wrist_max is the most-folded allowed (floor). Never go more negative.
        sol["wrist_angle"] = max(float(sol["wrist_angle"]), wmax)
        for _ in range(36):
            arm = JointGroupCommand()
            arm.name = "arm"
            arm.cmd = [float(sol[n]) for n in ARM[:4]]
            self._arm.publish(arm)
            grip = JointSingleCommand()
            grip.name = "gripper"
            grip.cmd = 1.65
            self._grip.publish(grip)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(1.0 / 12.0)
        print(
            f"reset → waist={sol['waist']:.2f} sh={sol['shoulder']:.2f} "
            f"wr={sol['wrist_angle']:.2f}",
            flush=True,
        )
        return 0


def main() -> int:
    rclpy.init()
    node = Reset()
    try:
        return node.run()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
