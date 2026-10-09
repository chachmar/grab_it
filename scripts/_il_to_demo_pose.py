#!/usr/bin/env python3
"""Slow move from real sleep to the shoulder-demo park.

Fold the shoulder to -1.90 while the wrist stays tucked, then rotate the
wrist to +1.40. Same speed caps as the GUI Sleep path. Aborts unless the
arm is still near sleep.
"""
from __future__ import annotations

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState

from interbotix_xs_msgs.msg import JointGroupCommand

NAMES = ["waist", "shoulder", "wrist_angle", "wrist_rotate"]
SPEED = {"waist": 0.35, "shoulder": 0.22, "wrist_angle": 0.30, "wrist_rotate": 0.40}
SLEEP = {"shoulder": -0.2321, "wrist_angle": -1.7279}
DEMO = [0.0, -1.90, 1.40, 0.0]
QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


def _ease(a: float) -> float:
    return a * a * (3.0 - 2.0 * a)


class Mover(Node):
    def __init__(self) -> None:
        super().__init__("demo_pose_move")
        self.pos: dict[str, float] = {}
        self.create_subscription(JointState, "/px150_no_elbow/joint_states", self._js, QOS)
        self.pub = self.create_publisher(
            JointGroupCommand, "/px150_no_elbow/commands/joint_group", 10
        )

    def _js(self, msg: JointState) -> None:
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.pos[str(name)] = float(msg.position[i])

    def wait_joints(self) -> list[float]:
        t0 = time.monotonic()
        while time.monotonic() - t0 < 5.0:
            rclpy.spin_once(self, timeout_sec=0.1)
            if all(n in self.pos for n in NAMES):
                return [self.pos[n] for n in NAMES]
        raise SystemExit("no joint states")

    def send(self, q: list[float]) -> None:
        msg = JointGroupCommand()
        msg.name = "arm"
        msg.cmd = [float(v) for v in q]
        self.pub.publish(msg)

    def segment(self, q0: list[float], q1: list[float]) -> list[float]:
        dq = [b - a for a, b in zip(q0, q1)]
        dur = 0.8
        for name, delta in zip(NAMES, dq):
            spd = SPEED[name]
            if spd > 1e-6:
                dur = max(dur, abs(delta) / spd)
        dur = min(20.0, dur)
        steps = max(8, int(dur * 20))
        dt = dur / steps
        print(
            f"segment {dur:.1f}s "
            + " ".join(f"{n}:{a:+.2f}->{b:+.2f}" for n, a, b in zip(NAMES, q0, q1)),
            flush=True,
        )
        q = list(q0)
        for s in range(1, steps + 1):
            a = _ease(s / steps)
            q = [c + a * d for c, d in zip(q0, dq)]
            self.send(q)
            time.sleep(dt)
        for _ in range(5):
            self.send(q1)
            time.sleep(0.05)
        return list(q1)


def main() -> None:
    rclpy.init()
    node = Mover()
    q = node.wait_joints()
    print("now " + " ".join(f"{n}:{v:+.3f}" for n, v in zip(NAMES, q)), flush=True)
    if abs(q[1] - SLEEP["shoulder"]) > 0.45 or abs(q[2] - SLEEP["wrist_angle"]) > 0.55:
        raise SystemExit(
            f"not at sleep shoulder={q[1]:+.2f} wrist={q[2]:+.2f} — not moving"
        )
    # Tuck the shoulder back before the wrist flips.
    folded = [0.0, -1.90, q[2], q[3]]
    q = node.segment(q, folded)
    q = node.segment(q, DEMO)
    time.sleep(0.4)
    got = node.wait_joints()
    print("arrived " + " ".join(f"{n}:{v:+.3f}" for n, v in zip(NAMES, got)), flush=True)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
