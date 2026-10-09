#!/usr/bin/env python3
"""Publish GUI /cmd_vel on Wi‑Fi so domain_bridge can reach Create 3.

The GUI ROS participant is crossover-only (so Nano joint_states match).
Drive sticks are forwarded here over UDP 127.0.0.1 and republished with
bridge_both.xml (writer has both locators — domain_bridge can match Wi‑Fi).

Run:
  FASTRTPS_DEFAULT_PROFILES_FILE=.../bridge_both.xml python3 cmd_vel_wifi_relay.py
"""
from __future__ import annotations

import os
import socket
import struct
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

PORT = int(os.environ.get("GRAB_IT_CMD_VEL_UDP_PORT", "25150"))
HOLD_S = 0.4  # zero once if the GUI process dies mid-stick


class CmdVelWifiRelay(Node):
    def __init__(self):
        super().__init__("cmd_vel_wifi_relay")
        self._pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("127.0.0.1", PORT))
        self._sock.setblocking(False)
        self._last_rx = 0.0
        self._driving = False
        self.create_timer(0.05, self._poll)
        self.get_logger().info(f"UDP 127.0.0.1:{PORT} -> /cmd_vel (Wi‑Fi DDS)")

    def _poll(self):
        now = time.monotonic()
        while True:
            try:
                data, _ = self._sock.recvfrom(128)
            except BlockingIOError:
                break
            except Exception:
                break
            if len(data) < 48:
                continue
            lx, ly, lz, ax, ay, az = struct.unpack("!6d", data[:48])
            msg = Twist()
            msg.linear.x = lx
            msg.linear.y = ly
            msg.linear.z = lz
            msg.angular.x = ax
            msg.angular.y = ay
            msg.angular.z = az
            self._pub.publish(msg)
            self._last_rx = now
            self._driving = abs(lx) > 1e-6 or abs(az) > 1e-6
        if self._driving and self._last_rx and (now - self._last_rx) > HOLD_S:
            self._pub.publish(Twist())
            self._driving = False


def main():
    rclpy.init()
    node = CmdVelWifiRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
