#!/usr/bin/env python3
"""Fan GUI arm / drive commands to ROS *and* a TCP socket for Windows Isaac.

Isaac Sim 6 ships Jazzy rclpy; the AGX is Humble. Those DDS graphs do not
match. Isaac therefore connects *out* to this TCP server (AGX:25151) and
reads newline JSON. UDP 127.0.0.1:25151 is still the hop from gui_node.

Run:
  FASTRTPS_DEFAULT_PROFILES_FILE=.../bridge_both.xml python3 arm_cmd_wifi_relay.py
"""
from __future__ import annotations

import json
import os
import socket
import threading

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import JointState

PORT = int(os.environ.get("GRAB_IT_ARM_CMD_UDP_PORT", "25151"))


class ArmCmdWifiRelay(Node):
    def __init__(self):
        super().__init__("arm_cmd_wifi_relay")
        self._js_pub = self.create_publisher(
            JointState, "/px150_no_elbow/commands/joint_targets", 10
        )
        self._vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("127.0.0.1", PORT))
        self._sock.setblocking(False)
        self._clients: list[socket.socket] = []
        self._lock = threading.Lock()
        threading.Thread(target=self._accept_loop, daemon=True).start()
        self.create_timer(0.02, self._poll)
        self.get_logger().info(
            f"UDP 127.0.0.1:{PORT} + TCP 0.0.0.0:{PORT} (Isaac LAN) "
            "-> joint_targets / cmd_vel"
        )

    def _accept_loop(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", PORT))
        srv.listen(4)
        while True:
            try:
                conn, addr = srv.accept()
            except Exception:
                continue
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self._lock:
                self._clients.append(conn)
            self.get_logger().info(f"Isaac LAN command client {addr[0]}:{addr[1]}")

    def _fanout(self, payload: dict) -> None:
        line = (json.dumps(payload, separators=(",", ":")) + "\n").encode()
        with self._lock:
            clients = list(self._clients)
        dead = []
        for c in clients:
            try:
                c.sendall(line)
            except Exception:
                dead.append(c)
        if dead:
            with self._lock:
                self._clients = [c for c in self._clients if c not in dead]
            for c in dead:
                try:
                    c.close()
                except Exception:
                    pass

    def _handle(self, payload: dict) -> None:
        kind = str(payload.get("t") or "arm")
        if kind == "vel":
            tw = Twist()
            tw.linear.x = float(payload.get("lx", 0.0))
            tw.angular.z = float(payload.get("az", 0.0))
            self._vel_pub.publish(tw)
            self._fanout({"t": "vel", "lx": tw.linear.x, "az": tw.angular.z})
            return
        names = payload.get("n") or payload.get("names") or []
        pos = payload.get("p") or payload.get("pos") or []
        if not names or len(names) != len(pos):
            return
        js = JointState()
        js.name = [str(n) for n in names]
        js.position = [float(x) for x in pos]
        self._js_pub.publish(js)
        self._fanout({"t": "arm", "n": js.name, "p": js.position})

    def _poll(self):
        while True:
            try:
                data, _ = self._sock.recvfrom(4096)
            except BlockingIOError:
                break
            except Exception:
                break
            try:
                payload = json.loads(data.decode("utf-8"))
            except Exception:
                continue
            if isinstance(payload, dict):
                self._handle(payload)


def main():
    rclpy.init()
    node = ArmCmdWifiRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
