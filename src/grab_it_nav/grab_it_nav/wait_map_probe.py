#!/usr/bin/env python3
"""Wait for OccupancyGrid without ros2cli daemon (crossover DDS breaks topic list/echo)."""
import os
import sys
import time

os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
os.environ.setdefault("ROS_DOMAIN_ID", "0")

import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

TOPIC = sys.argv[1] if len(sys.argv) > 1 else "/rtabmap/map"
TIMEOUT = float(sys.argv[2]) if len(sys.argv) > 2 else 90.0

qos = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
)

got = {"ok": False, "w": 0, "h": 0}


def cb(msg):
    got["ok"] = True
    got["w"] = msg.info.width
    got["h"] = msg.info.height


def main():
    rclpy.init()
    node = rclpy.create_node("wait_map_probe")
    node.create_subscription(OccupancyGrid, TOPIC, cb, qos)
    t0 = time.time()
    while not got["ok"] and time.time() - t0 < TIMEOUT:
        rclpy.spin_once(node, timeout_sec=0.5)
    try:
        node.destroy_node()
    except Exception:
        pass
    try:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass
    if got["ok"]:
        print(f"OK {TOPIC} {got['w']}x{got['h']}", flush=True)
        return 0
    print(f"FAIL {TOPIC} timeout={TIMEOUT}s", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
