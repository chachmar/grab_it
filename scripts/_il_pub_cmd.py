#!/usr/bin/env python3
"""One-shot /grab_it/il_command publisher (play|stop)."""
import json
import sys
import time

import rclpy
from std_msgs.msg import String


def main() -> None:
    cmd = (sys.argv[1] if len(sys.argv) > 1 else "play").strip().lower()
    rclpy.init()
    node = rclpy.create_node("il_cmd_once")
    pub = node.create_publisher(String, "/grab_it/il_command", 10)
    time.sleep(0.4)
    msg = String()
    msg.data = json.dumps({"cmd": cmd})
    for _ in range(5):
        pub.publish(msg)
        time.sleep(0.05)
    node.get_logger().info(f"published il_command cmd={cmd}")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
