#!/usr/bin/env python3
"""Publish odom -> base_link / base_footprint TF from /odom.

Create3's bridged /tf often lags wall-clock by several seconds even when /odom
stamps are fresh. Nav2/RViz/RTAB then look delayed or fail TF lookups.
Prefer this republisher and do NOT bridge Create3 /tf (tf_static is fine).
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class OdomToTf(Node):
    def __init__(self):
        super().__init__("odom_to_tf")
        self._br = TransformBroadcaster(self)
        # Create3 / domain_bridge odom is BEST_EFFORT
        qos = QoSProfile(
            depth=20,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(Odometry, "/odom", self._cb, qos)
        self._n = 0
        self.get_logger().info("odom_to_tf: publishing TF from /odom")

    def _cb(self, msg: Odometry):
        parent = msg.header.frame_id or "odom"
        child = msg.child_frame_id or "base_link"
        if parent.startswith("/"):
            parent = parent[1:]
        if child.startswith("/"):
            child = child[1:]

        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = parent
        t.child_frame_id = child
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation = msg.pose.pose.orientation

        # Create3 also publishes base_footprint; keep planar twin for costmaps.
        foot = TransformStamped()
        foot.header = t.header
        foot.child_frame_id = "base_footprint"
        foot.transform.translation.x = t.transform.translation.x
        foot.transform.translation.y = t.transform.translation.y
        foot.transform.translation.z = 0.0
        foot.transform.rotation = t.transform.rotation

        self._br.sendTransform([t, foot])
        self._n += 1
        if self._n == 1:
            self.get_logger().info(
                f"first TF {parent} -> {child} (+ base_footprint)"
            )


def main():
    rclpy.init()
    node = OdomToTf()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
