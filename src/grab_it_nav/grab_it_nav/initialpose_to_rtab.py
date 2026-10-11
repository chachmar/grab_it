#!/usr/bin/env python3
"""Relay /initialpose -> /rtabmap/initialpose with inflated covariance.

RViz "2D Pose Estimate" publishes /initialpose. RTAB localization needs
/rtabmap/initialpose. Extracted from nav_bringup.sh. Behavior matches that heredoc.
"""
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseWithCovarianceStamped

QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.VOLATILE,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
)

# ~0.5 m XY, ~25 deg yaw — helps RTAB lock when arrow is roughly right
COV_X = 0.25
COV_Y = 0.25
COV_YAW = (25.0 * math.pi / 180.0) ** 2


class PoseRelay(Node):
    def __init__(self):
        super().__init__('initialpose_to_rtab')
        self.pub = self.create_publisher(PoseWithCovarianceStamped, '/rtabmap/initialpose', QOS)
        self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self._cb, QOS)
        self.get_logger().info(
            'Relaying /initialpose -> /rtabmap/initialpose '
            f'(cov xy={COV_X:.2f} yaw_deg=25)'
        )

    def _cb(self, msg):
        out = PoseWithCovarianceStamped()
        out.header = msg.header
        out.pose.pose = msg.pose.pose
        cov = [0.0] * 36
        cov[0] = COV_X
        cov[7] = COV_Y
        cov[35] = COV_YAW
        out.pose.covariance = cov
        self.pub.publish(out)
        self.get_logger().info('Forwarded initial pose to RTAB-Map (inflated covariance)')


def main():
    rclpy.init()
    node = PoseRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.try_shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
