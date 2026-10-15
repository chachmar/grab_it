#!/usr/bin/env python3
"""Republish /scan_raw -> /scan in base_link, stamped to odom->base_link TF time.

depthimage_to_laserscan should use output_frame=camera_link (NOT optical).
LaserScan uses x=r*cos(th), y=r*sin(th) in that frame (X forward).

If a scan still arrives in an *_optical_frame, project with optical convention
(Z forward, X right): (r*sin(th), 0, r*cos(th)).
"""
import math
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener, TransformException


def q_rotate(q, v):
    x, y, z, w = q
    vx, vy, vz = v
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def scan_point_in_frame(frame_id: str, r: float, a: float):
    """Return (x,y,z) in the LaserScan header frame for range r at angle a."""
    if "optical" in (frame_id or "").lower():
        # optical: X right, Y down, Z forward
        return (r * math.sin(a), 0.0, r * math.cos(a))
    # standard ROS laser / camera_link: X forward, Y left
    return (r * math.cos(a), r * math.sin(a), 0.0)


class ScanStampFix(Node):
    def __init__(self):
        super().__init__("scan_stamp_fix")
        self.target = "base_link"
        self.buf = Buffer(cache_time=Duration(seconds=60.0))
        self.listener = TransformListener(self.buf, self)
        self.pub = self.create_publisher(LaserScan, "/scan", qos_profile_sensor_data)
        self.create_subscription(LaserScan, "/scan_raw", self._cb, qos_profile_sensor_data)
        self._ok = 0
        self._fail = 0
        self.get_logger().info(
            "scan_stamp_fix: /scan_raw -> /scan (base_link @ odom TF time)"
        )

    def _cb(self, msg: LaserScan):
        try:
            tf_cam = self.buf.lookup_transform(
                self.target, msg.header.frame_id, rclpy.time.Time(),
                timeout=Duration(seconds=0.05),
            )
            tf_odom = self.buf.lookup_transform(
                "odom", self.target, rclpy.time.Time(),
                timeout=Duration(seconds=0.05),
            )
        except TransformException as ex:
            self._fail += 1
            if self._fail % 50 == 1:
                self.get_logger().warn(f"TF fail: {ex}")
            return

        t = tf_cam.transform.translation
        q = (
            tf_cam.transform.rotation.x,
            tf_cam.transform.rotation.y,
            tf_cam.transform.rotation.z,
            tf_cam.transform.rotation.w,
        )

        pts = []
        for i, r in enumerate(msg.ranges):
            if r is None or math.isnan(r) or math.isinf(r):
                continue
            if r < msg.range_min or r > msg.range_max:
                continue
            a = msg.angle_min + i * msg.angle_increment
            px, py, pz = scan_point_in_frame(msg.header.frame_id, r, a)
            p = q_rotate(q, (px, py, pz))
            # Keep roughly waist-height obstacles in base_link (drop floor/ceiling)
            zb = p[2] + t.z
            if zb < -0.2 or zb > 1.8:
                continue
            pts.append((p[0] + t.x, p[1] + t.y))

        out = LaserScan()
        out.header.stamp = tf_odom.header.stamp
        out.header.frame_id = self.target
        out.angle_min = -math.pi
        out.angle_max = math.pi
        out.angle_increment = math.radians(1.0)
        out.time_increment = 0.0
        out.scan_time = 0.05
        out.range_min = msg.range_min
        out.range_max = msg.range_max
        bins = int(round((out.angle_max - out.angle_min) / out.angle_increment)) + 1
        ranges = [float("inf")] * bins
        for x, y in pts:
            ang = math.atan2(y, x)
            rng = math.hypot(x, y)
            idx = int(round((ang - out.angle_min) / out.angle_increment))
            if 0 <= idx < bins and rng < ranges[idx]:
                ranges[idx] = rng
        out.ranges = ranges
        self.pub.publish(out)
        self._ok += 1
        if self._ok == 1:
            valid = sum(1 for v in ranges if v < float("inf"))
            self.get_logger().info(
                f"First /scan published (frame in={msg.header.frame_id}, "
                f"pts={len(pts)}, bins_hit={valid})"
            )
        elif self._ok % 100 == 0:
            valid = sum(1 for v in ranges if v < float("inf"))
            self.get_logger().info(
                f"/scan ok={self._ok} in_frame={msg.header.frame_id} "
                f"pts={len(pts)} bins={valid}"
            )


def main():
    rclpy.init()
    node = ScanStampFix()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
