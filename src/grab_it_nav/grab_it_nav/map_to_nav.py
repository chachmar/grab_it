#!/usr/bin/env python3
"""Relay OccupancyGrid /rtabmap/map -> /map for Nav2, with light obstacle erosion.

Process name must NOT contain 'rtabmap'.
Erosion: peel ~ERODE_CELLS of occupied fringe into free space so corridors
that mapped slightly tight stay navigable (Create3 still protected by inflation).

Extracted from nav_bringup.sh. Behavior matches that heredoc.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import OccupancyGrid

MAP_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
)

# 1 cell @ 5 cm ≈ slight widen of mapped halls (2 was over-opening + wall-hugging plans)
ERODE_CELLS = 1
OCCUPIED_THRESH = 50


def erode_obstacles(msg: OccupancyGrid, iters: int) -> OccupancyGrid:
    w, h = msg.info.width, msg.info.height
    data = list(msg.data)
    if w < 3 or h < 3 or not data:
        return msg
    neigh = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for _ in range(iters):
        out = data[:]
        for y in range(h):
            row = y * w
            for x in range(w):
                i = row + x
                if data[i] < OCCUPIED_THRESH:
                    continue
                for dx, dy in neigh:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and data[ny * w + nx] == 0:
                        out[i] = 0
                        break
        data = out
    msg.data = data
    return msg


class MapToNav(Node):
    def __init__(self):
        super().__init__('map_to_nav')
        self.pub = self.create_publisher(OccupancyGrid, '/map', MAP_QOS)
        self.create_subscription(OccupancyGrid, '/rtabmap/map', self._cb, MAP_QOS)
        self._sent = False

    def _cb(self, msg):
        out = erode_obstacles(msg, ERODE_CELLS)
        self.pub.publish(out)
        if not self._sent:
            self.get_logger().info(
                f'Relaying map {out.info.width}x{out.info.height} @ '
                f'{out.info.resolution:.3f} m (erode={ERODE_CELLS} cells)'
            )
            self._sent = True


def main():
    rclpy.init()
    node = MapToNav()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
