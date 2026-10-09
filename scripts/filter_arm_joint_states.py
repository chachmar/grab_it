#!/usr/bin/env python3
"""Republish arm joint_states without extra joints MoveIt doesn't know (e.g. pan)."""
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

# Keep only joints that exist on the px150_no_elbow MoveIt / Interbotix arm model.
KEEP = {
    "waist",
    "shoulder",
    "wrist_angle",
    "wrist_rotate",
    "gripper",
    "left_finger",
    "right_finger",
}


class FilterArmJS(Node):
    def __init__(self):
        super().__init__("filter_arm_joint_states")
        # Match xs_sdk / MoveIt (BEST_EFFORT sensor data). RELIABLE pubs won't
        # connect to MoveIt's BEST_EFFORT joint_state subscription.
        self.pub = self.create_publisher(
            JointState, "/px150_no_elbow/joint_states_arm", qos_profile_sensor_data
        )
        # Prefer the AGX bridge topic (holds last-good). Live Nano topic is flaky
        # over DDS and caused MoveIt/RViz arm links to pulse.
        self.create_subscription(
            JointState,
            "/px150_no_elbow/joint_states_rsp",
            self._cb,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            "Filtering /px150_no_elbow/joint_states_rsp -> /px150_no_elbow/joint_states_arm "
            "(sensor_data QoS)"
        )

    def _cb(self, msg: JointState):
        out = JointState()
        out.header = msg.header
        for i, name in enumerate(msg.name):
            if name not in KEEP:
                continue
            out.name.append(name)
            if i < len(msg.position):
                out.position.append(msg.position[i])
            if i < len(msg.velocity):
                out.velocity.append(msg.velocity[i])
            if i < len(msg.effort):
                out.effort.append(msg.effort[i])
        if out.name:
            self.pub.publish(out)


def main():
    rclpy.init()
    node = FilterArmJS()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
