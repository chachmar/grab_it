#!/usr/bin/env bash
# Echo measured joint_states with sensor QoS during PLAY (3 short trials).
set -eo pipefail
docker exec -u admin -w /workspaces/isaac_ros-dev isaac_ros_dev-aarch64-container bash -lc '
set +u
source /opt/ros/humble/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
unset CYCLONEDDS_URI
export FASTRTPS_DEFAULT_PROFILES_FILE=/workspaces/isaac_ros-dev/bridge_both.xml

python3 - <<PY
import json, time, threading
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String

QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)

class N(Node):
    def __init__(self):
        super().__init__("il_js_probe")
        self.js=[]
        self.create_subscription(JointState, "/px150_no_elbow/joint_states", self._js, QOS)
        self.cmd = self.create_publisher(String, "/grab_it/il_command", 10)
    def _js(self, msg):
        names=list(msg.name); pos=list(msg.position)
        self.js.append({n: float(pos[i]) for i,n in enumerate(names) if i < len(pos)})
    def pub(self, c):
        m=String(); m.data=json.dumps({"cmd": c})
        for _ in range(5):
            self.cmd.publish(m); time.sleep(0.05)

rclpy.init(); node=N()
stop=False
def spin():
    while not stop:
        rclpy.spin_once(node, timeout_sec=0.05)
threading.Thread(target=spin, daemon=True).start()
time.sleep(1.0)

def arm(d):
    return [d.get(k) for k in ("waist","shoulder","wrist_angle","wrist_rotate","gripper")]

results=[]
for trial in range(1,4):
    node.js.clear()
    time.sleep(0.8)
    before = arm(node.js[-1]) if node.js else None
    node.pub("play")
    time.sleep(8.0)
    node.pub("stop")
    time.sleep(0.5)
    after = arm(node.js[-1]) if node.js else None
    delta=None
    moved=False
    if before and after:
        delta=[None if (x is None or y is None) else abs(float(x)-float(y)) for x,y in zip(before,after)]
        moved=any((d or 0)>0.03 for d in delta)
    results.append({"trial": trial, "js_n": len(node.js), "before": before, "after": after, "delta": delta, "moved": moved})
    print(json.dumps(results[-1]))
    time.sleep(2.0)

stop=True
ok=sum(1 for r in results if r.get("moved"))
print(json.dumps({"trials_moved": ok, "trials": len(results), "pass": ok >= 1}))
node.destroy_node(); rclpy.shutdown()
PY
'
