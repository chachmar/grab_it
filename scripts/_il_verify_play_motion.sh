#!/usr/bin/env bash
# Sample joint_states / joint_targets during ACT PLAY to verify motion.
set -eo pipefail
set +u
OUT=/tmp/il_play_verify_run.txt
: >"$OUT"

docker exec -u admin -w /workspaces/isaac_ros-dev isaac_ros_dev-aarch64-container bash -lc '
set +u
source /opt/ros/humble/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
unset CYCLONEDDS_URI
export FASTRTPS_DEFAULT_PROFILES_FILE=/workspaces/isaac_ros-dev/rtabmap_profile.xml
[[ -f $FASTRTPS_DEFAULT_PROFILES_FILE ]] || export FASTRTPS_DEFAULT_PROFILES_FILE=/workspaces/isaac_ros-dev/bridge_both.xml

python3 - <<PY
import json, time, threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from interbotix_xs_msgs.msg import JointGroupCommand
from std_msgs.msg import String

class Sampler(Node):
    def __init__(self):
        super().__init__("il_verify_sampler")
        self.js = []
        self.tgt = []
        self.grp = []
        self.create_subscription(JointState, "/px150_no_elbow/joint_states", self._js, 10)
        self.create_subscription(JointState, "/px150_no_elbow/commands/joint_targets", self._tgt, 10)
        self.create_subscription(JointGroupCommand, "/px150_no_elbow/commands/joint_group", self._grp, 10)
        self.cmd = self.create_publisher(String, "/grab_it/il_command", 10)
        self.status = []
        self.create_subscription(String, "/grab_it/il_status", self._st, 10)

    def _js(self, msg):
        names = list(msg.name); pos = list(msg.position)
        d = {n: float(pos[i]) for i, n in enumerate(names) if i < len(pos)}
        self.js.append((time.time(), d))

    def _tgt(self, msg):
        names = list(msg.name); pos = list(msg.position)
        d = {n: float(pos[i]) for i, n in enumerate(names) if i < len(pos)}
        self.tgt.append((time.time(), d))

    def _grp(self, msg):
        if (msg.name or "") == "arm" and len(msg.cmd) >= 4:
            self.grp.append((time.time(), [float(x) for x in msg.cmd[:4]]))

    def _st(self, msg):
        try:
            self.status.append(json.loads(msg.data))
        except Exception:
            self.status.append({"raw": msg.data})

    def pub_cmd(self, cmd):
        m = String(); m.data = json.dumps({"cmd": cmd})
        for _ in range(5):
            self.cmd.publish(m)
            time.sleep(0.05)

rclpy.init()
node = Sampler()
# spin in background
stop = False
def spin():
    while not stop:
        rclpy.spin_once(node, timeout_sec=0.05)
th = threading.Thread(target=spin, daemon=True); th.start()
time.sleep(1.0)
# baseline
t0 = time.time()
while time.time() - t0 < 1.5:
    time.sleep(0.1)
base_js = list(node.js)
node.pub_cmd("play")
print("PLAY sent")
t0 = time.time()
while time.time() - t0 < 12.0:
    time.sleep(0.1)
node.pub_cmd("stop")
print("STOP sent")
time.sleep(0.5)
stop = True
time.sleep(0.2)

def arm_vec(d):
    return [d.get(k) for k in ("waist","shoulder","wrist_angle","wrist_rotate","gripper")]

out = {
    "js_samples": len(node.js),
    "tgt_samples": len(node.tgt),
    "grp_samples": len(node.grp),
    "status_tail": node.status[-5:],
}
if base_js and node.js:
    b = arm_vec(base_js[0][1])
    a = arm_vec(node.js[-1][1])
    out["js_before"] = b
    out["js_after"] = a
    out["js_delta"] = [None if (x is None or y is None) else abs(x-y) for x,y in zip(b,a)]
    out["js_moved"] = any((d or 0) > 0.02 for d in out["js_delta"])
if node.grp:
    out["grp_first"] = node.grp[0][1]
    out["grp_last"] = node.grp[-1][1]
    out["grp_moved"] = any(abs(a-b) > 0.01 for a,b in zip(out["grp_first"], out["grp_last"]))
if node.tgt:
    out["tgt_last"] = arm_vec(node.tgt[-1][1])
print(json.dumps(out, indent=2))
node.destroy_node(); rclpy.shutdown()
PY
' | tee "$OUT"
