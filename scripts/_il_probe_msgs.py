#!/usr/bin/env python3
import os
import sys

cands = [
    os.path.expanduser(
        "~/dev-ws/install/interbotix_xs_msgs/local/lib/python3.10/dist-packages"
    ),
    os.path.expanduser("~/dev-ws/install/interbotix_xs_msgs/lib/python3.10/site-packages"),
    "/opt/ros/humble/local/lib/python3.10/dist-packages",
    "/opt/ros/humble/lib/python3.10/site-packages",
]
for c in cands:
    print("cand", c, "exists", os.path.isdir(c))
    if os.path.isdir(c):
        sys.path.insert(0, c)

print("sys.path[0:5]", sys.path[:5])
try:
    import interbotix_xs_msgs

    print("pkg", getattr(interbotix_xs_msgs, "__file__", None))
except Exception as exc:
    print("pkg fail", exc)

try:
    import interbotix_xs_msgs.msg as m

    print("msg", getattr(m, "__file__", None))
    print("attrs", [a for a in dir(m) if "Joint" in a or "joint" in a][:20])
    from interbotix_xs_msgs.msg import JointGroupCommand

    print("JointGroupCommand OK", JointGroupCommand)
except Exception as exc:
    print("import fail", type(exc).__name__, exc)

msg_init = os.path.expanduser(
    "~/dev-ws/install/interbotix_xs_msgs/local/lib/python3.10/dist-packages/"
    "interbotix_xs_msgs/msg/__init__.py"
)
print("msg_init exists", os.path.isfile(msg_init), msg_init)
if os.path.isfile(msg_init):
    with open(msg_init, encoding="utf-8", errors="replace") as f:
        print(f.read()[:500])
