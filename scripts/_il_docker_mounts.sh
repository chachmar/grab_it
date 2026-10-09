#!/usr/bin/env bash
docker inspect isaac_ros_dev-aarch64-container --format '{{json .Mounts}}' | python3 -m json.tool | head -80
echo "===="
# Import test inside container with host install paths (same bind)
docker exec -u admin -w /workspaces/isaac_ros-dev isaac_ros_dev-aarch64-container bash -lc '
set +u
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 - <<PY
from interbotix_xs_msgs.msg import JointGroupCommand
print("container msgs OK", JointGroupCommand)
PY
'
