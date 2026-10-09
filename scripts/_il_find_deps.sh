#!/usr/bin/env bash
set -eo pipefail
set +u
echo "=== find msgs ==="
find /opt/ros/humble /home/ubuntu/dev-ws/install /workspaces/isaac_ros-dev/install -name 'joint_group_command*' 2>/dev/null | head -15 || true
echo "=== install interbotix ==="
ls /home/ubuntu/dev-ws/install/interbotix_xs_msgs 2>&1 | head -8 || true
echo "=== container venv mount ==="
docker exec -u admin isaac_ros_dev-aarch64-container bash -lc 'ls /home/ubuntu/lerobot-venv/bin/activate; ls /workspaces/isaac_ros-dev/install/interbotix_xs_msgs 2>&1 | head -5; python3 -c "import sys; print(sys.version)"' 2>&1 | head -20
echo "=== host venv + PYTHONPATH trial ==="
source /opt/ros/humble/setup.bash
# Use only packages that exist
for p in interbotix_xs_msgs grab_it_il; do
  if [[ -f /home/ubuntu/dev-ws/install/$p/share/$p/package.sh ]]; then
    source /home/ubuntu/dev-ws/install/$p/share/$p/package.sh || true
  fi
done
source /home/ubuntu/lerobot-venv/bin/activate
# Add ROS message python paths if present
export PYTHONPATH="/home/ubuntu/dev-ws/install/interbotix_xs_msgs/local/lib/python3.10/dist-packages:${PYTHONPATH:-}"
export PYTHONPATH="/opt/ros/humble/lib/python3.10/site-packages:${PYTHONPATH}"
python3 - <<'PY'
import sys
print("py", sys.executable)
try:
    from interbotix_xs_msgs.msg import JointGroupCommand
    print("msgs OK")
except Exception as e:
    print("msgs FAIL", e)
try:
    import torch, lerobot
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
except Exception as e:
    print("torch FAIL", e)
PY
