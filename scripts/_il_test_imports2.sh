#!/usr/bin/env bash
set -eo pipefail
set +u
source /opt/ros/humble/setup.bash
# Do NOT activate venv — use system/humble python + venv site-packages
export PYTHONPATH="/home/ubuntu/lerobot-venv/lib/python3.10/site-packages:/home/ubuntu/dev-ws/install/interbotix_xs_msgs/local/lib/python3.10/dist-packages:${PYTHONPATH:-}"
export AMENT_PREFIX_PATH="/home/ubuntu/dev-ws/install/interbotix_xs_msgs:${AMENT_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="/home/ubuntu/dev-ws/install/interbotix_xs_msgs/lib:${LD_LIBRARY_PATH:-}"
which python3
python3 - <<'PY'
import sys
print("exe", sys.executable)
from interbotix_xs_msgs.msg import JointGroupCommand
import torch, lerobot
print("OK", JointGroupCommand, torch.__version__, torch.cuda.is_available())
PY
