#!/usr/bin/env bash
set -eo pipefail
set +u
source /opt/ros/humble/setup.bash
source "$HOME/lerobot-venv/bin/activate"
export PYTHONPATH="/home/ubuntu/dev-ws/install/interbotix_xs_msgs/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages:${PYTHONPATH:-}"
# grab_it_il is a pure python package under src or install
if [[ -d /home/ubuntu/dev-ws/install/grab_it_il/lib/python3.10/site-packages ]]; then
  export PYTHONPATH="/home/ubuntu/dev-ws/install/grab_it_il/lib/python3.10/site-packages:${PYTHONPATH}"
fi
export PYTHONPATH="/home/ubuntu/dev-ws/src/grab_it_il:${PYTHONPATH}"
python3 - <<'PY'
from interbotix_xs_msgs.msg import JointGroupCommand
import torch, lerobot, rclpy
print("OK", torch.__version__, "cuda", torch.cuda.is_available())
PY
