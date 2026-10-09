#!/usr/bin/env bash
set -eo pipefail
set +u
WS=/home/ubuntu/dev-ws
echo "=== dist-packages tree ==="
find "$WS/install/interbotix_xs_msgs" -type d -name 'dist-packages' 2>/dev/null
find "$WS/install/interbotix_xs_msgs" -name 'JointGroupCommand*' 2>/dev/null | head
find "$WS/install/interbotix_xs_msgs" -path '*interbotix_xs_msgs/msg*' 2>/dev/null | head -20

echo "=== try COLCON_CURRENT_PREFIX ==="
source /opt/ros/humble/setup.bash
export COLCON_CURRENT_PREFIX="$WS/install/interbotix_xs_msgs"
# shellcheck disable=SC1091
source "$WS/install/interbotix_xs_msgs/share/interbotix_xs_msgs/package.bash" 2>/dev/null \
  || source "$WS/install/interbotix_xs_msgs/share/interbotix_xs_msgs/package.sh" 2>/dev/null \
  || true
export COLCON_CURRENT_PREFIX="$WS/install/grab_it_il"
source "$WS/install/grab_it_il/share/grab_it_il/package.bash" 2>/dev/null \
  || source "$WS/install/grab_it_il/share/grab_it_il/package.sh" 2>/dev/null \
  || true
source "$HOME/lerobot-venv/bin/activate"
python3 - <<'PY'
from interbotix_xs_msgs.msg import JointGroupCommand
import torch, lerobot
print("HOST OK", torch.__version__, torch.cuda.is_available())
PY
