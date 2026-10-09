#!/bin/bash
# Launch D435i optical-frame calibration static TF from yaml (trial or permanent).
WS=${WS:-/workspaces/isaac_ros-dev}
CAL=${1:-$WS/config/calibration/d435i_optical_cal.yaml}
TRIAL=$WS/config/calibration/d435i_optical_cal_trial.yaml
if [ ! -f "$CAL" ] && [ -f "$TRIAL" ]; then
  CAL="$TRIAL"
fi
if [ ! -f "$CAL" ]; then
  exit 0
fi

source /opt/ros/humble/setup.bash 2>/dev/null || true
python3 - "$CAL" <<'PY'
import subprocess, sys, yaml
from pathlib import Path
cal = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
pf = cal.get("parent_frame")
cf = cal.get("child_frame")
xyz = cal.get("xyz") or [0, 0, 0]
rpy = cal.get("rpy") or [0, 0, 0]
if not pf or not cf:
    sys.exit(0)
subprocess.run(["pkill", "-f", f"static_transform_publisher.*{cf}"],
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.Popen([
    "ros2", "run", "tf2_ros", "static_transform_publisher",
    "--x", str(xyz[0]), "--y", str(xyz[1]), "--z", str(xyz[2]),
    "--roll", str(rpy[0]), "--pitch", str(rpy[1]), "--yaw", str(rpy[2]),
    "--frame-id", str(pf), "--child-frame-id", str(cf),
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(f"[grasp_cal_static_tf] {pf} -> {cf} from {sys.argv[1]}")
PY
