#!/bin/bash
# Ensure px150 URDF mesh filenames resolve.
# URDF references px150_N_*.stl; on-disk files use short names (shoulder.stl, ...).
set -e
WS=${WS:-/workspaces/isaac_ros-dev}
SRC_MESH=$WS/src/interbotix/interbotix_ros_manipulators/interbotix_ros_xsarms/interbotix_xsarm_descriptions/meshes
PX=$SRC_MESH/px150_meshes
INST_PX=$WS/install/interbotix_xsarm_descriptions/share/interbotix_xsarm_descriptions/meshes/px150_meshes

link_one() {
  local dir="$1" dest="$2" src="$3"
  if [ -e "$dir/$src" ] || [ -L "$dir/$src" ]; then
    ln -sfn "$src" "$dir/$dest"
  else
    echo "[meshes] WARN: missing $dir/$src"
  fi
}

for DIR in "$PX" "$INST_PX"; do
  [ -d "$DIR" ] || continue
  link_one "$DIR" px150_1_base.stl base.stl
  link_one "$DIR" px150_2_shoulder.stl shoulder.stl
  link_one "$DIR" px150_3_upper_arm.stl upper_arm.stl  # needed by fixed no_elbow URDF
  link_one "$DIR" px150_4_forearm.stl forearm.stl
  link_one "$DIR" px150_5_wrist.stl wrist.stl
  link_one "$DIR" px150_6_gripper.stl gripper.stl
  link_one "$DIR" px150_7_gripper_prop.stl gripper_prop.stl
  link_one "$DIR" px150_8_gripper_bar.stl gripper_bar.stl
  link_one "$DIR" px150_9_gripper_finger.stl gripper_finger.stl
  if [ -f "$SRC_MESH/ar_tag.stl" ]; then
    ln -sfn ../ar_tag.stl "$DIR/px150_10_ar_tag.stl" 2>/dev/null || \
      ln -sfn "$SRC_MESH/ar_tag.stl" "$DIR/px150_10_ar_tag.stl"
  fi
done
