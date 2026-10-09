#!/usr/bin/env bash
set +e
echo "=== docker images ==="
docker images --format '{{.Repository}}:{{.Tag}}' | head -30
echo "=== containers ==="
docker ps -a --format '{{.Names}} {{.Status}}' | head -20
echo "=== pkgs ==="
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
unset CYCLONEDDS_URI || true
if [[ -f /home/ubuntu/dev-ws/install/setup.bash ]]; then
  source /home/ubuntu/dev-ws/install/setup.bash
fi
ros2 pkg prefix interbotix_xsarm_moveit 2>&1 | head -3
ros2 pkg prefix grab_it_grasp 2>&1 | head -3
ros2 pkg prefix moveit_ros_move_group 2>&1 | head -3
echo "=== move_group running ==="
timeout 6 ros2 node list 2>&1 | grep -E 'move_group|grasp' || echo none
ls /opt/ros/humble/lib/moveit_ros_move_group/move_group 2>/dev/null || echo no_system_move_group
