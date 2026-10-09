#!/usr/bin/env bash
# Stop the live AGX stack: GUI, cameras, click-to-grasp, ACT.
# Does not stop the Nano cameras or the arm driver.
set +e
for pat in \
  'dev-ws/mjpeg_relay.py' \
  'grab_it_gui/gui_node.py' \
  'grab_it_il/play_policy.py' \
  'scripts/arm_joint_states_bridge.py' \
  'grab_it_description grab_it_launch.py' \
  'grab_it_robot_state_publisher' \
  'scripts/filter_arm_joint_states.py' \
  'moveit_ros_move_group/move_group' \
  'lib/grab_it_grasp/grasp_from_click' \
  'ros2 run grab_it_grasp grasp_from_click'
do
  pids=$(pgrep -f "$pat" || true)
  if [[ -n "$pids" ]]; then
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
    echo "stopped $pat: $pids"
  fi
done
sleep 1
for pat in \
  'dev-ws/mjpeg_relay.py' \
  'grab_it_gui/gui_node.py' \
  'grab_it_il/play_policy.py' \
  'scripts/arm_joint_states_bridge.py' \
  'grab_it_description grab_it_launch.py' \
  'grab_it_robot_state_publisher' \
  'scripts/filter_arm_joint_states.py' \
  'moveit_ros_move_group/move_group' \
  'lib/grab_it_grasp/grasp_from_click' \
  'ros2 run grab_it_grasp grasp_from_click'
do
  pids=$(pgrep -f "$pat" || true)
  if [[ -n "$pids" ]]; then
    # shellcheck disable=SC2086
    kill -9 $pids 2>/dev/null || true
  fi
done
docker rm -f grab_it_live >/dev/null 2>&1 && echo "container grab_it_live removed"
echo "AGX live stack stopped"
