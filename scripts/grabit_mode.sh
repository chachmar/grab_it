#!/usr/bin/env bash
# One GUI, one world. real and sim never run together.
#   bash ~/dev-ws/scripts/grabit_mode.sh real|sim|stop|status
set -eo pipefail
WS="${GRAB_IT_WS:-$HOME/dev-ws}"
MODE_FILE=/tmp/grabit_mode

stop_pids() {
  local pat="$1"
  local pids
  pids=$(pgrep -f "$pat" || true)
  if [ -n "$pids" ]; then
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
    echo "stopped $pat: $pids"
  fi
}

stop_sim_extras() {
  stop_pids 'scripts/agx_isaac_hub.py'
  stop_pids 'agx_isaac_hub.py'
  stop_pids 'cmd_vel_wifi_relay.py'
  stop_pids 'arm_cmd_wifi_relay.py'
  sleep 0.4
  stop_pids 'scripts/agx_isaac_hub.py'
  stop_pids 'cmd_vel_wifi_relay.py'
  stop_pids 'arm_cmd_wifi_relay.py'
}

start_host_gui() {
  local profile="$1"
  local gen="${WS}/build/interbotix_xs_msgs/rosidl_generator_py/interbotix_xs_msgs"
  local root="${WS}/build/interbotix_xs_msgs"
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
  export ROS_DOMAIN_ID=0
  unset CYCLONEDDS_URI || true
  export FASTRTPS_DEFAULT_PROFILES_FILE="$profile"
  export PYTHONPATH="/home/ubuntu/cosmos3-venv/lib/python3.10/site-packages:${WS}/build/interbotix_xs_msgs/rosidl_generator_py:/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages:${WS}/src/grab_it_gui:${PYTHONPATH:-}"
  export LD_LIBRARY_PATH="${gen}:${root}:${LD_LIBRARY_PATH:-}"
  nohup python3 "$WS/src/grab_it_gui/grab_it_gui/gui_node.py" >/tmp/gui_node.log 2>&1 &
  echo "gui_pid=$!"
}

cmd="${1:-status}"
case "$cmd" in
  real)
    bash "$WS/scripts/_il_real_stop.sh" || true
    stop_sim_extras
    echo real >"$MODE_FILE"
    bash "$WS/scripts/_il_real_restart.sh"
    echo "mode=real  GUI http://192.168.50.207:8765"
    ;;
  sim)
    bash "$WS/scripts/_il_real_stop.sh" || true
    stop_sim_extras
    echo sim >"$MODE_FILE"
    # Hub owns :8081 (cameras + /isaac/cmd). GUI owns :8765.
    nohup python3 "$WS/scripts/agx_isaac_hub.py" >/tmp/agx_isaac_hub.log 2>&1 &
    echo "hub_pid=$!"
    sleep 0.5
    export GRAB_IT_SIM=1
    unset GRAB_IT_IGNORE_ISAAC || true
    start_host_gui "$WS/bridge_both.xml"
    FASTRTPS_DEFAULT_PROFILES_FILE="$WS/bridge_both.xml" \
      nohup python3 "$WS/scripts/cmd_vel_wifi_relay.py" >/tmp/cmd_vel_wifi.log 2>&1 &
    echo "vel_pid=$!"
    FASTRTPS_DEFAULT_PROFILES_FILE="$WS/bridge_both.xml" \
      nohup python3 "$WS/scripts/arm_cmd_wifi_relay.py" >/tmp/arm_cmd_wifi.log 2>&1 &
    echo "arm_relay_pid=$!"
    echo "mode=sim  GUI http://192.168.50.207:8765  hub :8081"
    echo "Start Isaac: C:\\Users\\chach\\Projects\\grab_it-sim\\run_sim_nav.bat --env room"
    ;;
  stop)
    bash "$WS/scripts/_il_real_stop.sh" || true
    stop_sim_extras
    echo stopped >"$MODE_FILE"
    echo "mode=stopped  arm was not commanded home or sleep"
    ;;
  status)
    set +e
    mode="unset"
    if [ -f "$MODE_FILE" ]; then
      mode=$(tr -d '\r\n' <"$MODE_FILE")
    fi
    echo "mode=$mode"
    case "$mode" in
      real) echo "dds_profile=rtabmap_profile.xml (crossover 10.42.0.1)" ;;
      sim) echo "dds_profile=bridge_both.xml (sim / Create 3 only)" ;;
      *) echo "dds_profile=none" ;;
    esac
    echo "ckpt=${GRAB_IT_IL_CKPT:-$WS/datasets/lerobot/outputs/act_foam_pole_lookreach_office_shoulder/checkpoints/last/pretrained_model}"
    for label_pat in \
      "gui grab_it_gui/gui_node.py" \
      "mjpeg dev-ws/mjpeg_relay.py" \
      "hub agx_isaac_hub.py" \
      "act grab_it_il/play_policy.py" \
      "grasp grasp_from_click" \
      "vel_relay cmd_vel_wifi_relay.py" \
      "arm_relay arm_cmd_wifi_relay.py"
    do
      label=${label_pat%% *}
      pat=${label_pat#* }
      if pgrep -f "$pat" >/dev/null 2>&1; then
        echo "$label=up"
      else
        echo "$label=down"
      fi
    done
    ;;
  *)
    echo "Usage: $0 real|sim|stop|status" >&2
    exit 2
    ;;
esac
