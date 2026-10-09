# grab_it — Launch & Operations Guide

Working reference for bringing up the Nano, AGX, Create 3, cameras, arm, mapping, and Nav2 after a reboot or for day-to-day use.

## 1. System at a glance

| Machine | Role | Networks |
|---------|------|----------|
| **Nano** | D435i + PX150 (`grab_it_nano` container) | Wi-Fi + crossover `10.42.0.2` |
| **AGX** | RTAB-Map, Nav2, teleop bridge (`isaac_ros_dev-aarch64-container`) | Wi-Fi + crossover `10.42.0.1` |
| **Create 3** | Base odom / `cmd_vel` | Wi-Fi only, **ROS domain 5** |

- Main graph (cameras, arm, RTAB, Nav2): **domain 0**, Fast DDS (`rmw_fastrtps_cpp`).
- Create 3: **domain 5**; `domain_bridge` on the AGX relays odom/tf/`cmd_vel`.
- Images use the **crossover** Fast DDS profile; Create 3 needs **Wi-Fi** (use `bridge_both.xml` for the bridge / teleop).

Workspace mount: AGX host `~/dev-ws` ↔ container `/workspaces/isaac_ros-dev`.

---

## 2. Cold start after reboot

Do Nano first (sensors), then AGX (compute / nav).

### 2.1 Nano (host)

```bash
# Crossover IP (expect 10.42.0.2/24)
ip -br addr show enP8p1s0

# Device nodes (udev should persist)
ls -l /dev/ttyDXL /dev/video_wide

# 12V servo power ON, then:
sudo docker start grab_it_nano
sudo docker exec -it grab_it_nano bash
```

Inside the Nano container:

```bash
/root/nano_bringup.sh
```

Leave this running (D435i + wide USB cam; arm if `/dev/ttyDXL` / U2D2 is plugged in with 12V on).

If arm does not start: `lsusb` should show an FTDI U2D2; udev maps serial `FT6S4H3P` → `/dev/ttyDXL`. Plug the adapter in and re-run `nano_bringup.sh`.

### 2.2 AGX (host)

```bash
# Crossover IP (expect 10.42.0.1/24)
ip -br addr show eno1

# Preferred: start from the docker-committed grab_it snapshot
bash ~/dev-ws/run_grabit_dev.sh
```

If you intentionally want a Dockerfile rebuild instead:

```bash
cd ~/dev-ws/src/isaac_ros_common/scripts && ./run_dev.sh -d ~/dev-ws
```

Extra shell into an already-running container:

```bash
docker exec -it -u admin -w /workspaces/isaac_ros-dev isaac_ros_dev-aarch64-container bash
```

### 2.3 AGX — mapping or navigation

**Mapping (rebuild / extend map):**

```bash
bash /workspaces/isaac_ros-dev/remap_map.sh
# or: bash mapping_bringup.sh
```

Drive slowly with teleop (see §5). Ctrl‑C when finished.

**Navigation (existing `rtabmap.db`):**

```bash
bash /workspaces/isaac_ros-dev/nav_bringup.sh
```

### 2.4 Full operator session (Nav2 + GUI + RViz + teleop)

One script inside the Isaac container after Nano bringup + Create 3 are up:

```bash
bash /workspaces/isaac_ros-dev/grabit_session.sh
```

Starts, in order:

1. `nav_bringup.sh` (bridge, RTAB loc, scan, Nav2) — unless already running / `--skip-nav`
2. Arm/base web GUI (`gui_node`, **`bridge_both.xml`**) + MJPEG relay → **http://192.168.50.207:8765**
3. Nav2 RViz (`LIBGL_ALWAYS_SOFTWARE=1`, real time — not sim)
4. Keyboard teleop on **domain 0** + `bridge_both.xml` (foreground)

GUI must use `bridge_both` (not crossover-only) so `/cmd_vel` reaches Create 3 through `domain_bridge`. Drive stick works even if the arm is off; arm needs Nano `xsarm_control` / joint states. **Dock/Undock** run as a domain‑5 subprocess (actions are not bridged). **Go to location** uses `config/named_locations.yaml` + Nav2 `navigate_to_pose`.

Useful flags: `--no-gui` `--no-rviz` `--no-teleop` `--skip-nav` `--kill-nav` (tear down Nav2 on Ctrl‑C; default leaves nav running).

Logs: `/tmp/grabit_session/`. Cancel Nav2 goals before manual drive — teleop, GUI, and Nav2 share `/cmd_vel`.

---

## 3. RViz on AGX (inside the container)

RViz runs **inside** the Isaac ROS container (same graph as mapping/Nav2). Use a
**new** container shell — do not stop `nav_bringup` / `mapping_bringup`.

### 3.1 Open a container shell

If the container is already up (`run_grabit_dev.sh`):

```bash
docker exec -it -u admin -w /workspaces/isaac_ros-dev isaac_ros_dev-aarch64-container bash
```

Or start fresh:

```bash
bash ~/dev-ws/run_grabit_dev.sh
```

### 3.2 Env + launch RViz

```bash
source /opt/ros/humble/setup.bash
source /workspaces/isaac_ros-dev/install/setup.bash 2>/dev/null || true
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
unset CYCLONEDDS_URI
export FASTRTPS_DEFAULT_PROFILES_FILE=/workspaces/isaac_ros-dev/rtabmap_profile.xml

# Avoids blank / black RViz window on Jetson (NoMachine / GL)
export LIBGL_ALWAYS_SOFTWARE=1

ros2 topic hz /map          # confirm map is live (nav/mapping running)

# Nav2-ready RViz (goals, costmaps, etc.):
ros2 launch nav2_bringup rviz_launch.py

# Or plain RViz if you only need TF / scan / map displays:
# rviz2
```

### 3.3 RViz setup (Nav2)

1. Fixed Frame = `map`
2. Uncheck **Amcl Particle Swarm** (unused — you use RTAB)
3. **2D Pose Estimate** on the robot, correct heading; wait a few seconds
4. Send a Nav2 Goal, or use a named location (§4)

Useful displays if starting from plain `rviz2`: **TF**, **Map** (`/map` or `/rtabmap/map`), **LaserScan** / depth cloud as needed.

Optional: save layout to `~/.rviz2/grab_it.rviz` and reopen with
`rviz2 -d ~/.rviz2/grab_it.rviz` (keep `LIBGL_ALWAYS_SOFTWARE=1`).

---

## 4. Named locations

Poses live in `config/named_locations.yaml` (map frame).

| Name | How to use |
|------|------------|
| `kitchen` | `bash scripts/go_to_named.sh kitchen` |
| `living_room` | `bash scripts/go_to_named.sh living_room` |
| `dock_area` | add via save script if not present |
| `trash` | add via save script if not present |

**Go to a place** (nav up):

```bash
bash scripts/go_to_named.sh kitchen
```

**Save current pose** as a name:

```bash
# robot localized in map, standing at the spot:
bash scripts/save_named_location.sh trash
```

**Print pose** (for hand-editing YAML):

```bash
export FASTRTPS_DEFAULT_PROFILES_FILE=/workspaces/isaac_ros-dev/rtabmap_profile.xml
ros2 run tf2_ros tf2_echo map base_link
# use Translation x/y and RPY yaw (radians)
```

---

## 5. Teleop (Create 3)

`domain_bridge` must be running (`nav_bringup` or `mapping_bringup`).

**Recommended (domain 0, through the bridge):**

```bash
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
unset CYCLONEDDS_URI
export FASTRTPS_DEFAULT_PROFILES_FILE=/workspaces/isaac_ros-dev/bridge_both.xml

ros2 topic hz /odom    # ~20 Hz
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

Stay on **domain 0** for teleop. Do **not** run teleop (or other extra clients) on domain 5 — Create 3 only allows a few DDS peers; domain-5 teleop blocks `/dock` discovery.

**Joystick (if configured):**

```bash
ros2 run joy joy_node
python3 /workspaces/isaac_ros-dev/src/grab_it_gui/grab_it_gui/grab_it_joy.py
```

Drive slowly while mapping; pause at corners; revisit halls both ways.

### 5.1 Dock / undock Create 3

`/dock` and `/undock` live on the Create 3 (**domain 5**). They are **not** relayed by `create3_bridge.yaml` (only odom / tf / `cmd_vel`).

Create 3 Wi-Fi: `192.168.50.251` (`create3.local`) — web UI for ROS domain / RMW: `http://192.168.50.251/ros-config`.

**Before docking:**

1. Stop any **domain 5** teleop / extra ROS clients (Ctrl+C). Domain-0 teleop through the bridge is fine to stop too so it does not fight `cmd_vel`.
2. Cancel an active Nav2 goal if the base is still driving.
3. Be near the powered dock (action fails if the dock is not visible).

**Dock (AGX container):**

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=5
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
unset CYCLONEDDS_URI
export FASTRTPS_DEFAULT_PROFILES_FILE=/workspaces/isaac_ros-dev/bridge_both.xml

ros2 action list                    # should show /dock
ros2 action send_goal /dock irobot_create_msgs/action/Dock "{}"
```

**Undock:**

```bash
ros2 action send_goal /undock irobot_create_msgs/action/Undock "{}"
```

If `Waiting for an action server…` hangs: you are missing `bridge_both.xml`, or too many domain-5 peers are connected (stop teleop on d5, retry). Nav on domain 0 can stay running.

---

## 6. Cameras

How Nano video stays at capture rate (wide **30 Hz**, D435i color/depth as launched) without Wi-Fi / extra DDS readers stalling the stream: [NANO_CAMERA_BRIDGE.md](NANO_CAMERA_BRIDGE.md).

### Preferred (Nano bringup)

Use `/root/nano_bringup.sh` or the project dual-camera launch (848×480, aligned depth, sync, no `tf_prefix`).

### Manual D435i (standalone)

```bash
source /opt/ros/humble/setup.bash
source ~/realsense_ws/install/setup.bash   # if using a dedicated realsense_ws
ros2 launch realsense2_camera rs_launch.py \
  enable_depth:=true \
  enable_color:=true \
  enable_gyro:=true \
  enable_accel:=true \
  unite_imu_method:=2
```

### Wide USB cam

```bash
ros2 launch ~/usb_cam_config/dual_camera.launch.py
```

Confirm aligned depth when mapping/nav need it:

```bash
export FASTRTPS_DEFAULT_PROFILES_FILE=/workspaces/isaac_ros-dev/rtabmap_profile.xml
ros2 topic echo /d435i/d435i/aligned_depth_to_color/image_raw --once --qos-profile sensor_data --no-arr
```

---

## 7. Arm (PX150 no-elbow)

### Prerequisites

- `/dev/ttyDXL` present on Nano  
- **12V servo power ON**  
- Prefer launching arm from Nano (`xsarm_control` / `nano_bringup`) so the USB device is local  

### Fix delayed / starved commands (xs_sdk)

All subscriptions share a mutually exclusive callback group with the joint-state timer; at 100 Hz the serial read can starve commands. Drop the rate to **10 Hz**:

```bash
sed -i 's/update_rate: 100/update_rate: 10/' \
  ~/interbotix_ws/install/interbotix_xsarm_control/share/interbotix_xsarm_control/config/px150.yaml

sed -i 's/update_rate: 100/update_rate: 10/' \
  ~/interbotix_ws/src/interbotix_ros_core/interbotix_ros_xseries/interbotix_xsarm_control/config/px150.yaml 2>/dev/null

grep -A1 "joint_state_publisher" \
  ~/interbotix_ws/install/interbotix_xsarm_control/share/interbotix_xsarm_control/config/px150.yaml
```

Rebuild/relaunch arm control after editing. Persist via Docker commit (§8) if the change is only inside the image.

### Launch arm control

```bash
source /opt/ros/humble/setup.bash
source /home/admin/interbotix_ws/install/setup.bash   # or Nano equivalent path
unset CYCLONEDDS_URI
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

ros2 launch interbotix_xsarm_control xsarm_control.launch.py \
  robot_model:=px150_no_elbow hardware_type:=actual
```

### Arm GUI

Operator controls (modes, sticks, grasp, nav, IL, teach-cal): [GRABIT_GUI_GUIDE.md](GRABIT_GUI_GUIDE.md).

```bash
python3 /workspaces/isaac_ros-dev/src/grab_it_gui/grab_it_gui/gui_node.py
```

### MoveIt

```bash
source /opt/ros/humble/setup.bash
source ~/interbotix_ws/install/setup.bash
ros2 launch interbotix_xsarm_moveit xsarm_moveit.launch.py \
  robot_model:=px150_no_elbow \
  hardware_type:=actual \
  dxl_com_port:=/dev/ttyDXL
```

Preferred on AGX (driver stays on Nano):

```bash
bash /workspaces/isaac_ros-dev/scripts/grabit_arm_bringup.sh
# starts MoveIt + robot_description + grasp_from_click
```

### Click-to-grasp (GUI + D435i depth)

MVP tabletop grasp: click a point in the D435i pane of the web GUI, then **Grasp at point**.

1. Nano cameras + arm driver up; AGX MoveIt via `grabit_arm_bringup.sh` (includes `grasp_from_click`).
2. Open **http://192.168.50.207:8765**
3. Click the object in the **D435i** view (wide USB is view-only — no depth).
4. Pan locks to 0 rad so `config/camera_tf.env` stays valid; status shows XYZ in `px150_no_elbow/base_link`.
5. Press **Grasp at point** → open gripper → pre-grasp → approach → close → retreat.
6. **Cancel** aborts; **Unlock pan** re-enables the pan slider.

Config: `config/grasp_params.yaml` (workspace bounds, approach offsets, depth limits).  
Topics: `/grab_it/grasp_click`, `/grab_it/grasp_command`, `/grab_it/grasp_status`.  
Build once: `colcon build --packages-select grab_it_grasp --symlink-install`.

### Recovery services

Hardware error on startup (reboot motors):

```bash
ros2 service call /px150/reboot_motors interbotix_xs_msgs/srv/Reboot \
  "{cmd_type: 'group', name: 'all', enable: true, smart_reboot: true}"
```

Torque re-enable:

```bash
ros2 service call /px150/torque_enable interbotix_xs_msgs/srv/TorqueEnable \
  "{cmd_type: 'group', name: 'all', enable: true}"
```

Namespace may be `/px150_no_elbow/...` depending on launch — check with `ros2 service list | grep -i torque`.

Verify packages:

```bash
ros2 pkg list | grep -E "interbotix_xs_sdk|interbotix_xs_ros_control|interbotix_xsarm_moveit"
```

---

## 8. Docker: commit, restore, cleanup

Workspace files under `~/dev-ws` persist on the host. **Apt installs and in-image edits** need a commit or Dockerfile layer.

### Save current AGX container (recommended)

```bash
# on AGX host, after exiting or while running:
docker commit isaac_ros_dev-aarch64-container grab_it_isaac_ros_dev:latest
docker tag grab_it_isaac_ros_dev:latest grab_it_isaac_ros_dev:$(date +%Y%m%d_%H%M%S)
```

Then always start with:

```bash
bash ~/dev-ws/run_grabit_dev.sh
```

### Alternate (retag stock name)

```bash
docker commit isaac_ros_dev-aarch64-container isaac_ros_dev-aarch64
cd ~/dev-ws && ./src/isaac_ros_common/scripts/run_dev.sh --skip_image_build -d ~/dev-ws
```

### Backup / restore image to disk

```bash
# backup (5–15+ min)
docker save grab_it_isaac_ros_dev:latest | gzip > ~/grab_it_isaac_backup.tar.gz

# restore
docker load < ~/grab_it_isaac_backup.tar.gz
docker tag grab_it_isaac_ros_dev:latest isaac_ros_dev-aarch64:latest
```

### Nano commit

```bash
docker commit grab_it_nano grab_it_nano
```

### Cleanup dangling images

```bash
docker image prune -f
```

### Durable apt packages (optional)

`~/.isaac_ros_common-config` + `Dockerfile.grabit` (see `CONFIG_IMAGE_KEY=ros2_humble.grabit`) bake Nav2/RTAB packages into a rebuildable layer. Prefer that for package lists; use **docker commit** for ad-hoc state (arm rate fix, pip, etc.).

---

## 9. Imitation learning (tabletop pick)

Switching between Isaac Sim and the real PX150 is one GUI and two exclusive launches. Use [GRABIT_SIM_REAL.md](GRABIT_SIM_REAL.md) (`bash ~/dev-ws/scripts/grabit_mode.sh real|sim|stop|status`). Do not follow the container `il_play.sh` steps below on the real arm: that path can fold the shoulder into the demo park. Real PLAY stays in the current pose.

Collect GUI / click-to-grasp demos, train ACT on the **AGX host**, replay on the real PX150 or the Isaac Sim twin in `~/Projects/grab_it-sim` (same joint topics). This is **not** Isaac Lab yet — Mimic/Robomimic can wrap that USD later.

Action / observation (20 Hz): measured + commanded `[waist, shoulder, wrist_angle, wrist_rotate, gripper]`, D435i color, optional wrist cam.

### Record (Isaac ROS container, session / GUI up)

`grab_it_il` is in `src/grab_it_il` and the workspace install overlay. A new container shell only has Humble — **`ros2 launch grab_it_il …` fails until you source `install/setup.bash`**. Use the wrappers (they source it):

```bash
# AGX host
docker exec -it -u admin -w /workspaces/isaac_ros-dev isaac_ros_dev-aarch64-container bash

bash /workspaces/isaac_ros-dev/scripts/il_record.sh
```

Manual equivalent:

```bash
source /opt/ros/humble/setup.bash
source /workspaces/isaac_ros-dev/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
unset CYCLONEDDS_URI
export FASTRTPS_DEFAULT_PROFILES_FILE=/workspaces/isaac_ros-dev/rtabmap_profile.xml
ros2 launch grab_it_il record_demos.launch.py
```

If colcon says **unknown package**, this AGX tree does not have `src/grab_it_il` yet. Copy it onto the AGX host, then rebuild:

```bash
# from the Windows repo, once:
#   scp -r src/grab_it_il ubuntu@192.168.50.207:~/dev-ws/src/
ls /workspaces/isaac_ros-dev/src/grab_it_il/package.xml
colcon build --packages-select grab_it_il --symlink-install
source install/setup.bash
```

In the GUI: **REC** → demonstrate a pick → **SAVE** (success) or **DISCARD**. A finished click-to-grasp tags the take as `oracle`. Episodes land in `/workspaces/isaac_ros-dev/datasets/tabletop_pick/episode_XXXXXX/`. Aim for 30–50 successful picks (smoke test ~10).

### Train (AGX host — not this container)

```bash
# host venv (lerobot 0.4 writes/reads dataset v3.0 — not the old v2.1 layout)
source ~/lerobot-venv/bin/activate
python3 ~/dev-ws/scripts/il_to_lerobot.py \
  --input ~/dev-ws/datasets/tabletop_pick \
  --output ~/dev-ws/datasets/lerobot/grabit_tabletop_pick
# if ffmpeg/h264 encode fails on Jetson, add: --images
lerobot-train \
  --policy.type=act \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/grabit_tabletop_pick \
  --dataset.root=$HOME/dev-ws/datasets/lerobot/grabit_tabletop_pick \
  --output_dir=$HOME/dev-ws/datasets/lerobot/outputs/act_tabletop_pick_smoke \
  --policy.device=cuda \
  --batch_size=4 \
  --steps=1500
# or: bash ~/dev-ws/scripts/il_smoke.sh   /   bash ~/dev-ws/scripts/il_train.sh
```

### Play (container; GUI **STOP** armed)

```bash
bash /workspaces/isaac_ros-dev/scripts/il_play.sh
# optional checkpoint:
bash /workspaces/isaac_ros-dev/scripts/il_play.sh \
  /workspaces/isaac_ros-dev/datasets/lerobot/outputs/act_tabletop_pick/checkpoints/last/pretrained_model
```

Without a checkpoint the player replays the latest episode (open-loop). GUI **PLAY** / **STOP** publish `/grab_it/il_command`. While playing, arm sticks are ignored.

### Twin (`grab_it-sim` on the Windows Isaac Sim box)

```bat
run_sim_nav.bat --env room
```

The twin now SUB `/px150_no_elbow/commands/joint_group` + `joint_single` and PUB `/px150_no_elbow/joint_states`. Build `interbotix_xs_msgs` once with `build_arm_msgs.bat` if Isaac Sim cannot import those types. Arm IL uses wall-clock like the real GUI; Nav2 still needs `use_sim_time:=true`.

Isaac Lab Mimic is a later wrapper on the existing USD — do not re-import the robot.

### Quest 3 immersive teleop (sim + real)

Same joint topics as the GUI. The 4-DOF mapper lives in `grab_it_quest` — do **not** enable Isaac’s IK / Grasp / Locomotion panels on the PX150.

**Pair the headset (Windows):**

`python -m isaacteleop.cloudxr` will **not** run here. NVIDIA’s `isaacteleop` package is Linux-only (manylinux wheels; no Windows build). This PC’s default Python is also 3.14, which that stack does not support.

Use **Meta Quest Link** as the OpenXR runtime:

```bat
C:\Users\chach\Projects\grab_it-sim\start_cloudxr.bat
```

That script opens Quest Link if it is installed. Then: enable Quest Link / Air Link on the headset, and in Quest Link set **Settings → General → OpenXR Runtime → Set Meta**. No-headset fallback: Isaac Teleop **Session → Debug → Debug Tracking** (drag the Right marker; on AGX use `clutch_mode:=always`, sim only).

**Start the stereo twin:**

```bat
C:\Users\chach\Projects\grab_it-sim\run_sim_xr.bat
:: same as: run_sim_nav.bat --xr --env room
```

Press Play. Confirm a Right controller / `/Teleop/Markers` prim moves. Do **not** enable Isaac’s IK / Grasp / Locomotion panels. The bridge publishes `/grab_it/quest/right_pose` and `/grab_it/quest/joy`. A cyan ghost sphere tracks the commanded EE.

**Start the mapper (AGX container, domain 0):**

```bash
source /opt/ros/humble/setup.bash
source /workspaces/isaac_ros-dev/install/setup.bash
# first time after pull:
# colcon build --packages-select grab_it_quest && source install/setup.bash
ros2 launch grab_it_quest quest_teleop.launch.py
# or: bash /workspaces/isaac_ros-dev/scripts/quest_teleop.sh
```

Controls (right controller):

1. Hold **grip** (deadman / clutch). Release freezes the arm; re-grip snapshots a new offset.
2. Move the controller to drive EE XYZ (position-only IK).
3. **Roll** the controller to twist `wrist_rotate`.
4. **Trigger click** toggles the gripper.

Debug without buttons: `clutch_mode:=always` is **sim-only**.

```bash
ros2 launch grab_it_quest quest_teleop.launch.py clutch_mode:=always
```

**Real PX150** (same node; twin can stay up as the headset view):

```bash
ros2 launch grab_it_quest quest_teleop.launch.py target:=real
# both backends (explicit):
ros2 launch grab_it_quest quest_teleop.launch.py target:=both allow_both:=true
```

Real / both checklist: Nano `xs_sdk` up, 12V on, `/px150_no_elbow/joint_states` live, grip clutch required. Stale joint states block motion. Scale is 0.4 on real. GUI arm sticks are ignored while Quest is clutched; IL PLAY also blocks Quest.

---

## 10. Quick command cheat sheet

| Task | Command |
|------|---------|
| Start AGX from snapshot | `bash ~/dev-ws/run_grabit_dev.sh` |
| Extra AGX shell | `docker exec -it -u admin -w /workspaces/isaac_ros-dev isaac_ros_dev-aarch64-container bash` |
| Nano sensors | `/root/nano_bringup.sh` |
| Map | `bash remap_map.sh` |
| Navigate | `bash nav_bringup.sh` |
| Full session (nav+GUI+RViz+teleop) | `bash grabit_session.sh` (see §2.4) |
| RViz (in AGX container) | `LIBGL_ALWAYS_SOFTWARE=1` + `ros2 launch nav2_bringup rviz_launch.py` (+ rtabmap_profile); see §3 |
| Teleop | `teleop_twist_keyboard` (+ bridge_both, domain 0) |
| Arm/base GUI only | `bash agx_bringup.sh` or `ros2 run grab_it_gui gui_node` → :8765 |
| Dock Create 3 | domain 5 + bridge_both → `ros2 action send_goal /dock irobot_create_msgs/action/Dock "{}"` |
| Undock Create 3 | domain 5 + bridge_both → `ros2 action send_goal /undock irobot_create_msgs/action/Undock "{}"` |
| Go to kitchen | `bash scripts/go_to_named.sh kitchen` |
| Save pose | `bash scripts/save_named_location.sh <name>` |
| Commit AGX image | `docker commit isaac_ros_dev-aarch64-container grab_it_isaac_ros_dev:latest` |
| Record IL demos | container: `bash scripts/il_record.sh` + GUI REC/SAVE |
| Train ACT (host) | `bash ~/dev-ws/scripts/il_smoke.sh` |
| Play policy | container: `bash scripts/il_play.sh [checkpoint]` |
| Quest XR twin | Windows: Quest Link via `start_cloudxr.bat` + `run_sim_xr.bat` |
| Quest EE teleop | container: `bash scripts/quest_teleop.sh` (`target:=real` for the PX150) |

---

## 11. Common pitfalls

| Symptom | Likely cause |
|---------|----------------|
| No camera / RTAB “no data” | Crossover down; wrong Fast DDS profile; Nano bringup not running |
| `nav_bringup` FAIL D435i / “Terminated” at camera check | RealSense stuck (`Device or resource busy` in `/tmp/cameras.log`); restart cameras on Nano (pkill realsense + `nano_bringup`). Crossover ping can still be OK. Also kill leftover **sim** RViz (`use_sim_time:=true`) before real nav |
| Nav2 `component_container_isolated` exit −9 | Usually SIGKILL from a stale / double nav launch or OOM — `pkill` old nav2/rtabmap/rviz, then rerun `nav_bringup.sh` |
| No `/odom` | Create 3 domain 5 / undocked; bridge not up; use `bridge_both.xml` |
| `/dock` waits forever | Need domain 5 + `bridge_both.xml`; stop domain-5 teleop (Create 3 peer limit); `/dock` is not bridged to domain 0 |
| RViz arm “No transform” | Arm TF rooted at `world`, not linked to `base_link` (nav can still work) |
| Costmap “sensor origin out of bounds” (high Z) | Create 3 `/odom` Z drifted — reset Create 3 pose / reboot base |
| Arm commands delayed | `update_rate: 100` → set to `10` in `px150.yaml` |
| `ttyDXL` missing | Udev / cable / power; check `ls -l /dev/ttyDXL` |
| Lost apt packages after new container | Didn’t use `run_grabit_dev.sh` / forgot `docker commit` |
| IL recorder idle / no SAVE | `record_demos` not running; episode shorter than 1 s |
| Twin arm does not move | Use `/px150_no_elbow/commands/joint_targets` (GUI/Quest always publish it). `interbotix_xs_msgs` is optional on Windows |
| `isaacteleop.cloudxr` missing | Expected on Windows — package is Linux-only. Use Quest Link |
| Quest arm silent on real | stale `/joint_states`, `target:=both` without `allow_both:=true`, or IL PLAY |

---

## 12. Misc dependencies

Websockets (GUI / tools that need it):

```bash
pip3 install websockets --break-system-packages
```
