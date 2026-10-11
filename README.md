# grab_it

Operator stack for the Grab-IT robot: a Create 3 base, a PX150 arm, and a D435i on a pan mount. The AGX runs navigation, the web GUI, and click-to-grasp. This page is the front door. Longer notes are in `GRABIT_LAUNCH_GUIDE.md`, `GRABIT_GUI_GUIDE.md`, and `GRABIT_SIM_REAL.md`.

## Machines

| Machine | Role |
|---|---|
| **Nano** | D435i and PX150. Container `grab_it_nano`. Crossover address `10.42.0.2`. Start cameras and the arm driver here before the AGX: `docker exec -it grab_it_nano /root/nano_bringup.sh`. ROS domain 0. |
| **AGX** | RTAB-Map, Nav2, the web GUI, and grasp / MoveIt. Host workspace `~/dev-ws` is `/workspaces/isaac_ros-dev` inside the Isaac container. Crossover address `10.42.0.1`. ROS domain 0. |
| **Create 3** | Base odometry and `/cmd_vel`. Wi-Fi only, ROS domain 5. The AGX `domain_bridge` relays `/odom`, `/tf_static`, and `/cmd_vel`. Create 3 `/tf` is not bridged; `odom_to_tf` republishes it. |

Images need the crossover Fast DDS profile (`rtabmap_profile.xml`). The Create 3 bridge and the drive command use `bridge_both.xml` (crossover plus Wi-Fi).

## Three commands

Run these on the AGX, inside the Isaac container, after the Nano and the Create 3 are up.

1. **Navigation** — localize on an existing map and start Nav2:

   ```bash
   ros2 launch grab_it_nav nav.launch.py
   ```

   `bash /workspaces/isaac_ros-dev/nav_bringup.sh` is a wrapper around that launch, and `grabit_session.sh` still calls the wrapper. The first time this package is on the robot:

   ```bash
   cd /workspaces/isaac_ros-dev
   colcon build --packages-select grab_it_nav --symlink-install
   source install/setup.bash
   ```

   Rebuild a map (this deletes `rtabmap.db`):

   ```bash
   ros2 launch grab_it_nav mapping.launch.py
   ```

   `bash mapping_bringup.sh` and `bash remap_map.sh` are unchanged and do the same job.

2. **GUI** — arm, base, cameras, and named places. There is no `gui.launch.py` yet. Start the node directly, or start the whole operator session (nav + GUI + RViz + teleop):

   ```bash
   python3 /workspaces/isaac_ros-dev/src/grab_it_gui/grab_it_gui/gui_node.py
   bash /workspaces/isaac_ros-dev/grabit_session.sh
   ```

   The page is port 8765.

3. **Grasping** — click-to-grasp. The working bringup is the shell, which sets `GRAB_IT_GRASP_PARAMS` and starts the node. MoveIt has to be up first (`scripts/grabit_arm_bringup.sh`).

   ```bash
   bash /workspaces/isaac_ros-dev/scripts/grasp_bringup.sh
   ```

   The package launch file is `ros2 launch grab_it_grasp grasp_from_click.launch.py`. It does not yet load `config/grasp_params.yaml`; the shell does.

Named goals once Nav2 is up: `bash scripts/go_to_named.sh kitchen` (`config/named_locations.yaml`). The same command is installed as `ros2 run grab_it_nav go_to_named kitchen`.

## What this slice did not change

GUI, grasp, imitation-learning scripts, and the experiment scripts under `scripts/` are where they were. The Python helpers and yaml/xml files that navigation already used are still at the old paths. `src/grab_it_nav` has its own copies, and the nav wrapper passes the original paths so a running robot keeps the files it already edits.
