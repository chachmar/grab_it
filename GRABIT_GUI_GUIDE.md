# grab_IT operator GUI

Web control page for the PX150 arm, Create 3 base, cameras, click-to-grasp, nav, imitation learning, and teach-cal.

Open **http://192.168.50.207:8765** (phone or laptop). The header shows **ONLINE** when the websocket is connected. Pick a mode at the top; only that mode’s controls are shown. The last mode is remembered after a refresh.

Bring-up (cameras, arm, Nav2, grasp, IL, teach-cal) is in [GRABIT_LAUNCH_GUIDE.md](GRABIT_LAUNCH_GUIDE.md). This page is how to *use* the GUI once it is up.

---

## Always on screen

These stay visible in every mode.

| Control | What it does |
|---------|----------------|
| **D435i** pane | Depth camera. Blue overlay = object in reach. Green cube = planned grasp. Click here to pick a grasp point (not in Calibration). |
| **WIDE** pane | Jaw camera preview only. Not used for pose clicks. |
| **Cam Pan** | Pan the D435i. Disabled while a grasp locks pan. |
| **Gripper** | Open / close. Spacebar toggles open vs closed. |
| **HOME** | Arm to Home. Keyboard: **H**. |
| **SLEEP** | Arm to Sleep. Keyboard: **Z**. |
| **SERVO STATUS** | Per-joint oval (green / orange / red / blocked) and temperature. **reboot** power-cycles that servo — it drops torque briefly. |

If a pane says `unavailable` or `connecting…`, cameras / MJPEG relay are not up. If the header stays **OFFLINE**, `gui_node` is not running.

GUI on AGX (Isaac container):

```bash
bash /workspaces/isaac_ros-dev/agx_bringup.sh
# or full session (nav + GUI + RViz):
bash /workspaces/isaac_ros-dev/grabit_session.sh
```

Nano cameras + arm must already be up (`/root/nano_bringup.sh` in the Nano container).

### Isaac twin (Windows)

Same page: **http://192.168.50.207:8765**. Do not start a second GUI. Isaac and the AGX must be on the same Wi‑Fi.

Restart **both** after the AGX workspace has these files (Windows edits do not reach the AGX by themselves). Isaac Sim 6 is **Jazzy**; the AGX is **Humble** — ROS topics never match. Isaac uses the same HTTP ports as the browser:

- `GET http://192.168.50.207:8081/isaac/cmd` (arm + drive + grasp events)
- `GET http://192.168.50.207:8081/isaac/grasp_status` (browser poll — clears “Resolving depth…”)
- `POST http://192.168.50.207:8081/push/d435i` (and `/push/wide`)

On the AGX, after copying `scripts/agx_isaac_hub.py` into the container (or `/tmp`):

```bash
kill $(pgrep -f mjpeg_relay.py) 2>/dev/null || true
kill $(pgrep -f agx_isaac_hub.py) 2>/dev/null || true
python3 /workspaces/isaac_ros-dev/scripts/agx_isaac_hub.py
# leave running — serves :8081 and :8765
```

Hard-refresh the browser so it loads the new `gui_hook.js` (status poller).

**Click-to-grasp on the twin** is resolved on Windows (local depth + IK + HTTP status). Do **not** rely on AGX `grasp_bringup.sh` / MoveIt for the twin — Humble cannot see Isaac depth or `FollowJointTrajectory`.

```bash
# optional: real-robot grasp stack only (not for Isaac twin)
# bash /workspaces/isaac_ros-dev/scripts/grasp_bringup.sh
```

Windows: `run_sim_xr.bat` (or nav) with D435i on. After `Arm settle done`, click the D435i pane — status should leave “Resolving depth…” within ~1s (`Point ready` or a clear error). Then **Grasp at point**.

If a D435i click stays on Resolving depth for 6s, the hook times out with “Sim: no depth reply” — usually the hub was not restarted with the updated script, or Isaac is not running.

---

## Mode menu

| Mode | You get |
|------|---------|
| **User Control** | Teleop + click-to-grasp + handoff |
| **Autonomous** | Same as User Control, plus named locations and dock |
| **Imitation** | Same as User Control, plus REC / SAVE / PLAY for training grasps |
| **Calibration** | Teach-cal (tip pose + pan sweep). Click-to-grasp is off. |

---

## User Control

Drive the arm and base by hand, and run click-to-grasp.

### Teleop sticks

| Stick | Axes |
|-------|------|
| **Arm · Base** | Shoulder (up/down), waist (left/right). Keys: **W/S** shoulder, **A/D** waist. |
| **Drive** | Forward / turn. Arrow keys. |
| **Arm · Wrist** | Wrist angle / rotate. Keys: **I/K** angle, **J/L** rotate. |

Joint sliders (waist, shoulder, wrist angle, wrist rotate) match the sticks. Tap a degree readout to type an angle. While Quest teleop is clutched (`/grab_it/quest/status`), arm sticks are ignored.

Cancel any Nav2 goal before driving — teleop, GUI, and Nav2 share `/cmd_vel`.

### Click-to-grasp

**Isaac twin:** resolved on Windows over the HTTP hub (see Isaac twin section above). Restart updated `agx_isaac_hub.py`, hard-refresh the browser, run Isaac with D435i. No AGX MoveIt / `grasp_bringup` required.

**Real robot:** needs Nano cameras + arm and AGX MoveIt + `grasp_from_click`:

```bash
bash /workspaces/isaac_ros-dev/scripts/grabit_arm_bringup.sh
```

If MoveIt is already running and you only need the grasp node:

```bash
bash /workspaces/isaac_ros-dev/scripts/grasp_bringup.sh
```

1. Click the object in the **D435i** view. Wait for the green cube and status XYZ.
2. **Grasp at point** — open → pre-grasp → approach → close → retreat. Pan locks for the motion.
3. **Cancel** aborts the motion.
4. **Unlock pan** if you need the pan slider again.

Wide camera is overlay-only. Do not click there for a pose.

### Handoff (jaw camera)

| Button | Action |
|--------|--------|
| **Grab object** | Hold pose, open gripper (take from a person / table via jaw view). |
| **Release object** | Open and return toward Home. |

---

## Autonomous

Everything in User Control, plus navigation. Also launch the grasp stack if you will Grasp at point.

Nav2 on AGX (Isaac container):

```bash
bash /workspaces/isaac_ros-dev/nav_bringup.sh
# or nav + GUI together:
bash /workspaces/isaac_ros-dev/grabit_session.sh
```

Prereqs: existing map (`remap_map.sh` / `mapping_bringup.sh`), Nano cameras, Create 3 undocked on domain 5. Named poses: `config/named_locations.yaml`.

1. Choose a location (`kitchen`, `living_room`, `trash`, `dock_area`, …).
2. **Go to location** — Nav2 `NavigateToPose`. Confirm in the dialog.
3. **Cancel goal** if you need to take over with the Drive stick.
4. **Restart nav** if Nav2 is wedged (lifecycle Reset → Startup).

**DOCK** / **UNDOCK** talk to Create 3 on domain 5. Confirm the dialog. Stop extra domain-5 clients first or `/dock` may never appear.

Save a new pose from the robot’s current map pose:

```bash
bash /workspaces/isaac_ros-dev/scripts/save_named_location.sh <name>
```

---

## Imitation

Train and replay tabletop picks. Same teleop and click-to-grasp as User Control. If status stays “Recorder idle”, the recorder is not running. Against the Isaac twin (`GRAB_IT_SIM=1`), REC/SAVE uses the same D435i / wrist topics as the Nano.

`grab_it_il` lives in the workspace overlay. A fresh container shell only has Humble, so **do not** run bare `ros2 launch grab_it_il …` until `install/setup.bash` is sourced. The wrapper scripts do that for you.

**1. AGX host → Isaac container**

```bash
docker exec -it -u admin -w /workspaces/isaac_ros-dev isaac_ros_dev-aarch64-container bash
```

Host bind-mount: `~/dev-ws` ↔ `/workspaces/isaac_ros-dev`.

**2. Record** (leave this running)

```bash
bash /workspaces/isaac_ros-dev/scripts/il_record.sh
```

Oracle demos from Grasp at point also need the grasp stack (`grabit_arm_bringup.sh`).

**3. Play** (second container shell; GUI **STOP** armed)

```bash
bash /workspaces/isaac_ros-dev/scripts/il_play.sh
# optional ACT checkpoint:
bash /workspaces/isaac_ros-dev/scripts/il_play.sh \
  /workspaces/isaac_ros-dev/datasets/lerobot/outputs/act_tabletop_pick/checkpoints/last/pretrained_model
```

No checkpoint → open-loop replay of the latest episode.

**4. Train** (AGX **host**, not the container)

```bash
source ~/lerobot-venv/bin/activate
bash ~/dev-ws/scripts/il_smoke.sh
```

| Button | Action |
|--------|--------|
| **REC** | Start an episode. Drive with sticks or run click-to-grasp. |
| **SAVE** | Write the episode. A finished click-to-grasp is tagged `oracle`. |
| **DISCARD** | Drop the take. |
| **PLAY** | Run the policy (or replay the latest episode if there is no checkpoint). Arm sticks are ignored while it plays. |
| **STOP** | Stop recording or playback. |

Episodes: `/workspaces/isaac_ros-dev/datasets/tabletop_pick/episode_XXXXXX/`. Aim for 30–50 successful picks (smoke test ~10).

---

## Calibration

Click-to-grasp calibration (taught tip + pan sweep). D435i clicks do **not** send a grasp point in this mode.

Launch teach-cal (Isaac container):

```bash
bash /workspaces/isaac_ros-dev/scripts/calibration/teach_cal_bringup.sh
```

1. **Free arm** — torque off. Support the arm by hand. Sticks / joint sliders are inert while free.
2. Move the gripper tip to the taught landmark.
3. **Hold arm** (same button) — torque on.
4. **Save tip pose**.
5. **Sleep→taught** to check the saved pose (clear the workspace first).
6. Set **Samples** (default 8, range 2–40) and **Run pan cal**. The arm stays at the taught tip; camera pan sweeps.
7. **Abort cal** to stop a sweep or goto.

HOME / SLEEP still work. Joint readouts stay; teleop sticks are hidden.

After a good sweep, apply results so `grasp_params.yaml` / optical cal get the new numbers:

```bash
python3 /workspaces/isaac_ros-dev/scripts/calibration/grasp_cal_analyze.py teach_cal.jsonl
```

---

## Status colors and locks

- Header **ONLINE** / **OFFLINE** = websocket to `gui_node` (`:8766`).
- **Grasp at point** stays disabled until a D435i click resolves depth.
- Pan slider dims while a grasp holds pan lock.
- **Free arm**: arm sticks and joint sliders dim until Hold.
- **PLAY**: arm sticks dim until STOP.
- Servo oval **blocked** (purple pulse) = that joint is not accepting motion.

---

## Keyboard (when the page is focused)

| Keys | Action |
|------|--------|
| W/S, A/D | Shoulder, waist |
| I/K, J/L | Wrist angle, rotate |
| Arrows | Drive |
| Space | Toggle gripper |
| H | Home |
| Z | Sleep |
