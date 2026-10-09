# grab_it — Sim and real imitation

One GUI at `http://192.168.50.207:8765`. Launch **sim or real, never both**. The page, buttons, and joint order stay the same. What changes is which process owns ports `8765` and `8081`, which DDS profile is loaded, and where PLAY sends the arm.

The header badge reads **REAL** or **SIM**. That label comes from `/tmp/grabit_mode`, written by `scripts/grabit_mode.sh`.

## Why the stacks cannot run together

They share ROS domain 0, the same joint topics (`/px150_no_elbow/commands/joint_group`, `joint_single`, `joint_states`), and the same camera HTTP paths. A sim PLAY can fold the real shoulder toward the demo park (about −1.9 rad). That path already hit the electronics stack. A real GUI left on `bridge_both.xml` steals the Nano camera stream.

```text
One GUI :8765
        |
        +-- REAL --> Nano cameras + xs_sdk --> click approach --> ACT close
        |
        +-- SIM  --> Isaac Sim (Windows) --> agx_isaac_hub
        |
        both use the same ACT checkpoint (GRAB_IT_IL_CKPT)
```

## What is already shared

- GUI: `src/grab_it_gui/grab_it_gui/gui.html` and `gui_node.py`.
- Policy player: `src/grab_it_il/grab_it_il/play_policy.py`. State order is waist, shoulder, wrist_angle, wrist_rotate, gripper, pan.
- Wrist sign matches the real arm (`ISAAC_JOINT_SIGN` for wrist_angle is +1). Sleep on the real robot is about shoulder −13° and wrist −100°. The demo park (shoulder about −1.9, wrist about +1.4) is a different pose. It is inside the servo stops. The path between sleep and that park is unsafe on the real arm.
- Default checkpoint:

```text
~/dev-ws/datasets/lerobot/outputs/act_foam_pole_lookreach_office_shoulder/checkpoints/last/pretrained_model
```

- Sim twin on the Windows box: `C:\Users\chach\Projects\grab_it-sim\run_sim_nav.bat --env room`.

## One command to switch

On the AGX host (`ubuntu@jetson-agx`, workspace `~/dev-ws`):

```bash
bash ~/dev-ws/scripts/grabit_mode.sh real     # Nano cameras, click-to-grasp, idle ACT
bash ~/dev-ws/scripts/grabit_mode.sh sim      # Isaac hub + sim GUI; real stack stopped
bash ~/dev-ws/scripts/grabit_mode.sh stop     # both stacks down; arm stays where it is
bash ~/dev-ws/scripts/grabit_mode.sh status   # mode file, processes, expected DDS profile
```

`real` stops the Isaac hub and the sim Wi-Fi relays, writes `real` to `/tmp/grabit_mode`, then runs `scripts/_il_real_restart.sh`:

- MJPEG and GUI on `rtabmap_profile.xml` (AGX crossover `10.42.0.1` only)
- `GRAB_IT_IGNORE_ISAAC=1` so a sim JPEG push cannot replace the Nano cameras
- click-to-grasp in the `grab_it_live` container (`use_arm_driver:=false`; Nano `xs_sdk` owns `/dev/ttyDXL`)
- ACT player on the host, idle until PLAY or a click handoff, with `GRAB_IT_IL_FROM_CURRENT=1`

`sim` stops that real stack (it does not stop Nano), writes `sim`, and starts `scripts/agx_isaac_hub.py` on port `8081` plus the same GUI process with `GRAB_IT_SIM=1` and `bridge_both.xml`. The hub keeps cameras and `/isaac/cmd`. The page stays on `8765`. Wi-Fi command relays start so Isaac can read arm JSON. This start does not send a park pose. Inside the container, `agx_bringup.sh --sim --no-mjpeg` is the same split (hub owns `8081`).

`stop` runs the real stop script and then kills the hub and relays. Joints are not commanded Home or Sleep.

In **REAL**, PLAY starts the host ACT player from the current joints. In **SIM**, PLAY is posted to the Isaac hub. Grasp at point drives the real click node only in REAL. In SIM the same button is posted to the hub, and the GUI does not publish arm or grasp commands onto the real bus.

## Checkpoint

Both modes read `GRAB_IT_IL_CKPT`. If it is unset, the real player uses the shoulder lookreach checkpoint above. Point at a new train only when you mean to:

```bash
export GRAB_IT_IL_CKPT=$HOME/dev-ws/datasets/lerobot/outputs/act_tabletop_pick/checkpoints/last/pretrained_model
bash ~/dev-ws/scripts/grabit_mode.sh real
```

## Episode tag

Each saved episode is `datasets/tabletop_pick/episode_*/meta.json`.

- `source` stays `teleop` or `oracle` (how the pick was demonstrated).
- `world` is `sim` or `real`, read from `/tmp/grabit_mode` at save time.

Filter on `world` when you fine-tune. Observation contract, 20 Hz: joints in the order above, D435i color, wide/wrist image, pole XY in `px150_no_elbow/base_link`.

## Safety rules

- On the real arm, do not run `scripts/_il_to_demo_pose.sh` and do not fold the shoulder toward −1.9. Real ACT starts from the current joints. The shoulder cannot fold back past the pose at PLAY or at the click handoff. The close happens after the click approach.
- Real cameras use `rtabmap_profile.xml` (crossover `10.42.0.1` only). `bridge_both.xml` is for sim and for Create 3 navigation, not for a real grasp session.
- Nano `xs_sdk` owns `/dev/ttyDXL`. The AGX grasp container does not open that port.
- STOP holds the current joints. It does not go Home or Sleep.
- If joint states read about −π on every arm joint, that is a failed sync, not a pose. Do not PLAY.

## Workflow A — train in Isaac, run on the real arm

### A1. Sim only

On the AGX host:

```bash
bash ~/dev-ws/scripts/grabit_mode.sh sim
bash ~/dev-ws/scripts/grabit_mode.sh status
```

`status` must say `sim`, with `agx_isaac_hub.py` up and no `grasp_from_click` or real `play_policy`. On the Windows Isaac box:

```bat
C:\Users\chach\Projects\grab_it-sim\run_sim_nav.bat --env room
```

Press Play in Isaac. Open `http://192.168.50.207:8765`. The badge says SIM and the D435i pane is the sim camera.

### A2. Collect

In the GUI imitation row: REC, run a pick (sticks or Grasp at point), SAVE on a success, DISCARD on a miss. Episodes land in `~/dev-ws/datasets/tabletop_pick/episode_*` with `"world": "sim"`. Cover the pole positions you care about. The shoulder lookreach set that already trains is a separate dataset behind the default checkpoint. New episodes do not overwrite that checkpoint.

### A3. Convert and train on the AGX host

LeRobot is the host venv, not the ROS container.

```bash
source ~/lerobot-venv/bin/activate
python3 ~/dev-ws/scripts/il_to_lerobot.py \
  --input ~/dev-ws/datasets/tabletop_pick \
  --output ~/dev-ws/datasets/lerobot/grabit_tabletop_pick
lerobot-train \
  --policy.type=act \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/grabit_tabletop_pick \
  --dataset.root=$HOME/dev-ws/datasets/lerobot/grabit_tabletop_pick \
  --output_dir=$HOME/dev-ws/datasets/lerobot/outputs/act_tabletop_pick \
  --policy.device=cuda \
  --batch_size=4 \
  --steps=80000
```

Deploy those weights only when you choose to:

```bash
export GRAB_IT_IL_CKPT=$HOME/dev-ws/datasets/lerobot/outputs/act_tabletop_pick/checkpoints/last/pretrained_model
```

Until that export is set, real mode keeps the shoulder checkpoint.

### A4. Leave sim completely

Close Isaac Play. Then:

```bash
bash ~/dev-ws/scripts/grabit_mode.sh stop
bash ~/dev-ws/scripts/grabit_mode.sh status
```

No hub, no GUI, no `play_policy`.

### A5. Real only

Nano first (12 V on, U2D2 at `/dev/ttyDXL`):

```bash
sudo docker start grab_it_nano
sudo docker exec -it grab_it_nano bash
/root/nano_bringup.sh
```

Leave that running. On the AGX:

```bash
export GRAB_IT_IL_CKPT=...   # only when deploying a new checkpoint
bash ~/dev-ws/scripts/grabit_mode.sh real
bash ~/dev-ws/scripts/grabit_mode.sh status
```

`status` must say `real`. The grasp log (`/tmp/grabit_session/grasp.log` inside the container) includes `act_closes=True`. The ACT log `/tmp/il_real_play_host.log` shows the player idle (`IL player kind=act`) and the arm has not moved. Refresh `http://192.168.50.207:8765`. The badge says REAL. The D435i and wide panes must be updating. The wide pane stays blank until Nano `/root/nano_bringup.sh` is publishing `/wide_cam/image_raw`.

### A6. Apply

Click the pole in the D435i image, then **Grasp at point**. The arm opens, approaches that click, and ACT closes from the pose it reached. The shoulder will not fold back past that pose. STOP or Cancel holds the arm. PLAY is the same ACT player from the current pose. It does not move to the demo park.

## Workflow B — record on the real arm, train or replay in sim

### B1. Real session

Use A5. In the imitation row, REC a real pick, SAVE, DISCARD as needed. Episodes are tagged `"world": "real"` in the same `datasets/tabletop_pick` tree. Press STOP so ACT is idle before you switch.

### B2. Switch

```bash
bash ~/dev-ws/scripts/grabit_mode.sh stop
bash ~/dev-ws/scripts/grabit_mode.sh sim
```

Start Isaac with `run_sim_nav.bat --env room` and press Play. The badge says SIM. Nano may stay up. The real player and the real grasp node are stopped, and the sim GUI does not publish arm commands onto the real bus. The hub owns sim PLAY.

### B3. Replay or fine-tune

Convert with `il_to_lerobot.py` as in A3. To fine-tune the existing shoulder policy, point the train resume at the shoulder `pretrained_model` and use a **new** `--output_dir`. That leaves the old `last` checkpoint in place until you copy it on purpose. Retest in sim with hub PLAY. When the sim grasp is acceptable, export `GRAB_IT_IL_CKPT` to that new `last/pretrained_model` and return with Workflow A from A4.

## Day-to-day

- Real grasp session: `bash ~/dev-ws/scripts/grabit_mode.sh real`
- Sim collect or retest: `bash ~/dev-ws/scripts/grabit_mode.sh sim`, then the Windows bat
- Done, arm stays put: `bash ~/dev-ws/scripts/grabit_mode.sh stop`
- Which world is live: `bash ~/dev-ws/scripts/grabit_mode.sh status` and the GUI badge
