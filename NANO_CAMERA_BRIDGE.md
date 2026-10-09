# Nano camera path: keeping ~30 fps

How grab_IT moves D435i + wide USB video from the Jetson Nano to the AGX without the stream collapsing into multi-second lag. The goal is that **the Nano keeps publishing at capture rate** even when the browser, Nav2, or Create 3 Wi-Fi is slow.

The wide USB cam is launched at **640×480, 30 Hz**. The D435i in [`nano/dual_camera.launch.py`](nano/dual_camera.launch.py) is currently **848×480×15** (color + depth, aligned, hardware-synced). Mapping notes still describe a 30 Hz D435i profile; either rate only holds if the **crossover JPEG path** below is intact. The GUI display itself is *not* 30 fps (HTTP snapshots / MJPEG capped at 10 Hz) — that is on purpose so the display cannot backpressure the Nano.

---

## Why Wi-Fi alone cannot hold 30 Hz

Cameras, the PX150, RTAB/Nav2, and Create 3 all share ROS 2 domain **0** (Create 3 itself is domain **5**, bridged). If the Fast DDS participant that *reads* images also binds the laptop Wi-Fi NIC:

- JPEG + raw + aligned-depth fight Create 3 `/odom` / `/cmd_vel` on a slow radio.
- **RELIABLE** image subscriptions queue when a consumer is slow. The Nano then waits, and the GUI looks seconds late.
- A **second** Fast DDS reader on `/wide_cam/image_raw/compressed` (e.g. grasp + GUI both subscribed) has frozen the wide pane.
- Stacked `image_transport republish` processes send **N copies** of the same JPEG across the wire.

The fix is not “try harder on Wi-Fi”. It is **keep camera DDS on a dedicated gigabit hop**, **compress once**, and **never let a slow UI become a RELIABLE reader**.

---

## Physical split

| Link | Addresses | What rides it |
|------|-----------|----------------|
| **Gigabit crossover** | Nano `10.42.0.2` ↔ AGX `10.42.0.1` | Color/depth, compressed JPEG, arm `joint_states` / commands |
| **Wi-Fi LAN** | AGX `192.168.50.207`, Create 3 `192.168.50.251` | Create 3 domain 5, `domain_bridge`, laptop GUI HTTP |

Fast DDS **interface whitelists** pin participants so discovery and user traffic cannot wander onto the wrong NIC:

- Nano bringup: `FASTRTPS_DEFAULT_PROFILES_FILE=/root/crossover_only.xml` (crossover only).
- AGX camera/GUI/session: [`rtabmap_profile.xml`](rtabmap_profile.xml) whitelist **`10.42.0.1` only**.
- Create 3 / dual-homed tools: [`bridge_both.xml`](bridge_both.xml) whitelist **`10.42.0.1` + `192.168.50.207`**.

`useBuiltinTransports` is **false**, so there is no accidental “all interfaces” participant.

---

## Relays: GUI stays on crossover, Create 3 stays on Wi-Fi

`gui_node` is a **joint_states reader**. If it uses `bridge_both.xml`, Fast DDS advertises the Wi-Fi locator; the Nano `xs_sdk` cannot deliver there, and servo ovals flap. So the GUI participant stays **crossover-pinned**.

Drive and (for Isaac) arm commands still have to reach Wi-Fi peers. They do **not** go by putting the GUI on both NICs. They hop **UDP on localhost**, then a **second** ROS node publishes with `bridge_both.xml`:

```
GUI (crossover)  --UDP 127.0.0.1:25150-->  cmd_vel_wifi_relay  --DDS Wi-Fi-->  domain_bridge  --> Create 3
GUI (crossover)  --UDP 127.0.0.1:25151-->  arm_cmd_wifi_relay  --DDS / TCP-->  Isaac / other Wi-Fi readers
```

Scripts: [`scripts/cmd_vel_wifi_relay.py`](scripts/cmd_vel_wifi_relay.py), [`scripts/arm_cmd_wifi_relay.py`](scripts/arm_cmd_wifi_relay.py). Started from [`grabit_session.sh`](grabit_session.sh) / [`agx_bringup.sh`](agx_bringup.sh).

That is the “Nano bridge” in operator terms: **cameras and arm state stay on the crossover; only small command packets are relayed onto Wi-Fi.**

Create 3 odom/`cmd_vel` still use `ros2 run domain_bridge` + [`create3_bridge.yaml`](create3_bridge.yaml) (domain 5 → 0). **Do not bridge Create 3 `/tf`** (it lags); use [`scripts/odom_to_tf.py`](scripts/odom_to_tf.py) instead. Dock/undock stay on domain 5 and are not image-related.

---

## Nano: capture + exactly one JPEG republish

[`nano/nano_bringup.sh`](nano/nano_bringup.sh) (inside `grab_it_nano`):

1. Pins DDS to crossover.
2. **Kills leftover** `image_transport republish` processes. Duplicates saturate the Nano CPU and flood the crossover with N copies of the same stream.
3. Launches [`nano/dual_camera.launch.py`](nano/dual_camera.launch.py):
   - Wide: `usb_cam`, `/dev/video_wide`, **30 fps**, `mjpeg2rgb`.
   - D435i: matched color+depth **848×480**, `align_depth`, `enable_sync` (depth/color currently **15 Hz** in that file).
4. Starts **one** `image_transport republish raw compressed` per camera:

   - `/wide_cam/image_raw` → `/wide_cam/image_raw/compressed`
   - `/d435i/d435i/color/image_raw` → `/d435i/d435i/color/image_raw/compressed`

JPEG over gigabit is what makes 30 Hz wide (and a healthy D435i color rate) affordable. Raw `sensor_msgs/Image` plus aligned depth on the same hop is what used to stall the GUI.

Arm: one `xs_sdk` only; Nano `robot_state_publisher` is suppressed so AGX owns TF.

---

## AGX: one ROS image consumer

[`mjpeg_relay.py`](mjpeg_relay.py) is the **only** process that should subscribe to Nano compressed color over DDS.

| Rule | Why |
|------|-----|
| Subscribe **BEST_EFFORT**, history **KEEP_LAST**, **depth 1** | If the browser is slow, drop frames. Never wait on the Nano. RELIABLE + MJPEG write produced multi-second lag. |
| Prefer **CompressedImage**; raw is fallback only if compressed is stale (>1 s) | Avoid pulling uncompressed color when JPEG is already flowing. |
| Expose **HTTP** `:8081/d435i.jpg` and `/wide.jpg` (latest snapshot) | Browser and grasp pull JPEGs without a second Fast DDS reader. Multipart MJPEG is capped at **10 Hz** and jumps to the newest seq so a slow write cannot queue the ROS side. |

[`gui_node.py`](src/grab_it_gui/grab_it_gui/gui_node.py) **does not** subscribe to camera topics (those subscriptions are commented out). The page polls the relay’s `.jpg` URLs.

[`grasp_from_click.py`](src/grab_it_grasp/grab_it_grasp/grasp_from_click.py):

- Wide overlay comes from the **relay HTTP URL**, not `/wide_cam/image_raw/compressed`.
- **Continuous aligned-depth is off** by default. Depth over the crossover delayed GUI video. Grasp takes a **one-shot** aligned-depth image on click.

Do not add another `ros2 topic hz` / RViz Image / extra compressed subscriber on the crossover while you care about live rate.

---

## Data flow

```
Nano usb_cam / realsense
        |  raw (local)
        v
image_transport republish  (exactly one per stream)
        |  CompressedImage, crossover DDS 10.42.0.0/24
        v
AGX mjpeg_relay  (BEST_EFFORT, depth=1)
        |  HTTP JPEG snapshots
        +---> gui.html camera panes
        +---> grasp_from_click wide overlay
        +---> IL recorder (if it subscribes, keep it BEST_EFFORT depth 1)
```

Commands and Create 3 stay off this path (UDP relays + `domain_bridge`).

---

## What not to do

- Dual-home the GUI or mjpeg_relay onto Wi-Fi “so everything is on one profile” if Nano joints or video then drop. Prefer crossover pin + UDP relays.
- Leave old `republish` processes running across Nano bringups.
- Subscribe GUI **and** grasp **and** RViz to the same compressed topic.
- Stream aligned-depth continuously “just in case” while operating the GUI.
- Use RELIABLE + large history on image subscriptions.

---

## Bringup reminder

1. Crossover up: Nano `10.42.0.2`, AGX `10.42.0.1`.
2. Nano container: `/root/nano_bringup.sh`.
3. AGX: `mjpeg_relay` + GUI (`grabit_session.sh` or `agx_bringup.sh`) with the profiles above.
4. Check rate **on the crossover participant**, e.g. compressed color `ros2 topic hz` with `rtabmap_profile.xml` / `crossover_only.xml` — not a Wi-Fi-only shell.

Related ops: [GRABIT_LAUNCH_GUIDE.md](GRABIT_LAUNCH_GUIDE.md) §1 and §6.
