#!/usr/bin/env python3
"""grab_it_gui - 3-thumbstick teleop for PX150 arm + Create 3 base, with D435i camera."""
import asyncio, json, os, socket, struct, subprocess, threading, time
import math
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState, CompressedImage, Image
from cv_bridge import CvBridge
import cv2
from interbotix_xs_msgs.msg import JointGroupCommand, JointSingleCommand
from interbotix_xs_msgs.srv import OperatingModes, RegisterValues, Reboot, TorqueEnable
from action_msgs.srv import CancelGoal
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import String
try:
    from nav2_msgs.action import NavigateToPose
    from nav2_msgs.srv import ClearEntireCostmap, ManageLifecycleNodes
except ImportError:
    NavigateToPose = None
    ClearEntireCostmap = None
    ManageLifecycleNodes = None
from rclpy.action import ActionClient
import websockets
import yaml

# #region agent log
_AGENT_DBG_PATH = "/tmp/debug-286aa6.ndjson"
_AGENT_DBG_INGEST = "http://127.0.0.1:7812/ingest/4a996561-7160-477a-9524-6d3d0dc3ae15"
# Also try Windows LAN if AGX can reach the Cursor host (best-effort).
_AGENT_DBG_INGEST_LAN = os.environ.get(
    "GRAB_IT_DEBUG_INGEST",
    "http://192.168.50.100:7812/ingest/4a996561-7160-477a-9524-6d3d0dc3ae15",
)

def _agent_dbg(hypothesis_id, location, message, data=None, run_id="pre"):
    try:
        import json as _json
        import urllib.request as _url
        payload = {
            "sessionId": "286aa6",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data or {},
            "timestamp": int(time.time() * 1000),
        }
        line = _json.dumps(payload, separators=(",", ":")) + "\n"
        with open(_AGENT_DBG_PATH, "a", encoding="utf-8") as f:
            f.write(line)
        body = line.encode("utf-8")
        for url in (_AGENT_DBG_INGEST, _AGENT_DBG_INGEST_LAN):
            try:
                req = _url.Request(
                    url,
                    data=body,
                    headers={
                        "Content-Type": "application/json",
                        "X-Debug-Session-Id": "286aa6",
                    },
                    method="POST",
                )
                _url.urlopen(req, timeout=0.15)
                break
            except Exception:
                continue
    except Exception:
        pass
# #endregion

# Pan lock angle for grasp mode (rad). Must match config/grasp_params.yaml.
GRASP_PAN_LOCK = float(os.environ.get("GRAB_IT_GRASP_PAN_LOCK", "0.0"))

LOCATIONS_YAML = Path(os.environ.get(
    "GRAB_IT_LOCATIONS_YAML",
    "/workspaces/isaac_ros-dev/config/named_locations.yaml",
))
CREATE3_DDS_PROFILE = os.environ.get(
    "CREATE3_DDS_PROFILE",
    "/workspaces/isaac_ros-dev/bridge_both.xml",
)
# GUI ROS stays crossover-only; drive sticks also go UDP → cmd_vel_wifi_relay
# (bridge_both) so domain_bridge can reach Create 3 without dual-iface readers.
CMD_VEL_UDP = (
    "127.0.0.1",
    int(os.environ.get("GRAB_IT_CMD_VEL_UDP_PORT", "25150")),
)
# Same hop for arm commands so Isaac on Wi‑Fi sees joint_targets even when
# gui_node is still crossover-pinned for the Nano.
ARM_CMD_UDP = (
    "127.0.0.1",
    int(os.environ.get("GRAB_IT_ARM_CMD_UDP_PORT", "25151")),
)
# Same service the Nav2 RViz panel uses for Startup / Reset
NAV_LIFECYCLE_SRV = os.environ.get(
    "GRAB_IT_NAV_LIFECYCLE_SRV",
    "/lifecycle_manager_navigation/manage_nodes",
)

JOINT_NAMES = ["waist", "shoulder", "wrist_angle", "wrist_rotate"]
JOINT_LIMITS = {
    "waist":        (-2.0246, 2.0944),  # 64 deg .. 300 deg
    "shoulder":     (-1.9722, 1.9373),  # Interbotix mechanical (~-113 .. +111 deg)
    "wrist_angle":  (-2.0769, 1.4486),  # 61 deg .. 263 deg
    "wrist_rotate": (-1.5708, 1.5708),  # 90 deg .. 270 deg
}
# Boom-removed ready / stow. Sleep must never be a single jump from a
# forward grasp — that whip arcs into the AGX / camera mount.
HOME_JOINTS = [0.0, 0.1745, -1.309, 0.0]
SLEEP_JOINTS = [0.0, -0.2321, -1.7279, 0.0]
# Max rad/s per joint for named Home/Sleep moves (shoulder is the danger).
NAMED_POSE_MAX_SPEED = {
    "waist": 0.35,
    "shoulder": 0.22,
    "wrist_angle": 0.30,
    "wrist_rotate": 0.40,
}
SPEED = {"waist": 1.2, "shoulder": 0.8, "wrist_angle": 1.0, "wrist_rotate": 1.5}
MAX_LIN = 0.30
MAX_ANG = 1.40
PAN_MIN = -2.0944   # 60 deg
PAN_MAX = 1.6406    # 274 deg
UPDATE_HZ = 20

# GUI teleop uses position mode (like Wizard) so the horn *holds*.
# PWM + coast was springing back; also the linkage is non-monotonic — past
# ~1.65 rad the fingers close again ("bounce" near 3/4 travel).
GRIPPER_JOINT_CLOSED = float(os.environ.get("GRAB_IT_GRIPPER_JOINT_CLOSED", "-0.95"))
GRIPPER_JOINT_OPEN = float(os.environ.get("GRAB_IT_GRIPPER_JOINT_OPEN", "1.65"))
GRIPPER_REBOOT_COOLDOWN_S = 15.0
# left_finger metres at the joint limits above (for slider label only)
FINGER_OPEN   = 0.0375
FINGER_CLOSED = 0.0115

SERVO_LIST = ["waist", "shoulder", "shoulder_shadow", "wrist_angle", "wrist_rotate", "gripper", "pan"]
ACTIVE_WINDOW = 1.0   # seconds a servo stays "active" after last command
# map an arm joint to the servo(s) it drives (shoulder drives its shadow too)
JOINT_TO_SERVOS = {
    "waist": ["waist"],
    "shoulder": ["shoulder", "shoulder_shadow"],
    "wrist_angle": ["wrist_angle"],
    "wrist_rotate": ["wrist_rotate"],
}
TEMP_WARN = 55   # approaching overheat (XL430 trips ~80)
TEMP_FAIL = 72   # treat as error well before shutdown
TEMP_BLOCK   = 75
TEMP_CLEAR   = 68
LOAD_BLOCK   = 500
LOAD_CLEAR   = 250
LOAD_WARN    = 350   # approaching overload (display only)
BLOCK_COUNT  = 2
# Gripper XL430 overheats well below arm thresholds if stalled in position mode.
GRIPPER_TEMP_WARN = 50
GRIPPER_TEMP_FAIL = 65
# EEPROM Current_Limit (XL430 units). 280 was too low to move the jaws;
# restore a high cap and stop overheat by holding the stall pose, not by
# starving current.
GRIPPER_CURRENT_LIMIT = int(os.environ.get("GRAB_IT_GRIPPER_CURRENT_LIMIT", "900"))
GRIPPER_OP_MODE = "position"
# Register reads share the DXL bus with xs_sdk syncRead and traverse AGX↔Nano
# DDS. They are advisory only — joint_states is the connectivity source of truth.
HEALTH_POLL_S = 0.8
HEALTH_BROADCAST_S = 0.5
REG_READ_TIMEOUT_S = 1.5        # clear stuck inflight (service hang → total health death)
SILENT_FAIL_COUNT = 40          # ~50s of misses at idle heartbeat
JOINT_FRESH_S = 6.0             # joint_states younger than this => bus/servo alive
SILENT_BLOCK_IF_JOINTS_LIVE = False
# Still poll health while moving — that is when overheat/overload show up.
SKIP_REG_POLL_WHEN_ACTIVE = False
CAMERA_TOPIC = "/d435i/d435i/color/image_raw/compressed"
WIDE_TOPIC   = "/wide_cam/image_raw"

# joint_states name used to decide if a servo row is "connected"
SERVO_JS_NAME = {
    "waist": "waist",
    "shoulder": "shoulder",
    "shoulder_shadow": "shoulder",
    "wrist_angle": "wrist_angle",
    "wrist_rotate": "wrist_rotate",
    "gripper": "gripper",
    "pan": "pan",
}

_HERE = Path(__file__).parent
_INSTALLED = Path(__file__).parents[3] / "lib" / "grab_it_gui" / "gui.html"
_SRC_HTML = _HERE / "gui.html"
# Prefer the file next to this script so source-tree edits apply without colcon.
HTML_FILE = _SRC_HTML if _SRC_HTML.exists() else _INSTALLED

_ACT_NAME_FILE = Path("/tmp/grabit_act_name")
_COLLECTION_FILE = Path("/tmp/grabit_collection")
_DEFAULT_ACT_NAME = "act_foam_pole_lookreach_office_shoulder"
_DEFAULT_COLLECTION = "tabletop_pick"


def _act_output_roots() -> list:
    roots = []
    seen = set()
    for raw in (
        os.environ.get("GRAB_IT_IL_OUTPUTS", ""),
        "/home/ubuntu/dev-ws/datasets/lerobot/outputs",
        "/workspaces/isaac_ros-dev/datasets/lerobot/outputs",
    ):
        if not raw:
            continue
        path = Path(raw)
        key = str(path)
        if key in seen or not path.is_dir():
            continue
        seen.add(key)
        roots.append(path)
    return roots


def _act_catalog() -> list:
    """Named checkpoints: outputs/<name>/checkpoints/last/pretrained_model."""
    items = []
    seen = set()
    for root in _act_output_roots():
        labels = {}
        catalog_path = root / "catalog.json"
        if catalog_path.is_file():
            try:
                data = json.loads(catalog_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = None
            if isinstance(data, dict):
                labels = {str(k): str(v) for k, v in data.items() if v}
        for folder in sorted(root.iterdir(), key=lambda p: p.name):
            if not folder.is_dir() or folder.name in seen:
                continue
            ckpt = folder / "checkpoints" / "last" / "pretrained_model"
            if not ckpt.is_dir():
                continue
            if not ((ckpt / "config.json").is_file() or (ckpt / "model.safetensors").is_file()):
                continue
            seen.add(folder.name)
            items.append({
                "name": folder.name,
                "label": labels.get(folder.name, folder.name),
                "path": str(ckpt),
            })
    return items


def _selected_act_name() -> str:
    try:
        text = _ACT_NAME_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        text = ""
    names = {item["name"] for item in _act_catalog()}
    if text in names:
        return text
    if _DEFAULT_ACT_NAME in names:
        return _DEFAULT_ACT_NAME
    return next(iter(names), "")


def _collection_name_ok(name: str) -> bool:
    if not name or len(name) > 64 or not name[0].isalnum():
        return False
    return all(c.isalnum() or c in "_-" for c in name)


def _clean_collection_name(raw: str) -> str:
    """Turn a typed name into a folder name. Spaces become underscores."""
    text = str(raw or "").strip().lower().replace(" ", "_")
    cleaned = []
    for ch in text:
        if ch.isalnum() or ch in "_-":
            cleaned.append(ch)
        elif ch in " .":
            cleaned.append("_")
    name = "".join(cleaned).strip("_")
    while "__" in name:
        name = name.replace("__", "_")
    return name if _collection_name_ok(name) else ""


def _convert_episode_progress(message: str) -> str:
    """The converter stays on one status line while it encodes each take."""
    log = Path("/tmp/il_train_collection.log")
    if not log.is_file():
        return message
    try:
        tail = log.read_bytes()[-80000:].decode("utf-8", "replace")
    except OSError:
        return message
    last = ""
    for line in tail.splitlines():
        if line.startswith("episode_"):
            last = line.split(":", 1)[0]
    if not last:
        return message
    return f"{message} ({last})"


def _collection_name() -> str:
    try:
        text = _COLLECTION_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        text = ""
    return text if _collection_name_ok(text) else _DEFAULT_COLLECTION


def _collection_dir(name: str) -> Path:
    return Path("/home/ubuntu/dev-ws/datasets") / name


def _collection_count(name: str) -> int:
    root = _collection_dir(name)
    if not root.is_dir():
        return 0
    return sum(
        1
        for p in root.iterdir()
        if p.is_dir() and p.name.startswith("episode_") and (p / "actions.npy").is_file()
    )

# Shared latest camera frame (JPEG bytes)
_frame_lock = threading.Lock()
_latest_frame = [None]
_wide_lock = threading.Lock()
_latest_wide = [None]
# Isaac Sim (Jazzy) polls this over HTTP :8765 — same port as the page.
_isaac_cmd = {
    "t": "arm",
    "n": list(JOINT_NAMES),
    "p": [float(v) for v in HOME_JOINTS],
    "lx": 0.0,
    "az": 0.0,
    "seq": 0,
}
_isaac_cmd_lock = threading.Lock()


def _isaac_note(**kwargs) -> None:
    with _isaac_cmd_lock:
        _isaac_cmd.update(kwargs)
        _isaac_cmd["seq"] = int(_isaac_cmd.get("seq", 0)) + 1


class _GUIHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/grabit/mode":
            try:
                text = Path("/tmp/grabit_mode").read_text(encoding="utf-8").strip().lower()
            except OSError:
                text = "real"
            if text not in ("sim", "real"):
                text = "real"
            body = json.dumps({"world": text}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/isaac/cmd":
            with _isaac_cmd_lock:
                body = json.dumps(_isaac_cmd).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if path in ("/", "/index.html", ""):
            content = HTML_FILE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(content)
        elif path in ("/camera.mjpg", "/wide.mjpg"):
            buf = _latest_frame if path == "/camera.mjpg" else _latest_wide
            lock = _frame_lock if path == "/camera.mjpg" else _wide_lock
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with lock:
                        frame = buf[0]
                    if frame is not None:
                        self.wfile.write(b"--frame\r\n")
                        self.send_header_line("Content-Type", "image/jpeg")
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                    time.sleep(1.0 / 15.0)  # ~15 fps
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(404); self.end_headers()

    def send_header_line(self, k, v):
        self.wfile.write(f"{k}: {v}\r\n".encode())

    def log_message(self, *_): pass


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def _run_http(port):
    _ThreadingHTTPServer(("0.0.0.0", port), _GUIHandler).serve_forever()


class GrabItGUINode(Node):
    def __init__(self):
        super().__init__("grab_it_gui")
        self._pos = {j: float(v) for j, v in zip(JOINT_NAMES, HOME_JOINTS)}
        self._left_finger = None
        # Seed Home so sticks publish before the first joint_states (Isaac).
        self._target = {j: float(v) for j, v in zip(JOINT_NAMES, HOME_JOINTS)}
        # While grasp_from_click is moving the arm, GUI teleop must not publish
        # a stale Sleep/Home _target (that snaps the shoulder back after a miss).
        self._grasp_arm_busy = False
        self._named_pose_busy = False
        self._named_pose_gen = 0
        # stick state: arm sticks + drive stick
        self._lx = self._ly = 0.0   # arm: waist, shoulder
        self._rx = self._ry = 0.0   # arm: wrist_rotate, wrist_angle
        self._dx = self._dy = 0.0   # drive: angular, linear
        self._lock = threading.Lock()
        self._dt = 1.0 / UPDATE_HZ
        self._ws_clients = set()
        self._ws_lock = threading.Lock()
        self._ws_loop = None

        self._arm_pub = self.create_publisher(
            JointGroupCommand, "/px150_no_elbow/commands/joint_group", 10)
        self._gripper_pub = self.create_publisher(
            JointSingleCommand, "/px150_no_elbow/commands/joint_single", 10)
        self._js_cmd_pub = self.create_publisher(
            JointState, "/px150_no_elbow/commands/joint_targets", 10)
        self._cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self._cmd_vel_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._arm_cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if NavigateToPose is None:
            self.get_logger().warn("nav2_msgs missing — arm GUI up, navigation disabled")
            self._nav_client = None
            self._nav_cancel_cli = None
            self._clear_local_cli = None
            self._clear_global_cli = None
            self._nav_lifecycle_cli = None
        else:
            self._nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
            self._nav_cancel_cli = self.create_client(
                CancelGoal, "navigate_to_pose/_action/cancel_goal")
            self._clear_local_cli = self.create_client(
                ClearEntireCostmap, "/local_costmap/clear_entirely_local_costmap")
            self._clear_global_cli = self.create_client(
                ClearEntireCostmap, "/global_costmap/clear_entirely_global_costmap")
            self._nav_lifecycle_cli = self.create_client(
                ManageLifecycleNodes, NAV_LIFECYCLE_SRV)
        self._locations = self._load_locations()
        self._nav_busy = False
        self._nav_goal_handle = None
        self._nav_restarting = False
        self._drive_was_active = False
        self._pan_pos = None
        self._grasp_pan_locked = False

        self._grasp_click_pub = self.create_publisher(String, "/grab_it/grasp_click", 10)
        self._grasp_cmd_pub = self.create_publisher(String, "/grab_it/grasp_command", 10)
        self.create_subscription(
            String, "/grab_it/grasp_status", self._grasp_status_cb, 10)
        self._il_cmd_pub = self.create_publisher(String, "/grab_it/il_command", 10)
        self._policy_live = False
        self._quest_clutched = False
        self._teach_free = False
        self.create_subscription(String, "/grab_it/il_status", self._il_status_cb, 10)
        self.create_subscription(String, "/grab_it/quest/status", self._quest_status_cb, 10)
        self._cal_cmd_pub = self.create_publisher(String, "/grab_it/cal_command", 10)
        self.create_subscription(String, "/grab_it/cal_status", self._cal_status_cb, 10)
        self._sim = os.environ.get("GRAB_IT_SIM", "0").lower() in ("1", "true", "yes")
        if self._sim:
            self.get_logger().info(
                "GRAB_IT_SIM=1: skip Dynamixel mode/torque; publish joint_targets for pan"
            )

        self.create_subscription(
            JointState,
            "/px150_no_elbow/joint_states",
            self._js_cb,
            qos_profile_sensor_data,
        )
        # DISABLED: image consumed by mjpeg_relay (crossover-pinned) instead
        # self.create_subscription(
        #     CompressedImage, CAMERA_TOPIC, self._img_cb, 10)
        self._bridge = CvBridge()
        # DISABLED: wide image consumed by mjpeg_relay instead (was RAW over WiFi!)
        # self.create_subscription(
        #     Image, WIDE_TOPIC, self._wide_cb, 10)

        self._op_cli = self.create_client(OperatingModes, "/px150_no_elbow/set_operating_modes")
        self._reg_cli = self.create_client(RegisterValues, "/px150_no_elbow/get_motor_registers")
        self._reg_set_cli = self.create_client(RegisterValues, "/px150_no_elbow/set_motor_registers")
        self._reboot_cli = self.create_client(Reboot, "/px150_no_elbow/reboot_motors")
        self._tq_cli = self.create_client(TorqueEnable, "/px150_no_elbow/torque_enable")
        self._health = {s: {"temp": None, "err": None, "load": None, "read_fail": 0, "blocked": False, "trip": 0} for s in SERVO_LIST}
        self._poll_idx = 0
        self._active = {}       # servo -> last actuation timestamp
        self._hb_idx = 0        # heartbeat round-robin when idle
        self._js_stamp = 0.0    # monotonic time of last joint_states msg
        self._servo_js_stamp = {s: 0.0 for s in SERVO_LIST}
        self._reg_inflight = False
        self._reg_inflight_t = 0.0
        self.create_timer(HEALTH_POLL_S, self._poll_health)
        self.create_timer(HEALTH_BROADCAST_S, self._broadcast_health)
        self.create_timer(self._dt, self._control_loop)
        self.create_timer(0.5, self._reassert_teach_free)
        self._modes_ok = False
        self._gripper_mode = None  # GRIPPER_OP_MODE once confirmed
        self._gripper_joint = None
        self._gripper_cmd = None  # last commanded gripper joint (rad)
        self._gripper_last_reboot_t = 0.0
        self._gripper_active_effort = 0.0  # nonzero while a move is in flight
        self._pan_mode = None
        self._pan_mode_t = 0.0
        self._pan_cmd = None  # last commanded pan (rad); republished while active
        self._pan_init_timer = self.create_timer(4.0, self._init_pan_mode)
        # #region agent log
        self._dbg_js_gap_logged = False
        self.create_timer(2.0, self._dbg_comm_heartbeat)
        self.create_timer(1.0, self._broadcast_train_status)
        self.create_timer(5.0, self._broadcast_il_lists)
        _agent_dbg("H0", "gui_node.py:__init__", "gui_node started", {"pid": os.getpid()})
        # #endregion
        self.get_logger().info("grab_it_gui node started")

    def _publish_cmd_vel(self, twist: Twist):
        # Crossover DDS (Nav2 / anything on 10.42.0.1) plus UDP hop to the
        # Wi‑Fi relay so Create 3 still sees /cmd_vel on domain_bridge.
        self._cmd_vel_pub.publish(twist)
        _isaac_note(lx=float(twist.linear.x), az=float(twist.angular.z))
        try:
            self._cmd_vel_sock.sendto(
                struct.pack(
                    "!6d",
                    float(twist.linear.x),
                    float(twist.linear.y),
                    float(twist.linear.z),
                    float(twist.angular.x),
                    float(twist.angular.y),
                    float(twist.angular.z),
                ),
                CMD_VEL_UDP,
            )
        except Exception:
            pass
        try:
            self._arm_cmd_sock.sendto(
                json.dumps(
                    {
                        "t": "vel",
                        "lx": float(twist.linear.x),
                        "az": float(twist.angular.z),
                    }
                ).encode(),
                ARM_CMD_UDP,
            )
        except Exception:
            pass

    # #region agent log
    def _dbg_comm_heartbeat(self):
        now = time.monotonic()
        js_age = None if self._js_stamp <= 0 else (now - self._js_stamp)
        with self._ws_lock:
            n_ws = len(self._ws_clients)
        try:
            sub_n = int(self._gripper_pub.get_subscription_count())
        except Exception:
            sub_n = -1
        pan_age = now - self._servo_js_stamp.get("pan", 0.0)
        grip_age = now - self._servo_js_stamp.get("gripper", 0.0)
        _agent_dbg(
            "H1-H3",
            "gui_node.py:_dbg_comm_heartbeat",
            "comm heartbeat",
            {
                "js_age_s": None if js_age is None else round(js_age, 3),
                "js_live": self._joints_live(),
                "ws_clients": n_ws,
                "reg_inflight": self._reg_inflight,
                "reg_inflight_age_s": round(now - self._reg_inflight_t, 3) if self._reg_inflight else 0.0,
                "pan_locked": self._grasp_pan_locked,
                "pan_js_age_s": round(pan_age, 3),
                "gripper_js_age_s": round(grip_age, 3),
                "joint_single_subs": sub_n,
                "op_ready": self._op_cli.service_is_ready(),
                "reg_ready": self._reg_cli.service_is_ready(),
                "modes_ok": self._modes_ok,
            },
        )
        if js_age is not None and js_age > 2.0:
            _agent_dbg(
                "H3",
                "gui_node.py:_dbg_comm_heartbeat",
                "joint_states silent",
                {"js_age_s": round(js_age, 3), "ws_clients": n_ws},
            )
    # #endregion

    def _init_pan_mode(self):
        self._pan_init_timer.cancel()
        if self._sim:
            self.get_logger().info("GRAB_IT_SIM: skip Dynamixel pan/gripper operating modes")
            return
        if self._modes_ok:
            return
        if not self._op_cli.service_is_ready():
            self.get_logger().warn("set_operating_modes not ready; retrying in 2s")
            self._pan_init_timer = self.create_timer(2.0, self._init_pan_mode)
            return
        self._ensure_pan_mode(force=True)
        # After a gripper reboot, restore position + current cap + torque.
        self._ensure_gripper_position_mode(force=True)
        self._torque_arm_group()
        self._modes_ok = True

    def _torque_arm_group(self):
        if getattr(self, "_teach_free", False):
            return
        self._set_arm_torque(True)

    def _arm_servos(self):
        names = []
        for joint in JOINT_NAMES:
            for s in JOINT_TO_SERVOS.get(joint, [joint]):
                if s not in names:
                    names.append(s)
        return names

    def _set_arm_torque(self, enable: bool) -> bool:
        ok = False
        if self._tq_cli.service_is_ready():
            req = TorqueEnable.Request()
            req.cmd_type = "group"
            req.name = "arm"
            req.enable = bool(enable)
            self._tq_cli.call_async(req)
            ok = True
            for name in self._arm_servos():
                one = TorqueEnable.Request()
                one.cmd_type = "single"
                one.name = name
                one.enable = bool(enable)
                self._tq_cli.call_async(one)
        else:
            self.get_logger().warn("torque_enable not ready")
        # Mirror Torque_Enable in EEPROM/RAM via register write. Teach-free
        # was writing 0 here; Hold must write 1 or the bus stays limp after
        # Free (service-only enable often loses to the last register poke).
        if self._reg_set_cli.service_is_ready():
            for name in self._arm_servos():
                rr = RegisterValues.Request()
                rr.cmd_type = "single"
                rr.name = name
                rr.reg = "Torque_Enable"
                rr.value = 1 if enable else 0
                self._reg_set_cli.call_async(rr)
        self.get_logger().info(
            f"arm torque {'ON' if enable else 'OFF'} servos={self._arm_servos()} svc={ok}"
        )
        return ok

    def _reassert_teach_free(self) -> None:
        if getattr(self, "_teach_free", False):
            self._set_arm_torque(False)

    def _publish_js_targets(self, names, positions) -> None:
        js = JointState()
        js.name = list(names)
        js.position = [float(p) for p in positions]
        if self._real_bus():
            self._js_cmd_pub.publish(js)
        _isaac_note(t="arm", n=list(js.name), p=list(js.position))
        try:
            self._arm_cmd_sock.sendto(
                json.dumps(
                    {"t": "arm", "n": list(js.name), "p": list(js.position)}
                ).encode(),
                ARM_CMD_UDP,
            )
        except Exception:
            pass

    def _hold_arm_present(self) -> None:
        """After torque ON, latch Goal_Position to the live joints so the arm stiffens."""
        with self._lock:
            if not all(j in self._pos for j in JOINT_NAMES):
                return
            q = [float(self._pos[j]) for j in JOINT_NAMES]
            self._target = {j: float(v) for j, v in zip(JOINT_NAMES, q)}
        msg = JointGroupCommand()
        msg.name = "arm"
        msg.cmd = q
        for _ in range(3):
            self._publish_arm_group(msg)
            time.sleep(0.03)

    def _arm_gui_active(self) -> bool:
        return (
            (not self._policy_live)
            and (not self._quest_clutched)
            and (not self._teach_free)
            and (not self._grasp_arm_busy)
            and (not self._named_pose_busy)
        )

    def _sync_arm_target_from_pos(self) -> None:
        """Point teleop at the live pose so Sleep/Home aren't replayed after grasp."""
        with self._lock:
            if all(j in self._pos for j in JOINT_NAMES):
                self._target = {j: float(self._pos[j]) for j in JOINT_NAMES}
                self._lx = self._ly = self._rx = self._ry = 0.0

    def _broadcast_teach(self) -> None:
        self._broadcast(json.dumps({"type": "teach_free", "free": bool(self._teach_free)}))

    def set_arm_teach(self, free: bool) -> None:
        """Untorque the arm group for kinesthetic teaching, or hold the current pose."""
        free = bool(free)
        if free:
            # Flag first so the 0.5s reassert timer keeps torque OFF.
            self._teach_free = True
            with self._lock:
                self._lx = self._ly = self._rx = self._ry = 0.0
            if not self._set_arm_torque(False):
                self._teach_free = False
            self._broadcast_teach()
            return
        # Clear teach-free BEFORE enabling torque. Otherwise the reassert
        # timer (still seeing free=True) immediately calls torque OFF again.
        self._teach_free = False
        with self._lock:
            if all(j in self._pos for j in JOINT_NAMES):
                self._target = {j: float(self._pos[j]) for j in JOINT_NAMES}
        ok = self._set_arm_torque(True)
        if ok:
            self._hold_arm_present()
        else:
            self.get_logger().warn("Hold arm: torque_enable failed — arm may stay limp")
        self._broadcast_teach()
        if ok:
            self.get_logger().info("Hold arm: torque ON + present pose latched")

    def _torque_on(self, servo: str):
        if getattr(self, "_teach_free", False) and servo in self._arm_servos():
            return
        if not self._tq_cli.service_is_ready():
            return
        req = TorqueEnable.Request()
        req.cmd_type = "single"
        req.name = servo
        req.enable = True
        self._tq_cli.call_async(req)

    def _ensure_pan_mode(self, force: bool = False):
        """Pan is not in modes.yaml — xs_sdk often leaves torque OFF until we set it."""
        if self._pan_mode == "position" and not force:
            return
        if not self._op_cli.service_is_ready():
            return
        rp = OperatingModes.Request()
        rp.cmd_type = "single"
        rp.name = "pan"
        rp.mode = "position"
        rp.profile_type = "velocity"
        rp.profile_velocity = 60
        rp.profile_acceleration = 25
        self._op_cli.call_async(rp)
        self._torque_on("pan")
        self._pan_mode = "position"
        self._pan_mode_t = time.time()
        self.get_logger().info("pan operating mode -> position + torque")

    def _ensure_gripper_position_mode(self, force: bool = False):
        """Position mode with a restored Current_Limit so the jaws can move.

        Overheat is avoided by holding the stall pose after a pinch, not by
        dropping Current_Limit so low the motor cannot close.
        """
        if self._gripper_mode == GRIPPER_OP_MODE and not force:
            return
        if not self._op_cli.service_is_ready():
            return
        rg = OperatingModes.Request()
        rg.cmd_type = "single"
        rg.name = "gripper"
        rg.mode = GRIPPER_OP_MODE
        rg.profile_type = "velocity"
        rg.profile_velocity = 40
        rg.profile_acceleration = 15
        self._op_cli.call_async(rg)
        if self._reg_set_cli.service_is_ready():
            rr = RegisterValues.Request()
            rr.cmd_type = "single"
            rr.name = "gripper"
            rr.reg = "Current_Limit"
            rr.value = int(max(80, min(1193, GRIPPER_CURRENT_LIMIT)))
            self._reg_set_cli.call_async(rr)
        self._torque_on("gripper")
        self._gripper_mode = GRIPPER_OP_MODE
        self.get_logger().info(
            f"gripper operating mode -> {GRIPPER_OP_MODE} "
            f"Current_Limit={GRIPPER_CURRENT_LIMIT}"
        )

    def _js_cb(self, msg):
        # xs_sdk publishes ~-pi when Dynamixel reads fail (power off / bus error).
        def _valid(p):
            return abs(p + math.pi) > 0.05

        got_live = False
        now_m = time.monotonic()
        with self._lock:
            for name, pos in zip(msg.name, msg.position):
                # Topic arriving = bus is up. Keep last-good angles if this
                # sample is the -pi fail sentinel (mode change / PWM burst).
                for servo, js_name in SERVO_JS_NAME.items():
                    if js_name == name:
                        self._servo_js_stamp[servo] = now_m
                if name in self._pos and _valid(pos):
                    self._pos[name] = pos
                    got_live = True
                if name == "left_finger" and _valid(pos):
                    self._left_finger = pos
                if name == "gripper" and _valid(pos):
                    self._gripper_joint = pos
                    if (
                        self._gripper_cmd is not None
                        and abs(pos - self._gripper_cmd) < 0.05
                    ):
                        self._gripper_active_effort = 0.0
                if name == "pan" and _valid(pos):
                    self._pan_pos = pos
            if self._target is None and all(_valid(self._pos[j]) for j in JOINT_NAMES):
                self._target = dict(self._pos)
        # Connectivity follows the topic, not valid encoder values. Grasp
        # set_operating_modes briefly yields -pi; the GUI must stay green.
        self._js_stamp = now_m
        if got_live:
            for servo in SERVO_LIST:
                if self._servo_joint_live(servo):
                    h = self._health[servo]
                    if h.get("blocked") and h.get("err") in (None, 0):
                        h["blocked"] = False
                        h["trip"] = 0
                        h["read_fail"] = 0

    def _grasp_status_cb(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception:
            data = {"status": "unknown", "message": msg.data}
        st = str(data.get("status", "")).lower()
        if st in ("planning", "grasping", "busy"):
            self._grasp_arm_busy = True
        elif st in ("done", "error", "cancelled", "unreachable", "idle", "point"):
            was_busy = self._grasp_arm_busy
            self._grasp_arm_busy = False
            # Always resync after a grasp attempt — stale Sleep _target was
            # yanking the shoulder back the moment teleop published again.
            if was_busy or st in ("done", "error", "cancelled", "unreachable"):
                self._sync_arm_target_from_pos()
                self.get_logger().info(
                    f"arm teleop target synced after grasp status={st}"
                )
        if st in ("done", "error", "cancelled", "unreachable", "idle"):
            if self._grasp_pan_locked:
                self.unlock_pan()
        self._broadcast(json.dumps({"type": "grasp_status", **data}))

    def _il_status_cb(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception:
            data = {"message": msg.data}
        if "playing" in data:
            self._policy_live = bool(data.get("playing"))
        self._broadcast(json.dumps({"type": "il_status", **data}))

    def _quest_status_cb(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        clutched = bool(data.get("clutched"))
        if clutched != self._quest_clutched:
            self._quest_clutched = clutched
            if clutched:
                self._sync_arm_target_from_pos()
            self.get_logger().info(
                f"Quest teleop clutched={clutched} — GUI arm sticks "
                f"{'ignored' if clutched else 'active'}"
            )

    def _world_mode(self) -> str:
        """real or sim, from /tmp/grabit_mode (grabit_mode.sh)."""
        try:
            text = Path("/tmp/grabit_mode").read_text(encoding="utf-8").strip().lower()
        except OSError:
            text = ""
        if text in ("sim", "real"):
            return text
        return "sim" if self._sim else "real"

    def _real_bus(self) -> bool:
        """False in sim so sticks and PLAY do not command the physical arm."""
        return self._world_mode() != "sim"

    def _publish_arm_group(self, msg) -> None:
        if not self._real_bus():
            return
        publisher = self._arm_pub
        publisher.publish(msg)

    def publish_il_command(self, cmd, **extra):
        payload = {"cmd": (cmd or "").strip().lower()}
        payload.update(extra)
        self.get_logger().info(f"IL cmd {payload}")
        self._il_cmd_pub.publish(String(data=json.dumps(payload)))
        if payload["cmd"] in ("stop", "stop_play"):
            self._policy_live = False
        elif payload["cmd"] in ("play", "start_play"):
            self._policy_live = True

    def _kick_real_act(self) -> None:
        """Make sure the real-arm ACT player is up, then tell it to play."""
        import subprocess

        script = Path("/home/ubuntu/dev-ws/scripts/_il_real_ensure.sh")
        if script.is_file():
            try:
                subprocess.run(["bash", str(script)], check=False, timeout=180)
            except Exception as exc:
                self.get_logger().warn(f"real ACT ensure failed: {exc}")
        self.publish_il_command("play")

    def _broadcast_act_catalog(self) -> None:
        self._broadcast(json.dumps({
            "type": "act_catalog",
            "acts": _act_catalog(),
            "selected": _selected_act_name(),
        }))

    def _select_act(self, name: str) -> None:
        """Point the idle player at a named checkpoint. Does not move the arm."""
        import signal

        name = str(name or "").strip()
        match = next((item for item in _act_catalog() if item["name"] == name), None)
        if match is None:
            self.get_logger().warn(f"ACT name not in catalog: {name}")
            self._broadcast_act_catalog()
            return
        try:
            _ACT_NAME_FILE.write_text(name + "\n", encoding="utf-8")
        except OSError as exc:
            self.get_logger().warn(f"could not store ACT name: {exc}")
        script = Path("/home/ubuntu/dev-ws/scripts/_il_real_play_host.sh")
        if not script.is_file():
            script = Path("/workspaces/isaac_ros-dev/scripts/_il_real_play_host.sh")
        proc = subprocess.run(
            ["pgrep", "-f", "grab_it_il/play_policy.py"],
            check=False,
            capture_output=True,
            text=True,
        )
        pids = [int(p) for p in (proc.stdout or "").split() if p.isdigit()]
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
        if pids:
            time.sleep(0.6)
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
        if script.is_file():
            env = os.environ.copy()
            env["GRAB_IT_IL_CKPT"] = match["path"]
            subprocess.Popen(
                ["bash", "-lc", f'exec bash "{script}" >>/tmp/il_real_play_host.log 2>&1'],
                env=env,
                start_new_session=True,
            )
            self.get_logger().info(f"ACT selected {name}")
        else:
            self.get_logger().warn(f"ACT player script missing: {script}")
        self._broadcast_act_catalog()

    def _kick_recorder(self, source: str) -> None:
        """Start the demo recorder if it is down, then begin an episode."""
        import subprocess

        script = Path("/home/ubuntu/dev-ws/scripts/_il_real_record_host.sh")
        alt = Path("/workspaces/isaac_ros-dev/scripts/_il_real_record_host.sh")
        if not script.is_file():
            script = alt
        running = subprocess.run(
            ["pgrep", "-f", "grab_it_il/record_demos.py"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if running.returncode != 0 and script.is_file():
            self.get_logger().info("starting IL recorder")
            subprocess.Popen(
                ["bash", str(script)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            time.sleep(2.0)
        elif running.returncode != 0:
            self.get_logger().warn(f"IL recorder script missing: {script}")
        self.publish_il_command("start", source=source or "teleop")

    def _start_auto_collect(self, count) -> None:
        """Record N scripted grasps around the current click. Does not change the ACT."""
        try:
            count = int(count)
        except (TypeError, ValueError):
            count = 5
        count = max(1, min(15, count))
        script = Path("/home/ubuntu/dev-ws/scripts/_il_auto_collect.sh")
        if not script.is_file():
            script = Path("/workspaces/isaac_ros-dev/scripts/_il_auto_collect.sh")
        if not script.is_file():
            self.get_logger().warn("auto collect script missing")
            return
        running = subprocess.run(
            ["pgrep", "-f", "scripts/_il_auto_collect.py"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if running.returncode == 0:
            self.get_logger().info("auto collect already running")
            return
        Path("/tmp/grabit_auto_collect.stop").unlink(missing_ok=True)
        subprocess.Popen(
            ["bash", "-lc", f'exec bash "{script}" {count} >>/tmp/il_auto_collect.log 2>&1'],
            start_new_session=True,
        )
        self.get_logger().info(f"auto collect × {count}")

    def _stop_auto_collect(self) -> None:
        import signal

        Path("/tmp/grabit_auto_collect.stop").write_text("1\n", encoding="utf-8")
        if Path("/tmp/grabit_auto_collect.pid").is_file():
            self.publish_grasp_command("cancel")
        proc = subprocess.run(
            ["pgrep", "-f", "scripts/_il_auto_collect.py"],
            check=False,
            capture_output=True,
            text=True,
        )
        for pid in [int(p) for p in (proc.stdout or "").split() if p.isdigit()]:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass

    def _train_status_payload(self):
        path = Path("/tmp/il_train_status.json")
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        message = str(data.get("message") or "").strip()
        if not message:
            return None
        if "converting" in message:
            message = _convert_episode_progress(message)
        return {
            "type": "train_status",
            "running": bool(data.get("running")),
            "message": message,
        }

    def _broadcast_train_status(self) -> None:
        payload = self._train_status_payload()
        if payload is None:
            return
        key = (payload["message"], payload["running"])
        if key == getattr(self, "_train_status_key", None):
            return
        self._train_status_key = key
        self._broadcast(json.dumps(payload))
        if not payload["running"] and "done" in payload["message"]:
            self._broadcast_act_catalog()

    def _start_train(self, steps=20000) -> None:
        """Train the current collection into its own ACT folder."""
        try:
            steps = int(steps)
        except (TypeError, ValueError):
            steps = 20000
        steps = max(1000, min(100000, steps))
        name = _collection_name()
        script = Path("/home/ubuntu/dev-ws/scripts/_il_train_collection.py")
        if not script.is_file():
            script = Path("/workspaces/isaac_ros-dev/scripts/_il_train_collection.py")
        if not script.is_file():
            self.get_logger().warn("train script missing")
            return
        running = subprocess.run(
            ["pgrep", "-f", "scripts/_il_train_collection.py"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if running.returncode == 0:
            self._broadcast(json.dumps({
                "type": "train_status",
                "running": True,
                "message": f"training {name}: already running",
            }))
            return
        self._broadcast(json.dumps({
            "type": "train_status",
            "running": True,
            "message": f"training {name}: starting {steps} steps",
        }))
        subprocess.Popen(
            ["bash", "-lc", f'exec python3 "{script}" {name} {steps} >>/tmp/il_train_collection.log 2>&1'],
            start_new_session=True,
        )
        self.get_logger().info(f"training collection {name} steps={steps}")

    def _broadcast_il_lists(self) -> None:
        self._broadcast_act_catalog()
        self._broadcast_collection()

    def _broadcast_collection(self) -> None:
        name = _collection_name()
        self._broadcast(json.dumps({
            "type": "collection",
            "name": name,
            "episodes": _collection_count(name),
        }))

    def _new_collection(self, name: str) -> None:
        """Start a new episode folder. Does not change the loaded ACT."""
        import signal

        name = _clean_collection_name(name)
        if not name:
            self.get_logger().warn(f"rejected collection name {name!r}")
            self._broadcast(json.dumps({
                "type": "il_status",
                "recording": False,
                "message": "collection name: letters, numbers, _ or -",
            }))
            return
        folder = _collection_dir(name)
        try:
            folder.mkdir(parents=True, exist_ok=True)
            _COLLECTION_FILE.write_text(name + "\n", encoding="utf-8")
        except OSError as exc:
            self.get_logger().warn(f"collection folder failed: {exc}")
            return
        script = Path("/home/ubuntu/dev-ws/scripts/_il_real_record_host.sh")
        if not script.is_file():
            script = Path("/workspaces/isaac_ros-dev/scripts/_il_real_record_host.sh")
        proc = subprocess.run(
            ["pgrep", "-f", "grab_it_il/record_demos.py"],
            check=False,
            capture_output=True,
            text=True,
        )
        pids = [int(p) for p in (proc.stdout or "").split() if p.isdigit()]
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
        for _ in range(15):
            still = subprocess.run(
                ["pgrep", "-f", "grab_it_il/record_demos.py"],
                check=False,
                capture_output=True,
                text=True,
            )
            if not (still.stdout or "").strip():
                break
            time.sleep(0.2)
        if script.is_file():
            env = os.environ.copy()
            env["GRAB_IT_IL_DATASET"] = str(folder)
            subprocess.Popen(
                ["bash", "-lc", f'exec bash "{script}" >>/tmp/il_record_host.log 2>&1'],
                env=env,
                start_new_session=True,
            )
            self.get_logger().info(f"collection {name} → {folder}")
        self._broadcast_collection()
        self._broadcast(json.dumps({
            "type": "il_status",
            "recording": False,
            "message": f"collection {name} — REC to start",
        }))

    def _cal_status_cb(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception:
            data = {"status": "unknown", "message": msg.data}
        self._broadcast(json.dumps({"type": "cal_status", **data}))

    def publish_cal_command(self, cmd: str, **extra):
        payload = {"cmd": (cmd or "").strip().lower()}
        payload.update(extra)
        self.get_logger().info(f"cal_command {payload}")
        self._cal_cmd_pub.publish(String(data=json.dumps(payload)))

    def publish_grasp_click(self, u, v, width=None, height=None, cam="d435i", click_id=None):
        """Forward a normalized click (0..1) from D435i or wide to grasp_from_click."""
        if not self._real_bus():
            return
        payload = {"u": float(u), "v": float(v), "cam": str(cam or "d435i")}
        if width is not None:
            payload["width"] = int(width)
        if height is not None:
            payload["height"] = int(height)
        if click_id is not None:
            payload["click_id"] = int(click_id)
        self._grasp_click_pub.publish(String(data=json.dumps(payload)))

    def publish_grasp_json(self, payload):
        self._grasp_cmd_pub.publish(String(data=json.dumps(payload)))

    def publish_grasp_command(self, cmd):
        if not self._real_bus():
            return
        cmd = (cmd or "").strip().lower()
        # Lock only for real grasp execution — not on every D435i click.
        if cmd in ("execute", "grasp"):
            self._grasp_arm_busy = True
            self._sync_arm_target_from_pos()
            self.lock_pan_for_grasp()
        elif cmd in ("cancel", "unlock_pan", "unlock"):
            self._grasp_arm_busy = False
            self._sync_arm_target_from_pos()
            self.unlock_pan()
        elif cmd == "lock_pan":
            self.lock_pan_for_grasp()
        self._grasp_cmd_pub.publish(String(data=cmd))

    def lock_pan_for_grasp(self):
        """Hold current tilt during grasp. URDF TF follows the pan joint, so
        aiming tilted-down is OK — do not snap back to the nav calibration pose."""
        self._grasp_pan_locked = True
        self._ensure_pan_mode(force=True)
        if self._pan_pos is not None and abs(float(self._pan_pos) + math.pi) > 0.05:
            self._pan_cmd = float(self._pan_pos)
            self._publish_pan(self._pan_cmd, burst=3)
            self._active["pan"] = time.time()
        self._broadcast(json.dumps({
            "type": "pan_lock",
            "locked": True,
            "value": self.pan_slider_pct() or 50,
        }))

    def unlock_pan(self):
        self._grasp_pan_locked = False
        self._broadcast(json.dumps({"type": "pan_lock", "locked": False}))

    def _img_cb(self, msg):
        # msg.data is already JPEG bytes (compressed topic)
        with _frame_lock:
            _latest_frame[0] = bytes(msg.data)

    def _wide_cb(self, msg):
        try:
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            ok, jpg = cv2.imencode(".jpg", cv_img, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                with _wide_lock:
                    _latest_wide[0] = jpg.tobytes()
        except Exception as e:
            self.get_logger().warn(f"wide cam encode error: {e}")

    def gripper_slider_pct(self):
        # Prefer gripper joint — left_finger is non-monotonic past peak open.
        if self._gripper_joint is not None:
            span = GRIPPER_JOINT_OPEN - GRIPPER_JOINT_CLOSED
            if abs(span) < 1e-6:
                return None
            frac = (self._gripper_joint - GRIPPER_JOINT_CLOSED) / span
            return int(round(max(0.0, min(1.0, frac)) * 100))
        if self._left_finger is None:
            return None
        frac = (self._left_finger - FINGER_CLOSED) / (FINGER_OPEN - FINGER_CLOSED)
        return int(round(max(0.0, min(1.0, frac)) * 100))

    def _servo_joint_live(self, servo) -> bool:
        return (time.monotonic() - self._servo_js_stamp.get(servo, 0.0)) < JOINT_FRESH_S

    def _blocked(self, servo):
        """Teleop must not freeze on flaky register reads. Health is display-only."""
        return False

    def _control_loop(self):
        with self._lock:
            stick_age = time.monotonic() - getattr(self, "_stick_t", 0.0)
            if stick_age > 0.30:
                lx = ly = rx = ry = 0.0
                dx = dy = 0.0
            else:
                lx, ly, rx, ry = self._lx, self._ly, self._rx, self._ry
                dx, dy = self._dx, self._dy
            # Arm targets come from /joint_states. Drive must still work if the
            # arm is off / not on the graph (otherwise /cmd_vel never publishes).
            tgt = None if self._target is None else dict(self._target)

        def dz(v): return v if abs(v) > 0.05 else 0.0

        # --- Arm (sticks 1 & 2) ---
        if tgt is not None and self._arm_gui_active():
            deltas = {
                "waist":        dz(-lx) * SPEED["waist"]        * self._dt,
                "shoulder":     dz(-ly) * SPEED["shoulder"]      * self._dt,
                "wrist_angle":  dz(-ry) * SPEED["wrist_angle"]   * self._dt,
                "wrist_rotate": dz(rx)  * SPEED["wrist_rotate"]  * self._dt,
            }
            if any(abs(d) > 1e-6 for d in deltas.values()):
                now = time.time()
                for j in JOINT_NAMES:
                    if abs(deltas[j]) > 1e-6:
                        for s in JOINT_TO_SERVOS.get(j, [j]):
                            self._active[s] = now   # mark actuated servo(s)
                    if self._blocked(j):
                        continue  # hold blocked joint at its current target
                    lo, hi = JOINT_LIMITS[j]
                    tgt[j] = max(lo, min(hi, tgt[j] + deltas[j]))
                with self._lock:
                    self._target = tgt
                msg = JointGroupCommand()
                msg.name = "arm"
                msg.cmd = [tgt[j] for j in JOINT_NAMES]
                self._publish_arm_group(msg)
                self._publish_js_targets(JOINT_NAMES, msg.cmd)

        # --- Drive (stick 3) ---
        # Only publish while the stick is deflected. Continuous zeros here fight
        # Nav2 / teleop on /cmd_vel and cause jitter. Send one zero on release.
        twist = Twist()
        twist.linear.x = dz(-dy) * MAX_LIN
        twist.angular.z = dz(-dx) * MAX_ANG
        driving = abs(twist.linear.x) > 1e-6 or abs(twist.angular.z) > 1e-6
        if driving:
            self._publish_cmd_vel(twist)
            self._drive_was_active = True
        elif self._drive_was_active:
            self._publish_cmd_vel(Twist())
            self._drive_was_active = False

        # --- Hold pan Goal_Position (DDS drops + torque-off were making this flaky) ---
        self._service_pan()

        # --- Broadcast joint angles to GUI (~5 Hz) ---
        self._js_tick = getattr(self, "_js_tick", 0) + 1
        if self._js_tick >= 4:  # 20Hz / 4 = 5Hz
            self._js_tick = 0
            with self._lock:
                pos = dict(self._pos)
            slider_pcts = {}
            for j in JOINT_NAMES:
                lo, hi = JOINT_LIMITS[j]
                frac = (pos[j] - lo) / (hi - lo) if hi > lo else 0.0
                slider_pcts[j] = int(round(max(0.0, min(1.0, frac)) * 100))
            payload = json.dumps({
                "type": "joint_state",
                "joints": {j: round(math.degrees(pos[j]), 1) for j in JOINT_NAMES},
                "slider_pcts": slider_pcts,
                "gripper_pct": self.gripper_slider_pct(),
                "pan_pct": self.pan_slider_pct(),
            })
            self._broadcast(payload)

    def _joints_live(self) -> bool:
        return (time.monotonic() - self._js_stamp) < JOINT_FRESH_S

    def _read(self, servo, reg, cb):
        if self._reg_inflight:
            return
        self._reg_inflight = True
        self._reg_inflight_t = time.monotonic()
        r = RegisterValues.Request()
        r.cmd_type = "single"
        r.name = servo
        r.reg = reg

        def _done(fut, s=servo):
            self._reg_inflight = False
            cb(s, fut)

        fut = self._reg_cli.call_async(r)
        fut.add_done_callback(_done)

    def _reg_first_int(self, values):
        if values is None:
            return None
        if isinstance(values, (list, tuple)):
            if not values:
                return None
            return int(values[0])
        try:
            if hasattr(values, "__len__") and not isinstance(values, (str, bytes)):
                if len(values) == 0:
                    return None
                return int(values[0])
            return int(values)
        except (TypeError, ValueError, IndexError):
            return None

    def _err_label(self, err) -> str:
        if err in (None, 0):
            return ""
        err = int(err)
        bits = []
        if err & 1:
            bits.append("VIN")
        if err & 4:
            bits.append("OH")
        if err & 8:
            bits.append("ENC")
        if err & 16:
            bits.append("SHK")
        if err & 32:
            bits.append("OL")
        return "+".join(bits) if bits else f"E{err}"

    def _poll_health(self):
        # One register read at a time. Never stack calls — a timeout that
        # cleared _reg_inflight while the service was still pending wedged
        # xs_sdk. GUI is crossover-only now, so these should complete.
        if self._reg_inflight:
            if (time.monotonic() - self._reg_inflight_t) < 5.0:
                return
            self._reg_inflight = False
        if not self._reg_cli.service_is_ready():
            return
        n = len(SERVO_LIST)
        servo = SERVO_LIST[self._poll_idx % n]
        phase = (self._poll_idx // n) % 3
        self._poll_idx += 1
        if phase == 0:
            self._read(servo, "Present_Temperature", self._on_temp)
        elif phase == 1:
            self._read(servo, "Hardware_Error_Status", self._on_err)
        else:
            self._read(servo, "Present_Load", self._on_load)

    def _note_reg_fail(self, servo):
        # Empty/timeout replies are normal on AGX↔Nano; only count when that
        # servo's joint_states are also stale.
        if self._servo_joint_live(servo):
            return
        self._health[servo]["read_fail"] = self._health[servo].get("read_fail", 0) + 1

    def _on_temp(self, servo, fut):
        try:
            v = self._reg_first_int(fut.result().values)
            if v is not None:
                self._health[servo]["temp"] = v
                self._health[servo]["read_fail"] = 0
                warn_t = GRIPPER_TEMP_WARN if servo == "gripper" else TEMP_WARN
                if v >= warn_t:
                    self.get_logger().warn(f"{servo}: temp {v}C")
            else:
                self._note_reg_fail(servo)
            self._eval_block(servo)
        except Exception:
            self._note_reg_fail(servo)
            self._eval_block(servo)

    def _on_err(self, servo, fut):
        try:
            v = self._reg_first_int(fut.result().values)
            if v is not None:
                self._health[servo]["err"] = v
                self._health[servo]["read_fail"] = 0
                if v != 0:
                    self.get_logger().warn(
                        f"{servo}: Hardware_Error_Status={v} {self._err_label(v)}"
                    )
                    self._broadcast_health()
                    if servo == "gripper":
                        self._maybe_reboot_gripper_fault(v)
                self._eval_block(servo)
            else:
                self._note_reg_fail(servo)
                self._eval_block(servo)
        except Exception:
            self._note_reg_fail(servo)

    def _maybe_reboot_gripper_fault(self, err_code: int):
        now = time.time()
        if now - self._gripper_last_reboot_t < GRIPPER_REBOOT_COOLDOWN_S:
            return
        # Overload while moving: skip reboot thrash; re-assert position after.
        if abs(self._gripper_active_effort) > 1e-3:
            self.get_logger().warn(
                f"gripper HW fault {err_code} while commanding — re-hold position"
            )
            if self._gripper_cmd is not None:
                self._publish_gripper_pos(self._gripper_cmd)
            return
        self._gripper_last_reboot_t = now
        self.get_logger().warn(
            f"gripper HW fault {err_code} {self._err_label(err_code)} "
            "— rebooting motor (clears overload latch)"
        )
        self.reboot_servo("gripper")
        self._gripper_mode = None
        self._modes_ok = False
        self._pan_init_timer = self.create_timer(1.5, self._init_pan_mode)
        # Keep err visible in the GUI until the next clean Hardware_Error_Status.

    def _on_load(self, servo, fut):
        try:
            v = self._reg_first_int(fut.result().values)
            if v is not None:
                self._health[servo]["load"] = v
                self._health[servo]["read_fail"] = 0
                self._eval_block(servo)
            else:
                self._note_reg_fail(servo)
                self._eval_block(servo)
        except Exception:
            self._note_reg_fail(servo)

    def _hw_services_absent(self) -> bool:
        try:
            return not self._reg_cli.service_is_ready()
        except Exception:
            return True

    def _servo_status(self, h, servo=None):
        temp, err, load = h["temp"], h["err"], h.get("load")
        live = self._servo_joint_live(servo) if servo else self._joints_live()
        # Isaac / no xs_sdk: joint_states ticking + no motor services → connected.
        if (self._sim or self._hw_services_absent()) and self._joints_live():
            live = True
        warn_t = GRIPPER_TEMP_WARN if servo == "gripper" else TEMP_WARN
        fail_t = GRIPPER_TEMP_FAIL if servo == "gripper" else TEMP_FAIL
        # Purple only when truly blocked AND not receiving joint_states.
        if h.get("blocked") and not live:
            return "blocked"
        hot_fail = temp is not None and temp >= fail_t
        hot_warn = temp is not None and temp >= warn_t
        load_warn = load is not None and abs(load) >= LOAD_WARN
        if (err not in (None, 0)) or hot_fail:
            return "red"
        if hot_warn or load_warn:
            return "orange"
        if live:
            return "green"
        if h.get("read_fail", 0) >= 4:
            return "red"
        if temp is None and err is None:
            return "unknown"
        return "unknown"

    def _broadcast_health(self):
        health = {}
        for s in SERVO_LIST:
            h = self._health[s]
            blocked_ui = bool(h["blocked"] and not self._servo_joint_live(s))
            health[s] = {
                "status": self._servo_status(h, s),
                "temp": h["temp"],
                "err": h["err"],
                "err_label": self._err_label(h["err"]),
                "load": h["load"],
                "blocked": blocked_ui,
                "live": self._servo_joint_live(s),
            }
        self._broadcast(json.dumps({"type": "servo_health", "health": health}))
        self._broadcast(json.dumps({"type": "world", "world": self._world_mode()}))

    def _eval_block(self, servo):
        h = self._health[servo]
        temp = h["temp"]; load = h["load"]; err = h["err"]
        live = self._servo_joint_live(servo)
        hot   = (temp is not None and temp >= TEMP_BLOCK)
        # Gripper Present_Load spikes every close — do not purple-block teleop.
        heavy = (
            servo != "gripper"
            and load is not None
            and abs(load) >= LOAD_BLOCK
            and not live
        )
        # HW error latches are sticky; only freeze teleop if joints are dead too.
        errored = (err not in (None, 0)) and not live
        silent = (h.get("read_fail", 0) >= SILENT_FAIL_COUNT) and not live
        if silent and self._joints_live() and not SILENT_BLOCK_IF_JOINTS_LIVE:
            silent = False
        if h["blocked"]:
            cool = (temp is None or temp <= TEMP_CLEAR)
            light = (load is None or abs(load) <= LOAD_CLEAR)
            if live or (cool and light and not errored):
                h["blocked"] = False
                h["trip"] = 0
                if live:
                    h["read_fail"] = 0
                self.get_logger().info(f"{servo}: UNBLOCKED (live={live}, temp={temp}, load={load})")
        else:
            if hot or heavy or errored or silent:
                h["trip"] += 1
                if h["trip"] >= BLOCK_COUNT or errored or silent:
                    h["blocked"] = True
                    reason = "silent" if silent else ("error" if errored else ("hot" if hot else "load"))
                    self.get_logger().warn(f"{servo}: BLOCKED [{reason}] (temp={temp}, load={load}, err={err})")
                    if silent:
                        self.get_logger().warn(
                            "Arm bus quiet — check 12V + DXL; ensure only ONE xsarm_control on Nano"
                        )
            else:
                h["trip"] = 0

    def reboot_servo(self, servo):
        if servo not in SERVO_LIST:
            return
        if self._sim or not self._reboot_cli.service_is_ready():
            self.get_logger().info(f"{servo}: reboot skipped (sim or service absent)")
            return
        req = Reboot.Request()
        req.cmd_type = "single"; req.name = servo
        req.enable = True; req.smart_reboot = False
        self._reboot_cli.call_async(req)
        self.get_logger().info(f"reboot requested: {servo}")

    def _load_locations(self):
        try:
            data = yaml.safe_load(LOCATIONS_YAML.read_text(encoding="utf-8")) or {}
            locs = data.get("locations") or {}
            self.get_logger().info(
                f"Loaded {len(locs)} named locations from {LOCATIONS_YAML}"
            )
            return locs
        except Exception as e:
            self.get_logger().warn(f"Could not load locations YAML: {e}")
            return {}

    def location_names(self):
        return sorted(self._locations.keys())

    def _create3_action_subprocess(self, action_name, action_type):
        """Dock/undock live on Create3 domain 5; not bridged to domain 0."""
        def _run():
            env = os.environ.copy()
            env["ROS_DOMAIN_ID"] = "5"
            env["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
            env.pop("CYCLONEDDS_URI", None)
            env["FASTRTPS_DEFAULT_PROFILES_FILE"] = CREATE3_DDS_PROFILE
            cmd = [
                "ros2", "action", "send_goal",
                action_name, action_type, "{}",
            ]
            self.get_logger().info(f"{action_name}: launching on domain 5…")
            try:
                r = subprocess.run(
                    cmd, env=env, capture_output=True, text=True, timeout=180,
                )
                out = (r.stdout or "") + (r.stderr or "")
                if r.returncode == 0 and "SUCCEEDED" in out:
                    self.get_logger().info(f"{action_name}: SUCCEEDED")
                else:
                    self.get_logger().warn(
                        f"{action_name}: rc={r.returncode} "
                        f"{out.strip()[-400:]}"
                    )
            except Exception as e:
                self.get_logger().warn(f"{action_name} failed: {e}")
        threading.Thread(target=_run, daemon=True).start()

    def send_dock(self):
        self._create3_action_subprocess(
            "/dock", "irobot_create_msgs/action/Dock")

    def send_undock(self):
        self._create3_action_subprocess(
            "/undock", "irobot_create_msgs/action/Undock")

    def _nav_status(self, text):
        self.get_logger().info(text)
        self._broadcast(json.dumps({"type": "nav_status", "text": text}))

    def send_navigate(self, name: str):
        name = (name or "").strip()
        if name not in self._locations:
            self.get_logger().warn(
                f"Unknown location '{name}'. Known: {', '.join(self.location_names())}"
            )
            return
        if self._nav_restarting:
            self.get_logger().warn("Nav restart in progress — wait, then try again")
            return
        if self._nav_busy:
            self.get_logger().warn("Navigate already in progress — cancel first")
            return
        loc = self._locations[name]
        if self._nav_client is None:
            self.get_logger().warn("Navigation is not available on this GUI")
            return
        if not self._nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn("navigate_to_pose not available (is nav_bringup up?)")
            return
        goal = NavigateToPose.Goal()
        pose = PoseStamped()
        pose.header.frame_id = loc.get("frame_id", "map")
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(loc["x"])
        pose.pose.position.y = float(loc["y"])
        pose.pose.position.z = float(loc.get("z", 0.0))
        yaw = float(loc["yaw"])
        pose.pose.orientation.z = math.sin(yaw * 0.5)
        pose.pose.orientation.w = math.cos(yaw * 0.5)
        goal.pose = pose
        self._nav_busy = True
        self._nav_status(
            f"Navigating to '{name}' ({loc['x']:.2f}, {loc['y']:.2f})"
        )
        fut = self._nav_client.send_goal_async(goal)
        fut.add_done_callback(lambda f, n=name: self._nav_goal_cb(f, n))

    def _nav_goal_cb(self, future, name):
        try:
            handle = future.result()
            if not handle or not handle.accepted:
                self.get_logger().warn(f"Nav goal '{name}' rejected")
                self._nav_busy = False
                self._nav_goal_handle = None
                self._nav_status(f"Nav goal '{name}' rejected")
                return
            self._nav_goal_handle = handle
            handle.get_result_async().add_done_callback(
                lambda f, n=name: self._nav_result_cb(f, n))
        except Exception as e:
            self.get_logger().warn(f"Nav goal error: {e}")
            self._nav_busy = False
            self._nav_goal_handle = None

    def _nav_result_cb(self, future, name):
        try:
            result = future.result()
            self._nav_status(f"Nav '{name}' finished status={result.status}")
        except Exception as e:
            self.get_logger().warn(f"Nav result error: {e}")
        self._nav_busy = False
        self._nav_goal_handle = None

    def cancel_navigate(self):
        """Cancel active navigate_to_pose goal(s) from GUI or RViz."""
        self._nav_status("Canceling nav goal…")
        handle = self._nav_goal_handle
        if handle is not None:
            try:
                handle.cancel_goal_async()
            except Exception as e:
                self.get_logger().warn(f"goal_handle cancel failed: {e}")
        self._nav_goal_handle = None
        self._nav_busy = False

        # Stop base immediately so a cancel doesn't coast into obstacles
        try:
            self._publish_cmd_vel(Twist())
        except Exception:
            pass

        def _cancel_all():
            try:
                if self._nav_cancel_cli is None:
                    self._nav_status("Cancel service unavailable (Nav2 down?)")
                    return
                if self._nav_cancel_cli.wait_for_service(timeout_sec=2.0):
                    fut = self._nav_cancel_cli.call_async(CancelGoal.Request())
                    fut.add_done_callback(
                        lambda f: self._nav_status("Nav goal cancel requested"))
                else:
                    self._nav_status("Cancel service unavailable (Nav2 down?)")
            except Exception as e:
                self.get_logger().warn(f"cancel_goal service error: {e}")
        threading.Thread(target=_cancel_all, daemon=True).start()

        # Best-effort costmap clear so a stuck plan doesn't linger
        def _clear(cli, label):
            try:
                if cli is None or ClearEntireCostmap is None:
                    return
                if cli.wait_for_service(timeout_sec=1.0):
                    cli.call_async(ClearEntireCostmap.Request())
                    self.get_logger().info(f"cleared {label}")
            except Exception:
                pass
        threading.Thread(
            target=_clear, args=(self._clear_local_cli, "local_costmap"),
            daemon=True).start()
        threading.Thread(
            target=_clear, args=(self._clear_global_cli, "global_costmap"),
            daemon=True).start()

    def _lifecycle_call(self, command: int, label: str, timeout_sec: float = 20.0) -> bool:
        """Call Nav2 lifecycle_manager manage_nodes (same as RViz Startup/Reset)."""
        if self._nav_lifecycle_cli is None or ManageLifecycleNodes is None:
            self._nav_status("Lifecycle service missing (nav2_msgs not installed)")
            return False
        if not self._nav_lifecycle_cli.wait_for_service(timeout_sec=3.0):
            self._nav_status(f"Lifecycle service missing ({NAV_LIFECYCLE_SRV})")
            return False
        req = ManageLifecycleNodes.Request()
        req.command = command
        fut = self._nav_lifecycle_cli.call_async(req)
        t0 = time.time()
        while not fut.done() and (time.time() - t0) < timeout_sec:
            time.sleep(0.05)
        if not fut.done():
            self._nav_status(f"Nav {label} timed out")
            return False
        try:
            resp = fut.result()
            ok = bool(resp and resp.success)
            self.get_logger().info(f"lifecycle {label}: success={ok}")
            return ok
        except Exception as e:
            self.get_logger().warn(f"lifecycle {label} error: {e}")
            return False

    def restart_navigation(self):
        """RViz-style Nav2 restart: cancel goal, lifecycle Reset, then Startup."""
        if self._nav_restarting:
            self._nav_status("Nav reset already running")
            return

        self.cancel_navigate()
        self._nav_restarting = True
        self._nav_status("Resetting Nav2 (RViz Reset → Startup)…")

        def _run():
            try:
                # Match Nav2 RViz panel: Reset (deactivate+cleanup) then Startup
                # (configure+activate) so navigate_to_pose is usable again.
                ok_reset = self._lifecycle_call(
                    ManageLifecycleNodes.Request.RESET, "Reset")
                if not ok_reset:
                    self._nav_status("Nav2 Reset failed — is navigation_launch up?")
                    return
                ok_start = self._lifecycle_call(
                    ManageLifecycleNodes.Request.STARTUP, "Startup")
                if ok_start:
                    self._nav_status("Nav2 restarted (Reset + Startup)")
                else:
                    self._nav_status("Nav2 Reset OK but Startup failed")
            except Exception as e:
                self._nav_status(f"Nav restart failed: {e}")
            finally:
                self._nav_restarting = False

        threading.Thread(target=_run, daemon=True).start()

    def _broadcast(self, payload):
        if self._ws_loop is None:
            return
        with self._ws_lock:
            clients = list(self._ws_clients)
        if not clients:
            return
        async def _send():
            import asyncio as _a
            await _a.gather(*[c.send(payload) for c in clients],
                            return_exceptions=True)
        try:
            asyncio.run_coroutine_threadsafe(_send(), self._ws_loop)
        except Exception:
            pass

    def set_joint_degrees(self, joint, degrees):
        if not self._arm_gui_active():
            return
        if joint not in JOINT_NAMES:
            return
        if self._blocked(joint):
            return
        angle = math.radians(degrees)
        lo, hi = JOINT_LIMITS[joint]
        angle = max(lo, min(hi, angle))
        with self._lock:
            if self._target is None:
                self._target = dict(self._pos)
            self._target[joint] = angle
            tgt = dict(self._target)
        msg = JointGroupCommand()
        msg.name = "arm"
        msg.cmd = [tgt[j] for j in JOINT_NAMES]
        self._publish_arm_group(msg)

    def set_joint_angle(self, joint, frac):
        if not self._arm_gui_active():
            return
        if joint not in JOINT_NAMES:
            return
        if self._blocked(joint):
            return
        lo, hi = JOINT_LIMITS[joint]
        frac = max(0.0, min(1.0, frac))
        angle = lo + frac * (hi - lo)
        with self._lock:
            if self._target is None:
                self._target = dict(self._pos)
            self._target[joint] = angle
            tgt = dict(self._target)
        msg = JointGroupCommand()
        msg.name = "arm"
        msg.cmd = [tgt[j] for j in JOINT_NAMES]
        self._publish_arm_group(msg)

    def set_joint_degrees(self, joint, degrees):
        if not self._arm_gui_active():
            return
        if joint not in JOINT_NAMES:
            return
        if self._blocked(joint):
            return
        angle = math.radians(degrees)
        lo, hi = JOINT_LIMITS[joint]
        angle = max(lo, min(hi, angle))
        with self._lock:
            if self._target is None:
                self._target = dict(self._pos)
            self._target[joint] = angle
            tgt = dict(self._target)
        msg = JointGroupCommand()
        msg.name = "arm"
        msg.cmd = [tgt[j] for j in JOINT_NAMES]
        self._publish_arm_group(msg)

    def set_joint_angle(self, joint, frac):
        if not self._arm_gui_active():
            return
        if joint not in JOINT_NAMES:
            return
        if self._blocked(joint):
            return
        lo, hi = JOINT_LIMITS[joint]
        frac = max(0.0, min(1.0, frac))
        angle = lo + frac * (hi - lo)
        with self._lock:
            if self._target is None:
                self._target = dict(self._pos)
            self._target[joint] = angle
            tgt = dict(self._target)
        msg = JointGroupCommand()
        msg.name = "arm"
        msg.cmd = [tgt[j] for j in JOINT_NAMES]
        self._publish_arm_group(msg)

    def set_arm_sticks(self, lx, ly, rx, ry):
        if not self._arm_gui_active():
            return
        with self._lock:
            self._lx = max(-1.0, min(1.0, lx))
            self._ly = max(-1.0, min(1.0, ly))
            self._rx = max(-1.0, min(1.0, rx))
            self._ry = max(-1.0, min(1.0, ry))
            self._stick_t = time.monotonic()

    def set_drive_stick(self, dx, dy):
        with self._lock:
            self._dx = max(-1.0, min(1.0, dx))
            self._dy = max(-1.0, min(1.0, dy))
            self._stick_t = time.monotonic()

    def pan_slider_pct(self):
        if self._pan_pos is None:
            return None
        frac = (self._pan_pos - PAN_MIN) / (PAN_MAX - PAN_MIN)
        return int(round(max(0.0, min(1.0, frac)) * 100))

    def _publish_pan(self, cmd: float, burst: int = 1):
        cmd = float(max(PAN_MIN, min(PAN_MAX, cmd)))
        self._pan_cmd = cmd
        msg = JointSingleCommand()
        msg.name = "pan"
        msg.cmd = cmd
        for _ in range(max(1, int(burst))):
            if self._real_bus():
                self._gripper_pub.publish(msg)
            self._publish_js_targets(["pan"], [cmd])

    def _service_pan(self):
        """Re-assert pan goal while dragging or grasp-locked."""
        if Path("/tmp/grabit_auto_collect.pid").is_file():
            return
        if self._pan_cmd is None:
            return
        now = time.time()
        active = (now - self._active.get("pan", 0.0)) < 1.25
        if not active and not self._grasp_pan_locked:
            return
        # Throttle to ~10 Hz
        if (now - getattr(self, "_pan_last_pub_t", 0.0)) < 0.1:
            return
        self._pan_last_pub_t = now
        self._publish_pan(self._pan_cmd, burst=1)

    def set_pan(self, value):
        # #region agent log
        if self._blocked("pan") and not self._servo_joint_live("pan"):
            _agent_dbg(
                "H5",
                "gui_node.py:set_pan",
                "pan rejected: blocked + joint dead",
                {"value": value, "blocked": True},
            )
            return
        if self._grasp_pan_locked:
            _agent_dbg(
                "H5",
                "gui_node.py:set_pan",
                "pan rejected: grasp_pan_locked",
                {"value": value},
            )
            return
        # #endregion
        if self._blocked("pan"):
            # joints-live path already softens this; still try if only status flaked
            if not self._servo_joint_live("pan"):
                return
        if self._grasp_pan_locked:
            return
        now = time.time()
        # #region agent log
        forcing_mode = self._pan_mode != "position" or (now - self._pan_mode_t) > 8.0
        if forcing_mode:
            _agent_dbg(
                "H4",
                "gui_node.py:set_pan",
                "re-asserting pan mode/torque",
                {"pan_mode": self._pan_mode, "mode_age_s": round(now - self._pan_mode_t, 2)},
            )
        # #endregion
        if self._pan_mode != "position" or (now - self._pan_mode_t) > 8.0:
            self._ensure_pan_mode(force=True)
        self._active["pan"] = now
        # value: 0.0..1.0 maps PAN_MIN..PAN_MAX
        value = max(0.0, min(1.0, value))
        cmd = PAN_MIN + value * (PAN_MAX - PAN_MIN)
        self._publish_pan(cmd, burst=3)
        # #region agent log
        _agent_dbg(
            "H3-H5",
            "gui_node.py:set_pan",
            "pan command published",
            {
                "value": value,
                "cmd": round(cmd, 4),
                "subs": int(self._gripper_pub.get_subscription_count()),
                "js_live": self._joints_live(),
            },
        )
        # #endregion

    def set_gripper(self, value):
        if not self._real_bus():
            return
        """Slider 0=closed .. 1=open → gripper joint position (holds)."""
        if self._policy_live:
            return
        value = max(0.0, min(1.0, float(value)))
        # Grasp / other nodes may flip the servo mode — refresh current-based hold.
        now = time.time()
        if self._gripper_mode != GRIPPER_OP_MODE:
            self._ensure_gripper_position_mode(force=True)
            self._gripper_mode_t = now
        cmd = GRIPPER_JOINT_CLOSED + value * (GRIPPER_JOINT_OPEN - GRIPPER_JOINT_CLOSED)
        self._active["gripper"] = time.time()
        self._gripper_active_effort = 1.0
        self._publish_gripper_pos(cmd)

    def _publish_gripper_pos(self, cmd: float):
        cmd = float(max(GRIPPER_JOINT_CLOSED - 0.05, min(GRIPPER_JOINT_OPEN + 0.05, cmd)))
        self._gripper_cmd = cmd
        msg = JointSingleCommand()
        msg.name = "gripper"
        msg.cmd = cmd
        if self._real_bus():
            self._gripper_pub.publish(msg)
        self._publish_js_targets(["gripper"], [cmd])

    def go_home(self):
        if self._teach_free:
            self.set_arm_teach(False)
        if self._policy_live:
            return
        self._start_named_pose_move([list(HOME_JOINTS)], label="home")

    def go_sleep(self):
        if self._teach_free:
            self.set_arm_teach(False)
        if self._policy_live:
            return
        # Never jump straight from a forward grasp into Sleep — shoulder
        # whips back through the AGX / camera. Ease to Home, then Sleep.
        self._start_named_pose_move(
            [list(HOME_JOINTS), list(SLEEP_JOINTS)], label="sleep"
        )

    def _start_named_pose_move(self, waypoints, label: str) -> None:
        self._named_pose_gen += 1
        gen = int(self._named_pose_gen)
        self._named_pose_busy = True
        with self._lock:
            self._lx = self._ly = self._rx = self._ry = 0.0
        self.get_logger().info(
            f"named pose '{label}' slow path ({len(waypoints)} segment(s))"
        )
        threading.Thread(
            target=self._named_pose_worker,
            args=(waypoints, label, gen),
            daemon=True,
            name=f"named_pose_{label}",
        ).start()

    def _named_pose_worker(self, waypoints, label: str, gen: int) -> None:
        try:
            with self._lock:
                q = [float(self._pos.get(j, 0.0)) for j in JOINT_NAMES]
            for wp in waypoints:
                if gen != self._named_pose_gen:
                    return
                goal = [float(v) for v in wp[: len(JOINT_NAMES)]]
                if len(goal) < len(JOINT_NAMES):
                    goal.extend(q[len(goal) :])
                if not self._interpolate_arm(q, goal, gen):
                    return
                q = list(goal)
            with self._lock:
                self._target = {j: float(v) for j, v in zip(JOINT_NAMES, q)}
            # Hold final (group + joint_targets so Isaac without xs_msgs follows)
            msg = JointGroupCommand()
            msg.name = "arm"
            msg.cmd = [float(v) for v in q]
            for _ in range(3):
                if gen != self._named_pose_gen:
                    return
                self._publish_arm_group(msg)
                self._publish_js_targets(JOINT_NAMES, q)
                time.sleep(0.05)
            self.get_logger().info(f"named pose '{label}' complete")
        except Exception as e:
            self.get_logger().warn(f"named pose '{label}' failed: {e}")
        finally:
            if gen == self._named_pose_gen:
                self._named_pose_busy = False
                self._sync_arm_target_from_pos()

    def _interpolate_arm(self, q0, q1, gen: int) -> bool:
        """Publish a capped-speed linear joint path. False if cancelled."""
        dq = [float(b) - float(a) for a, b in zip(q0, q1)]
        # Duration from the slowest joint (shoulder capped lowest).
        dur = 0.35
        for j, d in zip(JOINT_NAMES, dq):
            spd = float(NAMED_POSE_MAX_SPEED.get(j, 0.25))
            if spd > 1e-6:
                dur = max(dur, abs(d) / spd)
        dur = min(12.0, max(0.8, dur))
        rate = float(UPDATE_HZ)
        steps = max(8, int(dur * rate))
        dt = dur / float(steps)
        for s in range(1, steps + 1):
            if gen != self._named_pose_gen:
                return False
            a = s / float(steps)
            # Ease in/out so the shoulder does not crack at the ends.
            a = a * a * (3.0 - 2.0 * a)
            q = [float(c) + a * float(d) for c, d in zip(q0, dq)]
            with self._lock:
                self._target = {j: float(v) for j, v in zip(JOINT_NAMES, q)}
            msg = JointGroupCommand()
            msg.name = "arm"
            msg.cmd = q
            self._publish_arm_group(msg)
            self._publish_js_targets(JOINT_NAMES, q)
            time.sleep(dt)
        return True

    async def ws_handler(self, ws):
        with self._ws_lock:
            self._ws_clients.add(ws)
        # #region agent log
        _agent_dbg(
            "H2",
            "gui_node.py:ws_handler",
            "ws client connected",
            {"clients": len(self._ws_clients)},
        )
        # #endregion
        try:
            await ws.send(json.dumps({
                "type": "locations",
                "names": self.location_names(),
            }))
            pct = self.gripper_slider_pct()
            if pct is not None:
                await ws.send(json.dumps({"type": "gripper_init", "value": pct}))
            ppct = self.pan_slider_pct()
            if ppct is not None:
                await ws.send(json.dumps({"type": "pan_init", "value": ppct}))
            await ws.send(json.dumps({"type": "teach_free", "free": bool(self._teach_free)}))
            await ws.send(json.dumps({"type": "world", "world": self._world_mode()}))
            train = self._train_status_payload()
            if train is not None:
                await ws.send(json.dumps(train))
            self._broadcast_act_catalog()
            self._broadcast_collection()
            self._broadcast_health()
        except Exception:
            pass
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                    t = msg.get("type")
                    if t == "arm":
                        self.set_arm_sticks(msg.get("lx",0), msg.get("ly",0),
                                            msg.get("rx",0), msg.get("ry",0))
                    elif t == "drive":
                        self.set_drive_stick(msg.get("dx",0), msg.get("dy",0))
                    elif t == "gripper":
                        self.set_gripper(msg.get("value", 1.0))
                    elif t == "pan":
                        self.set_pan(msg.get("value", 0.5))
                    elif t == "home":
                        self.go_home()
                    elif t == "sleep":
                        self.go_sleep()
                    elif t == "dock":
                        self.send_dock()
                    elif t == "undock":
                        self.send_undock()
                    elif t == "navigate":
                        self.send_navigate(msg.get("name", ""))
                    elif t == "cancel_nav":
                        self.cancel_navigate()
                    elif t == "restart_nav":
                        self.restart_navigation()
                    elif t == "joint_angle":
                        self.set_joint_angle(msg.get("joint", ""), msg.get("value", 0.5))
                    elif t == "joint_degrees":
                        self.set_joint_degrees(msg.get("joint", ""), msg.get("degrees", 0.0))
                    elif t == "reboot":
                        self.reboot_servo(msg.get("servo", ""))
                    elif t == "grasp_click":
                        self.publish_grasp_click(
                            msg.get("u", -1), msg.get("v", -1),
                            msg.get("width"), msg.get("height"),
                            cam=msg.get("cam", "d435i"),
                            click_id=msg.get("click_id"),
                        )
                    elif t == "grasp_execute":
                        self.publish_grasp_command("execute")
                    elif t == "grasp_cancel":
                        self.publish_grasp_command("cancel")
                    elif t == "grasp_lock_pan":
                        self.publish_grasp_command("lock_pan")
                    elif t == "grasp_unlock_pan":
                        self.unlock_pan()
                    elif t == "handoff_grab":
                        self.publish_grasp_command("handoff_grab")
                    elif t == "handoff_release":
                        self.publish_grasp_command("handoff_release")
                    elif t == "il_collection":
                        threading.Thread(
                            target=self._new_collection,
                            args=(str(msg.get("name") or ""),),
                            daemon=True,
                        ).start()
                    elif t == "il_train":
                        threading.Thread(
                            target=self._start_train,
                            args=(msg.get("steps", 20000),),
                            daemon=True,
                        ).start()
                    elif t == "il_start":
                        threading.Thread(
                            target=self._kick_recorder,
                            args=(str(msg.get("source") or "teleop"),),
                            daemon=True,
                        ).start()
                    elif t == "il_save":
                        self.publish_il_command("save", source=msg.get("source", "teleop"))
                    elif t == "il_discard":
                        self.publish_il_command("discard")
                    elif t == "il_act_select":
                        threading.Thread(
                            target=self._select_act,
                            args=(str(msg.get("name") or ""),),
                            daemon=True,
                        ).start()
                    elif t == "il_replay":
                        self.publish_il_command("play", mode="replay")
                    elif t == "il_play":
                        if self._world_mode() == "sim":
                            self.get_logger().info(
                                "SIM PLAY — Isaac hub owns the command, real arm stays"
                            )
                        else:
                            threading.Thread(
                                target=self._kick_real_act, daemon=True
                            ).start()
                    elif t == "il_auto_collect":
                        if self._world_mode() == "sim":
                            self.get_logger().info(
                                "auto collect stays off in sim"
                            )
                        else:
                            threading.Thread(
                                target=self._start_auto_collect,
                                args=(msg.get("count", 5),),
                                daemon=True,
                            ).start()
                    elif t == "il_stop":
                        self._stop_auto_collect()
                        if self._world_mode() == "sim":
                            self.get_logger().info("SIM STOP — Isaac hub")
                        else:
                            self.publish_il_command("stop")
                    elif t == "arm_free":
                        self.set_arm_teach(True)
                    elif t == "arm_hold":
                        self.set_arm_teach(False)
                    elif t == "cal_save":
                        self.publish_cal_command("save_pose")
                    elif t == "cal_goto":
                        self.publish_cal_command("goto_taught")
                    elif t == "cal_sweep":
                        try:
                            n = int(msg.get("n", 8))
                        except (TypeError, ValueError):
                            n = 8
                        self.publish_cal_command("run_sweep", n=n)
                    elif t == "cal_abort":
                        self.publish_cal_command("abort")
                except Exception as e:
                    self.get_logger().warn(f"WS error: {e}")
        except Exception:
            pass
        finally:
            with self._ws_lock:
                self._ws_clients.discard(ws)
            # #region agent log
            _agent_dbg(
                "H2",
                "gui_node.py:ws_handler",
                "ws client disconnected",
                {"clients": len(self._ws_clients)},
            )
            # #endregion


def _run_ws(node, port):
    async def ws_main():
        node._ws_loop = asyncio.get_running_loop()
        async with websockets.serve(node.ws_handler, "0.0.0.0", port):
            node.get_logger().info(f"WebSocket on ws://0.0.0.0:{port}")
            await asyncio.Future()
    asyncio.run(ws_main())


def main(args=None):
    rclpy.init(args=args)
    node = GrabItGUINode()
    HTTP_PORT = int(os.environ.get("GRAB_IT_HTTP_PORT", 8765))
    WS_PORT   = int(os.environ.get("GRAB_IT_WS_PORT",   8766))
    threading.Thread(target=_run_http, args=(HTTP_PORT,), daemon=True).start()
    node.get_logger().info(f"GUI available at http://0.0.0.0:{HTTP_PORT}")
    threading.Thread(target=_run_ws, args=(node, WS_PORT), daemon=True).start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
