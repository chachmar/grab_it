#!/usr/bin/env python3
"""Record N scripted grasps around the current click.

Each take is one row of GRASP_SPREAD: a small lateral, depth, height, and
standoff offset from the click the user just made. The approach side does
not change. Take 1 is the click itself. The same click always produces the
same offsets. A different grasp means the user moves the pole and clicks again.

A saved take is the approach and the scripted close. Holding the pole is
success. The jaws then open and the arm returns to the pose captured when
the batch started, with no lift. Failed takes return there too. Recording
is already stopped before that return.
"""
from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from grab_it_il.common import (
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    HZ,
    IL_COMMAND_TOPIC,
    IL_STATUS_TOPIC,
    JOINT_GROUP_TOPIC,
    JOINT_LIMITS,
    JOINT_SINGLE_TOPIC,
    JOINT_STATES_TOPIC,
    STATE_NAMES,
)

try:
    from interbotix_xs_msgs.msg import JointGroupCommand, JointSingleCommand
    from interbotix_xs_msgs.srv import OperatingModes, TorqueEnable
except ImportError:
    JointGroupCommand = None
    JointSingleCommand = None
    OperatingModes = None
    TorqueEnable = None

_SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

PAN_WINDOW = 0.20
START_TOL = 0.15
SHOULDER_HARD = -1.2
SHOULDER_SLACK = 0.05
BAD_JOINT = 3.0
PAN_TOL = 0.04
AUTO_RETURN_SHOULDER_MIN = -0.55
RATES = {
    "waist": 0.35,
    "shoulder": 0.22,
    "wrist_angle": 0.30,
    "wrist_rotate": 0.40,
    "pan": 0.25,
}
STOP_FILE = Path("/tmp/grabit_auto_collect.stop")
PID_FILE = Path("/tmp/grabit_auto_collect.pid")
GRASP_STATUS_TOPIC = "/grab_it/grasp_status"
GRASP_COMMAND_TOPIC = "/grab_it/grasp_command"

# lateral m, depth m, height m, standoff m. Fixed order, not a new draw.
# Lateral and depth stay inside ±1 cm. Height stays inside ±2.5 cm.
# Standoff stays inside 4–7 cm. The facing of the click is unchanged.
GRASP_SPREAD = (
    (0.00, 0.00, 0.000, 0.05),
    (0.01, 0.00, 0.000, 0.05),
    (-0.01, 0.00, 0.000, 0.05),
    (0.00, 0.01, 0.000, 0.05),
    (0.00, -0.01, 0.000, 0.05),
    (0.00, 0.00, 0.025, 0.05),
    (0.00, 0.00, -0.025, 0.05),
    (0.01, 0.00, 0.015, 0.04),
    (-0.01, 0.00, -0.015, 0.06),
    (0.00, 0.01, 0.020, 0.07),
    (0.00, -0.01, -0.020, 0.04),
    (0.01, -0.01, 0.000, 0.06),
    (-0.01, 0.01, 0.000, 0.05),
    (0.01, 0.01, -0.010, 0.07),
    (-0.01, -0.01, 0.010, 0.04),
)


def _collection_dir() -> Path:
    name = "tabletop_pick"
    try:
        text = Path("/tmp/grabit_collection").read_text(encoding="utf-8").strip()
    except OSError:
        text = ""
    if text and text[0].isalnum() and all(c.isalnum() or c in "_-" for c in text):
        name = text
    for root in (
        Path("/home/ubuntu/dev-ws/datasets"),
        Path("/workspaces/isaac_ros-dev/datasets"),
    ):
        folder = root / name
        if folder.is_dir():
            return folder
    return Path("/home/ubuntu/dev-ws/datasets") / name


def _episode_source(path: Path) -> str:
    meta = path / "meta.json"
    if not meta.is_file():
        return ""
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(data.get("source") or "")


def _seed_episode(root: Path) -> Path | None:
    """The newest take the user saved. Generated pan passes are not the seed."""
    if not root.is_dir():
        return None
    eps = sorted(
        [p for p in root.iterdir() if p.is_dir() and (p / "actions.npy").is_file()],
        key=lambda p: p.name,
    )
    human = [p for p in eps if _episode_source(p) != "auto"]
    if human:
        return human[-1]
    return eps[0] if eps else None


def _pan_offsets(count: int) -> list[float]:
    count = max(1, min(15, int(count)))
    if count == 1:
        return [0.0]
    return [float(x) for x in np.linspace(-PAN_WINDOW, PAN_WINDOW, count)]


def _stop_requested() -> bool:
    return STOP_FILE.is_file()


def _in_limits(name: str, value: float) -> bool:
    if name == "gripper":
        lo, hi = GRIPPER_CLOSED, GRIPPER_OPEN
        if lo > hi:
            lo, hi = hi, lo
    else:
        lo, hi = JOINT_LIMITS[name]
    return lo - 1e-3 <= float(value) <= hi + 1e-3


def _slew(current: float, target: float, rate: float, dt: float) -> float:
    step = float(rate) * dt
    delta = float(target) - float(current)
    if delta > step:
        return float(current) + step
    if delta < -step:
        return float(current) - step
    return float(target)


class AutoCollect(Node):
    def __init__(self, count: int) -> None:
        super().__init__("grab_it_il_auto_collect")
        self._count = max(1, min(15, int(count)))
        self._joints = [0.0] * len(STATE_NAMES)
        self._have = False
        self._status_pub = self.create_publisher(String, IL_STATUS_TOPIC, 10)
        self._cmd_pub = self.create_publisher(String, IL_COMMAND_TOPIC, 10)
        self._arm_pub = None
        self._single_pub = None
        self._pan_ready = False
        self._mode_cli = None
        self._torque_cli = None
        if JointGroupCommand is not None and JointSingleCommand is not None:
            self._arm_pub = self.create_publisher(JointGroupCommand, JOINT_GROUP_TOPIC, 10)
            self._single_pub = self.create_publisher(JointSingleCommand, JOINT_SINGLE_TOPIC, 10)
        if OperatingModes is not None and TorqueEnable is not None:
            self._mode_cli = self.create_client(
                OperatingModes, "/px150_no_elbow/set_operating_modes"
            )
            self._torque_cli = self.create_client(
                TorqueEnable, "/px150_no_elbow/torque_enable"
            )
        self.create_subscription(JointState, JOINT_STATES_TOPIC, self._js_cb, _SENSOR_QOS)
        self._grasp_pub = self.create_publisher(String, GRASP_COMMAND_TOPIC, 10)
        self._grasp_msg: dict | None = None
        self._grasp_n = 0
        self.create_subscription(String, GRASP_STATUS_TOPIC, self._grasp_cb, 10)

    def _grasp_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if isinstance(data, dict):
            self._grasp_msg = data
            self._grasp_n += 1

    def _js_cb(self, msg: JointState) -> None:
        names = list(msg.name)
        pos = list(msg.position)
        for i, joint in enumerate(STATE_NAMES):
            if joint in names:
                self._joints[i] = float(pos[names.index(joint)])
                self._have = True

    def _status(self, message: str) -> None:
        self.get_logger().info(message)
        self._status_pub.publish(String(data=json.dumps({
            "recording": False,
            "message": message,
        })))

    def _il(self, cmd: str) -> None:
        self._cmd_pub.publish(String(data=json.dumps({"cmd": cmd, "source": "auto"})))

    def _wait_joints(self, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if _stop_requested():
                return False
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._have:
                return True
        return False

    def _publish_arm(self, q: list[float]) -> None:
        if self._arm_pub is None or JointGroupCommand is None:
            return
        msg = JointGroupCommand()
        msg.name = "arm"
        msg.cmd = [float(v) for v in q[:4]]
        self._arm_pub.publish(msg)

    def _publish_single(self, name: str, value: float) -> None:
        if self._single_pub is None or JointSingleCommand is None:
            return
        msg = JointSingleCommand()
        msg.name = name
        msg.cmd = float(value)
        self._single_pub.publish(msg)

    def _check_seed(self, actions: np.ndarray, pan0: float) -> str | None:
        if actions.ndim != 2 or actions.shape[1] < 5 or actions.shape[0] < 2:
            return "seed episode is too short"
        if not self._have:
            return "no joint read"
        if any(abs(float(v)) > BAD_JOINT for v in self._joints[:4]):
            return "bad joint read"
        names = ["waist", "shoulder", "wrist_angle", "wrist_rotate", "gripper"]
        for col, name in enumerate(names):
            for value in actions[:, col]:
                if not _in_limits(name, float(value)):
                    return f"seed {name} outside joint limits"
        if float(np.min(actions[:, 1])) <= SHOULDER_HARD:
            return "seed shoulder too folded"
        start_shoulder = float(actions[0, 1])
        if float(np.min(actions[:, 1])) < start_shoulder - SHOULDER_SLACK:
            return "seed folds past its own start shoulder"
        if float(self._joints[1]) - SHOULDER_SLACK > start_shoulder:
            return "returning to the sample would fold the shoulder"
        if not _in_limits("pan", pan0):
            return "sample pan outside joint limits"
        return None

    def _at_sample_start(self, actions: np.ndarray) -> bool:
        for i in range(3):
            if abs(float(actions[0, i]) - float(self._joints[i])) > START_TOL:
                return False
        return True

    def _goto_start(self, actions: np.ndarray, floor: float) -> bool:
        """Rate-limited return to the sample's first pose. Shoulder cannot fold."""
        dt = 1.0 / float(HZ)
        cmd = [float(self._joints[i]) for i in range(4)]
        grip = float(self._joints[4])
        target = [float(actions[0, i]) for i in range(4)]
        target_grip = float(actions[0, 4])
        names = ["waist", "shoulder", "wrist_angle", "wrist_rotate"]
        deadline = time.monotonic() + 25.0
        self._status("returning to the sample start")
        while time.monotonic() < deadline:
            if _stop_requested():
                return False
            rclpy.spin_once(self, timeout_sec=0.0)
            done = True
            for i, name in enumerate(names):
                cmd[i] = _slew(cmd[i], target[i], RATES[name], dt)
                if abs(cmd[i] - target[i]) > 0.02:
                    done = False
                if not _in_limits(name, cmd[i]):
                    self._status(f"{name} command outside joint limits")
                    return False
            if cmd[1] < floor:
                self._status("shoulder fold blocked")
                return False
            grip = _slew(grip, target_grip, 2.0, dt)
            if not _in_limits("gripper", grip):
                self._status("gripper command outside joint limits")
                return False
            self._publish_arm(cmd)
            self._publish_single("gripper", grip)
            if done and self._at_sample_start(actions):
                return True
            time.sleep(dt)
        self._status("could not reach the sample start")
        return False

    def _call_ready(self, client, request, timeout: float = 2.0) -> bool:
        if client is None or not client.wait_for_service(timeout_sec=timeout):
            return False
        fut = client.call_async(request)
        deadline = time.monotonic() + timeout
        while not fut.done() and time.monotonic() < deadline:
            if _stop_requested():
                return False
            rclpy.spin_once(self, timeout_sec=0.05)
        return fut.done()

    def _enable_pan(self) -> bool:
        """Pan torque stays off until position mode is set. The GUI does this for the slider."""
        if self._pan_ready:
            return True
        if OperatingModes is None or TorqueEnable is None:
            self._status("pan mode messages are missing")
            return False
        mode = OperatingModes.Request()
        mode.cmd_type = "single"
        mode.name = "pan"
        mode.mode = "position"
        mode.profile_type = "velocity"
        mode.profile_velocity = 60
        mode.profile_acceleration = 25
        if not self._call_ready(self._mode_cli, mode):
            self._status("pan mode service not ready")
            return False
        torque = TorqueEnable.Request()
        torque.cmd_type = "single"
        torque.name = "pan"
        torque.enable = True
        if not self._call_ready(self._torque_cli, torque):
            self._status("pan torque service not ready")
            return False
        time.sleep(0.3)
        self._pan_ready = True
        return True

    def _slew_pan(self, target: float) -> bool:
        if not self._enable_pan():
            return False
        dt = 1.0 / float(HZ)
        cmd = float(self._joints[5])
        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline:
            if _stop_requested():
                return False
            rclpy.spin_once(self, timeout_sec=0.0)
            cmd = _slew(cmd, target, RATES["pan"], dt)
            if not _in_limits("pan", cmd):
                self._status("pan command outside joint limits")
                return False
            self._publish_single("pan", cmd)
            measured = float(self._joints[5])
            if abs(measured - target) <= PAN_TOL and abs(cmd - target) <= PAN_TOL:
                return True
            time.sleep(dt)
        self._status(
            f"pan did not reach the offset "
            f"(meas={self._joints[5]:.3f} cmd={cmd:.3f} target={target:.3f})"
        )
        return False

    def _replay(self, actions: np.ndarray, pan: float, floor: float) -> bool:
        dt = 1.0 / float(HZ)
        cmd = [float(self._joints[i]) for i in range(4)]
        grip = float(self._joints[4])
        names = ["waist", "shoulder", "wrist_angle", "wrist_rotate"]
        for row in actions:
            if _stop_requested():
                return False
            rclpy.spin_once(self, timeout_sec=0.0)
            for i, name in enumerate(names):
                cmd[i] = _slew(cmd[i], float(row[i]), RATES[name], dt)
                if not _in_limits(name, cmd[i]):
                    self._status(f"{name} command outside joint limits")
                    return False
            if cmd[1] < floor:
                self._status("shoulder fold blocked")
                return False
            grip = _slew(grip, float(row[4]), 2.0, dt)
            if not _in_limits("gripper", grip):
                self._status("gripper command outside joint limits")
                return False
            self._publish_arm(cmd)
            self._publish_single("gripper", grip)
            self._publish_single("pan", pan)
            time.sleep(dt)
        return True

    def _drain_grasp(self) -> int:
        """Drop status left over from the previous take before a new command."""
        for _ in range(30):
            rclpy.spin_once(self, timeout_sec=0.0)
        return int(self._grasp_n)

    def _grasp_cmd(self, payload) -> int:
        mark = self._drain_grasp()
        if isinstance(payload, dict):
            text = json.dumps(payload)
        else:
            text = str(payload)
        self._grasp_pub.publish(String(data=text))
        return mark

    def _wait_grasp(
        self, mark: int, timeout: float, terminal: tuple[str, ...]
    ) -> dict | None:
        """Wait for a status that arrived after `mark`. Older statuses are ignored."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if _stop_requested():
                return None
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._grasp_n <= mark or self._grasp_msg is None:
                continue
            status = str(self._grasp_msg.get("status") or "")
            if status in terminal:
                return dict(self._grasp_msg)
        return None

    def _stamp_pole(self, xy: tuple[float, float]) -> None:
        root = _collection_dir()
        if not root.is_dir():
            return
        eps = sorted(
            p for p in root.iterdir()
            if p.is_dir() and (p / "actions.npy").is_file()
        )
        if not eps or _episode_source(eps[-1]) != "auto":
            return
        np.save(eps[-1] / "pole_xy.npy", np.asarray(xy, dtype=np.float32))

    def _stop_motion(self) -> None:
        self._grasp_cmd("cancel")
        self._il("stop")

    def _return_to_batch_start(self, start_q: list[float]) -> bool:
        """Open the jaws and safely return all four arm joints to batch start."""
        mark = self._grasp_cmd({
            "cmd": "retreat",
            "return_joints": [float(v) for v in start_q],
        })
        result = self._wait_grasp(
            mark, 30.0, ("retreated", "error", "cancelled")
        )
        if _stop_requested() or (
            result is not None
            and str(result.get("status") or "") == "cancelled"
        ):
            return False
        if result is None:
            self._status("return to take start timed out")
            return False
        if str(result.get("status") or "") != "retreated":
            self._status(
                str(result.get("message") or "return to take start failed")
            )
            return False
        self._status(str(result.get("message") or "Returned to take start"))
        return True

    def run(self) -> int:
        if not self._wait_joints():
            self._status("no joint read" if not _stop_requested() else "auto collect stopped")
            return 1
        if any(abs(float(v)) > BAD_JOINT for v in self._joints[:4]):
            self._status("bad joint read")
            return 1
        batch_start = [float(v) for v in self._joints[:4]]
        names = ["waist", "shoulder", "wrist_angle", "wrist_rotate"]
        if any(not _in_limits(name, value) for name, value in zip(names, batch_start)):
            self._status("starting arm pose is outside joint limits")
            return 1
        if batch_start[1] < AUTO_RETURN_SHOULDER_MIN:
            self._status("starting shoulder is too folded for auto collect")
            return 1
        mark = self._grasp_cmd({"cmd": "anchor"})
        anchor_msg = self._wait_grasp(mark, 3.0, ("point", "error"))
        if anchor_msg is not None and str(anchor_msg.get("status") or "") == "error":
            self._status(str(anchor_msg.get("message") or "click the pole first"))
            return 1
        if _stop_requested():
            self._status("auto collect stopped")
            return 0
        if anchor_msg is None:
            self._status("click the pole, then Auto collect")
            return 1
        pole_xy = (float(anchor_msg["x"]), float(anchor_msg["y"]))
        spread = GRASP_SPREAD[: self._count]
        self._status(f"auto collect click × {len(spread)}")
        saved = 0
        for i, (lateral, depth, height, standoff) in enumerate(spread):
            if _stop_requested():
                self._stop_motion()
                self._status("auto collect stopped")
                return 0
            self._status(
                f"auto {i + 1}/{len(spread)} "
                f"lat {lateral:+.2f} dep {depth:+.2f} z {height:+.3f}"
            )
            self._il("start")
            time.sleep(0.35)
            mark = self._grasp_cmd({
                "cmd": "grasp",
                "lateral": lateral,
                "depth": depth,
                "height": height,
                "standoff": standoff,
            })
            done = self._wait_grasp(
                mark, 90.0, ("done", "error", "cancelled", "unreachable")
            )
            if done is None:
                self._stop_motion()
                self._status(
                    "auto collect stopped" if _stop_requested() else "grasp timed out"
                )
                return 0 if _stop_requested() else 1
            status = str(done.get("status") or "")
            message = str(done.get("message") or "")
            if status != "done" or not message.startswith("Grasp complete"):
                self._il("stop")
                if status == "cancelled" or _stop_requested():
                    self._status("auto collect stopped")
                    return 0
                self._status(f"auto {i + 1} discarded ({message})")
                if not self._return_to_batch_start(batch_start):
                    self._stop_motion()
                    self._status(
                        "auto collect stopped — could not return to take start"
                    )
                    return 1
                time.sleep(0.3)
                continue
            self._il("save")
            time.sleep(0.8)
            self._stamp_pole(pole_xy)
            saved += 1
            if _stop_requested():
                self._stop_motion()
                self._status("auto collect stopped")
                return 0
            if not self._return_to_batch_start(batch_start):
                if _stop_requested():
                    self._stop_motion()
                    self._status("auto collect stopped")
                    return 0
                self._stop_motion()
                self._status(
                    "auto collect stopped — could not return to take start"
                )
                return 1
            time.sleep(0.3)
        self._status(f"auto collect saved {saved}")
        return 0


def main() -> int:
    count = 5
    if len(sys.argv) > 1:
        try:
            count = int(sys.argv[1])
        except ValueError:
            count = 5
    STOP_FILE.unlink(missing_ok=True)
    PID_FILE.write_text(str(os.getpid()) + "\n", encoding="utf-8")

    def _on_term(_signum, _frame):
        STOP_FILE.write_text("1\n", encoding="utf-8")

    signal.signal(signal.SIGTERM, _on_term)
    rclpy.init()
    node = AutoCollect(count)
    try:
        return node.run()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
        PID_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
