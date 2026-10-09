#!/usr/bin/env python3
"""Record tabletop-pick demonstrations from GUI teleop or click-to-grasp."""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image, JointState
from std_msgs.msg import String

from grab_it_il.common import (
    ARM_JOINTS,
    D435I_COMPRESSED,
    D435I_RAW,
    DEFAULT_DATASET,
    GRASP_STATUS_TOPIC,
    HZ,
    IL_COMMAND_TOPIC,
    IL_STATUS_TOPIC,
    JOINT_GROUP_TOPIC,
    JOINT_SINGLE_TOPIC,
    JOINT_STATES_TOPIC,
    MIN_EPISODE_S,
    STATE_NAMES,
    TASK,
    WRIST_COMPRESSED,
    WRIST_RAW,
    count_episodes,
    next_episode_dir,
    parse_json_cmd,
)

try:
    from interbotix_xs_msgs.msg import JointGroupCommand, JointSingleCommand
except ImportError:  # pragma: no cover
    JointGroupCommand = None
    JointSingleCommand = None

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


_SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


def _jpeg_from_compressed(msg: CompressedImage) -> Optional[bytes]:
    if not msg.data:
        return None
    return bytes(msg.data)


def _jpeg_from_raw(msg: Image) -> Optional[bytes]:
    if cv2 is None or not msg.data:
        return None
    h, w = int(msg.height), int(msg.width)
    if h <= 0 or w <= 0:
        return None
    enc = (msg.encoding or "rgb8").lower()
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if enc in ("rgb8", "bgr8"):
        img = buf.reshape(h, w, 3)
        if enc == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif enc in ("mono8", "8uc1"):
        img = buf.reshape(h, w)
    else:
        return None
    ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return bytes(jpg) if ok else None


def _write_jpeg_dir(dest: Path, frames: list[bytes]) -> bool:
    if not frames:
        return False
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, blob in enumerate(frames):
        if not blob:
            continue
        (dest / f"{i:06d}.jpg").write_bytes(blob)
        n += 1
    return n > 0


def _write_mp4_sync(path: Path, frames: list[bytes], fps: int) -> bool:
    if cv2 is None or not frames:
        return False
    first = cv2.imdecode(np.frombuffer(frames[0], dtype=np.uint8), cv2.IMREAD_COLOR)
    if first is None:
        return False
    h, w = first.shape[:2]
    # Even size; downscale so Jetson encode stays short.
    scale = min(1.0, 320.0 / max(w, 1), 240.0 / max(h, 1))
    tw = max(2, int(w * scale) // 2 * 2)
    th = max(2, int(h * scale) // 2 * 2)
    attempts = (
        (path.with_suffix(".mp4"), "mp4v"),
        (path.with_suffix(".avi"), "MJPG"),
    )
    for out, fourcc in attempts:
        writer = None
        try:
            writer = cv2.VideoWriter(
                str(out), cv2.VideoWriter_fourcc(*fourcc), float(fps), (tw, th)
            )
            if not writer.isOpened():
                continue
            for blob in frames:
                img = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    img = first
                if img.shape[1] != tw or img.shape[0] != th:
                    img = cv2.resize(img, (tw, th))
                writer.write(img)
            writer.release()
            writer = None
            if out.is_file() and out.stat().st_size > 0:
                return True
        except Exception:
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass
    return False


def _write_mp4(path: Path, frames: list[bytes], fps: int, timeout_s: float = 20.0) -> bool:
    if os.environ.get("GRAB_IT_IL_SKIP_VIDEO", "").lower() in ("1", "true", "yes"):
        return False
    if not frames:
        return False
    done = threading.Event()
    ok = [False]

    def _run() -> None:
        try:
            ok[0] = _write_mp4_sync(path, frames, fps)
        except Exception:
            ok[0] = False
        finally:
            done.set()

    threading.Thread(target=_run, name="il-mp4", daemon=True).start()
    if not done.wait(timeout_s):
        return False
    return bool(ok[0])


class RecordDemos(Node):
    def __init__(self) -> None:
        super().__init__("grab_it_il_record")
        self.declare_parameter("dataset_dir", str(DEFAULT_DATASET))
        self.declare_parameter("source", "teleop")
        self._root = Path(self.get_parameter("dataset_dir").value)
        self._default_source = str(self.get_parameter("source").value or "teleop")
        self._lock = threading.Lock()
        self._recording = False
        self._t0 = 0.0
        self._source = self._default_source
        self._states: list[list[float]] = []
        self._actions: list[list[float]] = []
        self._d435i: list[bytes] = []
        self._wrist: list[bytes] = []
        self._state = [0.0] * len(STATE_NAMES)
        self._action = [0.0] * len(STATE_NAMES)
        self._have_state = False
        self._have_arm_cmd = False
        self._latest_d435i: Optional[bytes] = None
        self._latest_wrist: Optional[bytes] = None

        self._status_pub = self.create_publisher(String, IL_STATUS_TOPIC, 10)
        self.create_subscription(String, IL_COMMAND_TOPIC, self._cmd_cb, 10)
        self.create_subscription(String, GRASP_STATUS_TOPIC, self._grasp_cb, 10)
        self.create_subscription(
            JointState, JOINT_STATES_TOPIC, self._js_cb, _SENSOR_QOS
        )
        if JointGroupCommand is not None:
            self.create_subscription(
                JointGroupCommand, JOINT_GROUP_TOPIC, self._group_cb, 10
            )
        if JointSingleCommand is not None:
            self.create_subscription(
                JointSingleCommand, JOINT_SINGLE_TOPIC, self._single_cb, 10
            )
        self.create_subscription(
            CompressedImage, D435I_COMPRESSED, self._d435i_jpg_cb, _SENSOR_QOS
        )
        self.create_subscription(
            CompressedImage, WRIST_COMPRESSED, self._wrist_jpg_cb, _SENSOR_QOS
        )
        self.create_timer(1.0 / HZ, self._tick)
        self.create_timer(1.0, self._heartbeat)
        self._publish_status(recording=False, message="recorder idle")
        self.get_logger().info(f"IL recorder → {self._root}")

    def _publish_status(self, **extra) -> None:
        payload = {
            "recording": self._recording,
            "episodes": count_episodes(self._root),
            "source": self._source,
            "frames": len(self._states),
        }
        payload.update(extra)
        self._status_pub.publish(String(data=json.dumps(payload)))

    def _cmd_cb(self, msg: String) -> None:
        data = parse_json_cmd(msg.data)
        cmd = str(data.get("cmd") or data.get("type") or "").strip().lower()
        if cmd.startswith("il_"):
            cmd = cmd[3:]
        source = str(data.get("source") or self._default_source)
        self.get_logger().info(f"IL command {cmd!r}")
        if cmd in ("start", "rec", "record"):
            self._start(source)
        elif cmd in ("save", "success"):
            self._save(success=True, source=str(data.get("source") or self._source))
        elif cmd in ("discard", "cancel"):
            self._discard()
        elif cmd in ("stop",):
            if self._recording:
                self._discard()

    def _grasp_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if str(data.get("status", "")).lower() != "done":
            return
        with self._lock:
            if not self._recording:
                return
            self._source = "oracle"
        self._publish_status(
            recording=True,
            offer_oracle=True,
            message="Oracle grasp done — SAVE to keep",
        )

    def _js_cb(self, msg: JointState) -> None:
        names = list(msg.name)
        pos = list(msg.position)
        changed = False
        for i, joint in enumerate(STATE_NAMES):
            if joint in names:
                self._state[i] = float(pos[names.index(joint)])
                changed = True
        if changed:
            self._have_state = True
            if not self._have_arm_cmd:
                self._action = list(self._state)

    def _group_cb(self, msg) -> None:
        if (msg.name or "") != "arm" or len(msg.cmd) < 4:
            return
        for i, _ in enumerate(ARM_JOINTS):
            self._action[i] = float(msg.cmd[i])
        self._have_arm_cmd = True

    def _single_cb(self, msg) -> None:
        if (msg.name or "") != "gripper":
            return
        self._action[4] = float(msg.cmd)

    def _d435i_jpg_cb(self, msg: CompressedImage) -> None:
        blob = _jpeg_from_compressed(msg)
        if blob:
            self._latest_d435i = blob

    def _d435i_raw_cb(self, msg: Image) -> None:
        if self._latest_d435i is not None:
            return
        blob = _jpeg_from_raw(msg)
        if blob:
            self._latest_d435i = blob

    def _wrist_jpg_cb(self, msg: CompressedImage) -> None:
        blob = _jpeg_from_compressed(msg)
        if blob:
            self._latest_wrist = blob

    def _wrist_raw_cb(self, msg: Image) -> None:
        if self._latest_wrist is not None:
            return
        blob = _jpeg_from_raw(msg)
        if blob:
            self._latest_wrist = blob

    def _heartbeat(self) -> None:
        if not self._recording:
            self._publish_status(recording=False, message="recorder idle")

    def _tick(self) -> None:
        with self._lock:
            if not self._recording:
                return
            self._states.append(list(self._state))
            self._actions.append(list(self._action))
            if self._latest_d435i:
                self._d435i.append(self._latest_d435i)
            if self._latest_wrist:
                self._wrist.append(self._latest_wrist)
            n = len(self._states)
        if n > 0 and n % HZ == 0:
            self._publish_status(recording=True, message=f"recording {n / HZ:.1f}s")

    def _start(self, source: str) -> None:
        with self._lock:
            if self._recording:
                self._publish_status(recording=True, message="already recording")
                return
            self._recording = True
            self._t0 = time.monotonic()
            self._source = source or self._default_source
            self._states = []
            self._actions = []
            self._d435i = []
            self._wrist = []
        self.get_logger().info(f"REC start source={self._source}")
        self._publish_status(recording=True, message="recording")

    def _discard(self) -> None:
        with self._lock:
            self._recording = False
            n = len(self._states)
            self._states = []
            self._actions = []
            self._d435i = []
            self._wrist = []
        self.get_logger().info(f"REC discarded ({n} frames)")
        self._publish_status(recording=False, message="discarded")

    def _save(self, success: bool, source: str) -> None:
        with self._lock:
            if not self._recording:
                self.get_logger().warn("SAVE ignored — not recording")
                self._publish_status(recording=False, message="not recording")
                return
            self._recording = False
            states = np.asarray(self._states, dtype=np.float32)
            actions = np.asarray(self._actions, dtype=np.float32)
            d435i = list(self._d435i)
            wrist = list(self._wrist)
            t0 = self._t0
            src = source or self._source
            self._states = []
            self._actions = []
            self._d435i = []
            self._wrist = []
        duration = time.monotonic() - t0
        n = int(states.shape[0]) if states.size else 0
        min_frames = int(MIN_EPISODE_S * HZ)
        if n < 1 or (n < min_frames and duration < MIN_EPISODE_S):
            self.get_logger().warn(
                f"episode too short frames={n} ({duration:.2f}s) — not saved"
            )
            self._publish_status(
                recording=False,
                message=f"too short ({duration:.1f}s) — record >1s then SAVE",
            )
            return
        self._publish_status(recording=False, message="saving…")
        try:
            dest = next_episode_dir(self._root)
            dest.mkdir(parents=True, exist_ok=True)
            np.save(dest / "states.npy", states)
            np.save(dest / "actions.npy", actions)
            videos = []
            if _write_jpeg_dir(dest / "d435i", d435i):
                videos.append("d435i/")
            if _write_jpeg_dir(dest / "wrist", wrist):
                videos.append("wrist/")
            if _write_mp4(dest / "d435i.mp4", d435i, HZ):
                videos.append("d435i.mp4")
            if _write_mp4(dest / "wrist.mp4", wrist, HZ):
                videos.append("wrist.mp4")
            meta = {
                "task": TASK,
                "hz": HZ,
                "success": bool(success),
                "source": src,
                "world": _episode_world(),
                "duration": round(duration, 3),
                "frames": n,
                "state_names": list(STATE_NAMES),
                "videos": videos,
            }
            (dest / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception as exc:
            self.get_logger().error(f"SAVE failed: {exc}")
            self._publish_status(recording=False, message=f"save failed: {exc}")
            return
        self.get_logger().info(
            f"saved {dest.name} frames={n} source={src} world={meta['world']} videos={videos}"
        )
        self._publish_status(
            recording=False,
            saved=dest.name,
            message=f"saved {dest.name}",
        )


def _episode_world() -> str:
    """sim or real, from the mode file written by grabit_mode.sh."""
    try:
        text = Path("/tmp/grabit_mode").read_text(encoding="utf-8").strip().lower()
    except OSError:
        text = ""
    if text in ("sim", "real"):
        return text
    if os.environ.get("GRAB_IT_SIM", "").strip().lower() in ("1", "true", "yes"):
        return "sim"
    return "real"


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RecordDemos()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
