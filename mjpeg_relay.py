#!/usr/bin/env python3
"""Low-latency MJPEG relay for grab_it (crossover-pinned image consumer).

Run:
  FASTRTPS_DEFAULT_PROFILES_FILE=.../rtabmap_profile.xml python3 mjpeg_relay.py

Streams:
  http://<agx-ip>:8081/d435i
  http://<agx-ip>:8081/wide

Prefers CompressedImage (Nano or Isaac JPEG republish). If compressed is
silent, JPEG-encodes raw sensor_msgs/Image (Isaac RGB without compressed).
"""
from __future__ import annotations

import os
import socket
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

PORT = 8081
ISAAC_JPEG_PORT = int(os.environ.get("GRAB_IT_ISAAC_JPEG_PORT", "8083"))
_ISAAC_STREAM = {0: "/d435i", 1: "/wide"}
# Multipart MJPEG is capped; GUI should use /d435i.jpg + /wide.jpg snapshots.
MAX_FPS = 10.0
COMPRESSED_STALE_S = 1.0
TOPICS = {
    "/wide": "/wide_cam/image_raw/compressed",
    "/d435i": "/d435i/d435i/color/image_raw/compressed",
}
RAW_TOPICS = {
    "/wide": "/wide_cam/image_raw",
    "/d435i": "/d435i/d435i/color/image_raw",
}

# BEST_EFFORT + depth 1: never backpressure Nano if the browser is slow.
# (RELIABLE + MJPEG write = multi-second GUI lag on the crossover.)
_qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

_frames: dict[str, bytes] = {}
_conds = {p: threading.Condition() for p in TOPICS}
_seqs = {p: 0 for p in TOPICS}
_stamp = {p: 0.0 for p in TOPICS}
_compressed_stamp = {p: 0.0 for p in TOPICS}


def _jpeg_from_raw(msg: Image) -> bytes | None:
    if cv2 is None or np is None or not msg.data:
        return None
    h, w = int(msg.height), int(msg.width)
    if h <= 0 or w <= 0:
        return None
    try:
        enc = (msg.encoding or "rgb8").lower()
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        px = max(1, h * w)
        nch = buf.size // px
        step = int(getattr(msg, "step", 0) or 0)
        if nch >= 4 or enc in ("rgba8", "bgra8", "8uc4"):
            ch = 4
        elif nch == 1 or enc in ("mono8", "8uc1"):
            ch = 1
        else:
            ch = 3
        row = step if step >= w * ch else w * ch
        if buf.size < h * row:
            row = w * ch
            if buf.size < h * row:
                return None
        raw = buf[: h * row].reshape(h, row)[:, : w * ch]
        if ch == 4:
            img = cv2.cvtColor(
                raw.reshape(h, w, 4),
                cv2.COLOR_BGRA2BGR if enc.startswith("bgr") else cv2.COLOR_RGBA2BGR,
            )
        elif ch == 1:
            img = raw.reshape(h, w)
        else:
            img = raw.reshape(h, w, 3)
            if enc in ("rgb8", "8uc3") or enc.startswith("rgb"):
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return bytes(jpg) if ok else None
    except Exception:
        return None


class RelayNode(Node):
    def __init__(self):
        super().__init__("mjpeg_relay")
        for path, topic in TOPICS.items():
            self.create_subscription(
                CompressedImage,
                topic,
                lambda msg, p=path: self._cb_compressed(p, msg),
                _qos,
            )
            self.get_logger().info(f"{topic} -> http://0.0.0.0:{PORT}{path}")
        if cv2 is None:
            self.get_logger().warn("OpenCV missing — raw Image fallback disabled")

    def _store(self, path: str, frame: bytes) -> None:
        with _conds[path]:
            _frames[path] = frame
            _seqs[path] += 1
            _stamp[path] = time.monotonic()
            _conds[path].notify_all()

    def _cb_compressed(self, path: str, msg: CompressedImage):
        if not msg.data:
            return
        _compressed_stamp[path] = time.monotonic()
        self._store(path, bytes(msg.data))

    def _cb_raw(self, path: str, msg: Image):
        # Prefer compressed whenever it is ticking (Nano or Isaac republish).
        if (time.monotonic() - _compressed_stamp[path]) < COMPRESSED_STALE_S:
            return
        frame = _jpeg_from_raw(msg)
        if frame:
            self._store(path, frame)


class MJPEGHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        key = {"/push/d435i": "/d435i", "/push/wide": "/wide"}.get(path)
        if key is None:
            self.send_error(404)
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            n = 0
        body = self.rfile.read(n) if n > 0 else b""
        if not body:
            self.send_error(400)
            return
        _store_isaac(key, body)
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        # Latest-frame snapshot — no browser MJPEG pipeline buffer.
        if path.endswith(".jpg"):
            key = path[:-4]
            if key not in TOPICS:
                self.send_error(404)
                return
            frame = _frames.get(key)
            if not frame:
                self.send_error(503)
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(frame)))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(frame)
            return
        if path not in TOPICS:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header(
            "Content-Type", "multipart/x-mixed-replace; boundary=frame"
        )
        self.send_header("Connection", "close")
        self.end_headers()

        cond = _conds[path]
        last_seq = -1
        min_dt = 1.0 / MAX_FPS
        last_sent = 0.0
        try:
            while True:
                with cond:
                    cond.wait_for(lambda: _seqs[path] != last_seq, timeout=2.0)
                    # Jump to newest sequence (drop anything accumulated while writing).
                    frame = _frames.get(path)
                    last_seq = _seqs[path]
                if frame is None:
                    continue
                now = time.monotonic()
                wait = min_dt - (now - last_sent)
                if wait > 0:
                    time.sleep(wait)
                    # Re-read newest after sleep so we never send a stale buffer.
                    with cond:
                        frame = _frames.get(path)
                        last_seq = _seqs[path]
                    if frame is None:
                        continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(
                    f"Content-Length: {len(frame)}\r\n\r\n".encode()
                )
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                try:
                    self.wfile.flush()
                except Exception:
                    pass
                last_sent = time.monotonic()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def log_message(self, *args):
        pass


def _store_isaac(path: str, frame: bytes) -> None:
    # Real-arm GUI: keep the Nano cameras. Isaac's push would replace them.
    if os.environ.get("GRAB_IT_IGNORE_ISAAC", "").strip().lower() in ("1", "true", "yes"):
        return
    if path not in TOPICS or not frame:
        return
    with _conds[path]:
        _frames[path] = frame
        _seqs[path] += 1
        _stamp[path] = time.monotonic()
        _conds[path].notify_all()


def _isaac_jpeg_client(conn: socket.socket, addr) -> None:
    buf = b""
    try:
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                break
            buf += chunk
            while True:
                idx = buf.find(b"GIJ1")
                if idx < 0:
                    buf = buf[-3:]
                    break
                if idx:
                    buf = buf[idx:]
                if len(buf) < 9:
                    break
                sid = buf[4]
                n = struct.unpack(">I", buf[5:9])[0]
                if n <= 0 or n > 2_000_000:
                    buf = buf[4:]
                    continue
                if len(buf) < 9 + n:
                    break
                payload = buf[9 : 9 + n]
                buf = buf[9 + n :]
                path = _ISAAC_STREAM.get(sid)
                if path:
                    _store_isaac(path, payload)
    except Exception:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _isaac_jpeg_server() -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", ISAAC_JPEG_PORT))
    srv.listen(4)
    while True:
        try:
            conn, addr = srv.accept()
        except Exception:
            continue
        threading.Thread(
            target=_isaac_jpeg_client, args=(conn, addr), daemon=True
        ).start()


def main():
    rclpy.init()
    node = RelayNode()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), MJPEGHandler)
    # Avoid slow clients stalling accept() for everyone.
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    threading.Thread(target=_isaac_jpeg_server, daemon=True).start()
    node.get_logger().info(
        f"MJPEG relay up on :{PORT} (max {MAX_FPS:.0f} fps, qos depth=1); "
        f"Isaac JPEG TCP :{ISAAC_JPEG_PORT}"
    )
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
