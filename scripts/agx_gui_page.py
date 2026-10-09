#!/usr/bin/env python3
"""Serve the existing grab_IT HTML on :8765 without ROS / gui_node.

Leave agx_isaac_hub.py running on :8081. Start this in another shell.
Do not run agx_bringup.sh (it will fight the hub and can kill the page).
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = 8765
HTML_CANDIDATES = [
    Path("/workspaces/isaac_ros-dev/src/grab_it_gui/grab_it_gui/gui.html"),
    Path("/workspaces/isaac_ros-dev/install/grab_it_gui/lib/grab_it_gui/gui.html"),
    Path("/tmp/gui.html"),
]
HOOK = (
    b"<script>"
    b"(function(){var s=document.createElement('script');"
    b"s.src='http://'+location.hostname+':8081/gui_hook.js';"
    b"document.body.appendChild(s);})();"
    b"</script>"
)


def load_html() -> tuple[bytes, str]:
    for path in HTML_CANDIDATES:
        if not path.is_file():
            continue
        data = path.read_bytes()
        if b"gui_hook.js" not in data and b"8081/gui_hook.js" not in data:
            if b"</body>" in data:
                data = data.replace(b"</body>", HOOK + b"\n</body>", 1)
            else:
                data += HOOK
        return data, str(path)
    raise FileNotFoundError(
        "gui.html not found. Checked:\n  " + "\n  ".join(str(p) for p in HTML_CANDIDATES)
    )


class Page(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path not in ("/", "/index.html", ""):
            self.send_error(404)
            return
        try:
            body, _src = load_html()
        except FileNotFoundError as exc:
            body = f"<pre>{exc}</pre>".encode()
            self.send_response(500)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    body, src = load_html()
    httpd = ThreadingHTTPServer(("0.0.0.0", PORT), Page)
    httpd.daemon_threads = True
    print(
        f"agx_gui_page on :{PORT}  ({len(body)} bytes from {src})  "
        f"http://192.168.50.207:{PORT}/",
        flush=True,
    )
    httpd.serve_forever()


if __name__ == "__main__":
    main()
