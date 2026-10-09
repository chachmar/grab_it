#!/usr/bin/env python3
"""Diagnose + repair IL button wiring in the AGX web GUI (idempotent)."""
from __future__ import annotations

from pathlib import Path

SRC = Path("/workspaces/isaac_ros-dev/src")
HTML = SRC / "grab_it_gui" / "grab_it_gui" / "gui.html"
NODE = SRC / "grab_it_gui" / "grab_it_gui" / "gui_node.py"

WS_IL = '''                    elif t == "il_start":
                        self.publish_il_command("start", source=msg.get("source", "teleop"))
                    elif t == "il_save":
                        self.publish_il_command("save", source=msg.get("source", "teleop"))
                    elif t == "il_discard":
                        self.publish_il_command("discard")
                    elif t == "il_play":
                        self.publish_il_command("play")
                    elif t == "il_stop":
                        self.publish_il_command("stop")
'''

IL_METHODS = '''
    def _il_status_cb(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception:
            data = {"message": msg.data}
        if "playing" in data:
            self._policy_live = bool(data.get("playing"))
        self._broadcast(json.dumps({"type": "il_status", **data}))

    def publish_il_command(self, cmd, **extra):
        payload = {"cmd": (cmd or "").strip().lower()}
        payload.update(extra)
        self.get_logger().info(f"IL cmd {payload}")
        self._il_cmd_pub.publish(String(data=json.dumps(payload)))
        if payload["cmd"] in ("stop", "stop_play"):
            self._policy_live = False
        elif payload["cmd"] in ("play", "start_play"):
            self._policy_live = True

'''

PUBS = '''        self._il_cmd_pub = self.create_publisher(String, "/grab_it/il_command", 10)
        self.create_subscription(String, "/grab_it/il_status", self._il_status_cb, 10)
        self._policy_live = False
'''

WS_EXCEPT = '''                except Exception as e:
                    self.get_logger().warn(f"WS error: {e}")'''


def main() -> None:
    if not HTML.is_file() or not NODE.is_file():
        raise SystemExit(f"missing GUI files:\n  {HTML}\n  {NODE}")
    html = HTML.read_text(encoding="utf-8")
    node = NODE.read_text(encoding="utf-8")
    print("html il-row     ", 'id="il-row"' in html)
    print("html il_start   ", "il_start" in html)
    print("html setIlStatus", "function setIlStatus" in html)
    print("node _il_cmd_pub", "_il_cmd_pub" in node)
    print("node il_start   ", 't == "il_start"' in node or "t == 'il_start'" in node)
    print("node _il_status ", "def _il_status_cb" in node)

    if "_il_cmd_pub" not in node:
        needle = 'String, "/grab_it/grasp_status", self._grasp_status_cb, 10)'
        i = node.find(needle)
        if i < 0:
            raise SystemExit("cannot find grasp_status subscription in gui_node.py")
        nl = node.find("\n", i)
        node = node[: nl + 1] + PUBS + node[nl + 1 :]
        print("patched pubs")
    if "def _il_status_cb" not in node:
        needle = "    def publish_grasp_click("
        if needle not in node:
            raise SystemExit("cannot find publish_grasp_click in gui_node.py")
        node = node.replace(needle, IL_METHODS + needle, 1)
        print("patched methods")
    if 't == "il_start"' not in node and "t == 'il_start'" not in node:
        if WS_EXCEPT not in node:
            raise SystemExit("cannot find WS error handler in gui_node.py")
        node = node.replace(WS_EXCEPT, WS_IL + WS_EXCEPT, 1)
        print("patched ws branches")
    if "IL cmd" not in node:
        old = "        self._il_cmd_pub.publish(String(data=json.dumps(payload)))\n"
        new = '        self.get_logger().info(f"IL cmd {payload}")\n' + old
        if old in node:
            node = node.replace(old, new, 1)
            print("patched IL log")

    old_rec = "  send({type: 'il_start', source: ilSource});"
    new_rec = (
        "  setIlStatus({recording: true, message: 'REC sent — waiting for recorder'});\n"
        + old_rec
    )
    if "REC sent" not in html and old_rec in html:
        html = html.replace(old_rec, new_rec, 1)
        print("patched optimistic REC")

    HTML.write_text(html, encoding="utf-8")
    NODE.write_text(node, encoding="utf-8")
    import py_compile
    py_compile.compile(str(NODE), doraise=True)
    print("ok wrote files, gui_node.py SYNTAX_OK")
    print("restart gui_node, relaunch recorder WITH FastRTPS profile, hard-refresh")


if __name__ == "__main__":
    main()
