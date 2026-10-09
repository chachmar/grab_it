#!/usr/bin/env python3
"""Add Free arm / Hold arm under Cancel (idempotent AGX patch)."""
from __future__ import annotations

from pathlib import Path

SRC = Path("/workspaces/isaac_ros-dev/src")
HTML = SRC / "grab_it_gui" / "grab_it_gui" / "gui.html"
NODE = SRC / "grab_it_gui" / "grab_it_gui" / "gui_node.py"

BTN = """  <div id="teach-row">
    <button id="teach-free-btn" type="button">Free arm</button>
  </div>
"""

CSS = """
  #teach-row { display: flex; justify-content: center; }
  #teach-row #teach-free-btn { flex: 0 1 240px; min-width: 160px; width: 50%; }
  #teach-free-btn {
    padding: 8px 8px; border-radius: 8px; border: 1px solid var(--accent);
    background: rgba(88,166,255,.12); color: var(--accent); font-size: .62rem;
    font-family: inherit; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
    cursor: pointer; touch-action: manipulation;
  }
  #teach-free-btn.live { background: var(--accent); color: #0d1117; }
  body.teach-free #stick-arm1, body.teach-free #stick-arm2,
  body.teach-free .joint-slider { opacity: .35; pointer-events: none; }
"""

JS = r"""
let teachFree = false;
function setTeachFree(free){
  teachFree = !!free;
  const btn = document.getElementById('teach-free-btn');
  if (btn) {
    btn.classList.toggle('live', teachFree);
    btn.textContent = teachFree ? 'Hold arm' : 'Free arm';
  }
  document.body.classList.toggle('teach-free', teachFree);
}
document.getElementById('teach-free-btn').addEventListener('pointerdown', (e) => {
  e.preventDefault();
  send({ type: teachFree ? 'arm_hold' : 'arm_free' });
  if (graspStatusEl) {
    graspStatusEl.textContent = teachFree
      ? 'Holding arm at current pose…'
      : 'Arm free — support it by hand, then Hold arm';
  }
});
"""

METHODS = '''
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
        if (not enable) and getattr(self, "_reg_set_cli", None) and self._reg_set_cli.service_is_ready():
            for name in self._arm_servos():
                rr = RegisterValues.Request()
                rr.cmd_type = "single"
                rr.name = name
                rr.reg = "Torque_Enable"
                rr.value = 0
                self._reg_set_cli.call_async(rr)
        self.get_logger().info(
            f"arm torque {'ON' if enable else 'OFF'} servos={self._arm_servos()} svc={ok}"
        )
        return ok

    def _reassert_teach_free(self) -> None:
        if getattr(self, "_teach_free", False):
            self._set_arm_torque(False)

    def _arm_gui_active(self) -> bool:
        return (not getattr(self, "_policy_live", False)) and (not getattr(self, "_teach_free", False))

    def _broadcast_teach(self) -> None:
        self._broadcast(json.dumps({"type": "teach_free", "free": bool(self._teach_free)}))

    def set_arm_teach(self, free: bool) -> None:
        free = bool(free)
        if free:
            self._teach_free = True
            with self._lock:
                self._lx = self._ly = self._rx = self._ry = 0.0
            if not self._set_arm_torque(False):
                self._teach_free = False
            self._broadcast_teach()
            return
        with self._lock:
            if all(j in self._pos for j in JOINT_NAMES):
                self._target = {j: float(self._pos[j]) for j in JOINT_NAMES}
        self._set_arm_torque(True)
        self._teach_free = False
        self._broadcast_teach()

'''

WS = '''                    elif t == "arm_free":
                        self.set_arm_teach(True)
                    elif t == "arm_hold":
                        self.set_arm_teach(False)
'''


def main() -> None:
    html = HTML.read_text(encoding="utf-8")
    node = NODE.read_text(encoding="utf-8")
    if 'id="teach-free-btn"' not in html:
        if "</div>\n\n<!-- Joint readout -->" in html:
            html = html.replace(
                "</div>\n\n<!-- Joint readout -->",
                "</div>\n" + BTN + "\n<!-- Joint readout -->",
                1,
            )
        elif 'id="cancel-grasp-btn"' in html:
            html = html.replace(
                "</div>\n  <div class=\"grasp-title\">Handoff",
                "</div>\n" + BTN + "  <div class=\"grasp-title\">Handoff",
                1,
            )
        else:
            raise SystemExit("html: cannot place Free arm button")
        print("patched html button")
    if "#teach-free-btn" not in html:
        html = html.replace("  #cancel-grasp-btn {", CSS + "  #cancel-grasp-btn {", 1)
        print("patched html css")
    if "function setTeachFree" not in html:
        html = html.replace(
            "      if (msg.type === 'il_status') setIlStatus(msg);\n",
            "      if (msg.type === 'il_status') setIlStatus(msg);\n"
            "      if (msg.type === 'teach_free') setTeachFree(!!msg.free);\n",
            1,
        )
        if "function clickToNorm(" in html:
            html = html.replace("function clickToNorm(", JS + "\nfunction clickToNorm(", 1)
        else:
            html += "\n<script>" + JS + "</script>\n"
        print("patched html js")
    if "_teach_free" not in node:
        if "self._policy_live = False\n" in node:
            node = node.replace(
                "self._policy_live = False\n",
                "self._policy_live = False\n        self._teach_free = False\n",
                1,
            )
        else:
            node = node.replace(
                "self._grasp_pan_locked = False\n",
                "self._grasp_pan_locked = False\n        self._teach_free = False\n",
                1,
            )
        print("patched teach flag")
    if "def set_arm_teach(" not in node:
        needle = "    def _torque_arm_group(self):"
        if needle not in node:
            raise SystemExit("gui_node: no _torque_arm_group")
        node = node.replace(needle, METHODS + needle, 1)
        print("patched teach methods")
    elif "arm torque OFF" not in node:
        old = '''    def _set_arm_torque(self, enable: bool) -> bool:
        if not self._tq_cli.service_is_ready():
            self.get_logger().warn("torque_enable not ready")
            return False
        req = TorqueEnable.Request()
        req.cmd_type = "group"
        req.name = "arm"
        req.enable = bool(enable)
        self._tq_cli.call_async(req)
        self.get_logger().info(f"arm group torque {'enable' if enable else 'disable'}")
        return True
'''
        if old in node:
            node = node.replace(old, METHODS, 1)
            print("upgraded torque-off to per-servo + register write")
        else:
            print("WARN: old teach code present but could not upgrade _set_arm_torque")
    if "self.create_timer(0.5, self._reassert_teach_free)" not in node:
        node = node.replace(
            "self.create_timer(self._dt, self._control_loop)",
            "self.create_timer(self._dt, self._control_loop)\n"
            "        self.create_timer(0.5, self._reassert_teach_free)",
            1,
        )
        print("patched teach reassert timer")
    if 't == "arm_free"' not in node:
        marker = '''                except Exception as e:
                    self.get_logger().warn(f"WS error: {e}")'''
        if marker not in node:
            raise SystemExit("gui_node: no WS error handler")
        node = node.replace(marker, WS + marker, 1)
        print("patched ws")
    if 'not self._policy_live' in node and "_arm_gui_active" in node:
        node = node.replace(
            "if tgt is not None and not self._policy_live:",
            "if tgt is not None and self._arm_gui_active():",
            1,
        )
    HTML.write_text(html, encoding="utf-8")
    NODE.write_text(node, encoding="utf-8")
    import py_compile
    py_compile.compile(str(NODE), doraise=True)
    print("ok teach-free patch, gui_node SYNTAX_OK")
    print("restart gui_node and hard-refresh the browser")


if __name__ == "__main__":
    main()
