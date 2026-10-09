#!/usr/bin/env python3
"""Patch AGX grab_it_gui / grasp_from_click with IL REC/SAVE/PLAY controls."""
from __future__ import annotations

from pathlib import Path

SRC = Path("/workspaces/isaac_ros-dev/src")
HTML = SRC / "grab_it_gui" / "grab_it_gui" / "gui.html"
NODE = SRC / "grab_it_gui" / "grab_it_gui" / "gui_node.py"
GRASP = SRC / "grab_it_grasp" / "grab_it_grasp" / "grasp_from_click.py"

CSS = """
  #il-row {
    display: flex; flex-direction: column; align-items: stretch; gap: 8px;
    width: 100%; max-width: 640px;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 10px 14px;
  }
  #il-row .grasp-title {
    font-size: .6rem; color: var(--muted); letter-spacing: .08em; text-transform: uppercase;
  }
  #il-btn-row { display: flex; gap: 8px; flex-wrap: wrap; }
  #il-btn-row > button { flex: 1 1 0; min-width: 0; }
  #il-rec-btn, #il-save-btn, #il-discard-btn, #il-play-btn, #il-stop-btn {
    padding: 8px 8px; border-radius: 8px; border: 1px solid var(--border);
    background: var(--surface); color: var(--muted); font-size: .62rem;
    font-family: inherit; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
    cursor: pointer; touch-action: manipulation;
  }
  #il-rec-btn { border-color: #f85149; color: #f85149; background: rgba(248,81,73,.10); }
  #il-rec-btn.live { background: #f85149; color: #0d1117; }
  #il-save-btn { border-color: var(--accent2); color: var(--accent2); background: rgba(63,185,80,.12); }
  #il-discard-btn, #il-stop-btn { border-color: var(--warn); color: var(--warn); background: rgba(247,129,102,.10); }
  #il-play-btn { border-color: var(--accent); color: var(--accent); background: rgba(88,166,255,.12); }
  #il-play-btn.live { background: rgba(88,166,255,.35); }
  body.il-playing #stick-arm1, body.il-playing #stick-arm2 { opacity: .35; pointer-events: none; }
"""

ROW = """
<div id="il-row">
  <div class="grasp-title">Imitation learning · tabletop pick</div>
  <div id="il-btn-row">
    <button id="il-rec-btn" type="button">REC</button>
    <button id="il-save-btn" type="button">SAVE</button>
    <button id="il-discard-btn" type="button">DISCARD</button>
    <button id="il-play-btn" type="button">PLAY</button>
    <button id="il-stop-btn" type="button">STOP</button>
  </div>
  <div id="il-status">Recorder idle — start record_demos / play_policy</div>
</div>
"""

JS = r"""
const ilRecBtn = document.getElementById('il-rec-btn');
const ilPlayBtn = document.getElementById('il-play-btn');
const ilStatusEl = document.getElementById('il-status');
let ilRecording = false;
let ilPlaying = false;
let ilSource = 'teleop';

function setIlStatus(msg){
  if (!ilStatusEl) return;
  if (msg.recording != null) ilRecording = !!msg.recording;
  if (msg.playing != null) ilPlaying = !!msg.playing;
  if (ilRecBtn) ilRecBtn.classList.toggle('live', ilRecording);
  if (ilPlayBtn) ilPlayBtn.classList.toggle('live', ilPlaying);
  document.body.classList.toggle('il-playing', ilPlaying);
  const bits = [];
  if (msg.message) bits.push(msg.message);
  if (msg.episodes != null) bits.push(msg.episodes + ' demos');
  if (msg.frames != null && ilRecording) bits.push(msg.frames + ' frames');
  if (msg.kind) bits.push(msg.kind);
  if (msg.offer_oracle) bits.push('oracle');
  ilStatusEl.textContent = bits.join(' · ') || (ilRecording ? 'recording' : 'idle');
}

document.getElementById('il-rec-btn').addEventListener('click', () => {
  ilSource = 'teleop';
  send({type: 'il_start', source: ilSource});
});
document.getElementById('il-save-btn').addEventListener('click', () => {
  send({type: 'il_save', source: ilSource});
});
document.getElementById('il-discard-btn').addEventListener('click', () => {
  send({type: 'il_discard'});
});
document.getElementById('il-play-btn').addEventListener('click', () => {
  send({type: 'il_play'});
});
document.getElementById('il-stop-btn').addEventListener('click', () => {
  send({type: 'il_stop'});
});
"""

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
        self._il_cmd_pub.publish(String(data=json.dumps(payload)))
        if payload["cmd"] in ("stop", "stop_play"):
            self._policy_live = False
        elif payload["cmd"] in ("play", "start_play"):
            self._policy_live = True

'''

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


def _require(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    return path.read_text(encoding="utf-8")


def _once(text: str, needle: str, insert: str, label: str) -> str:
    if needle in text:
        return text
    if insert not in text:
        raise SystemExit(f"{label}: anchor not found:\n  {insert!r}")
    return text.replace(insert, insert + needle, 1)


def _insert_before(text: str, anchor: str, insert: str, label: str, already: str) -> str:
    if already in text:
        return text
    if anchor not in text:
        raise SystemExit(f"{label}: anchor not found:\n  {anchor!r}")
    return text.replace(anchor, insert + anchor, 1)


def patch_html(text: str) -> str:
    text = _insert_before(text, "  #pose-row {", CSS, "gui.html CSS", "#il-row {")
    text = _insert_before(
        text, "<!-- Joint readout -->", ROW + "\n", "gui.html row", 'id="il-row"'
    )
    text = _once(
        text,
        "      if (msg.type === 'il_status') setIlStatus(msg);\n",
        "      if (msg.type === 'grasp_status') setGraspStatus(msg);\n",
        "gui.html ws",
    )
    oracle = (
        "  if (st === 'done' && ilRecording) {\n"
        "    ilSource = 'oracle';\n"
        "    setIlStatus({recording: true, offer_oracle: true, "
        "message: 'Oracle grasp done — SAVE to keep'});\n"
        "  }\n"
    )
    if "offer_oracle" not in text:
        if "function setGraspStatus(msg)" in text:
            # append inside setGraspStatus, just before its final closing brace
            marker = None
            idx = text.find("function setGraspStatus(msg)")
            if idx < 0:
                raise SystemExit("gui.html: setGraspStatus missing")
            # find the function's closing "}\n\n" after the body
            rest = text[idx:]
            end = rest.find("\n}\n\n")
            if end < 0:
                raise SystemExit("gui.html: cannot find end of setGraspStatus")
            end += idx + 1  # points at '}'
            text = text[:end] + oracle + text[end:]
    text = _insert_before(text, "function clickToNorm(", JS + "\n", "gui.html JS", "function setIlStatus(")
    return text


def patch_node(text: str) -> str:
    pubs = (
        "        self._il_cmd_pub = self.create_publisher(String, \"/grab_it/il_command\", 10)\n"
        "        self.create_subscription(String, \"/grab_it/il_status\", self._il_status_cb, 10)\n"
        "        self._policy_live = False\n"
    )
    text = _once(
        text,
        pubs,
        '            String, "/grab_it/grasp_status", self._grasp_status_cb, 10)\n',
        "gui_node IL pubs",
    )
    if "def _il_status_cb(" not in text:
        # insert after _grasp_status_cb's broadcast line / method
        anchor = '        self._broadcast(json.dumps({"type": "grasp_status", **data}))\n'
        if anchor not in text:
            raise SystemExit("gui_node: grasp_status broadcast anchor missing")
        text = text.replace(anchor, anchor + "\n" + IL_METHODS, 1)
    if 't == "il_start"' not in text:
        for key in ('t == "handoff_release"', "t == 'handoff_release'"):
            if key in text:
                # insert after that branch's next line (publish call)
                i = text.find(key)
                nl = text.find("\n", i)
                nl2 = text.find("\n", nl + 1)
                text = text[: nl2 + 1] + WS_IL + text[nl2 + 1 :]
                break
        else:
            raise SystemExit("gui_node: handoff_release WS branch missing")
    old = "        if tgt is not None:"
    new = "        if tgt is not None and not getattr(self, \"_policy_live\", False):"
    if old in text and "not getattr(self, \"_policy_live\"" not in text and "not self._policy_live" not in text:
        text = text.replace(old, new, 1)
    guard = "        if getattr(self, \"_policy_live\", False):\n            return\n"
    for sig in (
        "    def set_joint_degrees(self, joint, degrees):",
        "    def set_joint_angle(self, joint, frac):",
        "    def set_arm_sticks(self, lx, ly, rx, ry):",
        "    def set_gripper(self, value):",
        "    def go_home(self):",
        "    def go_sleep(self):",
    ):
        if sig not in text:
            continue
        i = text.find(sig)
        nl = text.find("\n", i) + 1
        # skip existing docstring
        if text[nl : nl + 3] == '        """':
            end = text.find('"""', nl + 3)
            if end > 0:
                nl = text.find("\n", end + 3) + 1
        window = text[nl : nl + 80]
        if "_policy_live" in window:
            continue
        text = text[:nl] + guard + text[nl:]
    return text


def patch_grasp(text: str) -> str:
    if 'payload["oracle"]' in text or "payload['oracle']" in text:
        return text
    old = '        payload = {"status": status, "message": message}\n'
    new = (
        '        payload = {"status": status, "message": message}\n'
        '        if status == "done":\n'
        '            payload["oracle"] = True\n'
    )
    if old not in text:
        raise SystemExit("grasp_from_click: status payload anchor missing")
    return text.replace(old, new, 1)


def main() -> None:
    html = patch_html(_require(HTML))
    node = patch_node(_require(NODE))
    grasp = patch_grasp(_require(GRASP))
    HTML.write_text(html, encoding="utf-8")
    NODE.write_text(node, encoding="utf-8")
    GRASP.write_text(grasp, encoding="utf-8")
    print(f"ok patched {HTML}")
    print(f"ok patched {NODE}")
    print(f"ok patched {GRASP}")
    print("restart gui_node, then hard-refresh http://192.168.50.207:8765")


if __name__ == "__main__":
    main()
