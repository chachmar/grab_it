#!/usr/bin/env python3
"""No-ROS camera + command hub for Isaac Sim (Jazzy) ↔ AGX GUI (Humble).

The AGX copy of this repo is stale. Run THIS file on the AGX so the existing
GUI page (http://192.168.50.207:8765) can show Isaac cameras and drive the twin.

Cameras: Isaac POSTs JPEGs here; the browser GETs :8081/*.jpg.

Commands: the old GUI send() only talks WebSocket → Humble ROS, which Jazzy
Isaac never sees. This hub accepts the same GUI JSON (type=arm/home/drive/…)
on POST /isaac/cmd, integrates sticks like gui_node, and serves GET /isaac/cmd
for Isaac.

Click-to-grasp (twin): browser POSTs grasp_click/execute/cancel; Isaac polls
them via GET /isaac/cmd["grasp"], resolves depth locally, POSTs /isaac/grasp_status;
browser polls GET /isaac/grasp_status so the UI leaves "Resolving depth…".

On the AGX container (second shell, leave running):
  kill $(pgrep -f mjpeg_relay.py) 2>/dev/null || true
  kill $(pgrep -f agx_isaac_hub.py) 2>/dev/null || true
  python3 /tmp/agx_isaac_hub.py
  # or: python3 /workspaces/isaac_ros-dev/scripts/agx_isaac_hub.py
"""
from __future__ import annotations

import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = 8081
GUI_PORT = 8765
HTML_CANDIDATES = [
    Path("/workspaces/isaac_ros-dev/src/grab_it_gui/grab_it_gui/gui.html"),
    Path("/workspaces/isaac_ros-dev/install/grab_it_gui/lib/grab_it_gui/gui.html"),
    Path.home() / "dev-ws/src/grab_it_gui/grab_it_gui/gui.html",
    Path.home() / "dev-ws/install/grab_it_gui/lib/grab_it_gui/gui.html",
    Path("/tmp/gui.html"),
]
_PAGE_HOOK = (
    b"<script>"
    b"(function(){var s=document.createElement('script');"
    b"s.src='http://'+location.hostname+':8081/gui_hook.js';"
    b"document.body.appendChild(s);})();"
    b"</script>"
)
DT = 0.05
JOINT_NAMES = ["waist", "shoulder", "wrist_angle", "wrist_rotate"]
ALL_NAMES = JOINT_NAMES + ["gripper", "pan"]
JOINT_LIMITS = {
    "waist": (-2.0246, 2.0944),
    # Match Interbotix / URDF mechanical stops (was -0.26..1.66 and blocked aft fold).
    "shoulder": (-1.9722, 1.9373),
    "wrist_angle": (-2.0769, 1.4486),
    "wrist_rotate": (-1.5708, 1.5708),
}
SPEED = {"waist": 1.2, "shoulder": 0.8, "wrist_angle": 1.0, "wrist_rotate": 1.5}
HOME = [ -0.55, -1.90, 1.40, 0.0]  # twin lookreach PARK (waist right + tip-down)
SLEEP = [0.0, -0.2321, -1.7279, 0.0]
GRIP_CLOSED = -0.95
GRIP_OPEN = 1.65
PAN_MIN = -2.0944
PAN_MAX = 1.6406
MAX_LIN = 0.30
MAX_ANG = 1.40
GRASP_KINDS = frozenset(
    {
        "grasp_click",
        "grasp_execute",
        "grasp_cancel",
        "grasp_unlock_pan",
        "handoff_grab",
        "handoff_release",
        "grab_object",
        "release_object",
    }
)

_frames: dict[str, bytes] = {}
_pos = {
    "waist": HOME[0],
    "shoulder": HOME[1],
    "wrist_angle": HOME[2],
    "wrist_rotate": HOME[3],
    "gripper": GRIP_OPEN,
    "pan": 0.0,
}


def _load_hold_pose() -> None:
    """Keep the live arm pose across a hub restart, and drop the close-gate."""
    global _clear_gate
    path = Path("/tmp/il_hub_hold.json")
    if not path.is_file():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        for name in ALL_NAMES:
            if name in data:
                _pos[name] = float(data[name])
        _clear_gate = True
        pose = {n: round(_pos[n], 3) for n in ALL_NAMES}
        print(f"agx_isaac_hub: held arm pose {pose}", flush=True)
    except Exception as exc:
        print(f"agx_isaac_hub: hold pose ignored: {exc}", flush=True)
    try:
        path.unlink()
    except Exception:
        pass
_vel = {"lx": 0.0, "az": 0.0}
_seq = 0
_lock = threading.Lock()
_gui_ok = False
# Close command the fingers cannot reach. Same test as play_policy.
_grip_stall_t0: float | None = None

# Twin click-to-grasp relay (Isaac resolves; browser polls status)
_grasp_evt_seq = 0
_grasp_evt: dict | None = None
_grasp_status: dict = {
    "type": "grasp_status",
    "status": "idle",
    "message": "Ready — click D435i (Isaac twin)",
}
_grasp_status_seq = 0
_grasp_plan_t0 = 0.0
GRASP_PLAN_TIMEOUT_S = 6.0
_twin: dict = {"type": "twin", "ok": False}

# Served to the browser — posts cmds + polls grasp_status (unsticks Resolving depth)
_GUI_HOOK_JS = r"""
(function () {
  if (window._isaacSendHook) return;
  window._isaacSendHook = true;
  var host = location.hostname;
  var base = "http://" + host + ":8081";
  var lastStatusSeq = -1;
  var clickWatch = null;

  function applyStatus(msg) {
    if (!msg || typeof msg !== "object") return;
    if (typeof window.setGraspStatus === "function") {
      window.setGraspStatus(msg);
      return;
    }
    var el = document.getElementById("grasp-status");
    if (el && (msg.message || msg.status)) {
      el.textContent = msg.message || msg.status;
    }
    if (msg.status === "point" || msg.status === "done") {
      var btn = document.getElementById("grasp-btn");
      if (btn) btn.disabled = false;
    }
  }

  function armClickTimeout(clickId) {
    if (clickWatch) clearTimeout(clickWatch);
    clickWatch = setTimeout(function () {
      clickWatch = null;
      applyStatus({
        type: "grasp_status",
        status: "error",
        message: "Sim: no depth reply — is Isaac running / hub updated?",
        click_id: clickId,
      });
    }, 6000);
  }

  var orig = window.send;
  window.send = function (m) {
    m = m || {};
    try {
      fetch(base + "/isaac/cmd", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(m),
      }).catch(function () {});
    } catch (e) {}
    var t = m.type || m.t || "";
    if (t === "grasp_click") {
      armClickTimeout(m.click_id);
    }
    if (orig) return orig(m);
  };

  setInterval(function () {
    fetch(base + "/isaac/grasp_status", { cache: "no-store" })
      .then(function (r) {
        return r.ok ? r.json() : null;
      })
      .then(function (msg) {
        if (!msg) return;
        var seq = msg._seq != null ? Number(msg._seq) : -1;
        if (seq === lastStatusSeq && seq >= 0) return;
        if (seq >= 0) lastStatusSeq = seq;
        applyStatus(msg);
        var st = msg.status || "";
        if (
          st === "point" ||
          st === "error" ||
          st === "unreachable" ||
          st === "done" ||
          st === "cancelled"
        ) {
          if (clickWatch) {
            clearTimeout(clickWatch);
            clickWatch = null;
          }
        }
      })
      .catch(function () {});
  }, 100);
})();
"""


def _gui_html() -> bytes | None:
    for path in HTML_CANDIDATES:
        if not path.is_file():
            continue
        data = path.read_bytes()
        # Always inject poller unless this page already loads gui_hook.js.
        # (gui.html already POSTs to :8081/isaac/cmd — that must NOT skip the hook.)
        if b"8081/gui_hook.js" not in data and b"/gui_hook.js" not in data:
            if b"</body>" in data:
                data = data.replace(b"</body>", _PAGE_HOOK + b"\n</body>", 1)
            else:
                data += _PAGE_HOOK
        return data
    return None


def _dz(v: float) -> float:
    return v if abs(v) > 0.05 else 0.0


def _clamp_joint(name: str, value: float) -> float:
    lo, hi = JOINT_LIMITS[name]
    return max(lo, min(hi, float(value)))


def _snapshot() -> dict:
    out = {
        "t": "arm",
        "n": list(ALL_NAMES),
        "p": [float(_pos[n]) for n in ALL_NAMES],
        "lx": float(_vel["lx"]),
        "az": float(_vel["az"]),
        "seq": int(_seq),
    }
    if _clear_gate:
        out["clear_gate"] = True
    if _grasp_evt is not None:
        out["grasp"] = dict(_grasp_evt)
    out["grasp_status"] = dict(_grasp_status)
    out["grasp_status_seq"] = int(_grasp_status_seq)
    return out


def _bump() -> None:
    global _seq
    _seq += 1


def _set_grasp_status(payload: dict) -> None:
    global _grasp_status, _grasp_status_seq, _grasp_plan_t0
    data = dict(payload)
    data.setdefault("type", "grasp_status")
    _grasp_status_seq += 1
    data["_seq"] = int(_grasp_status_seq)
    _grasp_status = data
    if str(data.get("status") or "") == "planning":
        _grasp_plan_t0 = time.monotonic()
    else:
        _grasp_plan_t0 = 0.0


def _maybe_timeout_grasp_plan() -> None:
    """Clear stuck Resolving depth… if Isaac never POSTs a result."""
    global _grasp_plan_t0
    if not _grasp_plan_t0:
        return
    if str(_grasp_status.get("status") or "") != "planning":
        _grasp_plan_t0 = 0.0
        return
    if time.monotonic() - _grasp_plan_t0 < GRASP_PLAN_TIMEOUT_S:
        return
    _set_grasp_status(
        {
            "status": "error",
            "message": "Sim: no depth reply — fully quit/relaunch Isaac",
            "click_id": _grasp_status.get("click_id"),
        }
    )


def _queue_grasp_evt(kind: str, msg: dict) -> None:
    global _grasp_evt_seq, _grasp_evt
    _grasp_evt_seq += 1
    evt = {"evt": int(_grasp_evt_seq), "type": kind, "msg": dict(msg)}
    _grasp_evt = evt
    if kind == "grasp_click":
        cid = msg.get("click_id")
        _set_grasp_status(
            {
                "status": "planning",
                "message": "Resolving depth…",
                "click_id": cid,
            }
        )
    elif kind == "grasp_execute":
        _set_grasp_status({"status": "executing", "message": "Executing grasp…"})
    elif kind == "grasp_cancel":
        _set_grasp_status({"status": "cancelled", "message": "Cancel requested…"})
    elif kind == "grasp_unlock_pan":
        _set_grasp_status(
            {
                "status": _grasp_status.get("status") or "idle",
                "message": "Pan unlock",
                "click_id": _grasp_status.get("click_id"),
            }
        )
    elif kind in ("handoff_grab", "grab_object"):
        _set_grasp_status(
            {
                "status": "grasping",
                "message": "Grab: snapping nearest pole into jaws…",
            }
        )
    elif kind in ("handoff_release", "release_object"):
        _set_grasp_status({"status": "idle", "message": "Release requested…"})


def _merge_np(names, pos) -> bool:
    changed = False
    for name, raw in zip(names, pos):
        key = str(name).split("/")[-1]
        if key not in _pos:
            continue
        val = float(raw)
        if key in JOINT_LIMITS:
            val = _clamp_joint(key, val)
        elif key == "gripper":
            val = max(GRIP_CLOSED - 0.05, min(GRIP_OPEN + 0.05, val))
        elif key == "pan":
            val = max(PAN_MIN, min(PAN_MAX, val))
        if abs(_pos[key] - val) > 1e-6:
            _pos[key] = val
            changed = True
    return changed


_clear_gate = False
_load_hold_pose()
_il_status: dict = {
    "playing": False,
    "kind": "act",
    "message": "IL idle — PLAY centers waist then ACT · STOP centers waist",
}
_il_status_seq = 0
_il_job_lock = threading.Lock()
_il_job_busy = False

# Twin ACT lives on the AGX *host* (il_play_host.sh), not inside the Humble
# docker graph — so GUI play/stop must drive the host PARK+replay scripts.
_IL_WS = Path.home() / "dev-ws"
_IL_STOP_SH = _IL_WS / "scripts" / "_il_gui_stop.sh"
_IL_REPLAY_SH = _IL_WS / "scripts" / "_il_gui_play.sh"
_IL_ACT_CLOSE_SH = _IL_WS / "scripts" / "_il_act_close.sh"
_IL_DEFAULT_CKPT = (
    _IL_WS
    / "datasets"
    / "lerobot"
    / "outputs"
    / "act_foam_pole_lookreach_office_shoulder"
    / "checkpoints"
    / "last"
    / "pretrained_model"
)


def _set_il_status(**extra) -> None:
    global _il_status, _il_status_seq
    with _lock:
        _il_status_seq += 1
        payload = dict(_il_status)
        payload.update(extra)
        payload["_seq"] = int(_il_status_seq)
        _il_status = payload


def _il_ckpt_path() -> Path:
    import os

    raw = str(os.environ.get("GRAB_IT_IL_CKPT") or "").strip()
    if raw:
        return Path(raw)
    return _IL_DEFAULT_CKPT


def _publish_il_cmd(cmd: str) -> None:
    """GUI PLAY/STOP → host lookreach ACT (PARK + kill/restart), not docker ROS."""
    cmd = str(cmd or "").strip().lower()
    if cmd in ("start_play", "il_play"):
        cmd = "play"
    elif cmd in ("stop_play", "il_stop"):
        cmd = "stop"
    elif cmd in ("il_act_close",):
        cmd = "act_close"
    if cmd not in ("play", "stop", "act_close"):
        return

    def _run() -> None:
        global _il_job_busy
        import subprocess

        if not _il_job_lock.acquire(blocking=False):
            _set_il_status(
                playing=bool(_il_status.get("playing")),
                message="IL busy — wait for waist center to finish",
            )
            print(f"agx_isaac_hub: IL cmd={cmd} ignored (busy)", flush=True)
            return
        _il_job_busy = True
        try:
            if cmd == "stop":
                _set_il_status(playing=False, message="stopping ACT · holding the arm…")
                if not _IL_STOP_SH.is_file():
                    _set_il_status(playing=False, message=f"missing {_IL_STOP_SH}")
                    print(f"agx_isaac_hub: missing {_IL_STOP_SH}", flush=True)
                    return
                subprocess.run(
                    ["bash", str(_IL_STOP_SH)],
                    check=False,
                    timeout=60,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                _set_il_status(playing=False, kind="act", message="stopped · arm held")
                print("agx_isaac_hub: IL STOP → kill ACT, hold arm", flush=True)
                return

            if cmd == "act_close":
                _set_il_status(
                    playing=False,
                    kind="act",
                    message="ACT closing from the grasp point…",
                )
                if not _IL_ACT_CLOSE_SH.is_file():
                    _set_il_status(playing=False, message=f"missing {_IL_ACT_CLOSE_SH}")
                    print(f"agx_isaac_hub: missing {_IL_ACT_CLOSE_SH}", flush=True)
                    return
                subprocess.run(
                    ["bash", str(_IL_ACT_CLOSE_SH)],
                    check=False,
                    timeout=60,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                _set_il_status(playing=True, kind="act", message="ACT closing")
                print("agx_isaac_hub: ACT close from grasp point", flush=True)
                return

            # play / replay
            ckpt = _il_ckpt_path()
            if not _IL_REPLAY_SH.is_file():
                _set_il_status(playing=False, message=f"missing {_IL_REPLAY_SH}")
                print(f"agx_isaac_hub: missing {_IL_REPLAY_SH}", flush=True)
                return
            if not (ckpt / "config.json").is_file():
                _set_il_status(
                    playing=False,
                    message=f"checkpoint missing: {ckpt}",
                )
                print(f"agx_isaac_hub: IL PLAY missing ckpt {ckpt}", flush=True)
                return
            _set_il_status(
                playing=False,
                kind="act",
                message="moving to the grasp point, then ACT…",
                checkpoint=str(ckpt),
            )
            log = Path("/tmp/il_gui_replay.log")
            with log.open("ab", buffering=0) as lf:
                lf.write(f"\n--- gui PLAY {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n".encode())
                proc = subprocess.run(
                    ["bash", str(_IL_REPLAY_SH), str(ckpt)],
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    timeout=180,
                    check=False,
                )
            print(
                f"agx_isaac_hub: IL PLAY → {_IL_REPLAY_SH.name} ckpt={ckpt} rc={proc.returncode}",
                flush=True,
            )
            playing = False
            try:
                import subprocess as sp

                playing = (
                    sp.run(
                        ["pgrep", "-f", "/play_policy.py"],
                        check=False,
                        stdout=sp.DEVNULL,
                        stderr=sp.DEVNULL,
                    ).returncode
                    == 0
                )
            except Exception:
                pass
            _set_il_status(
                playing=playing,
                kind="act",
                message="ACT playing" if playing else "waist centered · ACT did not start",
                checkpoint=str(ckpt),
            )
        except Exception as exc:
            _set_il_status(playing=False, message=f"IL {cmd} failed: {exc}")
            print(f"agx_isaac_hub: IL cmd={cmd} failed: {exc}", flush=True)
        finally:
            _il_job_busy = False
            _il_job_lock.release()

    threading.Thread(target=_run, daemon=True, name=f"il-{cmd}").start()


def _annotate_grasp_stall(data: dict) -> None:
    """Expose simulated gripper resistance on the twin.

    ``arm_p`` gripper is the measured finger width. ``_pos`` is the command.
    A close command stuck well short of the fingers is a grasp stall.
    """
    global _grip_stall_t0
    names = list(data.get("arm_n") or [])
    pos = [float(x) for x in (data.get("arm_p") or [])]
    arm = {n: pos[i] for i, n in enumerate(names) if i < len(pos)}
    if "gripper" not in arm or "gripper" not in _pos:
        return
    meas = float(arm["gripper"])
    cmd = float(_pos["gripper"])
    err = meas - cmd
    data["grip_meas"] = meas
    data["grip_cmd"] = cmd
    data["grip_err"] = err
    now = time.monotonic()
    if cmd < 0.20 and err > 0.45:
        if _grip_stall_t0 is None:
            _grip_stall_t0 = now
        data["grasp_stall"] = (now - _grip_stall_t0) >= 0.60
    else:
        _grip_stall_t0 = None
        data["grasp_stall"] = False


def _apply_gui(msg: dict) -> bool:
    """Translate GUI websocket payloads into joint / cmd_vel targets."""
    global _gui_ok, _clear_gate
    changed = False
    kind = str(msg.get("type") or msg.get("t") or "")

    # Imitation PLAY/STOP → host lookreach ACT (PARK + restart), not joint sticks.
    if kind in (
        "il_play",
        "il_stop",
        "play",
        "stop",
        "start_play",
        "stop_play",
        "act_close",
        "il_act_close",
    ):
        _publish_il_cmd(kind)
        if not _gui_ok:
            _gui_ok = True
            print("agx_isaac_hub: GUI IL cmd hooked", flush=True)
        return True

    if kind in GRASP_KINDS:
        _queue_grasp_evt(kind, msg)
        if not _gui_ok:
            _gui_ok = True
            print("agx_isaac_hub: GUI cmd POST ok — sticks/HOME/grasp will move the twin", flush=True)
        return True

    if "n" in msg and "p" in msg:
        changed = _merge_np(msg["n"], msg["p"]) or changed
        # Arm posts own the close-gate clear flag. Waist-center sets it;
        # ACT relay posts omit it so the latch can arm again during play.
        _clear_gate = bool(msg.get("clear_gate"))
        # Always bump: twin soft rate-limit needs seq changes to keep stepping
        # toward a held PARK even when joint values are unchanged.
        changed = True
    elif kind in (
        "arm",
        "home",
        "sleep",
        "gripper",
        "pan",
        "joint_angle",
        "joint_degrees",
    ):
        # GUI sticks were ignored while the grasp latch froze the arm.
        _clear_gate = True

    if kind == "arm" or (kind != "drive" and {"ly", "rx", "ry"} & msg.keys()):
        lx = max(-1.0, min(1.0, float(msg.get("lx", 0.0))))
        ly = max(-1.0, min(1.0, float(msg.get("ly", 0.0))))
        rx = max(-1.0, min(1.0, float(msg.get("rx", 0.0))))
        ry = max(-1.0, min(1.0, float(msg.get("ry", 0.0))))
        deltas = {
            "waist": _dz(-lx) * SPEED["waist"] * DT,
            # Match joy_sim_teleop: shoulder + wrist_angle inverted vs legacy -ly/-ry.
            "shoulder": _dz(ly) * SPEED["shoulder"] * DT,
            "wrist_angle": _dz(ry) * SPEED["wrist_angle"] * DT,
            "wrist_rotate": _dz(rx) * SPEED["wrist_rotate"] * DT,
        }
        for joint, delta in deltas.items():
            if abs(delta) <= 1e-6:
                continue
            _pos[joint] = _clamp_joint(joint, _pos[joint] + delta)
            changed = True

    if kind == "drive" or "dx" in msg or "dy" in msg:
        if "lx" in msg or "az" in msg:
            # Direct twin teleop (Isaac probe / scripts) — meters/rad style caps.
            if "lx" in msg:
                _vel["lx"] = max(-MAX_LIN, min(MAX_LIN, float(msg["lx"])))
                changed = True
            if "az" in msg:
                _vel["az"] = max(-MAX_ANG, min(MAX_ANG, float(msg["az"])))
                changed = True
        else:
            dx = max(-1.0, min(1.0, float(msg.get("dx", 0.0))))
            dy = max(-1.0, min(1.0, float(msg.get("dy", 0.0))))
            _vel["lx"] = _dz(-dy) * MAX_LIN
            _vel["az"] = _dz(-dx) * MAX_ANG
            changed = True

    if kind == "home":
        for name, val in zip(JOINT_NAMES, HOME):
            _pos[name] = float(val)
        changed = True
    elif kind == "sleep":
        for name, val in zip(JOINT_NAMES, SLEEP):
            _pos[name] = float(val)
        changed = True
    elif kind == "gripper" and "value" in msg:
        frac = max(0.0, min(1.0, float(msg["value"])))
        _pos["gripper"] = GRIP_CLOSED + frac * (GRIP_OPEN - GRIP_CLOSED)
        changed = True
    elif kind == "pan" and "value" in msg:
        frac = max(0.0, min(1.0, float(msg["value"])))
        _pos["pan"] = PAN_MIN + frac * (PAN_MAX - PAN_MIN)
        changed = True
    elif kind == "joint_angle":
        joint = str(msg.get("joint") or "")
        if joint in JOINT_LIMITS:
            frac = max(0.0, min(1.0, float(msg.get("value", 0.5))))
            lo, hi = JOINT_LIMITS[joint]
            _pos[joint] = lo + frac * (hi - lo)
            changed = True
    elif kind == "joint_degrees":
        joint = str(msg.get("joint") or "")
        if joint in JOINT_LIMITS:
            _pos[joint] = _clamp_joint(joint, math.radians(float(msg.get("degrees", 0.0))))
            changed = True

    if kind not in ("arm", "drive") and "lx" in msg and "ly" not in msg:
        _vel["lx"] = float(msg["lx"])
        changed = True
    if kind not in ("arm", "drive") and "az" in msg:
        _vel["az"] = float(msg["az"])
        changed = True

    if changed:
        _bump()
        if not _gui_ok:
            _gui_ok = True
            print("agx_isaac_hub: GUI cmd POST ok — sticks/HOME will move the twin", flush=True)
    return changed


class Hub(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, code: int, obj: dict):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/grabit/mode":
            try:
                text = Path("/tmp/grabit_mode").read_text(encoding="utf-8").strip().lower()
            except OSError:
                text = "real"
            if text not in ("sim", "real"):
                text = "real"
            self._json(200, {"world": text})
            return
        if path in ("/", "/index.html", ""):
            body = _gui_html()
            if body is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/isaac/cmd":
            with _lock:
                body = json.dumps(_snapshot()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/isaac/grasp_status":
            with _lock:
                _maybe_timeout_grasp_plan()
                payload = dict(_grasp_status)
            self._json(200, payload)
            return
        if path == "/isaac/il_status":
            with _lock:
                payload = dict(_il_status)
            # Live process check so STOP/PLAY buttons reflect host play_policy.
            try:
                import subprocess as sp

                live = (
                    sp.run(
                        ["pgrep", "-f", "/play_policy.py"],
                        check=False,
                        stdout=sp.DEVNULL,
                        stderr=sp.DEVNULL,
                    ).returncode
                    == 0
                )
                payload["playing"] = bool(live)
                if live and not payload.get("message"):
                    payload["message"] = "ACT playing"
                elif not live and payload.get("playing"):
                    payload["message"] = payload.get("message") or "idle"
            except Exception:
                pass
            self._json(200, payload)
            return
        if path == "/isaac/twin":
            with _lock:
                payload = dict(_twin)
            self._json(200, payload)
            return
        if path == "/gui_hook.js":
            body = _GUI_HOOK_JS.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/javascript; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        key = {"/d435i.jpg": "d435i", "/wide.jpg": "wide", "/d435i": "d435i", "/wide": "wide"}.get(path)
        if key is None:
            self.send_error(404)
            return
        with _lock:
            frame = _frames.get(key)
        if not frame:
            self.send_error(503)
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(frame)))
        self._cors()
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(frame)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        if path == "/isaac/cmd":
            try:
                msg = json.loads(body.decode() or "{}")
            except Exception:
                self.send_error(400)
                return
            with _lock:
                _apply_gui(msg if isinstance(msg, dict) else {})
            self.send_response(204)
            self._cors()
            self.end_headers()
            return
        if path == "/isaac/grasp_status":
            try:
                msg = json.loads(body.decode() or "{}")
            except Exception:
                self.send_error(400)
                return
            if not isinstance(msg, dict):
                self.send_error(400)
                return
            with _lock:
                _set_grasp_status(msg)
            self.send_response(204)
            self._cors()
            self.end_headers()
            return
        if path == "/isaac/twin":
            try:
                msg = json.loads(body.decode() or "{}")
            except Exception:
                self.send_error(400)
                return
            if not isinstance(msg, dict):
                self.send_error(400)
                return
            with _lock:
                global _twin
                data = dict(msg)
                data["type"] = "twin"
                data["ok"] = True
                data["t"] = time.time()
                _annotate_grasp_stall(data)
                _twin = data
            self.send_response(204)
            self._cors()
            self.end_headers()
            return
        key = {"/push/d435i": "d435i", "/push/wide": "wide"}.get(path)
        if key is None or not body:
            self.send_error(404)
            return
        with _lock:
            _frames[key] = body
        self.send_response(204)
        self._cors()
        self.end_headers()


def main():
    httpd = ThreadingHTTPServer(("0.0.0.0", PORT), Hub)
    httpd.daemon_threads = True
    print(
        f"agx_isaac_hub on :{PORT}  GET /d435i.jpg /wide.jpg /isaac/cmd /isaac/grasp_status "
        f"/isaac/il_status /isaac/twin  "
        f"POST /push/* /isaac/cmd /isaac/grasp_status /isaac/twin",
        flush=True,
    )
    try:
        page = ThreadingHTTPServer(("0.0.0.0", GUI_PORT), Hub)
        page.daemon_threads = True
        threading.Thread(target=page.serve_forever, daemon=True, name="gui8765").start()
        html = _gui_html()
        print(
            f"agx_isaac_hub also serving GUI on :{GUI_PORT} "
            f"({0 if html is None else len(html)} bytes)",
            flush=True,
        )
    except OSError as exc:
        print(f"agx_isaac_hub: :{GUI_PORT} not bound ({exc}) — use :{PORT}/", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
