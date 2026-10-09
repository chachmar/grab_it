#!/usr/bin/env python3
"""Make IL SAVE/REC clickable via inline onclick; log recorder commands."""
from __future__ import annotations

from pathlib import Path

SRC = Path("/workspaces/isaac_ros-dev/src")
HTML = SRC / "grab_it_gui" / "grab_it_gui" / "gui.html"
NODE = SRC / "grab_it_gui" / "grab_it_gui" / "gui_node.py"
REC = SRC / "grab_it_il" / "grab_it_il" / "record_demos.py"

BTNS = {
    '<button id="il-rec-btn" type="button">REC</button>':
        '<button id="il-rec-btn" type="button" onclick="send({type:\'il_start\',source:(typeof ilSource!==\'undefined\'&&ilSource)||\'teleop\'})">REC</button>',
    '<button id="il-save-btn" type="button">SAVE</button>':
        '<button id="il-save-btn" type="button" onclick="send({type:\'il_save\',source:(typeof ilSource!==\'undefined\'&&ilSource)||\'teleop\'})">SAVE</button>',
    '<button id="il-discard-btn" type="button">DISCARD</button>':
        '<button id="il-discard-btn" type="button" onclick="send({type:\'il_discard\'})">DISCARD</button>',
    '<button id="il-play-btn" type="button">PLAY</button>':
        '<button id="il-play-btn" type="button" onclick="send({type:\'il_play\'})">PLAY</button>',
    '<button id="il-stop-btn" type="button">STOP</button>':
        '<button id="il-stop-btn" type="button" onclick="send({type:\'il_stop\'})">STOP</button>',
}


def main() -> None:
    html = HTML.read_text(encoding="utf-8")
    node = NODE.read_text(encoding="utf-8")
    rec = REC.read_text(encoding="utf-8") if REC.is_file() else ""
    print("html il_save send ", "il_save" in html)
    print("html save onclick ", "il_save" in html and "onclick" in html)
    print("node il_save     ", 't == "il_save"' in node or "t == 'il_save'" in node)
    print("node IL cmd log  ", "IL cmd" in node)
    print("rec cmd cb       ", "def _cmd_cb" in rec)

    n = 0
    for old, new in BTNS.items():
        if old in html:
            html = html.replace(old, new, 1)
            n += 1
    print("html buttons patched", n)

    if rec and "IL command" not in rec:
        old = "        cmd = str(data.get(\"cmd\") or data.get(\"type\") or \"\").strip().lower()\n"
        new = old + "        self.get_logger().info(f\"IL command {cmd!r}\")\n"
        if old in rec:
            rec = rec.replace(old, new, 1)
            REC.write_text(rec, encoding="utf-8")
            print("patched recorder command log")
        else:
            print("WARN: could not patch recorder log")

    HTML.write_text(html, encoding="utf-8")
    print("ok", HTML)
    print("hard-refresh the GUI (Ctrl+Shift+R). Restart recorder to pick up its log line.")


if __name__ == "__main__":
    main()
