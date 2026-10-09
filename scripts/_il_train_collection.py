#!/usr/bin/env python3
"""Convert and train the current grab_it collection. Does not touch other ACT folders."""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import subprocess

WS = Path.home() / "dev-ws"
VENV_PY = Path.home() / "lerobot-venv" / "bin" / "python3"
VENV_TRAIN = Path.home() / "lerobot-venv" / "bin" / "lerobot-train"
STATUS = Path("/tmp/il_train_status.json")
BATCH = "4"
SAVE_FREQ = 2500
PROTECTED = "act_foam_pole_lookreach_office_shoulder"
STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


def _status(name: str, running: bool, message: str) -> None:
    STATUS.write_text(
        json.dumps({"name": name, "running": running, "message": message}),
        encoding="utf-8",
    )
    print(message, flush=True)


def _run(cmd: list[str], name: str) -> int:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        text = line.strip()
        if not text:
            continue
        print(text, flush=True)
        if text.startswith("episode_"):
            ep = text.split(":", 1)[0]
            _status(name, True, f"training {name}: converting {ep}")
            continue
        match = STEP_RE.search(text)
        if match and "examples" not in text and not text.startswith("Map:"):
            _status(name, True, f"training {name}: step {match.group(1)}/{match.group(2)}")
        elif "error" in text.lower() or "traceback" in text.lower():
            _status(name, True, f"training {name}: {text[-180:]}")
    return proc.wait()


def _converted_count(ds: Path) -> int:
    info = ds / "meta" / "info.json"
    if not info.is_file():
        return -1
    try:
        data = json.loads(info.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return -1
    version = str(data.get("codebase_version") or "")
    if not version.startswith("v3"):
        return -1
    try:
        return int(data.get("total_episodes"))
    except (TypeError, ValueError):
        return -1


def _park_previous(out: Path) -> None:
    """Move this collection's last ACT aside. Never touches the shoulder policy."""
    if not out.exists():
        return
    if out.name == PROTECTED:
        raise SystemExit(f"refusing to replace {out}")
    prev = out.parent / f"{out.name}_previous"
    if prev.exists():
        shutil.rmtree(prev)
    out.rename(prev)


def _steps_arg() -> int:
    raw = sys.argv[2] if len(sys.argv) > 2 else "20000"
    try:
        n = int(raw)
    except ValueError:
        n = 20000
    return max(1000, min(100000, n))


def main() -> int:
    name = (sys.argv[1] if len(sys.argv) > 1 else "").strip()
    steps = _steps_arg()
    if not name or not name[0].isalnum() or not all(c.isalnum() or c in "_-" for c in name):
        _status(name or "?", False, "training refused: bad collection name")
        return 2
    if name == PROTECTED:
        _status(name, False, "training refused: that name is the shoulder ACT")
        return 2
    raw = WS / "datasets" / name
    ds = WS / "datasets" / "lerobot" / name
    out = WS / "datasets" / "lerobot" / "outputs" / name
    if not raw.is_dir():
        _status(name, False, f"training {name}: no collection folder")
        return 1
    n = sum(
        1
        for p in raw.iterdir()
        if p.is_dir() and p.name.startswith("episode_") and (p / "actions.npy").is_file()
    )
    if n < 1:
        _status(name, False, f"training {name}: no saved takes")
        return 1
    if not VENV_PY.is_file() or not VENV_TRAIN.is_file():
        _status(name, False, "training refused: lerobot venv is missing")
        return 1
    converted = _converted_count(ds)
    if converted == n:
        _status(name, True, f"training {name}: {n} takes already converted")
    else:
        _status(name, True, f"training {name}: converting {n} takes")
        rc = _run(
            [
                str(VENV_PY),
                str(WS / "scripts" / "il_to_lerobot.py"),
                "--input",
                str(raw),
                "--output",
                str(ds),
                "--repo-id",
                f"local/{name}",
            ],
            name,
        )
        if rc != 0:
            _status(name, False, f"training {name}: convert failed ({rc})")
            return rc
    _park_previous(out)
    save_freq = min(SAVE_FREQ, steps)
    _status(name, True, f"training {name}: step 0/{steps}")
    rc = _run(
        [
            str(VENV_TRAIN),
            "--policy.type=act",
            "--policy.push_to_hub=false",
            f"--dataset.repo_id=local/{name}",
            f"--dataset.root={ds}",
            "--dataset.video_backend=pyav",
            f"--output_dir={out}",
            "--policy.device=cuda",
            f"--batch_size={BATCH}",
            f"--steps={steps}",
            f"--save_freq={save_freq}",
        ],
        name,
    )
    if rc != 0:
        _status(name, False, f"training {name}: failed ({rc})")
        return rc
    _status(name, False, f"training {name}: done — select {name} in the ACT list")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
