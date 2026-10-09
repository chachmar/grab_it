#!/usr/bin/env python3
"""Resume grab_vertical_pole from checkpoints/last through 80000 steps."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

NAME = "grab_vertical_pole"
STEPS = 80000
WS = Path("/home/ubuntu/dev-ws")
OUT = WS / "datasets" / "lerobot" / "outputs" / NAME
DS = WS / "datasets" / "lerobot" / NAME
CFG = OUT / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
TRAIN = Path("/home/ubuntu/lerobot-venv/bin/lerobot-train")
STATUS = Path("/tmp/il_train_status.json")
STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


def status(running: bool, message: str) -> None:
    STATUS.write_text(
        json.dumps({"name": NAME, "running": running, "message": message}),
        encoding="utf-8",
    )
    print(message, flush=True)


def main() -> int:
    if not CFG.is_file():
        status(False, f"training {NAME}: no checkpoint to resume")
        return 1
    if not (DS / "meta" / "info.json").is_file():
        status(False, f"training {NAME}: dataset missing")
        return 1
    status(True, f"training {NAME}: resuming toward {STEPS}")
    cmd = [
        str(TRAIN),
        f"--config_path={CFG}",
        "--resume=true",
        f"--steps={STEPS}",
        f"--dataset.root={DS}",
        "--dataset.video_backend=pyav",
        f"--output_dir={OUT}",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
    ]
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
        match = STEP_RE.search(text)
        if match and "examples" not in text and not text.startswith("Map:"):
            cur = int(match.group(1))
            total = int(match.group(2))
            # A resumed run's progress bar counts only the steps still left.
            if 0 < total < STEPS:
                cur = (STEPS - total) + cur
                total = STEPS
            status(True, f"training {NAME}: step {cur}/{total}")
        elif "error" in text.lower() or "traceback" in text.lower():
            status(True, f"training {NAME}: {text[-180:]}")
    rc = proc.wait()
    if rc != 0:
        status(False, f"training {NAME}: resume failed ({rc})")
        return rc
    status(False, f"training {NAME}: done — select {NAME} in the ACT list")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
