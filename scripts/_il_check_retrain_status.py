#!/usr/bin/env python3
from pathlib import Path
import subprocess

p = Path("/tmp/il_retrain_lookreach_v2.log")
text = p.read_text(errors="replace") if p.exists() else ""
keys = (
    "chunk:",
    "Done.",
    "ERROR",
    "Traceback",
    "current checkpoint",
    "Moving old",
    "Rebuilding",
    "episodes after",
    "Setting nvpmodel",
    "=== chunked",
    "raw lookreach",
    "python=",
    "lerobot ok",
)
lines = []
for raw in text.splitlines():
    part = raw.split("\r")[-1].strip()
    if part and any(k in part for k in keys):
        lines.append(part[:240])
print("\n".join(lines[-60:]))
print("---")
r = subprocess.run(
    ["bash", "-lc", "pgrep -af 'il_retrain|lerobot-train|accelerate' || true"],
    capture_output=True,
    text=True,
)
print("PROC:", (r.stdout or "").strip() or "none")
out = Path.home() / "dev-ws/datasets/lerobot/outputs/act_foam_pole_lookreach/checkpoints"
print("CKPTS:", sorted(x.name for x in out.iterdir()) if out.exists() else "missing")
last = out / "last"
print("LAST:", str(last.resolve()) if last.exists() else "missing")
