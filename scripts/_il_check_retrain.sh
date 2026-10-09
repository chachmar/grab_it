#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
import json
from pathlib import Path
home = Path.home()
p = home / "dev-ws/datasets/lerobot/grabit_tabletop_pick/meta/info.json"
print("info_exists", p.exists())
if p.exists():
    d = json.loads(p.read_text())
    print("version", d.get("codebase_version"))
    print("total_episodes", d.get("total_episodes"))
    print("total_frames", d.get("total_frames"))
raw = home / "dev-ws/datasets/tabletop_pick"
print("raw_eps", sum(1 for x in raw.iterdir() if x.is_dir() and x.name.startswith("episode_")))
out = home / "dev-ws/datasets/lerobot/outputs"
if out.exists():
    print("outputs", sorted(x.name for x in out.iterdir()))
else:
    print("outputs", None)
ckpt = home / "dev-ws/datasets/lerobot/outputs/act_tabletop_pick/checkpoints/last/pretrained_model/model.safetensors"
print("ckpt", ckpt.exists())
PY
