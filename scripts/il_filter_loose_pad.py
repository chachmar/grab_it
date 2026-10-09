#!/usr/bin/env python3
"""Quarantine IL episodes whose saved pad_dist is too loose (early-close demos).

Moves episode_* dirs into <dataset>/_quarantine_loose_pad/ so retrain skips them.
Default max pad_dist matches IL_PAD_HIT_SLACK enclose (~0.022 m).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _pad_dist(meta: dict) -> float | None:
    if meta.get("pad_dist") is not None:
        return float(meta["pad_dist"])
    grasp = meta.get("grasp") or {}
    if grasp.get("pad_dist") is not None:
        return float(grasp["pad_dist"])
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        default=str(Path.home() / "dev-ws/datasets/tabletop_pick"),
    )
    ap.add_argument("--max-pad", type=float, default=0.028)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root = Path(args.dataset)
    trash = root / "_quarantine_loose_pad"
    kept = moved = missing = 0
    for ep in sorted(root.glob("episode_*")):
        if not ep.is_dir():
            continue
        meta_path = ep / "meta.json"
        if not meta_path.is_file():
            missing += 1
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        pad = _pad_dist(meta)
        if pad is None:
            # Older demos without pad_dist — keep (unknown).
            kept += 1
            continue
        if float(pad) <= float(args.max_pad):
            kept += 1
            continue
        moved += 1
        print(f"loose pad_d={pad:.4f} {ep.name}")
        if not args.dry_run:
            trash.mkdir(parents=True, exist_ok=True)
            dest = trash / ep.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(ep), str(dest))
    print(
        f"done kept={kept} moved={moved} missing_meta={missing} "
        f"max_pad={args.max_pad} dry_run={args.dry_run}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
