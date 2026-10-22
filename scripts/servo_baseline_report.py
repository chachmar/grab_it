#!/usr/bin/env python3
"""Summarize IK-servo trial logs — the stall/lift rates RL must beat.

Reads ``datasets/rl_goal/servo_trials/trials.jsonl`` (or ``--trials``).
Split by ``note`` containing ``nudge`` vs everything else (static).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ik_grasp_lib import load_trials, trials_dir


def _stats(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0, "stall": 0.0, "held": 0.0, "wide_ok": 0.0, "ik_live": 0.0}
    stall = sum(1 for r in rows if r.get("stalled"))
    held = sum(1 for r in rows if r.get("held"))
    wide = sum(1 for r in rows if r.get("wide_in_view"))
    ik = sum(1 for r in rows if r.get("ik_live", True))
    return {
        "n": n,
        "stall": stall / n,
        "held": held / n,
        "wide_ok": wide / n,
        "ik_live": ik / n,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=Path, default=None)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write JSON summary (default: servo_trials/baseline.json)",
    )
    args = ap.parse_args()
    rows = load_trials(args.trials)
    static = [r for r in rows if "nudge" not in str(r.get("note") or "").lower()]
    nudged = [r for r in rows if "nudge" in str(r.get("note") or "").lower()]
    summary = {
        "all": _stats(rows),
        "static": _stats(static),
        "nudged": _stats(nudged),
    }
    out = args.out or (trials_dir() / "baseline.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
