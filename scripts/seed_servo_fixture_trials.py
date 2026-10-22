#!/usr/bin/env python3
"""Write synthetic servo trials so the IL ranker can train offline.

Real hardware fills ``trials.jsonl`` via ``ik_grasp_servo.py``. This seed is
only for contract / unit paths when the arm is offline. Labels mix stall-held
successes and failures so ``ready_to_train`` is true.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ik_grasp_lib import ServoTrial, append_trial, trials_dir


def write_fixture(out: Path | None = None, *, clear: bool = True) -> Path:
    out = out or (trials_dir() / "fixture_trials.jsonl")
    if clear and out.is_file():
        out.unlink()

    # Successes: mid-height grasp, in view.
    for hz in (-0.02, 0.0, 0.02, 0.0):
        append_trial(
            ServoTrial(
                shaft_xy=[0.22, 0.0],
                shaft_z=0.17,
                shaft_yaw=0.87,
                chosen={
                    "phase": "grasp",
                    "tip": [0.284, 0.061, 0.227 + hz],
                    "yaw": 0.87,
                    "height_off": hz,
                    "geo_score": 1.0 - abs(hz) / 0.04,
                    "ik_err": 0.008,
                },
                pool_size=6,
                reachable=4,
                wide_in_view=True,
                wide_lost_any=False,
                ik_live=True,
                stalled=True,
                lifted=True,
                held=True,
                steps=40,
                note="static",
            ),
            out,
        )
    # Failures: lost view, drop after stall, nudge miss.
    append_trial(
        ServoTrial(
            chosen={
                "phase": "grasp",
                "tip": [0.30, 0.10, 0.25],
                "yaw": 0.87,
                "height_off": 0.02,
                "geo_score": 0.3,
                "ik_err": 0.025,
            },
            wide_in_view=False,
            wide_lost_any=True,
            stalled=False,
            held=False,
            note="static",
        ),
        out,
    )
    append_trial(
        ServoTrial(
            chosen={
                "phase": "pregrasp",
                "tip": [0.26, 0.05, 0.24],
                "yaw": 0.2,
                "height_off": -0.02,
                "geo_score": 0.4,
                "ik_err": 0.02,
            },
            wide_in_view=True,
            stalled=True,
            lifted=True,
            held=False,
            note="nudge",
        ),
        out,
    )
    append_trial(
        ServoTrial(
            chosen={
                "phase": "grasp",
                "tip": [0.29, 0.08, 0.22],
                "yaw": 1.2,
                "height_off": 0.0,
                "geo_score": 0.55,
                "ik_err": 0.015,
            },
            wide_in_view=True,
            stalled=False,
            held=False,
            note="nudge",
        ),
        out,
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="JSONL path (default: datasets/rl_goal/servo_trials/fixture_trials.jsonl)",
    )
    ap.add_argument("--clear", action="store_true", default=True)
    args = ap.parse_args()
    out = write_fixture(args.out, clear=args.clear)
    print(f"wrote fixture trials → {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
