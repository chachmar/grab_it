#!/usr/bin/env python3
"""Gate GrabIt-Goal-Timing vs IK-servo baseline on stall∧held rates.

Deploy on real only when the timing policy's stall and held rates are at least
as high as the servo baseline (static and nudged splits when present).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ik_grasp_lib import load_trials, trials_dir
from servo_baseline_report import _stats


def _pass(policy: dict, baseline: dict, *, eps: float = 0.0) -> bool:
    if baseline.get("n", 0) <= 0:
        return False
    return (
        float(policy.get("stall", 0.0)) + eps >= float(baseline.get("stall", 0.0))
        and float(policy.get("held", 0.0)) + eps >= float(baseline.get("held", 0.0))
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="servo baseline.json (default: servo_trials/baseline.json)",
    )
    ap.add_argument(
        "--timing-trials",
        type=Path,
        default=None,
        help="timing policy trials.jsonl (default: servo_trials/timing_trials.jsonl)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write gate verdict JSON",
    )
    ap.add_argument(
        "--eps",
        type=float,
        default=0.0,
        help="Allowed shortfall vs baseline (fraction)",
    )
    ap.add_argument(
        "--approach-only",
        action="store_true",
        help="Recompute servo baseline from recent Home-approach trials only "
        "(notes containing yaml_aim / approach-era held successes).",
    )
    args = ap.parse_args()

    base_path = args.baseline or (trials_dir() / "baseline.json")
    timing_path = args.timing_trials or (trials_dir() / "timing_trials.jsonl")

    if args.approach_only:
        rows = load_trials(trials_dir() / "trials.jsonl")
        # Prefer trials logged after the Home approach path (wide_ok + yaml_aim).
        recent = [
            r
            for r in rows
            if "yaml_aim" in str(r.get("note") or "")
            and (r.get("wide_in_view") or r.get("held") or r.get("stalled"))
        ]
        if len(recent) < 3:
            recent = [r for r in rows if "yaml_aim" in str(r.get("note") or "")]
        static = [r for r in recent if "nudge" not in str(r.get("note") or "").lower()]
        nudged = [r for r in recent if "nudge" in str(r.get("note") or "").lower()]
        baseline = {"all": _stats(recent), "static": _stats(static), "nudged": _stats(nudged)}
    else:
        baseline = json.loads(base_path.read_text(encoding="utf-8"))

    trows = load_trials(timing_path)
    t_static = [r for r in trows if "nudge" not in str(r.get("note") or "").lower()]
    t_nudged = [r for r in trows if "nudge" in str(r.get("note") or "").lower()]
    policy = {
        "all": _stats(trows),
        "static": _stats(t_static),
        "nudged": _stats(t_nudged),
    }

    checks = {}
    for split in ("static", "nudged", "all"):
        b = baseline.get(split) or {}
        p = policy.get(split) or {}
        if int(b.get("n", 0)) <= 0:
            checks[split] = {"skipped": True, "reason": "no baseline trials"}
            continue
        if int(p.get("n", 0)) <= 0:
            checks[split] = {"skipped": True, "reason": "no policy trials"}
            continue
        ok = _pass(p, b, eps=float(args.eps))
        checks[split] = {
            "ok": ok,
            "baseline": b,
            "policy": p,
        }

    # Require every split that has baseline trials to pass (when policy also has trials).
    required = [
        s
        for s in ("static", "nudged")
        if int((baseline.get(s) or {}).get("n", 0)) > 0
        and int((policy.get(s) or {}).get("n", 0)) > 0
    ]
    if not required:
        required = [
            s
            for s in ("all",)
            if int((baseline.get(s) or {}).get("n", 0)) > 0
            and int((policy.get(s) or {}).get("n", 0)) > 0
        ]
    deploy = bool(required) and all(checks[s].get("ok") for s in required)

    verdict = {
        "deploy": deploy,
        "required_splits": required,
        "checks": checks,
        "baseline": baseline,
        "policy": policy,
        "obs_dim": 21,
        "task": "GrabIt-Goal-Timing-Px150-v0",
    }
    out = args.out or (trials_dir() / "timing_gate.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(json.dumps(verdict, indent=2))
    print(
        f"{'PASS — deploy allowed' if deploy else 'FAIL — do not deploy'} → {out}",
        flush=True,
    )
    return 0 if deploy else 1


if __name__ == "__main__":
    raise SystemExit(main())
