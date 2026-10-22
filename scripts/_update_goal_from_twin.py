#!/usr/bin/env python3
"""Write config/rl_goal.yaml from the live hub twin click (reachable under wrist_max)."""
from __future__ import annotations

import json
import math
import sys
import urllib.request
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))

from ik_grasp_lib import GRIP_OPEN, tip_from_twin, wrist_max_of  # noqa: E402
from rl_goal_lib import ik_eef, load_goal  # noqa: E402


def main() -> int:
    hub = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8082"
    twin = json.loads(urllib.request.urlopen(f"{hub.rstrip('/')}/isaac/twin", timeout=5).read())
    goal = load_goal()
    wmax = wrist_max_of(goal)
    tip_raw, _ = tip_from_twin(twin, goal)
    taught = [float(goal["x"]), float(goal["y"]), float(goal["z"])]
    r = math.hypot(tip_raw[0], tip_raw[1])
    print(
        f"twin_xy={twin.get('pole_xy')} tip_raw_r={r:.3f} yaw={twin.get('pole_yaw')}",
        flush=True,
    )

    best = None
    for i in range(21):
        a = i / 20.0
        tip = [(1 - a) * tip_raw[j] + a * taught[j] for j in range(3)]
        for z in (0.2273, tip[2], 0.20, 0.24):
            t = [tip[0], tip[1], z]
            sol = ik_eef(t, 0.0, GRIP_OPEN, wmax)
            if sol is None:
                continue
            key = (a, float(sol.get("err", 0)), abs(z - 0.2273))
            if best is None or key < best[0]:
                best = (key, t, sol, a)
            break

    if best is None:
        print("FAIL: no reachable IK near click", flush=True)
        return 1

    _, tip, sol, a = best
    yaw = float(twin.get("pole_yaw") or 0.0)
    print(
        f"goal tip={[round(v, 4) for v in tip]} r={math.hypot(tip[0], tip[1]):.3f} "
        f"blend_a={a:.2f} ik_err={float(sol.get('err', 0)):.4f}",
        flush=True,
    )
    out = _SCRIPTS.parent / "config" / "rl_goal.yaml"
    out.write_text(
        "\n".join(
            [
                "# GrabIt-Goal from live pole click (projected reachable under wrist_max).",
                "# wrist_rotate field = shaft yaw for pool; joints_real.wrist_rotate = joint.",
                "",
                "frame: analytic_eef",
                "play_on_real: true",
                "source: pole_click",
                "",
                f"x: {tip[0]:.4f}",
                f"y: {tip[1]:.4f}",
                f"z: {tip[2]:.4f}",
                "gripper: -0.9500",
                f"wrist_rotate: {yaw:.4f}",
                f"wrist_angle_max: {wmax:.4f}",
                "",
                "joints_real:",
                f"  waist: {float(sol['waist']):.4f}",
                f"  shoulder: {float(sol['shoulder']):.4f}",
                f"  wrist_angle: {float(sol['wrist_angle']):.4f}",
                "  wrist_rotate: 0.0000",
                "  gripper: -0.9500",
                "",
                "xy_jitter: 0.0300",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
