#!/usr/bin/env python3
"""Offline checks for the IK-centered grasp pool, jaw window, and timing obs."""
from __future__ import annotations

import math
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from ik_grasp_lib import (  # noqa: E402
    JawView,
    append_trial,
    build_grasp_pool,
    jaw_bias_joints,
    pick_grasp,
    rank_candidates,
    ServoTrial,
    trials_dir,
)
from il_grasp_ranker import ready_to_train, train_logistic, _rows_from_trials  # noqa: E402
from rl_goal_lib import (  # noqa: E402
    OBS_DIM_TIMING,
    apply_joint_delta,
    build_timing_obs,
    ik_eef,
)


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(f"FAIL {msg}")
    print(f"ok {msg}", flush=True)


def main() -> int:
    tip = [0.284, 0.061, 0.227]
    yaw = 0.87
    # Folded wrist cap from the taught pole click.
    wmax = -1.4834
    pool = build_grasp_pool(tip, yaw, wrist_max=wmax)
    _check(len(pool) >= 1, f"reachable pool non-empty ({len(pool)})")
    for c in pool:
        _check(c.joints is not None, "every reachable cand has joints")
        _check(c.ik_err is not None and c.ik_err <= 0.03 + 1e-6, "ik_err within tol")
        sol = ik_eef(c.tip, c.yaw, c.joints["gripper"], wmax, seed=c.joints)
        _check(sol is not None, "re-IK of candidate tip succeeds")

    chosen, ranked = pick_grasp(tip, yaw, wrist_max=wmax, prefer_phase="grasp")
    _check(chosen is not None, "pick_grasp returns a candidate")
    _check(chosen.phase in ("grasp", "pregrasp"), "phase tagged")
    _check(ranked[0].geo_score >= ranked[-1].geo_score, "geo rank descending")

    # Unreachable tip far away → empty pool.
    far = build_grasp_pool([1.5, 1.5, 0.5], 0.0, wrist_max=wmax)
    _check(len(far) == 0, "far tip filtered by ik_eef")

    view = JawView.from_shaft({"u": 0.5, "v": 0.75, "from_vert": 0.1})
    _check(view.in_window and not view.lost, "jaw center is in window")
    lost = JawView.from_shaft({"u": 0.1, "v": 0.2, "from_vert": 0.0})
    _check(lost.lost and not lost.in_window, "outside window is lost")
    q = {
        "waist": 0.0,
        "shoulder": 0.2,
        "wrist_angle": -1.4,
        "wrist_rotate": 0.0,
        "gripper": 1.65,
    }
    biased = jaw_bias_joints(q, lost, wrist_max=-1.48)
    # Shaft high in frame (low v) → reach toward pole; never fold past taught floor.
    _check(biased["wrist_angle"] >= -1.48 - 1e-6, "never fold past taught wrist_max")
    _check(biased["shoulder"] > q["shoulder"], "high shaft reaches shoulder toward pole")
    mid = JawView.from_shaft({"u": 0.72, "v": 0.66, "from_vert": 0.0})
    mid_b = jaw_bias_joints(q, mid, wrist_max=-1.48)
    _check(mid_b["waist"] < q["waist"], "u-too-right decreases waist")
    _check(abs(mid_b["wrist_angle"] - q["wrist_angle"]) < 1e-6, "v-in-band leaves wrist alone")
    left = JawView.from_shaft({"u": 0.30, "v": 0.66, "from_vert": 0.0})
    left_b = jaw_bias_joints(q, left, wrist_max=-1.48)
    _check(left_b["waist"] > q["waist"], "u-too-left increases waist")
    blind = JawView.from_shaft(None)
    q_blind = dict(q)
    q_blind["shoulder"] = -1.40
    seek = jaw_bias_joints(q_blind, blind, wrist_max=-1.48)
    _check(seek["shoulder"] > q_blind["shoulder"], "blind seek unfolds shoulder")
    _check(seek["wrist_angle"] >= -1.48 - 1e-6, "blind seek respects wrist floor")

    tip_o = [0.25, 0.05, 0.22]
    grasp = [0.28, 0.06, 0.23]
    prev = [0.28, 0.06, 0.22]
    joints = [0.2, 0.5, -1.4, 0.8, 1.65]
    obs = build_timing_obs(
        tip_o, grasp, prev, joints, jaw_closed=False, wide_du=0.05, wide_dv=-0.1, lost_view=False
    )
    _check(len(obs) == OBS_DIM_TIMING, f"timing obs dim {OBS_DIM_TIMING}")
    nxt, close = apply_joint_delta(joints, [0.1, -0.1, 0.0, 0.0, 0.0], scale=0.15)
    _check(not close, "low channel 4 does not close")
    _check(abs(nxt[0] - (joints[0] + 0.015)) < 1e-6, "delta scale on waist")
    _, close2 = apply_joint_delta(joints, [0, 0, 0, 0, 0.9])
    _check(close2, "high channel 4 closes")

    # Trial log round-trip.
    path = trials_dir() / "_contract_trials.jsonl"
    if path.is_file():
        path.unlink()
    for held_ok, hz, geo in (
        (True, 0.0, 1.0),
        (True, -0.02, 0.9),
        (False, 0.02, 0.5),
        (False, 0.02, 0.4),
    ):
        append_trial(
            ServoTrial(
                shaft_xy=[0.22, 0.0],
                shaft_z=0.17,
                shaft_yaw=yaw,
                chosen={
                    "phase": "grasp",
                    "tip": tip,
                    "yaw": yaw,
                    "height_off": hz,
                    "geo_score": geo,
                    "ik_err": 0.01 if held_ok else 0.02,
                },
                stalled=held_ok,
                held=held_ok,
                wide_in_view=held_ok,
            ),
            path,
        )
    from ik_grasp_lib import load_trials

    rows = load_trials(path)
    _check(len(rows) == 4, "four trials logged")
    x, y = _rows_from_trials(rows)
    _check(ready_to_train(rows), "ready_to_train with both labels")
    scorer = train_logistic(x, y, epochs=200)
    _check(0.0 <= scorer(x[0]) <= 1.0, "scorer returns probability")
    ranked2 = rank_candidates(pool, scorer=scorer)
    _check(len(ranked2) == len(pool), "IL rank preserves pool size")
    path.unlink(missing_ok=True)

    # Baseline report on fixture trials + IL train end-to-end.
    from seed_servo_fixture_trials import write_fixture
    from servo_baseline_report import _stats

    fixture = trials_dir() / "_contract_fixture.jsonl"
    write_fixture(fixture, clear=True)
    from ik_grasp_lib import load_trials as _load

    frows = _load(fixture)
    _check(len(frows) >= 4, "fixture trials seeded")
    st = _stats([r for r in frows if "nudge" not in str(r.get("note") or "").lower()])
    _check(st["n"] >= 1 and st["held"] > 0.0, "static baseline has held rate")
    x_f, y_f = _rows_from_trials(frows)
    _check(ready_to_train(frows), "fixture ready_to_train")
    scorer_f = train_logistic(x_f, y_f, epochs=200)
    ranker_path = trials_dir() / "_contract_ranker.json"
    scorer_f.save(ranker_path)
    _check(ranker_path.is_file(), "ranker json written")
    ranker_path.unlink(missing_ok=True)
    fixture.unlink(missing_ok=True)

    print("ik grasp contract passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
