#!/usr/bin/env python3
"""Kick off GrabIt-Goal-Timing training: contract → timing BC warmstart → PPO.

Does not train the legacy absolute-joint GrabIt-Goal task. Deploy on real only
after the IK servo baseline is recorded and matched.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

WS = Path(r"C:\Users\chach\Projects\grab_it-dev-ws")
ISAAC = Path(r"C:\Users\chach\Projects\IsaacLab")
SCRIPTS = WS / "scripts"
GOAL = WS / "config" / "rl_goal.yaml"
LOGS = ISAAC / "logs" / "rsl_rl" / "grabit_goal_timing_px150"
RUNS = WS / "datasets" / "rl_goal" / "runs"
TRAIN = ISAAC / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"
TASK = "GrabIt-Goal-Timing-Px150-v0"


def _run(cmd: list[str], cwd: Path) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(cwd))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default=f"timing_{int(time.time()) % 100000}")
    ap.add_argument("--bc-samples", type=int, default=200)
    ap.add_argument("--bc-epochs", type=int, default=120)
    ap.add_argument("--ppo-iters", type=int, default=300)
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--skip-contract", action="store_true")
    ap.add_argument("--skip-ppo", action="store_true", help="BC warmstart only")
    args = ap.parse_args()

    py = sys.executable
    if not args.skip_contract:
        rc = _run([py, str(SCRIPTS / "test_ik_grasp_contract.py")], WS)
        if rc != 0:
            return rc

    run_dir = RUNS / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "rl_goal.yaml").write_text(GOAL.read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "name": args.name,
                "task": TASK,
                "created": int(time.time()),
                "action": "joint_delta_plus_close",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    warm = LOGS / "bc_warmstart" / "model_0.pt"
    warm.parent.mkdir(parents=True, exist_ok=True)
    run_bc = run_dir / "model_bc.pt"
    rc = _run(
        [
            py,
            str(SCRIPTS / "bc_pretrain_grasp_timing.py"),
            "--goal",
            str(GOAL),
            "--samples",
            str(args.bc_samples),
            "--epochs",
            str(args.bc_epochs),
            "--out",
            str(run_bc),
        ],
        WS,
    )
    if rc != 0:
        return rc
    shutil.copy2(run_bc, warm)
    shutil.copy2(run_bc, run_dir / "model.pt")
    print(f"warmstart → {warm}", flush=True)

    if args.skip_ppo:
        (run_dir / "train_ok.json").write_text(
            json.dumps({"checkpoint": "model_bc.pt", "ppo_iters": 0, "obs_dim": 21, "phase": "bc_only"}, indent=2),
            encoding="utf-8",
        )
        print("skip PPO — BC only", flush=True)
        return 0

    bat = ISAAC / "isaaclab.bat"
    rc = _run(
        [
            str(bat),
            "-p",
            str(TRAIN),
            "--task",
            TASK,
            "--num_envs",
            str(args.num_envs),
            "--headless",
            "--max_iterations",
            str(args.ppo_iters),
            "--resume",
            "--load_run",
            "bc_warmstart",
            "--checkpoint",
            "model_0.pt",
        ],
        ISAAC,
    )
    if rc != 0:
        print(f"PPO exited {rc}", flush=True)
        return rc

    # Prefer newest numbered ckpt under LOGS.
    ckpts = sorted(LOGS.glob("20*/model_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not ckpts:
        ckpts = sorted(LOGS.rglob("model_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if ckpts:
        shutil.copy2(ckpts[0], run_dir / "model.pt")
        shutil.copy2(ckpts[0], run_dir / ckpts[0].name)
        (run_dir / "train_ok.json").write_text(
            json.dumps(
                {
                    "checkpoint": ckpts[0].name,
                    "ppo_iters": args.ppo_iters,
                    "source": str(ckpts[0]),
                    "obs_dim": 21,
                    "task": TASK,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"saved {run_dir / 'model.pt'} from {ckpts[0]}", flush=True)
    else:
        print("PPO finished but no model_*.pt found under timing logs", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
