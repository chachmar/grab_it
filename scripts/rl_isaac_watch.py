#!/usr/bin/env python3
"""Poll the AGX hub for an RL job and run it on this Windows box.

TRAIN: write the queued analytic goal, BC-pretrain (with review episodes),
then PPO. Saves datasets/rl_goal/runs/<name>/ and copies to the AGX.
REPLAY: play a named checkpoint or episode on the live twin.
Does not kill Kit. Works while grabit_mode is real (tip mirror + queue).
"""
from __future__ import annotations

import atexit
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

# SIM hub :8081; REAL tip-mirror hub :8082 (mjpeg owns 8081).
HUB_HOST = "http://192.168.50.207"
ISAAC = Path(r"C:\Users\chach\Projects\IsaacLab")
WS = Path(r"C:\Users\chach\Projects\grab_it-dev-ws")
GOAL = WS / "config" / "rl_goal.yaml"
BC = ISAAC / "scripts" / "imitation_learning" / "isaaclab_mimic" / "bc_pretrain_grabit_goal.py"
TRAIN = ISAAC / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"
TWIN = WS / "scripts" / "play_goal_rl_on_twin.py"
LOGS = ISAAC / "logs" / "rsl_rl" / "grabit_goal_px150"
LOGS_TIMING = ISAAC / "logs" / "rsl_rl" / "grabit_goal_timing_px150"
RUNS = WS / "datasets" / "rl_goal" / "runs"
AGX_HOST = "jetson-agx"
AGX_RUNS = "~/dev-ws/datasets/rl_goal/runs"
TASK_ABS = "GrabIt-Goal-Px150-v0"
TASK_TIMING = "GrabIt-Goal-Timing-Px150-v0"
BC_TIMING = WS / "scripts" / "bc_pretrain_grasp_timing.py"


def _hub_api() -> str:
    """Prefer the port that answers /isaac/cmd (real=8082, sim=8081)."""
    for port in (8082, 8081):
        try:
            with urllib.request.urlopen(f"{HUB_HOST}:{port}/isaac/cmd", timeout=1.5) as r:
                json.loads(r.read().decode())
            return f"{HUB_HOST}:{port}"
        except Exception:
            continue
    return f"{HUB_HOST}:8082"


def _get(path: str) -> dict:
    with urllib.request.urlopen(_hub_api() + path, timeout=3) as r:
        return json.loads(r.read().decode())


def _post_status(text: str) -> None:
    data = json.dumps({"rl_status": text}).encode()
    req = urllib.request.Request(
        _hub_api() + "/isaac/rl_job",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=3).read()
    except Exception as exc:
        print("status post", exc, flush=True)


def _sanitize(name: str) -> str:
    text = "".join(c if c.isalnum() or c in "-_ " else "_" for c in str(name).strip())
    text = "_".join(text.split())
    return text[:64] or f"reach_{int(time.time())}"


def _ckpt_rank(path: Path) -> tuple[int, float]:
    """Prefer highest numbered model_N.pt; fall back to mtime."""
    stem = path.stem
    n = -1
    if stem.startswith("model_") and stem[6:].isdigit():
        n = int(stem[6:])
    elif stem == "model":
        n = -2  # explicit alias only when no numbered ckpt exists
    return (n, path.stat().st_mtime)


def _latest_ckpt(run_dir: Path | None = None) -> Path | None:
    """Newest / highest-iter checkpoint in a run dir or under LOGS."""
    if run_dir is not None:
        found = [p for p in run_dir.glob("model*.pt") if p.is_file()]
        if found:
            return max(found, key=_ckpt_rank)
        return None
    runs = sorted(LOGS.glob("20*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for run in runs:
        found = [p for p in run.glob("model_*.pt") if p.is_file()]
        if found:
            return max(found, key=_ckpt_rank)
    return None


def _run_stream(
    cmd: list[str],
    cwd: Path,
    status_re: re.Pattern | None = None,
    status_fn=None,
) -> int:
    print(" ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    last = ""
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        msg = None
        if status_fn is not None:
            msg = status_fn(line)
        elif status_re is not None:
            m = status_re.search(line)
            if m:
                msg = m.group(0).strip()
        if msg and msg != last:
            last = msg
            _post_status(msg)
    return int(proc.wait())


def _copy_to_agx(run_dir: Path) -> None:
    try:
        rc = subprocess.call(
            [
                "scp",
                "-o",
                "BatchMode=yes",
                "-r",
                str(run_dir),
                f"{AGX_HOST}:{AGX_RUNS}/",
            ]
        )
        print(f"scp to AGX rc={rc}", flush=True)
    except Exception as exc:
        print(f"scp failed: {exc}", flush=True)


def _replay_episode_on_twin(ep_dir: Path, hz: float = 15.0) -> int:
    actions_p = ep_dir / "actions.npy"
    if not actions_p.is_file():
        _post_status(f"missing actions in {ep_dir.name}")
        return 2
    actions = np.load(actions_p)
    names = ["waist", "shoulder", "wrist_angle", "wrist_rotate", "gripper"]
    dt = 1.0 / hz
    _post_status(f"twin episode {ep_dir.name}")
    for row in actions:
        payload = {
            "t": "arm",
            "n": names,
            "p": [float(row[i]) for i in range(5)],
            "clear_gate": True,
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            _hub_api() + "/isaac/cmd",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=1.0).read()
        except Exception as exc:
            print("twin post", exc, flush=True)
        time.sleep(dt)
    return 0


def _handle(job: dict) -> None:
    op = str(job.get("op") or "")
    name = _sanitize(str(job.get("name") or job.get("label") or f"reach_{int(time.time())}"))
    label = str(job.get("label") or name)
    episode = str(job.get("episode") or "")

    if op == "train":
        run_dir = RUNS / name
        # Never wipe a finished PPO run — duplicate hub jobs used to delete
        # model_999.pt mid-replay. Stamp a new name instead.
        if (run_dir / "train_ok.json").is_file() or (run_dir / "model.pt").is_file():
            stamped = f"{name}_{int(time.time()) % 100000}"
            _post_status(f"kept {name}; training as {stamped}")
            name = stamped
            run_dir = RUNS / name
        elif run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "episodes").mkdir(parents=True, exist_ok=True)

        goal = job.get("goal_text") or ""
        if goal:
            GOAL.write_text(goal, encoding="utf-8")
            (run_dir / "rl_goal.yaml").write_text(goal, encoding="utf-8")
            print(f"wrote {GOAL}", flush=True)
        elif GOAL.is_file():
            (run_dir / "rl_goal.yaml").write_text(GOAL.read_text(encoding="utf-8"), encoding="utf-8")

        # Timing path: joint deltas + pose-error obs (plan default going forward).
        job_task = str(job.get("task") or "").strip()
        timing = (
            job_task == TASK_TIMING
            or "timing" in name.lower()
            or "timing" in label.lower()
            or str(job.get("mode") or "").lower() == "timing"
        )
        task = TASK_TIMING if timing else TASK_ABS
        logs = LOGS_TIMING if timing else LOGS
        meta = {
            "name": name,
            "label": label,
            "created": int(time.time()),
            "task": task,
            "mode": "timing" if timing else "absolute",
        }
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        py = ISAAC / ".venv" / "Scripts" / "python.exe"
        if timing:
            warm = logs / "bc_warmstart" / "model_0.pt"
            warm.parent.mkdir(parents=True, exist_ok=True)
            _post_status(f"timing BC {name} starting")
            # Prefer system/workspace python for the lightweight timing BC.
            bc_py = str(py) if py.is_file() else sys.executable
            rc = _run_stream(
                [
                    bc_py,
                    str(BC_TIMING),
                    "--goal",
                    str(GOAL),
                    "--samples",
                    "200",
                    "--epochs",
                    "120",
                    "--out",
                    str(warm),
                ],
                WS,
                status_re=re.compile(r"timing BC epoch \d+/\d+|BC epoch \d+/\d+"),
            )
            load_run = "bc_warmstart"
            obs_dim = 21
            ppo_iters = 300
        else:
            warm = logs / "bc_warmstart_flat" / "model_0.pt"
            warm.parent.mkdir(parents=True, exist_ok=True)
            _post_status(f"BC {name} starting")
            rc = _run_stream(
                [
                    str(py),
                    str(BC),
                    "--goal",
                    str(GOAL),
                    "--trajectories",
                    "400",
                    "--epochs",
                    "200",
                    "--out",
                    str(warm),
                    "--episodes-dir",
                    str(run_dir / "episodes"),
                    "--save-episodes",
                    "8",
                ],
                ISAAC,
                status_re=re.compile(r"BC epoch \d+/\d+"),
            )
            load_run = "bc_warmstart_flat"
            obs_dim = 28
            ppo_iters = 1000
        if rc != 0:
            _post_status(f"BC failed {rc}")
            return
        # Review clips are samples for the Clip dropdown — not the train budget.
        if not timing:
            _post_status(f"review clips 8 (not train budget) — PPO {name} 0/{ppo_iters}")

        _post_status(f"PPO {name} starting 0/{ppo_iters} ({task})")
        bat = ISAAC / "isaaclab.bat"
        # rsl_rl prints "Learning iteration 12/80" (and variants).
        ppo_re = re.compile(
            r"Learning iteration\s+(\d+)\s*/\s*(\d+)|"
            r"(?:^|\s)[Ii]teration\s+(\d+)\s*/\s*(\d+)",
            re.M,
        )
        saw_iter = {"n": 0}
        ppo_t0 = time.time()

        def _ppo_status(line: str) -> str | None:
            if "RuntimeError" in line or "Traceback" in line:
                saw_iter["err"] = line.strip()[:160]
            m = ppo_re.search(line)
            if not m:
                return None
            saw_iter["n"] += 1
            if m.group(1) is not None:
                return f"PPO iter {m.group(1)}/{m.group(2)}"
            return f"PPO iter {m.group(3)}/{m.group(4)}"

        rc = _run_stream(
            [
                str(bat),
                "-p",
                str(TRAIN),
                "--task",
                task,
                "--num_envs",
                "256",
                "--headless",
                "--max_iterations",
                str(ppo_iters),
                "--resume",
                "--load_run",
                load_run,
                "--checkpoint",
                "model_0.pt",
            ],
            ISAAC,
            status_fn=_ppo_status,
        )
        if rc != 0 or saw_iter["n"] == 0:
            err = saw_iter.get("err") or f"rc={rc} iters_seen={saw_iter['n']}"
            _post_status(f"PPO failed — {err}")
            return

        ckpt = _latest_ckpt(logs if timing else None)
        if timing:
            # Search under timing experiment logs.
            runs = sorted(logs.glob("20*"), key=lambda p: p.stat().st_mtime, reverse=True)
            ckpt = None
            for run in runs:
                found = [p for p in run.glob("model_*.pt") if p.is_file()]
                if found:
                    ckpt = max(found, key=_ckpt_rank)
                    break
        if ckpt is None or ckpt.stat().st_mtime < (ppo_t0 - 2.0):
            _post_status("PPO finished but no fresh checkpoint")
            return
        dest = run_dir / "model.pt"
        shutil.copy2(ckpt, dest)
        # Keep the true iter name so replay does not silently fall back to mid-train.
        shutil.copy2(ckpt, run_dir / ckpt.name)
        (run_dir / "train_ok.json").write_text(
            json.dumps(
                {
                    "checkpoint": ckpt.name,
                    "ppo_iters": ppo_iters,
                    "iters_seen": saw_iter["n"],
                    "source": str(ckpt),
                    "obs_dim": obs_dim,
                    "task": task,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        _copy_to_agx(run_dir)
        _post_status(f"saved {name} ({ckpt.name}, PPO {ppo_iters})")
        return

    if op == "replay":
        run_dir = RUNS / name if name else None
        if episode and run_dir is not None:
            ep_dir = run_dir / "episodes" / episode
            if not ep_dir.is_dir():
                _post_status(f"missing episode {episode}")
                return
            rc = _replay_episode_on_twin(ep_dir)
            _post_status("episode replay done" if rc == 0 else f"episode failed {rc}")
            return
        ckpt = _latest_ckpt(run_dir) if run_dir else _latest_ckpt()
        if ckpt is None:
            _post_status("no checkpoint")
            return
        goal = (run_dir / "rl_goal.yaml") if run_dir and (run_dir / "rl_goal.yaml").is_file() else GOAL
        py = ISAAC / ".venv" / "Scripts" / "python.exe"
        _post_status(f"twin replay {ckpt.name}")
        rc = _run_stream(
            [
                str(py),
                str(TWIN),
                "--checkpoint",
                str(ckpt),
                "--goal",
                str(goal),
                "--start",
                "sleep",
                "--loops",
                "2",
                "--episode-s",
                "12",
            ],
            WS,
            status_re=None,
        )
        _post_status("twin replay done" if rc == 0 else f"replay failed {rc}")


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        import ctypes

        kernel = ctypes.windll.kernel32
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if handle:
            kernel.CloseHandle(handle)
            return True
    except Exception:
        pass
    return False


def _acquire_singleton() -> Path | None:
    """Only one watcher may poll the hub — duplicates double-train and confuse status."""
    lock_path = WS / "datasets" / "rl_goal" / ".rl_isaac_watch.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if lock_path.is_file():
            try:
                old = int(lock_path.read_text(encoding="utf-8").strip().split()[0])
            except Exception:
                old = 0
            if _pid_alive(old) and old != os.getpid():
                print(f"another rl_isaac_watch is running (pid {old}) — exit", flush=True)
                return None
        lock_path.write_text(f"{os.getpid()}\n", encoding="utf-8")
        return lock_path
    except Exception as exc:
        print("lock", exc, flush=True)
        return lock_path


def main() -> int:
    lock = _acquire_singleton()
    if lock is None:
        return 2

    def _clear() -> None:
        try:
            if lock.is_file() and lock.read_text(encoding="utf-8").startswith(str(os.getpid())):
                lock.unlink(missing_ok=True)
        except Exception:
            pass

    atexit.register(_clear)

    seen = 0
    print("watching", HUB_HOST, "ports 8082/8081", flush=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            snap = _get("/isaac/cmd")
            job = snap.get("rl_job") or {}
            jid = int(job.get("id") or 0)
            op = str(job.get("op") or "")
            if op in ("cancel", "clear", "idle", ""):
                continue
            if jid and jid != seen and op in ("train", "replay"):
                seen = jid
                print("job", op, job.get("name"), jid, flush=True)
                # Refuse tip-snapshot goals that still have an open jaw — those
                # come from TRAIN overwriting a pole click (fixed in gui_node).
                goal_text = str(job.get("goal_text") or "")
                if op == "train" and "source: pole_click" not in goal_text:
                    if "gripper:" in goal_text:
                        try:
                            import yaml as _yaml

                            g = _yaml.safe_load(goal_text) or {}
                            if float(g.get("gripper", -0.95)) > 0.0:
                                _post_status(
                                    "refused open-jaw goal — click the pole in D435i, then TRAIN"
                                )
                                print("refused open-jaw train", g.get("gripper"), flush=True)
                                continue
                        except Exception:
                            pass
                _handle(job)
        except Exception as exc:
            print("hub", exc, flush=True)
        time.sleep(2.0)


if __name__ == "__main__":
    raise SystemExit(main())
