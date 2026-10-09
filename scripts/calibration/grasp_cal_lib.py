"""Shared helpers for grasp / teach calibration analyze + apply."""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import yaml

WS = Path(os.environ.get("GRAB_IT_WS", "/workspaces/isaac_ros-dev"))
DEFAULT_GRASP = WS / "config" / "grasp_params.yaml"
OPTICAL_CAL = WS / "config" / "calibration" / "d435i_optical_cal.yaml"
OPTICAL_CAL_TRIAL = WS / "config" / "calibration" / "d435i_optical_cal_trial.yaml"
DEFAULT_JSONL = WS / "config" / "calibration" / "teach_cal.jsonl"
CAL_SUFFIX = "_cal"


def uncorrected_optical_frame(name: str) -> str:
    """Strip trial/permanent cal suffix so C is always parent→child, not _cal→_cal."""
    n = str(name)
    if n.endswith(CAL_SUFFIX):
        return n[: -len(CAL_SUFFIX)]
    return n


def corrected_optical_frame(name: str) -> str:
    base = uncorrected_optical_frame(name)
    return base if base.endswith(CAL_SUFFIX) else f"{base}{CAL_SUFFIX}"


@dataclass
class FitResult:
    t: np.ndarray
    R: np.ndarray
    roll: float
    pitch: float
    yaw: float
    parent_frame: str
    child_frame: str
    n_samples: int
    en_before_mean: float
    res_r_mean: float
    res_r_max: float
    loo_mean: Optional[float] = None


def load_samples(path: str | Path) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("source") == "teach_cal_session":
                continue
            if "truth_x" in r and "T_base_optical" in r:
                if "err_x" in r and "err_norm" not in r:
                    ex = float(r["err_x"])
                    ey = float(r.get("err_y", 0.0))
                    ez = float(r.get("err_z", 0.0))
                    r["err_norm"] = float(math.sqrt(ex * ex + ey * ey + ez * ez))
                    r["err_xy"] = float(math.hypot(ex, ey))
                rows.append(r)
    return rows


def mat_to_rpy(R: np.ndarray) -> Tuple[float, float, float]:
    sp = -float(R[2, 0])
    sp = max(-1.0, min(1.0, sp))
    pitch = math.asin(sp)
    if abs(sp) > 0.99999:
        roll = math.atan2(-float(R[1, 2]), float(R[1, 1]))
        yaw = 0.0
    else:
        roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
        yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    return roll, pitch, yaw


def umeyama_rigid(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    AA, BB = A - ca, B - cb
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = cb - R @ ca
    return R, t


def fit_correction(
    rows_fit: List[dict],
    parent_frame: Optional[str] = None,
    child_suffix: str = "_cal",
) -> Optional[FitResult]:
    if len(rows_fit) < 2:
        return None
    A, B = [], []
    for r in rows_fit:
        T = np.array(r["T_base_optical"], dtype=np.float64).reshape(4, 4)
        p_cam = np.array([r["cam_x"], r["cam_y"], r["cam_z"]], dtype=np.float64)
        p_true = np.array([r["truth_x"], r["truth_y"], r["truth_z"]], dtype=np.float64)
        try:
            q = (np.linalg.inv(T) @ np.append(p_true, 1.0))[:3]
        except np.linalg.LinAlgError:
            continue
        A.append(p_cam)
        B.append(q)
    if len(A) < 2:
        return None
    A = np.array(A)
    B = np.array(B)
    R, t = umeyama_rigid(A, B)
    res = np.linalg.norm((A @ R.T + t) - B, axis=1)
    roll, pitch, yaw = mat_to_rpy(R)
    en = np.array([float(r["err_norm"]) for r in rows_fit[: len(A)]])
    loo_mean = None
    if len(A) >= 6:
        loo = []
        for i in range(len(A)):
            m = np.ones(len(A), dtype=bool)
            m[i] = False
            Ri, ti = umeyama_rigid(A[m], B[m])
            loo.append(float(np.linalg.norm((Ri @ A[i] + ti) - B[i])))
        loo_mean = float(np.mean(loo))
    parent = uncorrected_optical_frame(
        parent_frame or rows_fit[-1].get("optical_frame", "d435i_color_optical_frame")
    )
    child = corrected_optical_frame(parent)
    return FitResult(
        t=t,
        R=R,
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        parent_frame=str(parent),
        child_frame=str(child),
        n_samples=len(A),
        en_before_mean=float(en.mean()),
        res_r_mean=float(res.mean()),
        res_r_max=float(res.max()),
        loo_mean=loo_mean,
    )


def fit_to_yaml_dict(fit: FitResult, source_jsonl: str = "") -> dict:
    return {
        "parent_frame": fit.parent_frame,
        "child_frame": fit.child_frame,
        "xyz": [round(float(x), 8) for x in fit.t],
        "rpy": [round(float(fit.roll), 8), round(float(fit.pitch), 8), round(float(fit.yaw), 8)],
        "fit_mean_err_before_m": round(fit.en_before_mean, 6),
        "fit_mean_err_after_m": round(fit.res_r_mean, 6),
        "n_samples": int(fit.n_samples),
        "source_jsonl": source_jsonl,
        "stamp": time.time(),
    }


def save_cal_yaml(path: Path, fit: FitResult, source_jsonl: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = fit_to_yaml_dict(fit, source_jsonl)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def load_cal_yaml(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        if d.get("parent_frame") and d.get("child_frame"):
            return d
    except Exception:
        pass
    return None


def stop_static_tf(child_frame: str) -> None:
    subprocess.run(
        ["pkill", "-f", f"static_transform_publisher.*{child_frame}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.3)


def launch_static_tf(cal: dict) -> subprocess.Popen:
    xyz = cal["xyz"]
    rpy = cal["rpy"]
    child = str(cal["child_frame"])
    stop_static_tf(child)
    cmd = [
        "ros2", "run", "tf2_ros", "static_transform_publisher",
        "--x", str(xyz[0]),
        "--y", str(xyz[1]),
        "--z", str(xyz[2]),
        "--roll", str(rpy[0]),
        "--pitch", str(rpy[1]),
        "--yaw", str(rpy[2]),
        "--frame-id", str(cal["parent_frame"]),
        "--child-frame-id", child,
    ]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def patch_grasp_params(path: Path, child_frame: str) -> None:
    text = path.read_text(encoding="utf-8")
    repl = [
        (r"(^(\s*)camera_frame:\s*).*$", rf"\1{child_frame}"),
        (r"(^(\s*)click_origin_offset:\s*).*$", r"\1[0.0, 0.0, 0.0]"),
        (r"(^(\s*)click_v_bias:\s*).*$", r"\10.0"),
        (r"(^(\s*)click_pan_scale:\s*).*$", r"\11.0"),
        (r"(^(\s*)grasp_farther_m:\s*).*$", r"\10.0"),
    ]
    for pat, sub in repl:
        text, n = re.subn(pat, sub, text, count=1, flags=re.MULTILINE)
    path.write_text(text, encoding="utf-8")


def mean_err(rows: List[dict]) -> float:
    if not rows:
        return float("inf")
    vals = []
    for r in rows:
        if "err_norm" in r:
            vals.append(float(r["err_norm"]))
        else:
            ex = float(r.get("err_x", 0.0))
            ey = float(r.get("err_y", 0.0))
            ez = float(r.get("err_z", 0.0))
            vals.append(math.sqrt(ex * ex + ey * ey + ez * ez))
    return float(np.mean(vals))


def wait_for_new_sweep(
    jsonl: Path,
    min_samples: int,
    baseline_count: int,
    baseline_mtime: float,
    timeout_s: float = 900.0,
    poll_s: float = 2.0,
    stable_s: float = 8.0,
) -> List[dict]:
    print(f"\n  Next: GUI → Run pan cal (N≥{min_samples}, same taught pose).")
    print(f"  Baseline file has {baseline_count} sample(s) — waiting for a NEW sweep.")
    try:
        input("  Press Enter when the sweep has finished (or to begin watching)… ")
    except (EOFError, KeyboardInterrupt):
        print()
    print(f"  Watching {jsonl} …  (Ctrl+C to cancel)\n")

    t0 = time.time()
    last_n = -1
    stable_at: Optional[float] = None
    sweep_started = baseline_count == 0
    last_wait_msg = 0.0

    while time.time() - t0 < timeout_s:
        if not jsonl.is_file():
            time.sleep(poll_s)
            continue
        mtime = float(jsonl.stat().st_mtime)
        rows = load_samples(jsonl)
        n = len(rows)

        if not sweep_started:
            if n < baseline_count or mtime > baseline_mtime + 0.5:
                sweep_started = True
                stable_at = None
                print("  New sweep detected — collecting samples …")
            else:
                if time.time() - last_wait_msg > 15.0:
                    print("  Still waiting for pan cal to start …")
                    last_wait_msg = time.time()
                time.sleep(poll_s)
                continue

        if n != last_n:
            print(f"     {n}/{min_samples} sample(s) …")
            last_n = n
            stable_at = None

        if n >= min_samples:
            if stable_at is None:
                stable_at = time.time()
            elif time.time() - stable_at >= stable_s:
                return rows
        time.sleep(poll_s)

    return load_samples(jsonl) if jsonl.is_file() else []


def prompt_yes_no(question: str, default: bool = False) -> bool:
    hint = "Y/n" if default else "y/N"
    try:
        ans = input(f"{question} [{hint}] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    if not ans:
        return default
    return ans in ("y", "yes")
