#!/usr/bin/env python3
"""Analyze grasp_cal / teach_cal output and optionally apply a verified correction.

Usage:
    python3 grasp_cal_analyze.py teach_cal.jsonl
    python3 grasp_cal_analyze.py teach_cal.jsonl --interactive
    python3 grasp_cal_analyze.py teach_cal.jsonl --interactive --verify-max-cm 2.0
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from grasp_cal_lib import (
    DEFAULT_GRASP,
    DEFAULT_JSONL,
    OPTICAL_CAL,
    OPTICAL_CAL_TRIAL,
    FitResult,
    uncorrected_optical_frame,
    corrected_optical_frame,
    fit_correction,
    launch_static_tf,
    load_samples,
    mean_err,
    patch_grasp_params,
    prompt_yes_no,
    save_cal_yaml,
    stop_static_tf,
    wait_for_new_sweep,
)


def truth_clusters(rows: List[dict], tol_m: float = 0.005):
    clusters: List[Tuple[Tuple[float, float, float], List[dict]]] = []
    for r in rows:
        pt = (float(r["truth_x"]), float(r["truth_y"]), float(r["truth_z"]))
        placed = False
        for i, (cent, members) in enumerate(clusters):
            if math.sqrt(sum((pt[j] - cent[j]) ** 2 for j in range(3))) <= tol_m:
                clusters[i] = (cent, members + [r])
                placed = True
                break
        if not placed:
            clusters.append((pt, [r]))
    clusters.sort(key=lambda c: -len(c[1]))
    return clusters


def cluster_mean_err(members: List[dict]) -> float:
    return mean_err(members)


def pick_fit_cluster(clusters, min_samples: int):
    eligible = [(c, m) for c, m in clusters if len(m) >= min_samples]
    if not eligible:
        return clusters[0][1] if clusters else [], False
    if len(eligible) == 1:
        return eligible[0][1], False
    best = min(eligible, key=lambda cm: cluster_mean_err(cm[1]))
    return best[1], True


def fmt_stats(name: str, v: np.ndarray, unit: str = "cm", scale: float = 100.0) -> str:
    v = v * scale
    return (f"  {name:<22} mean {v.mean():6.2f}  median {np.median(v):6.2f}  "
            f"max {v.max():6.2f}  {unit}")


def warn_plane_mismatch(rows: List[dict]) -> None:
    tp = [r for r in rows if r.get("method") == "table_plane"]
    if len(tp) < 3:
        return
    est_z = np.array([float(r["est_z"]) for r in tp])
    truth_z = np.array([float(r["truth_z"]) for r in tp])
    if float(est_z.std()) > 0.002:
        return
    dz = float(truth_z.mean() - est_z.mean())
    if abs(dz) <= 0.04:
        return
    print(
        f"\n  !! TABLE-PLANE HEIGHT MISMATCH: fallback hits z≈{est_z.mean():.3f} m "
        f"but taught tip is z≈{truth_z.mean():.3f} m ({dz * 100:+.0f} cm)."
    )
    print(
        "     Teach with tip ON the table/object, or use depth. "
        "Trial TF may still help grasp but verify carefully."
    )
    print(f"     ({len(tp)}/{len(rows)} samples used table_plane.)")


def bucket_report(rows: List[dict], key: str, label: str, edges: List[float]):
    vals, errs = [], []
    for r in rows:
        x = r.get(key)
        if x is None:
            continue
        vals.append(float(x))
        errs.append(float(r.get("err_norm", 0.0)))
    if len(vals) < 4:
        return
    vals = np.array(vals)
    errs = np.array(errs)
    print(f"\n  error vs {label}:")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (vals >= lo) & (vals < hi)
        if m.sum() == 0:
            continue
        print(f"    {lo:+7.2f} .. {hi:+7.2f} : n={m.sum():3d}  "
              f"mean |e| = {errs[m].mean() * 100:5.2f} cm")
    if vals.std() > 1e-6:
        c = float(np.corrcoef(vals, errs)[0, 1])
        verdict = ("STRONG" if abs(c) > 0.6 else "weak" if abs(c) > 0.3 else "none")
        print(f"    correlation with |error| : r = {c:+.2f}   ({verdict})")


def print_report(
    jsonl_path: str,
    rows: List[dict],
    rows_fit: List[dict],
    subset_fit: bool,
    fit: Optional[FitResult],
    optical_frame: Optional[str] = None,
) -> None:
    print("=" * 72)
    if subset_fit:
        print(f"  {len(rows)} samples in file; reporting fit cluster n={len(rows_fit)}")
    else:
        print(f"  {len(rows)} samples from {jsonl_path}")
    print("=" * 72)

    report = rows_fit
    err = np.array([[r["err_x"], r["err_y"], r["err_z"]] for r in report])
    en = np.linalg.norm(err, axis=1)

    print("\n--- 1. CURRENT ACCURACY -------------------------------------------")
    if subset_fit:
        print(f"  (fit cluster only — {len(rows) - len(rows_fit)} mixed-session "
              f"samples excluded)")
    print(fmt_stats("|error|", en))
    print(fmt_stats("horizontal error", np.hypot(err[:, 0], err[:, 1])))
    print(fmt_stats("|dz|", np.abs(err[:, 2])))
    warn_plane_mismatch(report)

    print("\n--- 2. WHERE THE ERROR COMES FROM ---------------------------------")
    bucket_report(report, "pan_deg", "pan angle (deg)",
                  [-90, -45, -20, -5, 5, 20, 45, 90])
    bucket_report(report, "truth_reach", "reach from base (m)",
                  [0.0, 0.15, 0.22, 0.28, 0.35, 0.45, 0.60])

    if fit is None:
        print("\n--- 3. FITTED CORRECTION ------------------------------------------")
        print("  not enough samples to fit.")
        print("=" * 72)
        return

    sample_frames = {str(r.get("optical_frame", "")) for r in rows_fit}
    if any(f.endswith("_cal") for f in sample_frames if f):
        print("\n  !! Samples used a _cal frame (trial TF was already active).")
        print("     Baseline fit should use uncorrected frames. Re-collect after:")
        print("       rm config/calibration/d435i_optical_cal_trial.yaml")
        print("       pkill -f static_transform_publisher.*_cal\n")

    print("\n--- 3. FITTED CORRECTION ------------------------------------------")
    print(f"  before fix               mean |e| = {fit.en_before_mean * 100:6.2f} cm")
    print(f"  after rigid fit          mean |e| = {fit.res_r_mean * 100:6.2f} cm   "
          f"(max {fit.res_r_max * 100:.2f})")
    if fit.loo_mean is not None:
        print(f"  leave-one-out            mean |e| = {fit.loo_mean * 100:6.2f} cm")
    print(f"\n  C (optical frame {fit.parent_frame} → {fit.child_frame}):")
    print(f"    xyz = [{fit.t[0]:+.5f}, {fit.t[1]:+.5f}, {fit.t[2]:+.5f}]   (m)")
    print(f"    rpy = [{fit.roll:+.5f}, {fit.pitch:+.5f}, {fit.yaw:+.5f}]   (rad)")
    print(f"        = [{math.degrees(fit.roll):+.2f}, {math.degrees(fit.pitch):+.2f}, "
          f"{math.degrees(fit.yaw):+.2f}] deg")

    frame = uncorrected_optical_frame(optical_frame or fit.parent_frame)
    print("\n--- 4. APPLY IT ---------------------------------------------------")
    print(f"  Trial:  python3 grasp_cal_analyze.py {jsonl_path} --interactive")
    print(f"  Manual: ros2 run tf2_ros static_transform_publisher \\")
    print(f"    --x {fit.t[0]:.5f} --y {fit.t[1]:.5f} --z {fit.t[2]:.5f} \\")
    print(f"    --roll {fit.roll:.5f} --pitch {fit.pitch:.5f} --yaw {fit.yaw:.5f} \\")
    print(f"    --frame-id {frame} --child-frame-id {fit.child_frame}")
    print("=" * 72)


def select_rows(args, rows: List[dict]):
    clusters = truth_clusters(rows)
    mixed = len(clusters) > 1
    rows_fit = rows
    subset_fit = False
    if mixed and not args.all_clusters:
        print("\n  !! MIXED SESSIONS — multiple taught poses in one file:")
        for cent, members in clusters:
            tr = math.hypot(cent[0], cent[1])
            print(
                f"     truth=({cent[0]:.3f},{cent[1]:.3f},{cent[2]:.3f}) "
                f"reach={tr:.3f}m  n={len(members):2d}  "
                f"mean |e|={cluster_mean_err(members) * 100:5.1f} cm"
            )
        rows_fit, subset_fit = pick_fit_cluster(clusters, args.min_samples)
        print(f"\n  Using best eligible cluster only (n={len(rows_fit)}).\n")
    return rows_fit, subset_fit


def run_interactive(args, jsonl: Path, rows_fit: List[dict], fit: FitResult) -> int:
    if args.optical_frame:
        fit.parent_frame = uncorrected_optical_frame(args.optical_frame)
        fit.child_frame = corrected_optical_frame(fit.parent_frame)

    baseline_cm = fit.en_before_mean * 100.0
    print("\n" + "=" * 72)
    print("  INTERACTIVE APPLY + VERIFY")
    print("=" * 72)
    print(f"  Baseline mean error: {baseline_cm:.2f} cm  (n={fit.n_samples})")
    print(f"  Expected after TF:   {fit.res_r_mean * 100:.2f} cm (fit residual)")

    if not prompt_yes_no("\n  Launch trial static TF now?", default=True):
        return 0

    cal_dict = {
        "parent_frame": fit.parent_frame,
        "child_frame": fit.child_frame,
        "xyz": [float(x) for x in fit.t],
        "rpy": [fit.roll, fit.pitch, fit.yaw],
    }
    save_cal_yaml(OPTICAL_CAL_TRIAL, fit, str(jsonl))
    proc = launch_static_tf(cal_dict)
    print(f"\n  Trial TF publishing {fit.parent_frame} → {fit.child_frame}  (pid={proc.pid})")
    print(f"  Trial cal written to {OPTICAL_CAL_TRIAL}")
    print("  teach_cal will use the _cal frame automatically on the next sweep.")

    baseline_mtime = float(jsonl.stat().st_mtime) if jsonl.is_file() else 0.0
    verify_rows = wait_for_new_sweep(
        jsonl,
        min_samples=args.min_samples,
        baseline_count=len(rows_fit),
        baseline_mtime=baseline_mtime,
        timeout_s=args.wait_timeout_s,
    )
    if len(verify_rows) < args.min_samples:
        print(
            f"\n  Need a new sweep with ≥{args.min_samples} samples "
            f"(got {len(verify_rows)}). Trial TF left running."
        )
        print(f"  Stop with: pkill -f 'static_transform_publisher.*{fit.child_frame}'")
        return 1

    verify_err_cm = mean_err(verify_rows) * 100.0
    print("\n" + "=" * 72)
    print("  VERIFICATION RESULT")
    print("=" * 72)
    print(f"  Before trial TF:  {baseline_cm:.2f} cm mean  (n={fit.n_samples}, baseline run)")
    print(f"  After trial TF:   {verify_err_cm:.2f} cm mean  (n={len(verify_rows)}, new run)")
    improved = verify_err_cm < baseline_cm * 0.6
    good = verify_err_cm <= args.verify_max_cm
    if good:
        print(f"  ✓ Passed threshold (≤ {args.verify_max_cm:.1f} cm).")
    elif improved:
        print(f"  ~ Improved but still > {args.verify_max_cm:.1f} cm — review before applying.")
    else:
        print("  ✗ Did not improve enough — not applying permanently.")
        if prompt_yes_no("  Stop trial TF?", default=True):
            stop_static_tf(fit.child_frame)
            OPTICAL_CAL_TRIAL.unlink(missing_ok=True)
        return 1

    if not prompt_yes_no("\n  Apply correction permanently to grasp_params + cal yaml?", default=True):
        print("  Trial TF still running. Re-run with --interactive to apply later.")
        return 0

    save_cal_yaml(OPTICAL_CAL, fit, str(jsonl))
    OPTICAL_CAL_TRIAL.unlink(missing_ok=True)
    patch_grasp_params(DEFAULT_GRASP, fit.child_frame)
    stop_static_tf(fit.child_frame)
    launch_static_tf({
        "parent_frame": fit.parent_frame,
        "child_frame": fit.child_frame,
        "xyz": [float(x) for x in fit.t],
        "rpy": [fit.roll, fit.pitch, fit.yaw],
    })

    print("\n  Applied:")
    print(f"    {OPTICAL_CAL}")
    print(f"    {DEFAULT_GRASP}  (camera_frame → {fit.child_frame}, fudge factors zeroed)")
    print("  Restart grasp node to pick up camera_frame:")
    print("    bash scripts/grasp_bringup.sh")
    print("  grasp_bringup also launches static TF when cal yaml exists.")
    print("=" * 72)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", nargs="?", default=str(DEFAULT_JSONL))
    ap.add_argument("--optical-frame", default=None)
    ap.add_argument("--min-samples", type=int, default=6)
    ap.add_argument("--all-clusters", action="store_true")
    ap.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="launch trial TF, wait for re-run, verify, optionally apply permanently",
    )
    ap.add_argument(
        "--verify-max-cm", type=float, default=2.5,
        help="max mean error (cm) after trial TF to pass verification",
    )
    ap.add_argument(
        "--wait-timeout-s", type=float, default=900.0,
        help="seconds to wait for a new pan-cal sweep",
    )
    args = ap.parse_args()

    jsonl = Path(args.jsonl)
    rows = load_samples(jsonl)
    if not rows:
        print(f"no completed samples in {jsonl}", file=sys.stderr)
        return 1

    rows_fit, subset_fit = select_rows(args, rows)
    if len(rows_fit) < args.min_samples:
        print(f"need >= {args.min_samples} samples (have {len(rows_fit)})", file=sys.stderr)
        return 1

    parent = args.optical_frame or rows_fit[-1].get("optical_frame", "d435i_color_optical_frame")
    fit = fit_correction(rows_fit, parent_frame=parent)
    print_report(str(jsonl), rows, rows_fit, subset_fit, fit, args.optical_frame)

    if args.interactive:
        if fit is None:
            return 1
        return run_interactive(args, jsonl, rows_fit, fit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
