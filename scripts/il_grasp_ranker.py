#!/usr/bin/env python3
"""IL ranker for the IK grasp pool.

Trains a tiny logistic scorer on servo trial logs. Features are the candidate
pose / IK residual; the label is stall-and-held. Until both successes and
failures exist, callers should keep using geometric ranking.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ik_grasp_lib import (
    GraspCandidate,
    candidate_features,
    load_trials,
    trials_dir,
)

FEAT_DIM = 10


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


class LogisticScorer:
    def __init__(self, w: np.ndarray, b: float):
        self.w = np.asarray(w, dtype=np.float64).reshape(-1)
        self.b = float(b)

    def __call__(self, feats: list[float] | np.ndarray) -> float:
        x = np.asarray(feats, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.w.shape[0]:
            raise ValueError(f"feat dim {x.shape[0]} != {self.w.shape[0]}")
        return float(_sigmoid(np.dot(self.w, x) + self.b))

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps({"w": self.w.tolist(), "b": self.b, "feat_dim": FEAT_DIM}),
            encoding="utf-8",
        )


def load_scorer(path: Path) -> LogisticScorer:
    data = json.loads(path.read_text(encoding="utf-8"))
    return LogisticScorer(np.asarray(data["w"], dtype=np.float64), float(data["b"]))


def _rows_from_trials(trials: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    xs: list[list[float]] = []
    ys: list[float] = []
    for t in trials:
        chosen = t.get("chosen") or {}
        if not chosen:
            continue
        # Rebuild a GraspCandidate-like feature vector from the log.
        c = GraspCandidate(
            x=float(chosen.get("tip", [0, 0, 0])[0]),
            y=float(chosen.get("tip", [0, 0, 0])[1]),
            z=float(chosen.get("tip", [0, 0, 0])[2]),
            yaw=float(chosen.get("yaw", 0.0)),
            phase=str(chosen.get("phase") or "grasp"),
            height_off=float(chosen.get("height_off", 0.0)),
            ik_err=float(chosen.get("ik_err") or 0.0),
            geo_score=float(chosen.get("geo_score") or 0.0),
            joints={
                "shoulder": 0.0,
                "wrist_angle": 0.0,
            },
        )
        label = 1.0 if (t.get("stalled") and t.get("held")) else 0.0
        # Also treat stall-without-hold as a weak negative (dropped).
        if t.get("stalled") and not t.get("held"):
            label = 0.0
        xs.append(candidate_features(c))
        ys.append(label)
    if not xs:
        return np.zeros((0, FEAT_DIM)), np.zeros((0,))
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def train_logistic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = 400,
    lr: float = 0.05,
) -> LogisticScorer:
    if x.shape[0] < 2:
        raise ValueError("need at least 2 labeled trials")
    if len(np.unique(y)) < 2:
        raise ValueError(
            "need both success and failure labels before training the IL ranker"
        )
    # Standardize features for stable logistic fit.
    mu = x.mean(axis=0)
    sig = x.std(axis=0)
    sig[sig < 1e-6] = 1.0
    xn = (x - mu) / sig
    w = np.zeros(x.shape[1], dtype=np.float64)
    b = 0.0
    n = float(x.shape[0])
    for _ in range(epochs):
        p = _sigmoid(xn @ w + b)
        err = p - y
        w -= lr * (xn.T @ err) / n
        b -= lr * float(err.mean())
    # Store weights in original feature space: w_orig = w / sig, b_orig = b - w·(mu/sig)
    w_orig = w / sig
    b_orig = b - float(np.dot(w, mu / sig))
    return LogisticScorer(w_orig, b_orig)


def ready_to_train(trials: list[dict] | None = None) -> bool:
    trials = trials if trials is not None else load_trials()
    _, y = _rows_from_trials(trials)
    return y.size >= 4 and len(np.unique(y)) >= 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--trials",
        type=Path,
        default=None,
        help="JSONL from ik_grasp_servo (default: datasets/rl_goal/servo_trials/trials.jsonl)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Scorer JSON path (default: datasets/rl_goal/servo_trials/ranker.json)",
    )
    ap.add_argument("--status", action="store_true", help="Print label counts and exit")
    args = ap.parse_args()
    trials = load_trials(args.trials)
    x, y = _rows_from_trials(trials)
    n_pos = int((y > 0.5).sum()) if y.size else 0
    n_neg = int((y <= 0.5).sum()) if y.size else 0
    print(f"trials={len(trials)} labeled={y.size} success={n_pos} fail={n_neg}")
    if args.status:
        print("ready" if ready_to_train(trials) else "need both success and failure")
        return 0
    if not ready_to_train(trials):
        print(
            "IL ranker waits for both successes and failures — keep geometric ranking",
            flush=True,
        )
        return 2
    scorer = train_logistic(x, y)
    out = args.out or (trials_dir() / "ranker.json")
    scorer.save(out)
    # Quick train accuracy
    preds = np.array([scorer(row) for row in x])
    acc = float(((preds > 0.5) == (y > 0.5)).mean())
    print(f"wrote {out} train_acc={acc:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
