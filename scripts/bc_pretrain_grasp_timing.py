#!/usr/bin/env python3
"""Warm-start the approach-timing actor from IK pose-error demos.

Does not clone absolute joint tracks. Each sample is an EARL-style timing
observation; the action is a joint delta toward ``ik_eef(grasp)`` plus a
close gate when the tip is in the pad.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ik_grasp_lib import pick_grasp, wrist_max_of
from rl_goal_lib import (
    CLOSED_GRIP,
    OBS_DIM_TIMING,
    apply_joint_delta,
    build_timing_obs,
    eef_of,
    ik_eef,
    load_goal,
    PAD_RADIUS,
)


class TimingActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(OBS_DIM_TIMING, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 5),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)


def _qdict(q: list[float]) -> dict:
    return {
        "waist": float(q[0]),
        "shoulder": float(q[1]),
        "wrist_angle": float(q[2]),
        "wrist_rotate": float(q[3]),
        "gripper": float(q[4]),
    }


def _make_pairs(n: int, goal: dict, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    wmax = wrist_max_of(goal)
    yaw0 = float(goal.get("wrist_rotate", 0.0))
    tip0 = [float(goal["x"]), float(goal["y"]), float(goal["z"])]
    xs: list[list[float]] = []
    ys: list[list[float]] = []
    for _ in range(n):
        tip = [
            tip0[0] + float(rng.uniform(-0.03, 0.03)),
            tip0[1] + float(rng.uniform(-0.03, 0.03)),
            tip0[2] + float(rng.uniform(-0.02, 0.02)),
        ]
        yaw = yaw0 + float(rng.uniform(-0.2, 0.2))
        # Random start joints near a solvable grasp.
        chosen, _ = pick_grasp(tip, yaw, wrist_max=wmax, prefer_phase="grasp")
        if chosen is None or chosen.joints is None:
            continue
        tgt = chosen.joints
        # Start farther: nudge joints away from the IK solution.
        q = [
            float(tgt["waist"]) + float(rng.uniform(-0.25, 0.25)),
            float(tgt["shoulder"]) + float(rng.uniform(-0.3, 0.3)),
            float(tgt["wrist_angle"]) + float(rng.uniform(-0.2, 0.2)),
            float(tgt["wrist_rotate"]) + float(rng.uniform(-0.3, 0.3)),
            1.65,
        ]
        grasp_prev = list(tip)
        for _step in range(8):
            eef = eef_of(q)
            # Fake wide-cam: closer tip → smaller lost probability.
            dist = math.dist(eef, tip)
            lost = dist > 0.08
            du = float(rng.normal(0.0, 0.05 if not lost else 0.2))
            dv = float(rng.normal(0.0, 0.05 if not lost else 0.2))
            obs = build_timing_obs(
                eef,
                tip,
                grasp_prev,
                q,
                jaw_closed=q[4] < 0.0,
                wide_du=du,
                wide_dv=dv,
                lost_view=lost,
            )
            # Teacher delta: toward IK joints, scaled to ~[-1,1].
            delta = [
                (float(tgt["waist"]) - q[0]) / 0.15,
                (float(tgt["shoulder"]) - q[1]) / 0.15,
                (float(tgt["wrist_angle"]) - q[2]) / 0.15,
                (float(tgt["wrist_rotate"]) - q[3]) / 0.15,
                1.0 if dist < PAD_RADIUS else -1.0,
            ]
            delta = [max(-2.0, min(2.0, d)) for d in delta]
            xs.append(obs)
            ys.append(delta)
            nxt, close = apply_joint_delta(q, delta, scale=0.15)
            if close:
                nxt[4] = CLOSED_GRIP
            q = nxt
            grasp_prev = list(tip)
            # Refresh IK target under wrist cap from the current seed.
            sol = ik_eef(tip, yaw, CLOSED_GRIP if close else 1.65, wmax, seed=_qdict(q))
            if sol is not None:
                tgt = sol
    if not xs:
        raise SystemExit("no timing pairs — check goal IK reachability")
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--goal", type=Path, default=None)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "datasets"
        / "rl_goal"
        / "runs"
        / "timing_bc"
        / "model.pt",
    )
    args = ap.parse_args()
    goal = load_goal(args.goal)
    x_np, y_np = _make_pairs(args.samples, goal, args.seed)
    print(f"pairs={len(x_np)} obs_dim={x_np.shape[1]}", flush=True)
    assert x_np.shape[1] == OBS_DIM_TIMING, x_np.shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(x_np, device=device)
    y = torch.tensor(y_np, device=device)
    actor = TimingActor().to(device)
    critic = nn.Sequential(
        nn.Linear(OBS_DIM_TIMING, 64),
        nn.ELU(),
        nn.Linear(64, 64),
        nn.ELU(),
        nn.Linear(64, 1),
    ).to(device)
    opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    n = x.shape[0]
    for epoch in range(args.epochs):
        perm = torch.randperm(n, device=device)
        losses = []
        for i in range(0, n, args.batch):
            idx = perm[i : i + args.batch]
            loss = torch.mean((actor(x[idx]) - y[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"timing BC epoch {epoch + 1}/{args.epochs} mse={float(np.mean(losses)):.5f}", flush=True)

    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=args.lr)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor_state_dict": {
                "distribution.std_param": torch.full((5,), 0.25, dtype=torch.float32),
                "mlp.0.weight": actor.mlp[0].weight.detach().cpu(),
                "mlp.0.bias": actor.mlp[0].bias.detach().cpu(),
                "mlp.2.weight": actor.mlp[2].weight.detach().cpu(),
                "mlp.2.bias": actor.mlp[2].bias.detach().cpu(),
                "mlp.4.weight": actor.mlp[4].weight.detach().cpu(),
                "mlp.4.bias": actor.mlp[4].bias.detach().cpu(),
            },
            "critic_state_dict": {
                "mlp.0.weight": critic[0].weight.detach().cpu(),
                "mlp.0.bias": critic[0].bias.detach().cpu(),
                "mlp.2.weight": critic[2].weight.detach().cpu(),
                "mlp.2.bias": critic[2].bias.detach().cpu(),
                "mlp.4.weight": critic[4].weight.detach().cpu(),
                "mlp.4.bias": critic[4].bias.detach().cpu(),
            },
            "optimizer_state_dict": opt.state_dict(),
            "iter": 0,
            "infos": {
                "source": "bc_pretrain_grasp_timing",
                "obs_dim": OBS_DIM_TIMING,
                "action": "joint_delta_plus_close",
            },
        },
        args.out,
    )
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
