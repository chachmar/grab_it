#!/usr/bin/env python3
"""Slow point-to-click-style IK approach, then wrist align, then timing close.

This is **not** the MoveIt ``grasp_from_click`` node. It matches the simpler
Gate-C / ``ik_eef`` path:

1. Solve ``ik_eef`` once for the taught tip with **wrist_rotate held at 0**
   (yaml ``wrist_rotate`` is shaft *yaw*, not the joint — streaming it yanked
   the arm off the pole).
2. **Slow slew** to that pose (time-parameterized, not a hard first step).
3. Servo ``wrist_rotate`` from the wide-cam shaft tilt so foam sits in the jaws.
4. Timing policy only for the final close gate (+ small deltas); vision still
   owns wrist roll.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from ik_grasp_lib import (  # noqa: E402
    CLOSED_GRIP,
    GRIP_OPEN,
    JAW_V,
    JawView,
    ServoTrial,
    append_trial,
    project_reachable_tip,
    tip_from_twin,
    trials_dir,
    wrist_max_of,
)
from ik_grasp_servo import wide_shaft_from_jpeg  # noqa: E402
from rl_goal_lib import (  # noqa: E402
    OBS_DIM_TIMING,
    actor_obs_dim,
    apply_joint_delta,
    build_timing_obs,
    eef_of,
    ik_eef,
    load_goal,
)

try:
    import rclpy
    from interbotix_xs_msgs.msg import JointGroupCommand, JointSingleCommand
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
except ImportError:  # pragma: no cover
    rclpy = None

ARM = ["waist", "shoulder", "wrist_angle", "wrist_rotate", "gripper"]
# Match _servo_go_home — no hard first jump.
MAX_RATE = {
    "waist": 0.45,
    "shoulder": 0.22,
    "wrist_angle": 0.35,
    "wrist_rotate": 0.5,
    "gripper": 1.2,
}


class Actor(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.obs_dim, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 5),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)


def _load_actor(path: Path) -> Actor:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("actor_state_dict") or ck
    mapped = {}
    for k, v in sd.items():
        if k.startswith("mlp."):
            mapped[k] = v
        elif k.startswith("actor.mlp."):
            mapped[k[len("actor.") :]] = v
    actor = Actor(actor_obs_dim(mapped))
    actor.load_state_dict(mapped, strict=False)
    actor.eval()
    if actor.obs_dim != OBS_DIM_TIMING:
        raise SystemExit(f"need {OBS_DIM_TIMING}-D timing ckpt, got {actor.obs_dim}")
    return actor


def _http_bytes(url: str, timeout: float = 2.0) -> bytes | None:
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read()
    except Exception:
        return None


def _http_json(url: str, timeout: float = 2.0) -> dict | None:
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


class IkTimingClose(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("ik_timing_close")
        self.args = args
        self.goal = load_goal(args.goal)
        self.wmax = wrist_max_of(self.goal)
        self.actor = _load_actor(args.checkpoint)
        self._js: dict[str, float] = {}
        self.create_subscription(
            JointState, "/px150_no_elbow/joint_states", self._on_js, qos_profile_sensor_data
        )
        self._arm = self.create_publisher(
            JointGroupCommand, "/px150_no_elbow/commands/joint_group", 10
        )
        self._grip = self.create_publisher(
            JointSingleCommand, "/px150_no_elbow/commands/joint_single", 10
        )

    def _on_js(self, msg: JointState) -> None:
        for n, p in zip(msg.name, msg.position):
            self._js[str(n)] = float(p)

    def _pub(self, q: dict[str, float]) -> None:
        arm = JointGroupCommand()
        arm.name = "arm"
        arm.cmd = [float(q[n]) for n in ARM[:4]]
        self._arm.publish(arm)
        grip = JointSingleCommand()
        grip.name = "gripper"
        grip.cmd = float(q["gripper"])
        self._grip.publish(grip)

    def _wait_q(self) -> dict[str, float]:
        t0 = time.time()
        while time.time() - t0 < 12:
            rclpy.spin_once(self, timeout_sec=0.1)
            if all(j in self._js for j in ARM):
                return {j: float(self._js[j]) for j in ARM}
        raise SystemExit("no joint_states")

    def _spin_q(self) -> dict[str, float]:
        rclpy.spin_once(self, timeout_sec=0.0)
        if all(j in self._js for j in ARM):
            return {j: float(self._js[j]) for j in ARM}
        return {j: float(self._js.get(j, 0.0)) for j in ARM}

    def _aim(self) -> list[float]:
        if self.args.yaml_aim:
            return [float(self.goal["x"]), float(self.goal["y"]), float(self.goal["z"])]
        twin = _http_json(f"{self.args.hub.rstrip('/')}/isaac/twin") or {}
        tip_raw, yaw_raw = tip_from_twin(twin, self.goal)
        tip, _yaw, _ = project_reachable_tip(
            tip_raw, yaw_raw, wrist_max=self.wmax, goal=self.goal
        )
        return tip

    def _view(self) -> JawView:
        cam = (self.args.cam or self.args.hub).rstrip("/")
        return JawView.from_shaft(wide_shaft_from_jpeg(_http_bytes(f"{cam}/wide.jpg")))

    def _slow_slew(self, start: dict[str, float], target: dict[str, float]) -> dict[str, float]:
        """Time-parameterized slew with per-joint rate caps (no hard first step)."""
        hz = 20.0
        dt = 1.0 / hz
        dur = 1.2
        for j in ARM:
            dur = max(
                dur,
                abs(float(target[j]) - float(start[j])) / max(MAX_RATE[j], 1e-3) + 0.8,
            )
        steps = max(1, int(dur * hz))
        self.get_logger().info(
            f"slow slew {dur:.1f}s ({steps} steps) → "
            f"sh={target['shoulder']:.2f} wr={target['wrist_angle']:.2f} "
            f"rot={target['wrist_rotate']:.2f}"
        )
        last = dict(start)
        for i in range(1, steps + 1):
            a = i / float(steps)
            cmd = {j: (1.0 - a) * float(start[j]) + a * float(target[j]) for j in ARM}
            self._pub(cmd)
            last = cmd
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(dt)
        # Hold briefly so the arm settles.
        for _ in range(int(0.4 * hz)):
            self._pub(target)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(dt)
        return self._spin_q()

    def run(self) -> int:
        q = self._wait_q()
        tip = self._aim()
        self.get_logger().info(
            f"hybrid: slow ik_eef → wrist align → timing close | "
            f"tip=({tip[0]:.3f},{tip[1]:.3f},{tip[2]:.3f}) wmax={self.wmax:.3f}"
        )
        trial = ServoTrial(
            shaft_xy=[tip[0] - 0.06, tip[1]],
            shaft_z=tip[2] - 0.096,
            shaft_yaw=float(self.goal.get("wrist_rotate", 0.0)),
            note=f"ik_timing|{self.args.note or 'static'}",
        )

        # --- 1) One IK solve; wrist_rotate joint = 0 (not yaml shaft yaw). ---
        seed = dict(q)
        sol = ik_eef(tip, 0.0, GRIP_OPEN, float(self.wmax), seed=seed)
        if sol is None:
            self.get_logger().error("ik_eef failed for taught tip")
            return 3
        tip_up = max(float(self.wmax) + 0.30, -1.15)
        target = {
            "waist": float(sol["waist"]),
            "shoulder": float(sol["shoulder"]),
            "wrist_angle": max(float(self.wmax), max(float(sol["wrist_angle"]), tip_up)),
            "wrist_rotate": 0.0,
            "gripper": GRIP_OPEN,
        }
        trial.reachable = 1
        trial.ik_live = True
        trial.chosen = {
            "phase": "ik_eef",
            "tip": tip,
            "joints": {k: target[k] for k in ARM[:4]},
            "ik_err": float(sol.get("err", 0.0)),
        }
        q = self._slow_slew(q, target)
        tip_now = eef_of(
            [q["waist"], q["shoulder"], q["wrist_angle"], q["wrist_rotate"], q["gripper"]]
        )
        dist = math.dist(tip_now, tip)
        self.get_logger().info(f"arrived dist={dist:.3f} — aligning wrist_rotate")

        # --- 2) Align pads with shaft (vision). Hold shoulder / wrist_angle / waist. ---
        dt = 1.0 / max(float(self.args.hz), 1.0)
        hold = dict(q)
        hold["gripper"] = GRIP_OPEN
        align_hits = 0
        t_align = time.time()
        while time.time() - t_align < float(self.args.align_s):
            q = self._spin_q()
            view = self._view()
            if view.in_window:
                trial.wide_in_view = True
            else:
                trial.wide_lost_any = True
            cmd = dict(hold)
            if view.detected:
                cmd["wrist_rotate"] = max(-1.2, min(1.2, -float(view.from_vert)))
                # Small capped waist steps — large detector jumps thrash the base.
                du = max(-0.06, min(0.06, 0.50 - float(view.u)))
                cmd["waist"] = max(
                    -1.0, min(1.0, float(hold["waist"]) + 0.35 * du)
                )
            self._pub(cmd)
            hold["wrist_rotate"] = cmd["wrist_rotate"]
            hold["waist"] = cmd["waist"]
            trial.steps += 1
            jaw_ok = view.detected and abs(view.u - 0.5) < 0.08 and (
                view.in_window or ((JAW_V[0] - 0.22) <= view.v <= JAW_V[1])
            )
            if jaw_ok:
                align_hits += 1
            else:
                align_hits = max(0, align_hits - 1)
            if trial.steps % 12 == 1:
                self.get_logger().info(
                    f"align u={view.u:.2f} v={view.v:.2f} det={view.detected} "
                    f"rot={cmd['wrist_rotate']:.2f} wa={cmd['waist']:.2f} "
                    f"hits={align_hits}"
                )
            if align_hits >= 10:
                break
            time.sleep(dt)

        self.get_logger().info(
            f"→ timing close (align_hits={align_hits} wide_ok={trial.wide_in_view})"
        )

        # --- 3) Timing policy for close; vision keeps wrist_rotate. ---
        close_started = False
        stall_hits = 0
        close_t0: float | None = None
        prev_grip: float | None = None
        timing_t0 = time.time()
        freeze = dict(hold)
        grasp_prev = list(tip)
        last_cmd = dict(hold)

        while time.time() - timing_t0 < float(self.args.timing_s):
            q = self._spin_q()
            view = self._view()
            tip_now = eef_of(
                [q["waist"], q["shoulder"], q["wrist_angle"], q["wrist_rotate"], q["gripper"]]
            )
            q5 = [q[n] for n in ARM]
            vec = build_timing_obs(
                tip_now,
                tip,
                grasp_prev,
                q5,
                jaw_closed=q5[4] < 0.0,
                wide_du=view.du,
                wide_dv=view.dv,
                lost_view=view.lost,
            )
            grasp_prev = list(tip)
            with torch.inference_mode():
                action = self.actor(torch.tensor(vec, dtype=torch.float32)).tolist()
            nxt, want_close = apply_joint_delta(q5, action, scale=0.08)
            # Cap deltas — policy must not yank away from the IK pose.
            for i, name in enumerate(ARM[:4]):
                lo = float(freeze[name]) - 0.12
                hi = float(freeze[name]) + 0.12
                nxt[i] = max(lo, min(hi, float(nxt[i])))
            cmd = {ARM[i]: float(nxt[i]) for i in range(5)}
            if view.detected:
                cmd["wrist_rotate"] = max(-1.2, min(1.2, -float(view.from_vert)))
            cmd["wrist_angle"] = max(float(self.wmax), float(cmd["wrist_angle"]))

            jaw_ok = view.detected and abs(view.u - 0.5) < 0.14
            timed = (time.time() - timing_t0) > float(self.args.timing_close_s)
            # Close on policy, jaws ready, or tip-timeout even if wide lost the shaft.
            if want_close or close_started or timed:
                if not close_started:
                    if want_close:
                        why = "policy"
                    elif jaw_ok:
                        why = "timeout+jaws"
                    else:
                        why = "timeout+tip"
                    self.get_logger().info(f"close ({why})")
                close_started = True
                cmd = dict(freeze)
                cmd["gripper"] = CLOSED_GRIP
                if view.detected:
                    cmd["wrist_rotate"] = max(
                        -1.2, min(1.2, -float(view.from_vert))
                    )
                    freeze["wrist_rotate"] = cmd["wrist_rotate"]
            else:
                cmd["gripper"] = GRIP_OPEN

            self._pub(cmd)
            last_cmd = cmd
            trial.steps += 1

            if close_started:
                if close_t0 is None:
                    close_t0 = time.time()
                grip_now = float(q["gripper"])
                settled = prev_grip is not None and abs(grip_now - prev_grip) < 0.04
                foam_gap = grip_now > (CLOSED_GRIP + 0.40)
                if (
                    (time.time() - close_t0) >= 0.9
                    and cmd["gripper"] < -0.4
                    and foam_gap
                    and settled
                ):
                    stall_hits += 1
                else:
                    stall_hits = max(0, stall_hits - 1)
                prev_grip = grip_now
                if stall_hits >= 10:
                    trial.stalled = True
                    self.get_logger().info(f"foam stall joint={grip_now:.3f} — lifting")
                    break
            time.sleep(dt)

        if trial.stalled:
            squeeze = dict(last_cmd)
            squeeze["gripper"] = CLOSED_GRIP
            for _ in range(int(0.8 * self.args.hz)):
                self._pub(squeeze)
                rclpy.spin_once(self, timeout_sec=0.0)
                time.sleep(dt)
            lift = dict(squeeze)
            sh0 = float(lift["shoulder"])
            n_lift = max(int(3.0 * self.args.hz), 16)
            for i in range(n_lift):
                lift["shoulder"] = sh0 - 0.12 * ((i + 1) / float(n_lift))
                lift["gripper"] = CLOSED_GRIP
                self._pub(lift)
                rclpy.spin_once(self, timeout_sec=0.0)
                time.sleep(dt)
            trial.lifted = True
            grips = []
            for _ in range(8):
                rclpy.spin_once(self, timeout_sec=0.05)
                if "gripper" in self._js:
                    grips.append(float(self._js["gripper"]))
                time.sleep(0.05)
            grip_med = float(np.median(grips)) if grips else CLOSED_GRIP
            trial.held = grip_med > (CLOSED_GRIP + 0.12)
            self.get_logger().info(
                f"post-lift grip_med={grip_med:.3f} held={trial.held}"
            )

        path = append_trial(trial, path=trials_dir() / "ik_timing_trials.jsonl")
        self.get_logger().info(
            f"logged {path} stalled={trial.stalled} held={trial.held} "
            f"wide_ok={trial.wide_in_view} steps={trial.steps}"
        )
        return 0 if trial.held else (1 if trial.stalled else 2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--goal", type=Path, default=None)
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("datasets/rl_goal/runs/timing_gate/model.pt"),
    )
    ap.add_argument("--hub", default="http://127.0.0.1:8082")
    ap.add_argument("--cam", default="http://127.0.0.1:8081")
    ap.add_argument("--hz", type=float, default=12.0)
    ap.add_argument("--align-s", type=float, default=8.0)
    ap.add_argument("--timing-s", type=float, default=12.0)
    ap.add_argument("--timing-close-s", type=float, default=3.5)
    ap.add_argument("--yaml-aim", action="store_true")
    ap.add_argument("--note", default="static")
    args = ap.parse_args()
    if args.goal is None:
        from rl_goal_lib import goal_yaml_path

        args.goal = goal_yaml_path()
    if rclpy is None:
        print("rclpy missing", file=sys.stderr)
        return 3
    rclpy.init()
    node = IkTimingClose(args)
    try:
        return node.run()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
