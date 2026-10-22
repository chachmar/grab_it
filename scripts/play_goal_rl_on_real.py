#!/usr/bin/env python3
"""Smoke-test a GrabIt-Goal checkpoint on the real PX150 (AGX host).

Starts from the *current* joints (never ramps to Isaac sleep — that path can
fold into the electronics from Interbotix factory sleep).

Default mode (legacy absolute-joint actor): the network invents joints; this
script FK→ik_eef under yaml wrist_angle_max so Isaac numbers never stream.

Timing mode (``--timing``): EARL-style pose-error obs (``build_timing_obs``).
The actor outputs joint deltas + a close gate; commands are applied with
``apply_joint_delta``. No re-IK of a joint target the network invented. Deploy
only after the IK servo baseline on the same poles is beaten.

Requires: grabit_mode=real, Nano xs_sdk publishing joint_states.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from interbotix_xs_msgs.msg import JointGroupCommand, JointSingleCommand

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from ik_grasp_lib import JawView, ServoTrial, append_trial, trials_dir  # noqa: E402
from rl_goal_lib import (  # noqa: E402
    CLOSED_GRIP,
    DEFAULT_WRIST_MAX,
    OBS_DIM_TIMING,
    actor_obs_dim,
    apply_joint_delta,
    build_goal_obs,
    build_timing_obs,
    eef_of,
    fk_horizon_command,
    ik_eef,
)

ARM_JOINTS = ["waist", "shoulder", "wrist_angle", "wrist_rotate", "gripper"]
ARM_OFFSET = (0.06, 0.0, 0.096)
TOOL = 0.008
NEAR_RADIUS = 0.025
GRIP_TOL = 0.15
LIMITS = {
    "waist": (-2.0246, 2.0944),
    "shoulder": (-1.9722, 1.9373),
    "wrist_angle": (-1.90, 1.4486),
    "wrist_rotate": (-1.5708, 1.5708),
    "gripper": (-0.95, 1.65),
}
_SHOULDER_Z = 0.06566 + 0.03891
_L1 = 0.15
_L2 = 0.065 + 0.043 + 0.023 + 0.027575


def _fk_px150(waist: float, shoulder: float, wrist: float, tool: float = TOOL):
    c, s = math.cos(waist), math.sin(waist)
    l2 = _L2 + float(tool)
    r = _L1 * math.cos(shoulder) + l2 * math.cos(shoulder + wrist)
    z = -_L1 * math.sin(shoulder) - l2 * math.sin(shoulder + wrist)
    return c * r, s * r, _SHOULDER_Z + z


def _clamp(name: str, v: float) -> float:
    lo, hi = LIMITS[name]
    return float(min(hi, max(lo, v)))


def _http_json(url: str, timeout: float = 1.5) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read()
            if not body:
                return {}
            return json.loads(body.decode("utf-8"))
    except Exception:
        return None


def _load_goal(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {
        "x": float(data.get("x", 0.22)),
        "y": float(data.get("y", 0.0)),
        "z": float(data.get("z", 0.26)),
        "gripper": float(data.get("gripper", -0.95)),
        "wrist_rotate": float(data.get("wrist_rotate", 0.0)),
        "wrist_angle_max": float(data.get("wrist_angle_max", DEFAULT_WRIST_MAX)),
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


def _load_actor(path: Path) -> tuple[Actor, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("actor_state_dict") or ck
    mapped = {}
    for k, v in sd.items():
        if k.startswith("mlp."):
            mapped[k] = v
        elif k.startswith("actor.mlp."):
            mapped[k[len("actor.") :]] = v
    obs_dim = actor_obs_dim(mapped)
    actor = Actor(obs_dim)
    actor.load_state_dict(mapped, strict=False)
    actor.eval()
    infos = ck.get("infos") if isinstance(ck, dict) else None
    return actor, infos if isinstance(infos, dict) else {}


def _wide_view_from_hub(hub: str | None, cam: str | None = None) -> JawView:
    # REAL: cameras on :8081, tip-mirror hub on :8082.
    base = (cam or hub or "").rstrip("/")
    if not base:
        return JawView.from_shaft(None)
    try:
        from ik_grasp_servo import wide_shaft_from_jpeg

        with urllib.request.urlopen(f"{base}/wide.jpg", timeout=1.5) as resp:
            jpeg = resp.read()
        return JawView.from_shaft(wide_shaft_from_jpeg(jpeg))
    except Exception:
        return JawView.from_shaft(None)


class GoalSmoke(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("play_goal_rl_on_real")
        self.args = args
        self.goal = _load_goal(args.goal)
        self.actor, ck_infos = _load_actor(args.checkpoint)
        # Timing checkpoints share the 21-D width with legacy v1. Prefer
        # checkpoint infos, else require --timing.
        self.timing = bool(args.timing) or (
            str(ck_infos.get("source", "")).endswith("timing")
            or str(ck_infos.get("action", "")) == "joint_delta_plus_close"
        )
        self._shaft_yaw = float(self.goal["wrist_rotate"])
        self._grasp_prev = [
            float(self.goal["x"]),
            float(self.goal["y"]),
            float(self.goal["z"]),
        ]
        self._js: dict[str, float] = {}
        self._js_t = 0.0
        self.create_subscription(
            JointState,
            "/px150_no_elbow/joint_states",
            self._on_js,
            qos_profile_sensor_data,
        )
        self._arm_pub = self.create_publisher(
            JointGroupCommand, "/px150_no_elbow/commands/joint_group", 10
        )
        self._grip_pub = self.create_publisher(
            JointSingleCommand, "/px150_no_elbow/commands/joint_single", 10
        )
        mode = "timing-delta" if self.timing else "absolute-reIK"
        self.get_logger().info(f"loaded {args.checkpoint} mode={mode}")
        self.get_logger().info(
            f"goal xyz=({self.goal['x']:.3f},{self.goal['y']:.3f},{self.goal['z']:.3f}) "
            f"grip={self.goal['gripper']:.3f} wrist_max={self.goal['wrist_angle_max']:.3f} "
            f"obs={self.actor.obs_dim}"
        )
        if self.timing and self.actor.obs_dim != OBS_DIM_TIMING:
            raise SystemExit(
                f"--timing needs a {OBS_DIM_TIMING}-D checkpoint, got {self.actor.obs_dim}"
            )

    def _on_js(self, msg: JointState) -> None:
        for n, p in zip(msg.name, msg.position):
            self._js[str(n)] = float(p)
        self._js_t = time.time()

    def _wait_js(self, timeout: float = 15.0) -> list[float]:
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if all(j in self._js for j in ARM_JOINTS) and (time.time() - self._js_t) < 1.0:
                q = [self._js[j] for j in ARM_JOINTS]
                if all(abs(v + math.pi) < 0.05 for v in q[:4]):
                    raise SystemExit(
                        "joint_states look like failed sync (~-pi) — check Nano xs_sdk / 12V"
                    )
                return q
        raise SystemExit("no live /px150_no_elbow/joint_states — is Nano xs_sdk up?")

    def _publish(self, q5: list[float]) -> None:
        arm = JointGroupCommand()
        arm.name = "arm"
        arm.cmd = [float(q5[i]) for i in range(4)]
        self._arm_pub.publish(arm)
        grip = JointSingleCommand()
        grip.name = "gripper"
        grip.cmd = float(q5[4])
        self._grip_pub.publish(grip)

    def _aim(self) -> list[float]:
        """Live shaft aim (base-link pole + ARM_OFFSET) or yaml tip."""
        g = [self.goal["x"], self.goal["y"], self.goal["z"]]
        self._shaft_yaw = float(self.goal["wrist_rotate"])
        hub = getattr(self.args, "hub", None)
        if not hub:
            return g
        twin = _http_json(f"{hub.rstrip('/')}/isaac/twin")
        if not twin:
            return g
        pole_xy = twin.get("pole_xy")
        if isinstance(pole_xy, (list, tuple)) and len(pole_xy) >= 2:
            # Twin pole_xy is px150 base_link; yaml tips are create3 (+ ARM_OFFSET).
            g[0] = float(pole_xy[0]) + ARM_OFFSET[0]
            g[1] = float(pole_xy[1]) + ARM_OFFSET[1]
        if twin.get("pole_z") is not None:
            g[2] = float(twin["pole_z"]) + ARM_OFFSET[2]
        if twin.get("pole_yaw") is not None:
            self._shaft_yaw = float(twin["pole_yaw"])
        return g

    def _obs(self, q: list[float], last_a: list[float]) -> np.ndarray:
        tip = _fk_px150(q[0], q[1], q[2], tool=TOOL)
        eef = [tip[i] + ARM_OFFSET[i] for i in range(3)]
        g = self._aim()
        if self.timing:
            view = _wide_view_from_hub(
                getattr(self.args, "hub", None),
                getattr(self.args, "cam", None),
            )
            vec = build_timing_obs(
                eef,
                g,
                self._grasp_prev,
                q,
                jaw_closed=q[4] < 0.0,
                wide_du=view.du,
                wide_dv=view.dv,
                lost_view=view.lost,
            )
            self._grasp_prev = list(g)
            return np.asarray(vec, dtype=np.float32)
        vec = build_goal_obs(
            eef,
            g,
            q[4],
            self.goal["gripper"],
            q,
            last_a,
            shaft_yaw=self._shaft_yaw,
            obs_dim=self.actor.obs_dim,
        )
        return np.asarray(vec, dtype=np.float32)

    def _cmd_from_action(
        self,
        action: list[float],
        last_good: list[float],
        seed: list[float] | None = None,
    ) -> list[float]:
        """FK actor joints → tip, then ik_eef under taught wrist cap."""
        tip = eef_of(action)
        sol = ik_eef(
            tip,
            float(action[3]),
            float(action[4]),
            float(self.goal["wrist_angle_max"]),
            seed=seed if seed is not None else last_good,
        )
        if sol is None:
            self.get_logger().warn(
                f"ik_eef failed for tip ({tip[0]:.3f},{tip[1]:.3f},{tip[2]:.3f}) "
                f"under wrist_max={self.goal['wrist_angle_max']:.3f} — holding"
            )
            return list(last_good)
        return [
            _clamp("waist", sol["waist"]),
            _clamp("shoulder", sol["shoulder"]),
            _clamp("wrist_angle", min(sol["wrist_angle"], float(self.goal["wrist_angle_max"]))),
            _clamp("wrist_rotate", sol["wrist_rotate"]),
            _clamp("gripper", sol["gripper"]),
        ]

    def _cmd_from_timing_delta(self, q: list[float], action: list[float]) -> list[float]:
        """Apply joint deltas; close gate triggers foam-stall close. No re-IK."""
        nxt, close = apply_joint_delta(q, action, scale=0.15)
        if close:
            nxt[4] = CLOSED_GRIP
        return [_clamp(ARM_JOINTS[i], nxt[i]) for i in range(5)]

    def run(self) -> int:
        q = self._wait_js()
        self.get_logger().info(f"start from current {[round(x, 3) for x in q]}")
        # Factory sleep is ~-1.8 to -pi. Taught pole clicks often sit near
        # wrist_angle_max (~-1.48) — that is a valid grasp pose, not sleep.
        sleep_floor = min(float(self.goal["wrist_angle_max"]) - 0.12, -1.62)
        if q[2] < sleep_floor:
            self.get_logger().error(
                f"Arm wrist still in sleep fold ({q[2]:.2f} rad < {sleep_floor:.2f}). "
                "Move the wrist up toward the taught grasp, then re-run REPLAY."
            )
            return 3

        dt = 1.0 / max(self.args.hz, 1.0)
        last_a = [0.0] * 5
        hold = 0
        best = 1e9
        t0 = time.time()
        step = 0
        cmd = list(q)
        stall_hits = 0
        close_t0: float | None = None
        prev_grip: float | None = None
        stalled = False
        held = False
        while time.time() - t0 < self.args.episode_s:
            started = time.perf_counter()
            rclpy.spin_once(self, timeout_sec=0.0)
            if all(j in self._js for j in ARM_JOINTS):
                q = [self._js[j] for j in ARM_JOINTS]
            obs = self._obs(q, last_a)
            with torch.inference_mode():
                action = self.actor(torch.tensor(obs)).tolist()

            if self.timing:
                # Pose-error policy: deltas only. Foam stall → lift → held gate.
                cmd = self._cmd_from_timing_delta(q, action)
                self._publish(cmd)
                last_a = [float(a) for a in action]
                tip = _fk_px150(q[0], q[1], q[2], tool=TOOL)
                eef = [tip[i] + ARM_OFFSET[i] for i in range(3)]
                aim = self._aim()
                dist = math.dist(eef, aim)
                best = min(best, dist)
                step += 1
                if cmd[4] < -0.4:
                    if close_t0 is None:
                        close_t0 = time.time()
                    grip_now = float(q[4])
                    settled = (
                        prev_grip is not None and abs(grip_now - prev_grip) < 0.04
                    )
                    foam_gap = grip_now > (CLOSED_GRIP + 0.25)
                    if (
                        (time.time() - close_t0) >= 0.7
                        and foam_gap
                        and settled
                    ):
                        stall_hits += 1
                    else:
                        stall_hits = max(0, stall_hits - 1)
                    prev_grip = grip_now
                else:
                    stall_hits = max(0, stall_hits - 1)
                    close_t0 = None
                    prev_grip = None
                if step % 15 == 1:
                    self.get_logger().info(
                        f"timing step {step} dist={dist:.3f} stall_hits={stall_hits} "
                        f"q={[round(x, 2) for x in cmd]}"
                    )
                if stall_hits >= 8:
                    stalled = True
                    self.get_logger().info(
                        f"foam stall joint={q[4]:.3f} — lifting"
                    )
                    break
            else:
                # Legacy absolute actor → tip re-IK.
                alpha = float(self.args.blend)
                blended = [
                    (1.0 - alpha) * q[i] + alpha * float(action[i]) for i in range(4)
                ] + [float(action[4])]
                dist = math.sqrt(sum((obs[6 + i]) ** 2 for i in range(3)))
                if dist < max(NEAR_RADIUS * 2.5, 0.06):
                    blended[4] = float(self.goal["gripper"])
                blended[0] = max(-0.55, min(0.55, blended[0]))
                aim = self._aim()
                if dist > NEAR_RADIUS:
                    blended = fk_horizon_command(
                        blended,
                        q,
                        aim,
                        float(self.goal["wrist_angle_max"]),
                        hz=float(self.args.hz),
                    )
                    if dist < 0.08:
                        blended[4] = float(self.goal["gripper"])
                cmd = self._cmd_from_action(blended, cmd, seed=q)
                self._publish(cmd)
                last_a = [float(a) for a in action]
                step += 1
                grip_err = abs(float(obs[9]) - float(obs[10]))
                best = min(best, dist)
                at = dist < NEAR_RADIUS and grip_err < GRIP_TOL
                hold = hold + 1 if at else 0
                if step % 15 == 1:
                    self.get_logger().info(
                        f"step {step} dist={dist:.3f} grip_err={grip_err:.3f} "
                        f"hold={hold} tip re-IK q={[round(x, 2) for x in cmd]}"
                    )
                if hold >= 12:
                    self.get_logger().info("goal held — smoke success")
                    break
            rem = dt - (time.perf_counter() - started)
            if rem > 0:
                time.sleep(rem)

        if self.timing and stalled:
            # Same squeeze + slow lift used by ik_grasp_servo for a fair gate.
            squeeze = list(cmd)
            squeeze[4] = CLOSED_GRIP
            for _ in range(int(0.8 * self.args.hz)):
                self._publish(squeeze)
                rclpy.spin_once(self, timeout_sec=0.0)
                time.sleep(dt)
            lift = list(squeeze)
            shoulder0 = float(lift[1])
            n_lift = max(int(3.0 * self.args.hz), 16)
            for i in range(n_lift):
                frac = (i + 1) / float(n_lift)
                lift[1] = shoulder0 - 0.12 * frac
                lift[4] = CLOSED_GRIP
                self._publish(lift)
                rclpy.spin_once(self, timeout_sec=0.0)
                time.sleep(dt)
            grips: list[float] = []
            for _ in range(8):
                rclpy.spin_once(self, timeout_sec=0.05)
                if "gripper" in self._js:
                    grips.append(float(self._js["gripper"]))
                time.sleep(0.05)
            grip_med = float(np.median(grips)) if grips else CLOSED_GRIP
            held = grip_med > (CLOSED_GRIP + 0.12)
            self.get_logger().info(
                f"post-lift grip_med={grip_med:.3f} held={held} "
                f"thr={CLOSED_GRIP + 0.12:.3f}"
            )
            if held:
                hold = 12

        aim = self._aim()
        if self.timing:
            trial = ServoTrial(
                shaft_xy=[aim[0] - 0.06, aim[1]],
                shaft_z=aim[2] - 0.096,
                shaft_yaw=float(self.goal.get("wrist_rotate", 0.0)),
                stalled=stalled,
                lifted=stalled,
                held=held,
                steps=step,
                ik_live=True,
                wide_in_view=False,
                note=f"timing|{self.args.note or 'gate'}",
            )
            path = append_trial(trial, path=trials_dir() / "timing_trials.jsonl")
            self.get_logger().info(
                f"timing trial {path} stalled={stalled} held={held} steps={step}"
            )

        self.get_logger().info(
            f"done steps={step} best_dist={best:.3f} hold={hold} "
            f"stalled={stalled} held={held}"
        )
        hold_cmd = list(cmd)
        end = time.time() + 1.0
        while time.time() < end:
            self._publish(hold_cmd)
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(0.1)
        if self.timing:
            return 0 if held else (1 if stalled else 2)
        return 0 if hold >= 12 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path.home()
        / "dev-ws/logs/rsl_rl/grabit_goal_px150/model_50_smoke.pt",
    )
    ap.add_argument(
        "--goal",
        type=Path,
        default=Path.home() / "dev-ws/config/rl_goal.yaml",
    )
    ap.add_argument(
        "--hub",
        default="http://127.0.0.1:8082",
        help="AGX hub for live pole_xy (empty string disables).",
    )
    ap.add_argument(
        "--cam",
        default="http://127.0.0.1:8081",
        help="wide.jpg host for timing jaw-window obs (REAL MJPEG).",
    )
    ap.add_argument("--hz", type=float, default=12.0)
    ap.add_argument("--episode-s", type=float, default=16.0)
    ap.add_argument(
        "--blend",
        type=float,
        default=0.35,
        help="0..1 blend of policy into current joints before tip re-IK.",
    )
    ap.add_argument(
        "--timing",
        action="store_true",
        help="EARL timing mode: pose-error obs, joint deltas, no absolute re-IK.",
    )
    ap.add_argument(
        "--note",
        default="gate",
        help="Logged on timing trials (static / nudged) for baseline compare.",
    )
    args = ap.parse_args()
    if args.hub == "":
        args.hub = None
    if not args.checkpoint.is_file():
        print(f"missing checkpoint {args.checkpoint}", file=sys.stderr)
        return 2
    rclpy.init()
    node = GoalSmoke(args)
    try:
        return node.run()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
