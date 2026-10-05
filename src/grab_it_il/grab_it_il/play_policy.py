#!/usr/bin/env python3
"""Play a tabletop-pick policy (ACT checkpoint or recorded-episode replay)."""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image, JointState
from std_msgs.msg import String

from grab_it_il.common import (
    D435I_COMPRESSED,
    D435I_RAW,
    DEFAULT_DATASET,
    HZ,
    IL_COMMAND_TOPIC,
    IL_STATUS_TOPIC,
    JOINT_GROUP_TOPIC,
    JOINT_SINGLE_TOPIC,
    JOINT_STATES_TOPIC,
    STATE_NAMES,
    WRIST_COMPRESSED,
    WRIST_RAW,
    clamp_action,
    clamp_action_envelope,
    load_joint_envelope,
    parse_json_cmd,
)

try:
    from interbotix_xs_msgs.msg import JointGroupCommand, JointSingleCommand
except ImportError:  # pragma: no cover
    # Twin host play only needs sensor_msgs/JointState → il_twin_cmd_relay.
    JointGroupCommand = None  # type: ignore[misc, assignment]
    JointSingleCommand = None  # type: ignore[misc, assignment]

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


_SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

HUB = os.environ.get("GRAB_IT_AGX_HUB_BASE", "http://127.0.0.1:8081").rstrip("/")


def _bgr_from_jpeg(blob: Optional[bytes]) -> Optional[np.ndarray]:
    if cv2 is None or not blob:
        return None
    img = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img


def _bgr_from_raw(msg: Image) -> Optional[np.ndarray]:
    if cv2 is None or not msg.data:
        return None
    h, w = int(msg.height), int(msg.width)
    enc = (msg.encoding or "rgb8").lower()
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    try:
        if enc == "bgr8":
            return buf.reshape(h, w, 3)
        if enc == "rgb8":
            return cv2.cvtColor(buf.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
    except ValueError:
        return None
    return None


def _http_bytes(url: str, timeout: float = 0.25) -> Optional[bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read()
    except Exception:
        return None


def _fetch_pole_xy() -> Optional[list[float]]:
    """Pole base XY. Twin hub in sim; env fallback only when nothing measured it."""
    try:
        with urllib.request.urlopen(f"{HUB}/isaac/twin", timeout=0.25) as resp:
            twin = json.loads(resp.read().decode())
    except Exception:
        twin = None
    if isinstance(twin, dict):
        xy = twin.get("pole_xy")
        if isinstance(xy, (list, tuple)) and len(xy) >= 2:
            return [float(xy[0]), float(xy[1])]
    raw = os.environ.get("GRAB_IT_IL_POLE_XY", "").strip()
    if raw:
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) >= 2:
            try:
                return [float(parts[0]), float(parts[1])]
            except ValueError:
                return None
    return None


# rad/s. Shoulder is the slowest so a policy delta cannot fold into the stack.
_SLEW_RAD_S = (0.35, 0.22, 0.30, 0.40, 2.0, 0.40)


class ReplayPolicy:
    """Open-loop replay of a recorded episode (no torch required)."""

    def __init__(self, episode_dir: Path) -> None:
        actions = np.load(episode_dir / "actions.npy")
        if actions.ndim != 2 or actions.shape[1] < 5:
            raise ValueError(f"bad actions.npy in {episode_dir}")
        self._actions = actions.astype(np.float32)
        self._i = 0
        self.needs_env_state = False

    def reset(self) -> None:
        self._i = 0

    def act(self, state, images, env_state=None) -> Optional[np.ndarray]:
        if self._i >= len(self._actions):
            return None
        a = self._actions[self._i]
        self._i += 1
        return a

    @property
    def done(self) -> bool:
        return self._i >= len(self._actions)


class ActPolicy:
    """LeRobot ACT checkpoint. Imported only when a checkpoint is given."""

    def __init__(
        self,
        checkpoint: Path,
        device: str = "cuda",
        *,
        n_action_steps: int = 1,
        temporal_ensemble_coeff: float = 0.01,
    ) -> None:
        try:
            from lerobot.common.policies.act.modeling_act import (
                ACTPolicy,
                ACTTemporalEnsembler,
            )
        except ImportError:
            # lerobot.policies.__init__ imports Groot, which fails to load on
            # this venv. Register the package without running that init.
            import sys
            import types

            import lerobot

            policies_dir = Path(lerobot.__file__).resolve().parent / "policies"
            if "lerobot.policies" not in sys.modules:
                pkg = types.ModuleType("lerobot.policies")
                pkg.__path__ = [str(policies_dir)]
                pkg.__package__ = "lerobot.policies"
                sys.modules["lerobot.policies"] = pkg
            from lerobot.policies.act.modeling_act import (  # type: ignore
                ACTPolicy,
                ACTTemporalEnsembler,
            )
        self._policy = ACTPolicy.from_pretrained(str(checkpoint))
        # Open-loop 100-step chunks teach early close + pecking. Original ACT
        # queries every step with temporal ensembling (coeff=0.01, n_action_steps=1).
        cfg = self._policy.config
        cfg.n_action_steps = int(max(1, n_action_steps))
        if temporal_ensemble_coeff is not None and float(temporal_ensemble_coeff) > 0.0:
            cfg.temporal_ensemble_coeff = float(temporal_ensemble_coeff)
            cfg.n_action_steps = 1  # required by LeRobot when ensembling
            from collections import deque

            self._policy.temporal_ensembler = ACTTemporalEnsembler(
                cfg.temporal_ensemble_coeff, cfg.chunk_size
            )
            self._policy._action_queue = deque([], maxlen=1)
        else:
            cfg.temporal_ensemble_coeff = None
            from collections import deque

            self._policy._action_queue = deque([], maxlen=cfg.n_action_steps)
        self._policy.to(device)
        self._policy.eval()
        self._device = device
        self._pre = None
        self._post = None
        self._held_inputs: dict = {}
        self.held_names: list[str] = []
        if (checkpoint / "policy_preprocessor.json").is_file():
            self._load_processors(checkpoint, device)
        img_feats = getattr(cfg, "image_features", None)
        if isinstance(img_feats, dict):
            self._image_keys = list(img_feats.keys())
        elif img_feats:
            try:
                self._image_keys = list(img_feats)
            except TypeError:
                self._image_keys = []
        else:
            self._image_keys = []
        if not self._image_keys:
            feats = getattr(cfg, "input_features", {}) or {}
            self._image_keys = [
                k for k in feats if str(k).startswith("observation.images.")
            ]
        if not self._image_keys:
            self._image_keys = ["observation.images.d435i"]

        feats = getattr(cfg, "input_features", {}) or {}
        self.needs_env_state = (
            "observation.environment_state" in feats
            or getattr(cfg, "env_state_feature", None) is not None
        )
        env_feat = getattr(cfg, "env_state_feature", None)
        if env_feat is not None and hasattr(env_feat, "shape"):
            self._env_dim = int(env_feat.shape[0])
        elif "observation.environment_state" in feats:
            shape = getattr(feats["observation.environment_state"], "shape", (2,))
            self._env_dim = int(shape[0]) if shape else 2
        else:
            self._env_dim = 2

    def _load_processors(self, checkpoint: Path, device: str) -> None:
        """LeRobot >= 0.4 keeps normalization beside the weights, not inside the model.

        Without these steps the network sees raw radians and its normalized
        output is sent to the servos as radians. A joint whose demos barely
        moved (std ~0.1 deg) then turns ~57 deg per unit of output.
        """
        import torch
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.processor.converters import (
            batch_to_transition,
            policy_action_to_transition,
            transition_to_batch,
            transition_to_policy_action,
        )
        from safetensors.torch import load_file

        self._pre = PolicyProcessorPipeline.from_pretrained(
            str(checkpoint),
            config_filename="policy_preprocessor.json",
            overrides={"device_processor": {"device": device}},
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
        )
        self._post = PolicyProcessorPipeline.from_pretrained(
            str(checkpoint),
            config_filename="policy_postprocessor.json",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
        # An input that never varied in the demos has a near-zero std, so a
        # small live offset normalizes to tens of sigma. Feed the demo value.
        files = sorted(
            checkpoint.glob("policy_preprocessor_step_*_normalizer_processor.safetensors")
        )
        if not files:
            return
        stats = load_file(str(files[0]))
        for key, floor, names in (
            ("observation.state", 0.01, STATE_NAMES),
            ("observation.environment_state", 0.005, ["pole_x", "pole_y"]),
        ):
            std = stats.get(f"{key}.std")
            mean = stats.get(f"{key}.mean")
            if std is None or mean is None:
                continue
            mask = std.reshape(-1) < floor
            if not bool(mask.any()):
                continue
            self._held_inputs[key] = (
                mask.to(device),
                mean.reshape(-1).to(device=device, dtype=torch.float32),
            )
            self.held_names.extend(
                names[i] if i < len(names) else f"{key}[{i}]"
                for i in range(mask.numel())
                if bool(mask[i])
            )

    def reset(self) -> None:
        reset = getattr(self._policy, "reset", None)
        if callable(reset):
            reset()

    def act(self, state, images, env_state=None) -> Optional[np.ndarray]:
        import torch

        obs = {
            "observation.state": torch.as_tensor(state, dtype=torch.float32)
            .unsqueeze(0)
            .to(self._device)
        }

        def _to_chw(img) -> "torch.Tensor":
            if img is None or cv2 is None:
                rgb = np.zeros((240, 320, 3), dtype=np.uint8)
            else:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (320, 240))
            return torch.as_tensor(rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0

        cam_alias = {
            "observation.images.d435i": "d435i",
            "observation.images.wrist": "wrist",
        }
        for key in self._image_keys:
            alias = cam_alias.get(key, key.rsplit(".", 1)[-1])
            obs[key] = _to_chw(images.get(alias)).unsqueeze(0).to(self._device)

        if self.needs_env_state:
            if env_state is None:
                env_vec = [0.0] * self._env_dim
            else:
                env_vec = [float(v) for v in list(env_state)[: self._env_dim]]
                while len(env_vec) < self._env_dim:
                    env_vec.append(0.0)
            obs["observation.environment_state"] = (
                torch.as_tensor(env_vec, dtype=torch.float32)
                .unsqueeze(0)
                .to(self._device)
            )

        for key, (mask, mean) in self._held_inputs.items():
            if key in obs:
                obs[key] = torch.where(mask, mean, obs[key])

        with torch.inference_mode():
            if self._pre is not None:
                obs = self._pre(obs)
            action = self._policy.select_action(obs)
            if self._post is not None:
                action = self._post(action)
        if hasattr(action, "detach"):
            action = action.detach().cpu().numpy()
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        return action[: len(STATE_NAMES)]


class PlayPolicy(Node):
    def __init__(self) -> None:
        super().__init__("grab_it_il_play")
        self.declare_parameter("checkpoint", "")
        self.declare_parameter("replay_episode", "")
        self.declare_parameter("dataset_dir", str(DEFAULT_DATASET))
        self.declare_parameter("autoplay", False)
        self.declare_parameter("device", "cuda")
        self.declare_parameter("n_action_steps", 1)
        self.declare_parameter("temporal_ensemble_coeff", 0.01)
        self.declare_parameter("joint_envelope", "")

        ckpt = str(self.get_parameter("checkpoint").value or "").strip()
        replay = str(self.get_parameter("replay_episode").value or "").strip()
        root = Path(self.get_parameter("dataset_dir").value)
        self._dataset_dir = root
        self._replay_policy: Optional[ReplayPolicy] = None
        self._replay_active = False
        device = str(self.get_parameter("device").value or "cuda")
        n_action_steps = int(self.get_parameter("n_action_steps").value or 1)
        te_coeff = float(self.get_parameter("temporal_ensemble_coeff").value or 0.0)
        env_path = str(self.get_parameter("joint_envelope").value or "").strip()

        self._policy = None
        self._kind = "none"
        ckpt_path = Path(ckpt) if ckpt else None
        if ckpt:
            self._policy = ActPolicy(
                Path(ckpt),
                device=device,
                n_action_steps=n_action_steps,
                temporal_ensemble_coeff=te_coeff,
            )
            self._kind = "act"
            self.get_logger().info(
                f"ACT play n_action_steps={self._policy._policy.config.n_action_steps} "
                f"temporal_ensemble={self._policy._policy.config.temporal_ensemble_coeff} "
                f"env_state={self._policy.needs_env_state} "
                f"normalizer={'saved' if self._policy._pre is not None else 'none'} "
                f"held_inputs={self._policy.held_names or '-'}"
            )
        elif replay:
            self._policy = ReplayPolicy(Path(replay))
            self._kind = "replay"
        else:
            latest = _latest_episode(root)
            if latest is not None:
                self._policy = ReplayPolicy(latest)
                self._kind = "replay"
                self.get_logger().info(f"no checkpoint — will replay {latest.name}")
            else:
                self.get_logger().warn(
                    "no checkpoint or episode — play will hold the current pose"
                )

        # Dataset joint envelope: bad actions hold last in-range command.
        if not env_path and ckpt_path is not None:
            for cand in (
                ckpt_path / "joint_envelope.json",
                ckpt_path.parent / "joint_envelope.json",
                ckpt_path.parent.parent / "joint_envelope.json",
                ckpt_path.parent.parent.parent / "joint_envelope.json",
            ):
                if cand.is_file():
                    env_path = str(cand)
                    break
        self._envelope = load_joint_envelope(env_path)
        self._last_ok_action: Optional[list[float]] = None
        # Shoulder demos lock the waist after the aim. Without this, ACT's
        # waist channel (and the ±0.16 dataset box) yaws the arm back to
        # center and only then reaches.
        lock_raw = os.environ.get("GRAB_IT_IL_WAIST_LOCK", "").strip()
        self._waist_lock: Optional[float] = None
        self._waist_lock_hold = False
        self._sleep_hold = os.environ.get("GRAB_IT_IL_SLEEP_HOLD", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        self._sleep_joints: Optional[tuple[float, float, float]] = None
        self._from_current = os.environ.get("GRAB_IT_IL_FROM_CURRENT", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        self._rel_action: Optional[list[float]] = None
        self._rel_state: Optional[list[float]] = None
        self._pole_from_click = False
        self._grip_absolute = False
        self._shoulder_floor: Optional[float] = None
        self._pending_play: dict = {}
        self._jpeg_poll_t = 0.0
        if lock_raw.lower() in ("hold", "live", "now"):
            # Freeze whatever waist the arm already has. A number sampled from
            # the twin can still be the folded park pose and yaws off the pole.
            self._waist_lock_hold = True
            self.get_logger().info("waist will stay where it is when ACT starts")
        elif lock_raw:
            try:
                self._waist_lock = float(lock_raw)
                self.get_logger().info(f"waist locked at {self._waist_lock:.3f} for this play")
            except ValueError:
                self.get_logger().warn(f"ignoring GRAB_IT_IL_WAIST_LOCK={lock_raw!r}")
        # Jaw roll is set by the pre-ACT wrist align. The dataset box is only
        # ±0.16 rad, so without this lock ACT spins the jaws off the shaft.
        rot_raw = os.environ.get("GRAB_IT_IL_WRIST_ROT_LOCK", "").strip()
        self._wrist_rot_lock: Optional[float] = None
        if rot_raw:
            try:
                self._wrist_rot_lock = float(rot_raw)
                self.get_logger().info(
                    f"wrist_rotate locked at {self._wrist_rot_lock:.3f} for this play"
                )
            except ValueError:
                self.get_logger().warn(f"ignoring GRAB_IT_IL_WRIST_ROT_LOCK={rot_raw!r}")
        self._waist_lock_env = self._waist_lock
        self._waist_lock_hold_env = self._waist_lock_hold
        if self._envelope:
            self.get_logger().info(f"dataset joint envelope from {env_path}")
        else:
            self.get_logger().info("no dataset joint envelope — hard joint limits only")

        self._lock = threading.Lock()
        self._playing = False
        self._state = [0.0] * len(STATE_NAMES)
        self._have_state = False
        # Finger width is measured. A close command the fingers cannot reach
        # is the sim stand-in for gripper current: the jaws have stalled on
        # the pole. Hold that error briefly so a normal close-in-progress
        # does not look like a grasp.
        self._stall_prev_t: Optional[float] = None
        self._stall_prev_meas: Optional[float] = None
        self._stall_t0: Optional[float] = None
        self._d435i: Optional[np.ndarray] = None
        self._wrist: Optional[np.ndarray] = None
        self._pole_xy: Optional[list[float]] = None

        self._arm_pub = None
        self._grip_pub = None
        if JointGroupCommand is not None and JointSingleCommand is not None:
            self._arm_pub = self.create_publisher(
                JointGroupCommand, JOINT_GROUP_TOPIC, 10
            )
            self._grip_pub = self.create_publisher(
                JointSingleCommand, JOINT_SINGLE_TOPIC, 10
            )
        else:
            self.get_logger().warn(
                "interbotix_xs_msgs missing — publishing joint_targets only (twin relay)"
            )
        self._js_cmd_pub = self.create_publisher(
            JointState, "/px150_no_elbow/commands/joint_targets", 10
        )
        self._status_pub = self.create_publisher(String, IL_STATUS_TOPIC, 10)
        self.create_subscription(String, IL_COMMAND_TOPIC, self._cmd_cb, 10)
        self.create_subscription(String, "/grab_it/pole_xy", self._pole_cb, 10)
        self.create_subscription(
            JointState, JOINT_STATES_TOPIC, self._js_cb, _SENSOR_QOS
        )
        self.create_subscription(
            CompressedImage, D435I_COMPRESSED, self._d435i_jpg_cb, _SENSOR_QOS
        )
        self.create_subscription(
            CompressedImage, WRIST_COMPRESSED, self._wrist_jpg_cb, _SENSOR_QOS
        )
        self.create_timer(1.0 / HZ, self._tick)
        self._autoplay = bool(self.get_parameter("autoplay").value)
        self._autoplay_armed = False
        if self._autoplay:
            self.get_logger().info(
                "autoplay armed — waiting for joint_states + camera frames"
            )
            self._publish_status(playing=False, message=f"autoplay wait ({self._kind})")
        else:
            self._publish_status(playing=False, message=f"player idle ({self._kind})")
        self.get_logger().info(f"IL player kind={self._kind}")

    def _obs_ready(self) -> bool:
        need_env = bool(
            getattr(self._policy, "needs_env_state", False)
        )
        if need_env and self._pole_xy is None:
            # Try one fetch before declaring not ready.
            self._pole_xy = _fetch_pole_xy()
            if self._pole_xy is None:
                return False
        return bool(self._have_state) and self._d435i is not None

    def _publish_status(self, **extra) -> None:
        payload = {"playing": self._playing, "kind": self._kind}
        payload.update(extra)
        self._status_pub.publish(String(data=json.dumps(payload)))

    def _pole_cb(self, msg: String) -> None:
        data = parse_json_cmd(msg.data)
        xy = data.get("pole_xy")
        if not (isinstance(xy, (list, tuple)) and len(xy) >= 2):
            if "x" in data and "y" in data:
                xy = [data.get("x"), data.get("y")]
            else:
                return
        try:
            live = [float(xy[0]), float(xy[1])]
        except (TypeError, ValueError):
            return
        self._pole_xy = live
        self._pole_from_click = True

    def _fill_images_http(self) -> None:
        now = time.monotonic()
        if now - self._jpeg_poll_t < 0.15:
            return
        self._jpeg_poll_t = now
        if self._d435i is None:
            img = _bgr_from_jpeg(_http_bytes(f"{HUB}/d435i.jpg"))
            if img is not None:
                self._d435i = img
        if self._wrist is None:
            img = _bgr_from_jpeg(_http_bytes(f"{HUB}/wide.jpg"))
            if img is not None:
                self._wrist = img

    def _cmd_cb(self, msg: String) -> None:
        data = parse_json_cmd(msg.data)
        cmd = str(data.get("cmd") or data.get("type") or "").strip().lower()
        if cmd in ("play", "start_play"):
            self._pending_play = data
            self._start()
        elif cmd in ("stop", "stop_play"):
            self._stop("stopped")

    def _js_cb(self, msg: JointState) -> None:
        names = list(msg.name)
        pos = list(msg.position)
        for i, joint in enumerate(STATE_NAMES):
            if joint in names:
                self._state[i] = float(pos[names.index(joint)])
                self._have_state = True

    def _d435i_jpg_cb(self, msg: CompressedImage) -> None:
        img = _bgr_from_jpeg(bytes(msg.data) if msg.data else None)
        if img is not None:
            self._d435i = img

    def _d435i_raw_cb(self, msg: Image) -> None:
        if self._d435i is not None:
            return
        img = _bgr_from_raw(msg)
        if img is not None:
            self._d435i = img

    def _wrist_jpg_cb(self, msg: CompressedImage) -> None:
        img = _bgr_from_jpeg(bytes(msg.data) if msg.data else None)
        if img is not None:
            self._wrist = img

    def _wrist_raw_cb(self, msg: Image) -> None:
        if self._wrist is not None:
            return
        img = _bgr_from_raw(msg)
        if img is not None:
            self._wrist = img

    def _slew(self, action: list[float], last: Optional[list[float]]) -> list[float]:
        if not last:
            return action
        dt = 1.0 / float(HZ)
        out = list(action)
        for i, rate in enumerate(_SLEW_RAD_S):
            if i >= len(out) or i >= len(last):
                break
            step = float(rate) * dt
            out[i] = max(float(last[i]) - step, min(float(last[i]) + step, float(out[i])))
        return out

    def _start(self) -> None:
        data = self._pending_play or {}
        self._pending_play = {}
        with self._lock:
            if self._playing:
                return
            pole = data.get("pole_xy")
            if isinstance(pole, (list, tuple)) and len(pole) >= 2:
                try:
                    self._pole_xy = [float(pole[0]), float(pole[1])]
                    self._pole_from_click = True
                except (TypeError, ValueError):
                    pass
            if not self._obs_ready():
                self.get_logger().warn(
                    "PLAY ignored — need joint_states + d435i frame first"
                )
                self._publish_status(
                    playing=False,
                    message="wait_obs",
                    have_state=self._have_state,
                    have_d435i=self._d435i is not None,
                    have_pole_xy=self._pole_xy is not None,
                )
                return
            if self._from_current and any(abs(float(v)) > 3.0 for v in self._state[:4]):
                self.get_logger().warn(
                    "PLAY ignored — joint read failed, not starting from a bad pose"
                )
                self._publish_status(playing=False, message="bad_joints")
                return
            self._playing = True
            self._last_ok_action = list(self._state)
            self._sleep_joints = None
            self._rel_action = None
            self._rel_state = None
            self._replay_active = str(data.get("mode") or "") == "replay"
            self._replay_policy = None
            if self._replay_active:
                latest = None
                search = []
                try:
                    col = Path("/tmp/grabit_collection").read_text(encoding="utf-8").strip()
                except OSError:
                    col = ""
                if col and col[0].isalnum() and all(c.isalnum() or c in "_-" for c in col):
                    search.append(Path("/home/ubuntu/dev-ws/datasets") / col)
                    search.append(Path("/workspaces/isaac_ros-dev/datasets") / col)
                search.append(self._dataset_dir)
                search.extend((
                    Path("/home/ubuntu/dev-ws/datasets/tabletop_pick"),
                    Path("/workspaces/isaac_ros-dev/datasets/tabletop_pick"),
                ))
                seen = set()
                for extra in search:
                    key = str(extra)
                    if key in seen:
                        continue
                    seen.add(key)
                    latest = _latest_episode(extra)
                    if latest is not None:
                        break
                if latest is None:
                    self._playing = False
                    self._replay_active = False
                    self.get_logger().warn("REPLAY ignored — no saved episode")
                    self._publish_status(playing=False, message="no episode to replay")
                    return
                self._replay_policy = ReplayPolicy(latest)
                self._grip_absolute = False
                self.get_logger().info(f"REPLAY {latest.name} from the current pose")
            else:
                self._grip_absolute = str(data.get("from") or "") == "click"
            if (self._from_current or self._replay_active) and len(self._state) > 1:
                self._shoulder_floor = float(self._state[1])
            else:
                self._shoulder_floor = None
            if self._grip_absolute:
                self._waist_lock = None
                self._waist_lock_hold = True
                floor = self._shoulder_floor
                xy = self._pole_xy
                self.get_logger().info(
                    "ACT close from click"
                    + (f" pole=({xy[0]:.3f},{xy[1]:.3f})" if xy else "")
                    + (f" shoulder floor={floor:.3f}" if floor is not None else "")
                )
            if (
                self._sleep_hold
                and not self._from_current
                and len(self._state) >= 4
            ):
                shoulder = float(self._state[1])
                wrist = float(self._state[2])
                # Real sleep is folded back. ACT's reach from there swings the
                # forearm into the electronics. Turn the waist only.
                if abs(shoulder - (-0.2321)) < 0.55 and abs(wrist - (-1.7279)) < 0.55:
                    self._sleep_joints = (
                        shoulder,
                        wrist,
                        float(self._state[3]),
                    )
                    self._waist_lock = None
                    self._waist_lock_hold = False
                    self.get_logger().info(
                        "sleep pose — rotating waist only, arm stays folded"
                    )
            if self._waist_lock_hold:
                self._waist_lock = None
            if self._policy is not None and not self._replay_active:
                self._policy.reset()
        kind = "replay" if self._replay_active else "act"
        self.get_logger().info(f"policy PLAY ({kind})")
        self._publish_status(
            playing=True,
            message="replaying last take" if self._replay_active else "playing",
            kind=kind,
        )

    def _stop(self, reason: str) -> None:
        with self._lock:
            self._playing = False
            self._replay_active = False
            self._replay_policy = None
            self._grip_absolute = False
            self._waist_lock = self._waist_lock_env
            self._waist_lock_hold = self._waist_lock_hold_env
        self.get_logger().info(f"policy STOP ({reason})")
        self._publish_status(playing=False, message=reason)

    def _grasp_stalled(self, meas: float, cmd: float) -> bool:
        """True when a close command is stuck against something in the jaws.

        An empty close reaches the command. A pole stops the fingers short of
        it and the measured width stops changing. That position error is the
        simulated motor resistance.
        """
        now = time.monotonic()
        err = float(meas) - float(cmd)
        closing = float(cmd) < 0.20 and err > 0.45
        rate = None
        if self._stall_prev_t is None:
            self._stall_prev_t = now
            self._stall_prev_meas = float(meas)
        else:
            dt = now - float(self._stall_prev_t)
            prev = float(self._stall_prev_meas if self._stall_prev_meas is not None else meas)
            if abs(float(meas) - prev) > 0.02 or dt >= 0.25:
                rate = (prev - float(meas)) / max(dt, 1e-3)
                self._stall_prev_t = now
                self._stall_prev_meas = float(meas)
        if not closing:
            self._stall_t0 = None
            return False
        if rate is None:
            return False
        if rate >= 0.15:
            self._stall_t0 = None
            return False
        if self._stall_t0 is None:
            self._stall_t0 = now
            return False
        return (now - float(self._stall_t0)) >= 0.60

    def _tick(self) -> None:
        self._fill_images_http()
        # A click measurement wins over the fixed 0.22,0 fallback.
        if getattr(self._policy, "needs_env_state", False) and not self._pole_from_click:
            xy = _fetch_pole_xy()
            if xy is not None:
                self._pole_xy = xy
        if self._autoplay and not self._autoplay_armed and self._obs_ready():
            self._autoplay_armed = True
            self.get_logger().info("autoplay: obs ready — starting PLAY")
            self._start()
        with self._lock:
            if not self._playing:
                return
            if not self._obs_ready():
                return
            state = list(self._state)
            images = {"d435i": self._d435i, "wrist": self._wrist}
            env_state = list(self._pole_xy) if self._pole_xy is not None else None
            if self._replay_active and self._replay_policy is not None:
                policy = self._replay_policy
            else:
                policy = self._policy
            last_ok = list(self._last_ok_action) if self._last_ok_action else None
            replay_active = self._replay_active
        if policy is None:
            action = state
        else:
            action = policy.act(state, images, env_state=env_state)
            if action is None:
                self._stop("episode done")
                return
        if self._from_current or replay_active:
            raw = [float(v) for v in action]
            if self._rel_action is None:
                self._rel_action = list(raw)
                self._rel_state = list(state)
                self.get_logger().info(
                    "from current pose — not moving to the demo park"
                )
            n = min(len(raw), len(self._rel_action), len(self._rel_state))
            action = [
                float(self._rel_state[i]) + (float(raw[i]) - float(self._rel_action[i]))
                for i in range(n)
            ]
            # The close is the policy gripper target. A relative delta stays
            # open when the first prediction is already "closed".
            if self._grip_absolute and len(raw) > 4 and len(action) > 4:
                action[4] = float(raw[4])
            action = clamp_action(action)
            new_ok = list(action)
            held = False
        else:
            hard = clamp_action(action)
            action, new_ok, held = clamp_action_envelope(
                hard, self._envelope, last_ok=last_ok
            )
        if self._waist_lock_hold and self._waist_lock is None and len(state) > 0:
            self._waist_lock = float(state[0])
            self.get_logger().info(
                f"waist locked at {self._waist_lock:.3f} — already facing, not turning"
            )
        if self._waist_lock is not None or self._wrist_rot_lock is not None:
            action = [float(v) for v in action]
            if self._waist_lock is not None:
                action[0] = float(self._waist_lock)
            if self._wrist_rot_lock is not None:
                action[3] = float(self._wrist_rot_lock)
            new_ok = list(action)
        if self._sleep_joints is not None:
            action = [float(v) for v in action]
            action[1], action[2], action[3] = self._sleep_joints
            new_ok = list(action)
        if self._shoulder_floor is not None and len(action) > 1:
            action = [float(v) for v in action]
            action[1] = max(float(action[1]), float(self._shoulder_floor))
            new_ok = list(action)
        if self._from_current or replay_active:
            action = self._slew(list(action), last_ok)
            new_ok = list(action)
        with self._lock:
            self._last_ok_action = new_ok
        if held:
            # Rare; do not spam — status carries a flag for debugging.
            self._publish_status(playing=True, message="envelope_hold")
        meas = float(state[4]) if len(state) > 4 else 0.0
        cmd = float(action[4]) if len(action) > 4 else meas
        if self._grasp_stalled(meas, cmd):
            self.get_logger().info(
                f"grasp stall meas={meas:.2f} cmd={cmd:.2f} "
                f"err={meas - cmd:.2f} — stopping ACT"
            )
            self._stop("grasp stall")
            return
        if self._arm_pub is not None and JointGroupCommand is not None:
            arm = JointGroupCommand()
            arm.name = "arm"
            arm.cmd = [float(action[i]) for i in range(4)]
            self._arm_pub.publish(arm)
        if self._grip_pub is not None and JointSingleCommand is not None:
            grip = JointSingleCommand()
            grip.name = "gripper"
            grip.cmd = float(action[4])
            self._grip_pub.publish(grip)
        js = JointState()
        js.name = list(STATE_NAMES)
        js.position = [float(v) for v in action[: len(STATE_NAMES)]]
        self._js_cmd_pub.publish(js)


def _latest_episode(root: Path) -> Optional[Path]:
    if not root.is_dir():
        return None
    eps = sorted(
        [p for p in root.iterdir() if p.is_dir() and (p / "actions.npy").is_file()],
        key=lambda p: p.name,
    )
    return eps[-1] if eps else None


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PlayPolicy()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
