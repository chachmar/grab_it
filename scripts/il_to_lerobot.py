#!/usr/bin/env python3
"""Pack grab_it episode folders into a LeRobot dataset (AGX host).

LeRobot >= 0.4 expects dataset format **v3.0**. When `lerobot` is importable
this script writes v3.0 via `LeRobotDataset.create`. Otherwise it writes the
legacy v2.1 layout (not trainable with 0.4.x).

Example (AGX host venv, not isaac_ros_dev):
  python3 scripts/il_to_lerobot.py \\
    --input ~/dev-ws/datasets/tabletop_pick \\
    --output ~/dev-ws/datasets/lerobot/grabit_tabletop_pick
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

SCRIPT_ID = "il_to_lerobot v3-writer+env_state"

STATE_NAMES = ["waist", "shoulder", "wrist_angle", "wrist_rotate", "gripper", "pan"]
ENV_STATE_NAMES = ["pole_x", "pole_y"]
IMAGE_H, IMAGE_W = 240, 320
REPO_ID = "local/grabit_tabletop_pick"


def _pad_sa(vec: np.ndarray) -> np.ndarray:
    """Ensure state/action length matches STATE_NAMES (pad pan=0 for old 5-DoF eps)."""
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    need = len(STATE_NAMES)
    if v.shape[0] == need:
        return v
    out = np.zeros((need,), dtype=np.float32)
    out[: min(need, v.shape[0])] = v[: min(need, v.shape[0])]
    return out


def _load_pole_xy(ep: Path, n: int) -> np.ndarray | None:
    """Load per-frame pole XY → observation.environment_state (n, 2)."""
    path = ep / "pole_xy.npy"
    if not path.is_file():
        return None
    arr = np.asarray(np.load(path), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None
    out = np.zeros((n, 2), dtype=np.float32)
    m = min(n, arr.shape[0])
    out[:m] = arr[:m, :2]
    if m < n:
        out[m:] = out[m - 1]
    return out


def _episodes(src: Path) -> list[Path]:
    eps = []
    for p in sorted(src.iterdir()):
        if not p.is_dir() or not p.name.startswith("episode_"):
            continue
        if not (p / "states.npy").is_file() or not (p / "actions.npy").is_file():
            continue
        meta = {}
        if (p / "meta.json").is_file():
            meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))
        if meta.get("success") is False:
            continue
        eps.append(p)
    return eps


def _try_resize_jpegs(src: Path, dest: Path, fps: int) -> bool:
    jpgs = sorted(src.glob("*.jpg")) if src.is_dir() else []
    if not jpgs:
        return False
    try:
        import cv2
    except ImportError:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(dest), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (IMAGE_W, IMAGE_H)
    )
    if not writer.isOpened():
        return False
    ok = False
    for p in jpgs:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        writer.write(cv2.resize(img, (IMAGE_W, IMAGE_H)))
        ok = True
    writer.release()
    return ok and dest.is_file()


def _try_resize_video(src: Path, dest: Path, fps: int) -> bool:
    try:
        import cv2
    except ImportError:
        shutil.copy2(src, dest)
        return True
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(dest), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (IMAGE_W, IMAGE_H)
    )
    ok = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(cv2.resize(frame, (IMAGE_W, IMAGE_H)))
        ok = True
    cap.release()
    writer.release()
    return ok


def _write_parquet(out: Path, rows: list[dict]) -> bool:
    try:
        import pandas as pd
    except ImportError:
        return False
    df = pd.DataFrame(rows)
    chunk = out / "data" / "chunk-000"
    chunk.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(chunk / "file-000.parquet", index=False)
        return True
    except Exception:
        df.to_json(chunk / "file-000.jsonl", orient="records", lines=True)
        return False


def _cam_exists(ep: Path, stem: str) -> bool:
    if (ep / f"{stem}.mp4").is_file() or (ep / f"{stem}.avi").is_file():
        return True
    folder = ep / stem
    return folder.is_dir() and any(folder.glob("*.jpg"))


def _bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    import cv2

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[0] != IMAGE_H or rgb.shape[1] != IMAGE_W:
        rgb = cv2.resize(rgb, (IMAGE_W, IMAGE_H), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def _load_cam_rgb(ep: Path, stem: str, n: int) -> np.ndarray | None:
    """n frames of RGB uint8 (H, W, 3), or None if this episode has no camera."""
    try:
        import cv2
    except ImportError:
        return None
    frames: list[np.ndarray] = []
    video = ep / f"{stem}.mp4"
    if not video.is_file():
        video = ep / f"{stem}.avi"
    if video.is_file():
        cap = cv2.VideoCapture(str(video))
        if cap.isOpened():
            while True:
                ret, bgr = cap.read()
                if not ret:
                    break
                frames.append(_bgr_to_rgb(bgr))
        cap.release()
    else:
        folder = ep / stem
        if folder.is_dir():
            for jpg in sorted(folder.glob("*.jpg")):
                bgr = cv2.imread(str(jpg), cv2.IMREAD_COLOR)
                if bgr is not None:
                    frames.append(_bgr_to_rgb(bgr))
    if not frames:
        return None
    while len(frames) < n:
        frames.append(frames[-1])
    return np.stack(frames[:n], axis=0)


def _black(n: int) -> np.ndarray:
    return np.zeros((n, IMAGE_H, IMAGE_W, 3), dtype=np.uint8)


def _looks_like_codec_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    needles = (
        "ffmpeg",
        "codec",
        "encoder",
        "libx264",
        "libsvtav1",
        "h264",
        "unknown encoder",
        "av_write",
    )
    return any(n in text for n in needles)


def _write_v30(
    dest: Path,
    episodes: list[Path],
    fps: int,
    task: str,
    repo_id: str,
    images_only: bool,
    vcodec: str | None,
) -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    has_d435i = any(_cam_exists(ep, "d435i") for ep in episodes)
    has_wrist = any(_cam_exists(ep, "wrist") for ep in episodes)
    has_env = any((ep / "pole_xy.npy").is_file() for ep in episodes)
    img_dtype = "image" if images_only else "video"
    features: dict = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": STATE_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": STATE_NAMES,
        },
    }
    if has_env:
        features["observation.environment_state"] = {
            "dtype": "float32",
            "shape": (len(ENV_STATE_NAMES),),
            "names": ENV_STATE_NAMES,
        }
    if has_d435i:
        features["observation.images.d435i"] = {
            "dtype": img_dtype,
            "shape": (IMAGE_H, IMAGE_W, 3),
            "names": ["height", "width", "channel"],
        }
    if has_wrist:
        features["observation.images.wrist"] = {
            "dtype": img_dtype,
            "shape": (IMAGE_H, IMAGE_W, 3),
            "names": ["height", "width", "channel"],
        }

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)

    create_kw: dict = {
        "repo_id": repo_id,
        "fps": fps,
        "root": dest,
        "robot_type": "grab_it_px150",
        "features": features,
        "use_videos": not images_only,
        "image_writer_threads": 4,
    }
    if not images_only and vcodec:
        create_kw["vcodec"] = vcodec
    mode = "images" if images_only else f"video/{vcodec or 'default'}"
    print(
        f"writing LeRobot v3.0 ({mode}) d435i={has_d435i} wrist={has_wrist} "
        f"env_state={has_env}"
    )
    ds = LeRobotDataset.create(**create_kw)
    total = 0
    try:
        for ep_i, ep in enumerate(episodes):
            states = np.load(ep / "states.npy")
            actions = np.load(ep / "actions.npy")
            n = min(len(states), len(actions))
            meta = {}
            if (ep / "meta.json").is_file():
                meta = json.loads((ep / "meta.json").read_text(encoding="utf-8"))
            ep_task = str(meta.get("task") or task)
            d435i = _load_cam_rgb(ep, "d435i", n) if has_d435i else None
            wrist = _load_cam_rgb(ep, "wrist", n) if has_wrist else None
            env_xy = _load_pole_xy(ep, n) if has_env else None
            if has_d435i and d435i is None:
                d435i = _black(n)
            if has_wrist and wrist is None:
                wrist = _black(n)
            if has_env and env_xy is None:
                env_xy = np.zeros((n, 2), dtype=np.float32)
            for t in range(n):
                frame = {
                    "observation.state": _pad_sa(states[t]),
                    "action": _pad_sa(actions[t]),
                    "task": ep_task,
                }
                if has_env:
                    frame["observation.environment_state"] = np.asarray(
                        env_xy[t], dtype=np.float32
                    )
                if has_d435i:
                    frame["observation.images.d435i"] = d435i[t]
                if has_wrist:
                    frame["observation.images.wrist"] = wrist[t]
                ds.add_frame(frame)
            ds.save_episode()
            total += n
            print(f"  {ep.name}: {n} frames env={env_xy is not None}")
        ds.finalize()
    except Exception:
        try:
            ds.finalize()
        except Exception:
            pass
        raise
    return total


def _write_v21(dest: Path, episodes: list[Path], fps: int, task: str) -> int:
    dest.mkdir(parents=True)
    rows: list[dict] = []
    ep_meta: list[dict] = []
    index = 0
    has_d435i = False
    has_wrist = False
    for ep_i, ep in enumerate(episodes):
        states = np.load(ep / "states.npy")
        actions = np.load(ep / "actions.npy")
        n = min(len(states), len(actions))
        meta = {}
        if (ep / "meta.json").is_file():
            meta = json.loads((ep / "meta.json").read_text(encoding="utf-8"))
        ep_fps = int(meta.get("hz") or fps)
        d435i_out = dest / "videos" / "chunk-000" / "observation.images.d435i" / f"episode_{ep_i:06d}.mp4"
        wrist_out = dest / "videos" / "chunk-000" / "observation.images.wrist" / f"episode_{ep_i:06d}.mp4"
        if (ep / "d435i.mp4").is_file() and _try_resize_video(ep / "d435i.mp4", d435i_out, ep_fps):
            has_d435i = True
        elif (ep / "d435i.avi").is_file() and _try_resize_video(ep / "d435i.avi", d435i_out, ep_fps):
            has_d435i = True
        elif _try_resize_jpegs(ep / "d435i", d435i_out, ep_fps):
            has_d435i = True
        if (ep / "wrist.mp4").is_file() and _try_resize_video(ep / "wrist.mp4", wrist_out, ep_fps):
            has_wrist = True
        elif (ep / "wrist.avi").is_file() and _try_resize_video(ep / "wrist.avi", wrist_out, ep_fps):
            has_wrist = True
        elif _try_resize_jpegs(ep / "wrist", wrist_out, ep_fps):
            has_wrist = True
        for t in range(n):
            rows.append(
                {
                    "observation.state": _pad_sa(states[t]).tolist(),
                    "action": _pad_sa(actions[t]).tolist(),
                    "timestamp": t / float(ep_fps),
                    "episode_index": ep_i,
                    "frame_index": t,
                    "index": index,
                    "task_index": 0,
                }
            )
            index += 1
        ep_meta.append(
            {
                "episode_index": ep_i,
                "length": n,
                "source": meta.get("source", "teleop"),
                "task": meta.get("task", task),
            }
        )

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [len(STATE_NAMES)],
            "names": STATE_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": [len(STATE_NAMES)],
            "names": STATE_NAMES,
        },
    }
    if has_d435i:
        features["observation.images.d435i"] = {
            "dtype": "video",
            "shape": [IMAGE_H, IMAGE_W, 3],
            "names": ["height", "width", "channel"],
        }
    if has_wrist:
        features["observation.images.wrist"] = {
            "dtype": "video",
            "shape": [IMAGE_H, IMAGE_W, 3],
            "names": ["height", "width", "channel"],
        }

    info = {
        "codebase_version": "v2.1",
        "robot_type": "grab_it_px150",
        "fps": fps,
        "total_episodes": len(episodes),
        "total_frames": index,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    meta_dir = dest / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    with (meta_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for row in ep_meta:
            f.write(json.dumps(row) + "\n")
    (meta_dir / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": task}) + "\n", encoding="utf-8"
    )
    wrote_parquet = _write_parquet(dest, rows)
    (dest / "convert_manifest.json").write_text(
        json.dumps(
            {
                "input": str(dest),
                "episodes": [p.name for p in episodes],
                "parquet": wrote_parquet,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not wrote_parquet:
        print("note: pandas/pyarrow missing — install them before lerobot-train")
    print(
        "WARNING: wrote v2.1 (lerobot not importable). "
        "lerobot-train 0.4+ needs v3.0 — re-run this script in ~/lerobot-venv."
    )
    return index


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert grab_it IL episodes to LeRobot layout")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--task", default="tabletop_pick")
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument(
        "--images",
        action="store_true",
        help="Store cameras as images (skip ffmpeg). Use if video encode fails on Jetson.",
    )
    ap.add_argument(
        "--vcodec",
        default="h264",
        help="ffmpeg codec for LeRobot v3 video (default h264; libsvtav1 often missing on Jetson).",
    )
    ap.add_argument(
        "--legacy-v21",
        action="store_true",
        help="Force the old v2.1 writer (not usable with lerobot>=0.4).",
    )
    args = ap.parse_args()
    print(SCRIPT_ID)

    src = args.input.expanduser().resolve()
    dest = args.output.expanduser().resolve()
    episodes = _episodes(src)
    if not episodes:
        raise SystemExit(f"no successful episodes in {src}")

    if args.legacy_v21:
        if dest.exists():
            shutil.rmtree(dest)
        index = _write_v21(dest, episodes, args.fps, args.task)
        print(f"wrote {len(episodes)} episodes / {index} frames → {dest}")
        return

    try:
        import lerobot  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "lerobot is not importable in this python. Activate the host venv first:\n"
            "  source ~/lerobot-venv/bin/activate\n"
            "  which python3   # must be .../lerobot-venv/bin/python3\n"
            f"({exc})"
        ) from exc

    try:
        index = _write_v30(
            dest,
            episodes,
            args.fps,
            args.task,
            args.repo_id,
            images_only=args.images,
            vcodec=None if args.images else args.vcodec,
        )
    except TypeError as exc:
        if args.images or "vcodec" not in str(exc):
            raise
        print("LeRobotDataset.create rejected vcodec=; retrying with library default")
        index = _write_v30(
            dest,
            episodes,
            args.fps,
            args.task,
            args.repo_id,
            images_only=False,
            vcodec=None,
        )
    except Exception as exc:
        if args.images or not _looks_like_codec_error(exc):
            raise
        print(f"video encode failed ({exc}); retrying with --images")
        index = _write_v30(
            dest,
            episodes,
            args.fps,
            args.task,
            args.repo_id,
            images_only=True,
            vcodec=None,
        )

    print(f"wrote {len(episodes)} episodes / {index} frames → {dest}")


if __name__ == "__main__":
    main()
