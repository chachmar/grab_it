#!/usr/bin/env python3
"""v3-only grab_it → LeRobot converter. Banner: il_to_lerobot v3-writer"""
from __future__ import annotations

import argparse, json, shutil
from pathlib import Path
import numpy as np

SCRIPT_ID = "il_to_lerobot v3-writer"
NAMES = ["waist", "shoulder", "wrist_angle", "wrist_rotate", "gripper"]
H, W = 240, 320


def episodes(src: Path):
    out = []
    for p in sorted(src.iterdir()):
        if not (p.is_dir() and p.name.startswith("episode_")):
            continue
        if not (p / "states.npy").is_file() or not (p / "actions.npy").is_file():
            continue
        meta = json.loads((p / "meta.json").read_text()) if (p / "meta.json").is_file() else {}
        if meta.get("success") is False:
            continue
        out.append(p)
    return out


def cam_exists(ep: Path, stem: str) -> bool:
    if (ep / f"{stem}.mp4").is_file() or (ep / f"{stem}.avi").is_file():
        return True
    d = ep / stem
    return d.is_dir() and any(d.glob("*.jpg"))


def load_rgb(ep: Path, stem: str, n: int):
    import cv2
    frames = []
    vid = ep / f"{stem}.mp4"
    if not vid.is_file():
        vid = ep / f"{stem}.avi"
    if vid.is_file():
        cap = cv2.VideoCapture(str(vid))
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(cv2.resize(rgb, (W, H)))
        cap.release()
    else:
        folder = ep / stem
        if folder.is_dir():
            for jpg in sorted(folder.glob("*.jpg")):
                bgr = cv2.imread(str(jpg), cv2.IMREAD_COLOR)
                if bgr is not None:
                    frames.append(cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (W, H)))
    if not frames:
        return None
    while len(frames) < n:
        frames.append(frames[-1])
    return np.stack(frames[:n]).astype(np.uint8)


def write_v30(dest: Path, eps, fps, task, repo_id, images_only, vcodec):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    has_d = any(cam_exists(e, "d435i") for e in eps)
    has_w = any(cam_exists(e, "wrist") for e in eps)
    dtype = "image" if images_only else "video"
    feats = {
        "observation.state": {"dtype": "float32", "shape": (5,), "names": NAMES},
        "action": {"dtype": "float32", "shape": (5,), "names": NAMES},
    }
    img = {"dtype": dtype, "shape": (H, W, 3), "names": ["height", "width", "channel"]}
    if has_d:
        feats["observation.images.d435i"] = dict(img)
    if has_w:
        feats["observation.images.wrist"] = dict(img)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    kw = dict(repo_id=repo_id, fps=fps, root=dest, robot_type="grab_it_px150",
              features=feats, use_videos=not images_only, image_writer_threads=4)
    if not images_only and vcodec:
        kw["vcodec"] = vcodec
    print(f"writing LeRobot v3.0 ({'images' if images_only else 'video/'+str(vcodec)}) d435i={has_d} wrist={has_w}")
    ds = LeRobotDataset.create(**kw)
    total = 0
    try:
        for ep in eps:
            st, ac = np.load(ep / "states.npy"), np.load(ep / "actions.npy")
            n = min(len(st), len(ac))
            meta = json.loads((ep / "meta.json").read_text()) if (ep / "meta.json").is_file() else {}
            d435 = load_rgb(ep, "d435i", n) if has_d else None
            wrist = load_rgb(ep, "wrist", n) if has_w else None
            black = np.zeros((n, H, W, 3), np.uint8)
            if has_d and d435 is None:
                d435 = black
            if has_w and wrist is None:
                wrist = black
            for t in range(n):
                fr = {"observation.state": st[t].astype(np.float32),
                      "action": ac[t].astype(np.float32),
                      "task": str(meta.get("task") or task)}
                if has_d:
                    fr["observation.images.d435i"] = d435[t]
                if has_w:
                    fr["observation.images.wrist"] = wrist[t]
                ds.add_frame(fr)
            ds.save_episode()
            total += n
            print(f"  {ep.name}: {n} frames")
        ds.finalize()
    except Exception:
        try:
            ds.finalize()
        except Exception:
            pass
        raise
    return total


def main():
    print(SCRIPT_ID)
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--task", default="tabletop_pick")
    ap.add_argument("--repo-id", default="local/grabit_tabletop_pick")
    ap.add_argument("--images", action="store_true")
    ap.add_argument("--vcodec", default="h264")
    args = ap.parse_args()
    try:
        import lerobot  # noqa: F401
    except ImportError as e:
        raise SystemExit("activate ~/lerobot-venv first: " + str(e))
    eps = episodes(args.input.expanduser().resolve())
    if not eps:
        raise SystemExit("no episodes")
    dest = args.output.expanduser().resolve()
    try:
        n = write_v30(dest, eps, args.fps, args.task, args.repo_id, args.images,
                      None if args.images else args.vcodec)
    except Exception as e:
        if args.images or not any(x in str(e).lower() for x in ("ffmpeg", "codec", "encoder", "h264")):
            raise
        print("video encode failed; retrying --images")
        n = write_v30(dest, eps, args.fps, args.task, args.repo_id, True, None)
    print(f"wrote {len(eps)} episodes / {n} frames → {dest}")


if __name__ == "__main__":
    main()
