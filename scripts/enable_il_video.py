#!/usr/bin/env python3
"""Re-enable IL camera video on AGX record_demos (jpeg folders + timed mp4)."""
from __future__ import annotations

from pathlib import Path

REC = Path("/workspaces/isaac_ros-dev/src/grab_it_il/grab_it_il/record_demos.py")

HELPERS = '''
def _write_jpeg_dir(dest: Path, frames: list[bytes]) -> bool:
    if not frames:
        return False
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, blob in enumerate(frames):
        if not blob:
            continue
        (dest / f"{i:06d}.jpg").write_bytes(blob)
        n += 1
    return n > 0


def _write_mp4_sync(path: Path, frames: list[bytes], fps: int) -> bool:
    if cv2 is None or not frames:
        return False
    first = cv2.imdecode(np.frombuffer(frames[0], dtype=np.uint8), cv2.IMREAD_COLOR)
    if first is None:
        return False
    h, w = first.shape[:2]
    scale = min(1.0, 320.0 / max(w, 1), 240.0 / max(h, 1))
    tw = max(2, int(w * scale) // 2 * 2)
    th = max(2, int(h * scale) // 2 * 2)
    for out, fourcc in ((path.with_suffix(".mp4"), "mp4v"), (path.with_suffix(".avi"), "MJPG")):
        writer = None
        try:
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*fourcc), float(fps), (tw, th))
            if not writer.isOpened():
                continue
            for blob in frames:
                img = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    img = first
                if img.shape[1] != tw or img.shape[0] != th:
                    img = cv2.resize(img, (tw, th))
                writer.write(img)
            writer.release()
            writer = None
            if out.is_file() and out.stat().st_size > 0:
                return True
        except Exception:
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass
    return False


def _write_mp4(path: Path, frames: list[bytes], fps: int, timeout_s: float = 20.0) -> bool:
    if os.environ.get("GRAB_IT_IL_SKIP_VIDEO", "").lower() in ("1", "true", "yes"):
        return False
    if not frames:
        return False
    done = threading.Event()
    ok = [False]

    def _run() -> None:
        try:
            ok[0] = _write_mp4_sync(path, frames, fps)
        except Exception:
            ok[0] = False
        finally:
            done.set()

    threading.Thread(target=_run, name="il-mp4", daemon=True).start()
    if not done.wait(timeout_s):
        return False
    return bool(ok[0])


'''


def main() -> None:
    t = REC.read_text(encoding="utf-8")
    t = t.replace("    return False  # skip mp4 — SAVE was hanging on VideoWriter\n", "")
    t = t.replace("    return False  # skip mp4 so SAVE cannot hang on Jetson encode\n", "")
    if "def _write_jpeg_dir(" not in t:
        i = t.find("def _write_mp4(")
        j = t.find("class RecordDemos")
        if i < 0 or j < 0:
            raise SystemExit("cannot find _write_mp4 / RecordDemos")
        t = t[:i] + HELPERS + t[j:]
        print("replaced video writers")
    old = """            np.save(dest / "states.npy", states)
            np.save(dest / "actions.npy", actions)
            videos = []
            if _write_mp4(dest / "d435i.mp4", d435i, HZ):
                videos.append("d435i.mp4")
            if _write_mp4(dest / "wrist.mp4", wrist, HZ):
                videos.append("wrist.mp4")
"""
    new = """            np.save(dest / "states.npy", states)
            np.save(dest / "actions.npy", actions)
            videos = []
            if _write_jpeg_dir(dest / "d435i", d435i):
                videos.append("d435i/")
            if _write_jpeg_dir(dest / "wrist", wrist):
                videos.append("wrist/")
            if _write_mp4(dest / "d435i.mp4", d435i, HZ):
                videos.append("d435i.mp4")
            if _write_mp4(dest / "wrist.mp4", wrist, HZ):
                videos.append("wrist.mp4")
"""
    if old in t and "d435i/" not in t:
        t = t.replace(old, new, 1)
        print("patched SAVE to write jpeg folders + mp4")
    if "import os" not in t:
        t = t.replace("import json\n", "import json\nimport os\n", 1)
    REC.write_text(t, encoding="utf-8")
    import py_compile
    py_compile.compile(str(REC), doraise=True)
    print("ok video enabled", REC)
    print("unset GRAB_IT_IL_SKIP_VIDEO; restart record_demos")


if __name__ == "__main__":
    main()
