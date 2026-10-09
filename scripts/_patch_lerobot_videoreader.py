#!/usr/bin/env python3
"""Patch lerobot video_utils so pyav works without torchvision.io.VideoReader."""
from __future__ import annotations

from pathlib import Path

TARGET = Path.home() / "lerobot-venv/lib/python3.10/site-packages/lerobot/datasets/video_utils.py"

OLD = '''    # set a video stream reader
    # TODO(rcadene): also load audio stream at the same time
    reader = torchvision.io.VideoReader(video_path, "video")

    # set the first and last requested timestamps
    # Note: previous timestamps are usually loaded, since we need to access the previous key frame
    first_ts = min(timestamps)
    last_ts = max(timestamps)

    # access closest key frame of the first requested frame
    # Note: closest key frame timestamp is usually smaller than `first_ts` (e.g. key frame can be the first frame of the video)
    # for details on what `seek` is doing see: https://pyav.basswood-io.com/docs/stable/api/container.html?highlight=inputcontainer#av.container.InputContainer.seek
    reader.seek(first_ts, keyframes_only=keyframes_only)

    # load all frames until last requested frame
    loaded_frames = []
    loaded_ts = []
    for frame in reader:
        current_ts = frame["pts"]
        if log_loaded_timestamps:
            logging.info(f"frame loaded at timestamp={current_ts:.4f}")
        loaded_frames.append(frame["data"])
        loaded_ts.append(current_ts)
        if current_ts >= last_ts:
            break

    if backend == "pyav":
        reader.container.close()

    reader = None
'''

NEW = '''    # set the first and last requested timestamps
    first_ts = min(timestamps)
    last_ts = max(timestamps)

    loaded_frames = []
    loaded_ts = []

    # Jetson / some torchvision builds lack torchvision.io.VideoReader.
    # Decode with PyAV directly when that happens.
    use_tv_reader = hasattr(torchvision.io, "VideoReader")
    if use_tv_reader:
        reader = torchvision.io.VideoReader(video_path, "video")
        reader.seek(first_ts, keyframes_only=keyframes_only)
        for frame in reader:
            current_ts = frame["pts"]
            if log_loaded_timestamps:
                logging.info(f"frame loaded at timestamp={current_ts:.4f}")
            loaded_frames.append(frame["data"])
            loaded_ts.append(current_ts)
            if current_ts >= last_ts:
                break
        if backend == "pyav":
            reader.container.close()
        reader = None
    else:
        import av
        import numpy as np

        container = av.open(video_path)
        stream = container.streams.video[0]
        # Seek near first_ts (seconds -> stream time_base)
        tb = stream.time_base
        seek_pts = int(first_ts / float(tb)) if tb is not None else 0
        try:
            container.seek(seek_pts, any_frame=False, backward=True, stream=stream)
        except Exception:
            container.seek(0, stream=stream)
        for frame in container.decode(stream):
            current_ts = float(frame.pts * tb) if frame.pts is not None else 0.0
            if current_ts < first_ts - 1e-3 and loaded_frames:
                # still before window after seek — keep going
                pass
            if log_loaded_timestamps:
                logging.info(f"frame loaded at timestamp={current_ts:.4f}")
            arr = frame.to_ndarray(format="rgb24")  # HWC uint8
            tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW
            loaded_frames.append(tensor)
            loaded_ts.append(current_ts)
            if current_ts >= last_ts:
                break
        container.close()
'''


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")
    if "use_tv_reader = hasattr(torchvision.io, \"VideoReader\")" in text:
        print("already patched:", TARGET)
        return
    if OLD not in text:
        raise SystemExit(f"expected block not found in {TARGET}")
    TARGET.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    bak = TARGET.with_suffix(".py.bak_videoreader")
    if not bak.exists():
        bak.write_text(text, encoding="utf-8")
    print("patched:", TARGET)


if __name__ == "__main__":
    main()
