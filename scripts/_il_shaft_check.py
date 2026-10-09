#!/usr/bin/env python3
import importlib.util
import json
import urllib.request

spec = importlib.util.spec_from_file_location(
    "fin", "/home/ubuntu/dev-ws/scripts/_il_wrist_finish.py"
)
fin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fin)
d = fin.twin()
jpeg = urllib.request.urlopen("http://127.0.0.1:8081/wide.jpg", timeout=2).read()
open("/tmp/wide_now.jpg", "wb").write(jpeg)
shaft = fin.shaft_pose(jpeg)
print("pad", d.get("pad_d"), "pz", d.get("pole_z"), "z", d.get("z"), "yaw", d.get("yaw"))
print("arm", dict(zip(d.get("arm_n") or [], d.get("arm_p") or [])))
print("shaft", shaft)
print("in_jaws", fin._in_jaws(shaft))
import cv2
import numpy as np

img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
print("shape", None if img is None else img.shape)
if img is not None:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    print(
        "H", int(np.percentile(h, 10)), int(np.median(h)), int(np.percentile(h, 90)),
        "S", int(np.percentile(s, 10)), int(np.median(s)), int(np.percentile(s, 90)),
        "V", int(np.percentile(v, 10)), int(np.median(v)), int(np.percentile(v, 90)),
    )
    mask = (h > 85) & (h < 130) & (s > 25) & (s < 140) & (v > 70) & (v < 230)
    print("foam_px", int(mask.sum()), "of", int(mask.size))
    rows, cols = 6, 8
    rh, cw = img.shape[0] // rows, img.shape[1] // cols
    print("grid V/S")
    for r in range(rows):
        cells = []
        for c in range(cols):
            sls = s[r * rh : (r + 1) * rh, c * cw : (c + 1) * cw]
            slv = v[r * rh : (r + 1) * rh, c * cw : (c + 1) * cw]
            cells.append(f"{int(np.median(slv)):3d}/{int(np.median(sls)):02d}")
        print(" ".join(cells))
