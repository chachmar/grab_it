#!/usr/bin/env python3
import json
import urllib.request

raw = urllib.request.urlopen("http://127.0.0.1:8081/isaac/twin", timeout=3).read()
d = json.loads(raw)
names = list(d.get("arm_n") or [])
pos = [float(x) for x in (d.get("arm_p") or [])]
arm = {n: round(pos[i], 3) for i, n in enumerate(names) if i < len(pos)}
print(
    json.dumps(
        {
            "x": d.get("x"),
            "y": d.get("y"),
            "yaw": d.get("yaw"),
            "lx": d.get("lx"),
            "az": d.get("az"),
            "pole_xy": d.get("pole_xy"),
            "pole_z": d.get("pole_z"),
            "pad_d": d.get("pad_d"),
            "arm": arm,
        }
    )
)
