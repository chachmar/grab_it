#!/usr/bin/env python3
"""Fold the arm so reaction torque stops spinning the base."""
import json
import time
import urllib.request

HUB = "http://127.0.0.1:8081"
NAMES = ["waist", "shoulder", "wrist_angle", "wrist_rotate", "gripper", "pan"]
POSE = [-0.2, -1.90, 1.40, 0.0, 1.20, 0.78]


def twin():
    return json.loads(urllib.request.urlopen(HUB + "/isaac/twin", timeout=2).read())


def post():
    body = json.dumps({"t": "arm", "n": NAMES, "p": POSE, "clear_gate": True}).encode()
    req = urllib.request.Request(
        HUB + "/isaac/cmd",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    urllib.request.urlopen(req, timeout=1).read()


d0 = twin()
print("start", round(d0.get("yaw", 0), 3), round(d0.get("x", 0), 3), round(d0.get("y", 0), 3), flush=True)
t0 = time.monotonic()
while time.monotonic() - t0 < 8.0:
    try:
        post()
    except Exception as exc:
        print("post", exc, flush=True)
    time.sleep(0.15)
d1 = twin()
print("end", round(d1.get("yaw", 0), 3), round(d1.get("x", 0), 3), round(d1.get("y", 0), 3), flush=True)
print("pole", d1.get("pole_xy"), "pad", d1.get("pad_d"), flush=True)
