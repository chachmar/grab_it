#!/usr/bin/env python3
import importlib.util
import time

spec = importlib.util.spec_from_file_location(
    "wf", "/home/ubuntu/dev-ws/scripts/_il_wrist_finish.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

d = m.twin()
arm = m.arm_of(d)
print("t0", round(d.get("t", 0), 2), "pad", d.get("pad_d"), "sh", round(arm.get("shoulder", 0), 3), "wa", round(arm.get("wrist_angle", 0), 3))
arm["shoulder"] = float(arm.get("shoulder", -1.0)) + 0.45
arm["gripper"] = m.GRIP_HOLD_OPEN
for i in range(8):
    m.post(arm, clear_gate=True)
    time.sleep(0.4)
    d = m.twin()
    got = m.arm_of(d)
    print(i, "t", round(d.get("t", 0), 2), "pad", d.get("pad_d"), "sh", round(got.get("shoulder", 0), 3), "grip", round(got.get("gripper", 0), 3))
