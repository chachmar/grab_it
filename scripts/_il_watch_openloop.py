#!/usr/bin/env python3
"""Sample twin arm_p during open-loop / ACT play for pass/fail."""
import json
import subprocess
import time
import urllib.request


def twin():
    d = json.loads(
        urllib.request.urlopen("http://127.0.0.1:8081/isaac/twin", timeout=2)
        .read()
        .decode()
    )
    return dict(
        zip(d.get("arm_n", []), [float(x) for x in d.get("arm_p", [])])
    ), d.get("playing")


print("t waist shoulder wrist pan grip")
samples = []
for i in range(40):
    by, _pl = twin()
    row = (
        i * 0.5,
        round(by.get("waist", 0), 3),
        round(by.get("shoulder", 0), 3),
        round(by.get("wrist_angle", 0), 3),
        round(by.get("pan", 0), 3),
        round(by.get("gripper", 0), 3),
    )
    samples.append(row)
    print("%4.1f %7.3f %8.3f %7.3f %7.3f %7.3f" % row)
    time.sleep(0.5)

waists = [s[1] for s in samples]
shoulders = [s[2] for s in samples]
pans = [s[4] for s in samples]
grips = [s[5] for s in samples]
look = any(w < -0.35 for w in waists[:10])
align = any(abs(w) < 0.25 for w in waists[8:])
reach = min(shoulders) < -0.5 and max(shoulders) < 0.5
lookdown = sum(1 for p in pans if p >= 0.55) >= len(pans) // 2
close = min(grips) < 0.4
print(
    "HEUR look_waist",
    look,
    "align",
    align,
    "reach_neg_sh",
    reach,
    "pan_down",
    lookdown,
    "grip_close",
    close,
)
print(
    "sh range",
    min(shoulders),
    max(shoulders),
    "pan range",
    min(pans),
    max(pans),
    "waist range",
    min(waists),
    max(waists),
)
print("play:", subprocess.getoutput("pgrep -af play_policy.py")[:300])
print(
    "log:",
    subprocess.getoutput(
        "grep -E 'episode done|PLAY|Error|Traceback|replay|done' "
        "/tmp/il_replay_episode.log | tail -25"
    ),
)
ok = look and align and reach and lookdown
print("STACK", "PASS" if ok else "FAIL")
