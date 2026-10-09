#!/usr/bin/env python3
"""Pan joint reality check: commanded vs REPORTED vs physically measured.

The click data says the camera's tilt does not match the pan joint angle the
URDF is given.  Three different things could cause that, and they need
different fixes:

  1. the servo never reaches the commanded angle   -> tune/geometry on the servo
  2. it reaches it, but the camera moves by a different amount (linkage/ratio)
     -> mimic joint or ratio in the xacro
  3. it moves the right amount about the WRONG AXIS -> fix axis in the xacro

Case 1 is invisible if you only compare "commanded" against a phone reading,
which is all we have done so far.  This sweeps pan, reads the ACTUAL joint
position back from /joint_states at each step, and asks you for the phone
angle, then reports which case you are in.

Sweeps up and then back down so backlash / hysteresis / gravity sag show up
as a difference between the two directions at the same commanded angle.

Usage (inside the Isaac container, same env as the rest of the stack):
    export FASTRTPS_DEFAULT_PROFILES_FILE=/workspaces/isaac_ros-dev/rtabmap_profile.xml
    python3 pan_check.py
    python3 pan_check.py --angles 0.0 0.15 0.3 0.45 0.6 --settle 3.0

At each step: lay the phone on the SAME camera face, the SAME way, every time.
Type the angle it shows (a bare number, sign included), or 's' to skip.
"""
from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

try:
    from interbotix_xs_msgs.msg import JointSingleCommand
except ImportError:
    print("interbotix_xs_msgs not found — source the workspace first.", file=sys.stderr)
    raise


class PanCheck(Node):
    def __init__(self, args):
        super().__init__("pan_check")
        self.args = args
        self._pan: Optional[float] = None
        self._seen = False
        self.create_subscription(
            JointState, args.joint_states_topic, self._js_cb, qos_profile_sensor_data
        )
        self._pub = self.create_publisher(JointSingleCommand, args.command_topic, 10)

    def _js_cb(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            if name == self.args.joint:
                p = float(pos)
                if abs(p + math.pi) <= 0.05:      # xs_sdk syncRead failure sentinel
                    return
                self._pan = p
                self._seen = True

    def command(self, value: float):
        msg = JointSingleCommand()
        msg.name = self.args.joint
        msg.cmd = float(value)
        for _ in range(3):
            self._pub.publish(msg)
            time.sleep(0.05)

    def reported(self, settle_s: float) -> Optional[float]:
        """Wait for motion to stop, then return the joint's own position."""
        t0 = time.monotonic()
        last = self._pan
        stable = 0
        while time.monotonic() - t0 < max(1.0, settle_s) + 4.0:
            time.sleep(0.15)
            cur = self._pan
            if cur is None:
                continue
            if last is not None and abs(cur - last) < 0.002:
                stable += 1
                if stable >= 4 and (time.monotonic() - t0) >= settle_s:
                    return cur
            else:
                stable = 0
            last = cur
        return self._pan


def analyse(rows: List[dict]):
    good = [r for r in rows if r.get("phone") is not None]
    print("\n" + "=" * 68)
    print("  cmd(rad)  cmd(deg)  reported(deg)  phone(deg)   dir")
    print("-" * 68)
    for r in rows:
        ph = "  --  " if r.get("phone") is None else f"{r['phone']:+7.1f}"
        rep = "  --  " if r.get("rep") is None else f"{math.degrees(r['rep']):+9.2f}"
        print(f"  {r['cmd']:+7.3f}  {math.degrees(r['cmd']):+7.2f}  {rep}      "
              f"{ph}    {r['dir']}")
    print("=" * 68)

    if len(good) < 3:
        print("\nNeed at least 3 phone readings to say anything. ")
        return

    # 1. Does the servo reach what we ask for?
    print("\n1. DOES THE SERVO REACH THE COMMAND?")
    worst = 0.0
    for r in rows:
        if r.get("rep") is None:
            continue
        d = math.degrees(r["rep"] - r["cmd"])
        worst = max(worst, abs(d))
        print(f"   cmd {math.degrees(r['cmd']):+7.2f}deg -> reported "
              f"{math.degrees(r['rep']):+7.2f}deg   (miss {d:+.2f}deg)")
    if worst > 3.0:
        print(f"   => NO. Off by up to {worst:.1f} deg. The servo is not tracking;")
        print("      fix that first — the URDF may be innocent.")
    else:
        print(f"   => yes, within {worst:.1f} deg. The reported angle is trustworthy,")
        print("      so any camera mismatch is in the URDF, not the servo.")

    # 2. Is the physical motion proportional to the reported motion?
    print("\n2. IS CAMERA MOTION PROPORTIONAL TO JOINT MOTION?")
    base = None
    for r in rows:
        if r.get("phone") is not None and r.get("rep") is not None:
            base = r
            break
    ratios = []
    for r in good:
        if r is base or r.get("rep") is None:
            continue
        dj = math.degrees(r["rep"] - base["rep"])
        dp = r["phone"] - base["phone"]
        if abs(dj) < 2.0:
            continue
        ratios.append(dp / dj)
        print(f"   joint moved {dj:+7.2f}deg, camera moved {dp:+7.2f}deg"
              f"   ratio {dp / dj:+.3f}")
    if len(ratios) >= 2:
        lo, hi = min(ratios), max(ratios)
        spread = abs(hi - lo)
        mean = sum(ratios) / len(ratios)
        print(f"\n   ratio range {lo:+.3f} .. {hi:+.3f}   (mean {mean:+.3f})")
        if spread < 0.15:
            print(f"   => CONSTANT ratio ~{mean:.3f}. A fixed transmission.")
            print(f"      Fix: drive the camera joint from pan with a "
                  f"{mean:.3f} multiplier in the xacro.")
            if abs(mean - 1.0) < 0.08:
                print("      (ratio ~1.0 — the joint is faithful; look at the AXIS "
                      "or the mount origin instead.)")
        else:
            print("   => NOT constant. Not a simple gear/belt ratio.")
            print("      Suspects: a four-bar linkage (non-linear by design),")
            print("      gravity sag that varies with angle, or a rotation axis")
            print("      not aligned with the one the URDF declares.")

    # 3. Hysteresis: same command, different approach direction.
    print("\n3. HYSTERESIS / SAG (same command, approached from each side)")
    ups = {round(r["cmd"], 4): r for r in rows if r["dir"] == "up" and r.get("phone") is not None}
    downs = {round(r["cmd"], 4): r for r in rows if r["dir"] == "down" and r.get("phone") is not None}
    shared = sorted(set(ups) & set(downs))
    if not shared:
        print("   (no command visited in both directions)")
    else:
        worst_h = 0.0
        for c in shared:
            dp = downs[c]["phone"] - ups[c]["phone"]
            worst_h = max(worst_h, abs(dp))
            print(f"   cmd {math.degrees(c):+7.2f}deg : up {ups[c]['phone']:+6.1f}  "
                  f"down {downs[c]['phone']:+6.1f}   diff {dp:+.1f}deg")
        if worst_h > 3.0:
            print(f"   => {worst_h:.1f} deg of hysteresis. Mechanical slop or the")
            print("      bracket sagging under the camera's weight. No URDF change")
            print("      can fix this — it is not a repeatable function of angle.")
        else:
            print(f"   => negligible ({worst_h:.1f} deg). Motion is repeatable.")
    print()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--angles", type=float, nargs="+",
                    default=[0.0, 0.15, 0.30, 0.45, 0.60],
                    help="commanded pan values in radians (swept up then down)")
    ap.add_argument("--joint", default="pan")
    ap.add_argument("--settle", type=float, default=2.5)
    ap.add_argument("--command-topic",
                    default="/px150_no_elbow/commands/joint_single")
    ap.add_argument("--joint-states-topic",
                    default="/px150_no_elbow/joint_states")
    ap.add_argument("--out", default="pan_check.csv")
    args = ap.parse_args()

    rclpy.init()
    node = PanCheck(args)
    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()

    print(__doc__)
    t0 = time.time()
    while not node._seen and time.time() - t0 < 8.0:
        time.sleep(0.2)
    if not node._seen:
        print(f"!! no '{args.joint}' in {args.joint_states_topic} — is the arm up, "
              f"and is FASTRTPS_DEFAULT_PROFILES_FILE set?")
        rclpy.try_shutdown()
        return 1

    seq = [(a, "up") for a in args.angles]
    seq += [(a, "down") for a in reversed(args.angles[:-1])]

    rows: List[dict] = []
    try:
        for cmd, direction in seq:
            print(f"\n--- commanding {args.joint} = {cmd:+.3f} rad "
                  f"({math.degrees(cmd):+.1f} deg)  [{direction}] ---")
            node.command(cmd)
            rep = node.reported(args.settle)
            if rep is None:
                print("   (no joint feedback)")
            else:
                print(f"   reported: {rep:+.4f} rad ({math.degrees(rep):+.2f} deg)"
                      f"   miss {math.degrees(rep - cmd):+.2f} deg")
            try:
                raw = input("   phone angle (number, or 's' to skip): ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            phone = None
            if raw.lower() not in ("s", "skip", ""):
                try:
                    phone = float(raw)
                except ValueError:
                    print("   (not a number — skipped)")
            rows.append({"cmd": cmd, "rep": rep, "phone": phone, "dir": direction})
    finally:
        if rows:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write("cmd_rad,cmd_deg,reported_rad,reported_deg,phone_deg,dir\n")
                for r in rows:
                    rep = "" if r["rep"] is None else f"{r['rep']:.6f}"
                    repd = "" if r["rep"] is None else f"{math.degrees(r['rep']):.3f}"
                    ph = "" if r["phone"] is None else f"{r['phone']:.2f}"
                    f.write(f"{r['cmd']:.4f},{math.degrees(r['cmd']):.3f},"
                            f"{rep},{repd},{ph},{r['dir']}\n")
            print(f"\nsaved {len(rows)} rows to {args.out}")
            analyse(rows)
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
