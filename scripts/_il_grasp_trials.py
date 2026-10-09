"""Place the pole at a few spots and score the grasp finish."""
import json
import pathlib
import subprocess
import time

ROOT = pathlib.Path(r"C:\Users\chach\Projects\grab_it-sim\_live_probe")
DONE = ROOT / "place_pole_done.json"
PLACE = ROOT / "place_pole.json"
SPOTS = [(0.18, 0.0), (0.22, -0.06), (0.20, 0.05), (0.16, -0.04)]


def place(x: float, y: float) -> None:
    PLACE.write_text(
        json.dumps({"x": x, "y": y, "pole_z": 0.24, "yaw_deg": 0.0}),
        encoding="utf-8",
    )
    t0 = time.time()
    while time.time() - t0 < 8:
        if DONE.exists() and DONE.stat().st_mtime >= t0 - 0.2:
            print("placed", DONE.read_text().strip(), flush=True)
            return
        time.sleep(0.25)
    print("place timeout", flush=True)


def run_one(x: float, y: float) -> int:
    print(f"\n=== trial world ({x}, {y}) ===", flush=True)
    place(x, y)
    time.sleep(0.4)
    proc = subprocess.run(
        [
            "ssh",
            "-o",
            "ConnectTimeout=8",
            "ubuntu@jetson-agx",
            "GRAB_IT_FINISH_NOW=1 GRAB_IT_RESET_APPROACH=1 python3 /home/ubuntu/dev-ws/scripts/_il_wrist_finish.py",
        ],
        text=True,
        capture_output=True,
    )
    text = (proc.stdout or "") + (proc.stderr or "")
    for line in text.splitlines():
        if line.startswith(("seat", "coarse", "try", "grasp")):
            print(line, flush=True)
    ok = "grasp OK" in text
    print("RESULT", "OK" if ok else "MISS", "rc", proc.returncode, flush=True)
    return 1 if ok else 0


def main() -> None:
    wins = 0
    for x, y in SPOTS:
        wins += run_one(x, y)
    print(f"\n{wins}/{len(SPOTS)} grabs", flush=True)


if __name__ == "__main__":
    main()
