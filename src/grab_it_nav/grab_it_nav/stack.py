"""Start the nav or mapping process list in the same order as the shell scripts.

Long-running commands come from bringup_spec. Gate checks (camera, TF, topics,
map) follow nav_bringup.sh / mapping_bringup.sh. On a failed gate the processes
already started are left running, matching those scripts' `exit 1` (they have
no trap). Helpers are killed only after Nav2 returns, or after mapping's
RTAB-Map process returns.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from grab_it_nav.bringup_spec import (
    CAMERA_HZ_TIMEOUT_S,
    COLOR_TOPIC,
    DEPTH_TOPIC,
    MAP_RELAY_WAIT_TIMEOUT_S,
    MAP_WAIT_TIMEOUT_S,
    MAPPING_PKILL,
    MAPPING_PKILL_SLEEP_S,
    MAPPING_STATIC_TF,
    NAV_PKILL,
    NAV_PKILL_SLEEP_S,
    NAV_REQUIRED_PACKAGES,
    SCAN_WAIT_TIMEOUT_S,
    SLEEP_AFTER_BRIDGE_S,
    SLEEP_AFTER_MAP_RELAY_S,
    SLEEP_AFTER_ODOM_S,
    SLEEP_AFTER_POSE_RELAY_S,
    SLEEP_AFTER_RTABMAP_S,
    SLEEP_AFTER_SCAN_START_S,
    SLEEP_AFTER_STATIC_TF_MAPPING_S,
    SLEEP_AFTER_STATIC_TF_NAV_S,
    TF_ATTEMPTS,
    TF_ECHO_TIMEOUT_S,
    TOPIC_HZ_TIMEOUT_S,
    depth_scan_cmd,
    domain_bridge_cmd,
    load_camera_tf,
    nav2_cmd,
    python_cmd,
    rtabmap_mapping_cmd,
    rtabmap_nav_cmd,
    static_tf_cmd,
)


class BringupFailure(SystemExit):
    """Gate failed. Processes started so far are intentionally still running."""


# Set by the SIGTERM handler. Session teardown sends SIGTERM to this process
# and historically leaves the nav children running (nav_bringup.sh has no trap).
_TERM = {"flag": False}


def _script(name: str) -> str:
    path = Path(__file__).resolve().parent / name
    if not path.is_file():
        raise BringupFailure(f"[nav] ERROR: missing helper {path}")
    return str(path)


def _env(profile: str | None) -> dict[str, str]:
    env = os.environ.copy()
    env["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
    env["ROS_DOMAIN_ID"] = "0"
    env.pop("CYCLONEDDS_URI", None)
    if profile:
        env["FASTRTPS_DEFAULT_PROFILES_FILE"] = profile
    return env


def _popen(
    cmd: list[str],
    profile: str | None,
    log_path: str | None,
    *,
    own_session: bool = True,
) -> subprocess.Popen:
    """Start a process. Background helpers get their own session.

    nav_bringup.sh uses `set -m`, so its `&` jobs do not receive terminal
    SIGINT. The foreground Nav2 / mapping RTAB-Map process stays in this
    process group (`own_session=False`) so Ctrl-C still stops it.
    """
    log = open(log_path, "w", encoding="utf-8") if log_path else None
    stdout = log if log is not None else None
    stderr = subprocess.STDOUT if log is not None else None
    proc = subprocess.Popen(
        cmd,
        env=_env(profile),
        stdout=stdout,
        stderr=stderr,
        start_new_session=own_session,
    )
    proc._grabit_log = log  # type: ignore[attr-defined]
    return proc


def _pkill(patterns: tuple[str, ...], sleep_s: float) -> None:
    for pattern in patterns:
        subprocess.run(["pkill", "-9", "-f", pattern], check=False)
    time.sleep(sleep_s)


def _topic_hz(topic: str, profile: str, timeout_s: float) -> bool:
    proc = subprocess.run(
        ["timeout", str(timeout_s), "ros2", "topic", "hz", topic],
        env=_env(profile),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return "average rate" in (proc.stdout or "")


def _pkg_prefix(pkg: str) -> bool:
    proc = subprocess.run(
        ["ros2", "pkg", "prefix", pkg],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.returncode == 0


def _require_packages() -> None:
    for pkg in NAV_REQUIRED_PACKAGES:
        if not _pkg_prefix(pkg):
            print(f"[nav] ERROR: missing package '{pkg}'")
            print("       Rebuild Isaac image with Dockerfile.grabit (nav2 lines), then re-run run_dev.sh")
            raise BringupFailure(1)


def _dds_profile(crossover: str, bridge: str) -> str:
    """Return the TF/camera profile. Falls back to bridge_both when crossover is unusable."""
    log_path = "/tmp/nav_xprof_probe.log"
    with open(log_path, "w", encoding="utf-8") as log:
        subprocess.run(
            ["timeout", "3", "ros2", "topic", "list"],
            env=_env(crossover),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    # Same rule as nav_bringup.sh: fall back only when the probe log says the
    # whitelist dropped every interface, whether `ros2 topic list` exited 0 or not.
    if "All whitelist interfaces were filtered out" in text:
        print("  [warn] crossover DDS profile unusable (10.42.0.1 down?) — using bridge_both.xml")
        return bridge
    return crossover


def _camera_color_visible(crossover: str) -> bool:
    return _topic_hz(COLOR_TOPIC, crossover, CAMERA_HZ_TIMEOUT_S)


def _print_camera_fail() -> None:
    print("  [FAIL] no D435i color on crossover.")
    print("         AGX host: ip -br addr | grep 10.42 ; ping -c2 10.42.0.2")
    print("         Nano:     sudo docker start grab_it_nano && docker exec -it grab_it_nano /root/nano_bringup.sh")
    print("         If bringup already running but camera stuck (device busy):")
    print("           docker exec grab_it_nano bash -lc 'pkill -9 -f realsense2_camera; sleep 2; /root/nano_bringup.sh'")
    print("         Check: docker exec grab_it_nano tail -30 /tmp/cameras.log")
    print("         (eno1/crossover is currently required — WiFi alone is not enough for images)")


def _tf_connected(profile: str, parent: str, child: str) -> bool:
    proc = subprocess.run(
        ["timeout", str(TF_ECHO_TIMEOUT_S), "ros2", "run", "tf2_ros", "tf2_echo", parent, child],
        env=_env(profile),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return "Translation:" in (proc.stdout or "")


def _wait_tf(profile: str, child: str) -> bool:
    for i in range(1, TF_ATTEMPTS + 1):
        if _tf_connected(profile, "base_link", child):
            print(f"  [OK]   TF tree connected (try {i})")
            return True
        time.sleep(1)
    return False


def _check_topic(topic: str, label: str, profile: str) -> bool:
    if _topic_hz(topic, profile, TOPIC_HZ_TIMEOUT_S):
        print(f"  [OK]   {label}")
        return True
    print(f"  [FAIL] {label} ({topic})")
    return False


def _wait_map(script: str, topic: str, timeout_s: float, profile: str) -> bool:
    proc = subprocess.run(
        ["python3", script, topic, str(timeout_s)],
        env=_env(profile),
        check=False,
    )
    return proc.returncode == 0


def _line_buffer() -> None:
    """Session redirects nav_bringup.sh to a file. Shell echo is line-oriented."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(line_buffering=True)
        except (OSError, ValueError):
            pass


def _term(procs: list[subprocess.Popen]) -> None:
    """SIGTERM, same as the shell's `kill $pid` (not SIGKILL)."""
    for proc in procs:
        if proc.poll() is None:
            proc.terminate()
    for proc in procs:
        log = getattr(proc, "_grabit_log", None)
        if log is not None:
            log.close()


def _nav_cleanup_pkill() -> None:
    for pattern in ("map_to_nav.py", "scan_stamp_fix.py", "initialpose_to_rtab.py", "odom_to_tf.py"):
        subprocess.run(["pkill", "-f", pattern], check=False)


def _install_term_handler():
    previous = signal.getsignal(signal.SIGTERM)
    _TERM["flag"] = False

    def _on_term(signum, frame):
        _TERM["flag"] = True
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _on_term)
    return previous


def run_nav(cfg: dict[str, str]) -> None:
    """Run the localization stack. cfg keys match nav.launch.py arguments."""
    previous = _install_term_handler()
    started: list[subprocess.Popen] = []
    try:
        _run_nav(cfg, started)
    except BringupFailure:
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
        if _TERM["flag"]:
            print("[nav] SIGTERM: leaving the nav stack running")


def _run_nav(cfg: dict[str, str], started: list[subprocess.Popen]) -> None:
    database = cfg["database_path"]
    params = cfg["params_file"]
    bridge_yaml = cfg["bridge_yaml"]
    crossover = cfg["crossover_profile"]
    bridge = cfg["bridge_profile"]
    if not os.path.isfile(database):
        print(f"[nav] ERROR: no map database at {database}")
        print(f"       Remap first: bash {cfg['workspace']}/mapping_bringup.sh   (or bash {cfg['workspace']}/remap_map.sh)")
        raise BringupFailure(1)
    if not os.path.isfile(params):
        print(f"[nav] ERROR: missing {params}")
        raise BringupFailure(1)
    for path, label in (
        (bridge_yaml, "bridge yaml"),
        (crossover, "crossover DDS profile"),
        (bridge, "bridge DDS profile"),
        (cfg["camera_tf_env"], "camera_tf.env"),
    ):
        if not os.path.isfile(path):
            print(f"[nav] ERROR: missing {label}: {path}")
            raise BringupFailure(1)
    _require_packages()
    cam = load_camera_tf(cfg["camera_tf_env"])
    cam_frame = cam["CAM_FRAME"]
    cam_optical = cam["CAM_OPTICAL"]

    _pkill(NAV_PKILL, NAV_PKILL_SLEEP_S)

    print("[nav] starting domain_bridge (Create3 d5 <-> d0; odom+tf_static+cmd_vel, no /tf)...")
    bridge_proc = _popen(domain_bridge_cmd(bridge_yaml), bridge, "/tmp/nav_bridge.log")
    started.append(bridge_proc)
    time.sleep(SLEEP_AFTER_BRIDGE_S)

    print("[nav] publishing TF from /odom (avoids lagged Create3 /tf)...")
    odom_proc = _popen(python_cmd(_script("odom_to_tf.py")), bridge, "/tmp/nav_odom_tf.log")
    started.append(odom_proc)
    time.sleep(SLEEP_AFTER_ODOM_S)

    tf_prof = _dds_profile(crossover, bridge)

    print("[nav] checking Nano camera (needs crossover 10.42.0.2 + nano_bringup)...")
    if not _camera_color_visible(crossover):
        _print_camera_fail()
        raise BringupFailure(1)
    print("  [OK]   camera color visible")

    print(
        f"[nav] starting static TF base_link -> {cam_frame}"
        f" (x={cam['CAM_TF_X']} z={cam['CAM_TF_Z']} pitch={cam['CAM_TF_PITCH']};"
        " from config/camera_tf.env; DDS=bridge_both)"
    )
    tf_proc = _popen(
        static_tf_cmd(
            cam["CAM_TF_X"], cam["CAM_TF_Y"], cam["CAM_TF_Z"],
            cam["CAM_TF_ROLL"], cam["CAM_TF_PITCH"], cam["CAM_TF_YAW"],
            cam_frame,
        ),
        bridge,
        "/tmp/nav_static_tf.log",
    )
    started.append(tf_proc)
    time.sleep(SLEEP_AFTER_STATIC_TF_NAV_S)

    print(f"[nav] waiting for camera TF + base_link -> {cam_optical}...")
    if not _wait_tf(tf_prof, cam_optical):
        print("  [FAIL] TF tree NOT connected after retries.")
        print(f"         Need RealSense TF ({cam_frame} -> {cam_optical}) from Nano (no tf_prefix).")
        print("         See /tmp/nav_static_tf.log")
        listed = subprocess.run(
            ["timeout", "5", "ros2", "topic", "list"],
            env=_env(tf_prof),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        shown = 0
        for line in (listed.stdout or "").splitlines():
            if "d435i" in line or "tf" in line:
                print(line)
                shown += 1
                if shown >= 15:
                    break
        raise BringupFailure(1)

    print("[nav] pre-flight...")
    fail = False
    if not _check_topic(COLOR_TOPIC, "camera color", crossover):
        if not _check_topic(COLOR_TOPIC, "camera color", tf_prof):
            fail = True
    if not _check_topic(DEPTH_TOPIC, "aligned depth", crossover):
        if not _check_topic(DEPTH_TOPIC, "aligned depth", tf_prof):
            fail = True
    if not _check_topic("/odom", "odom (bridge)", bridge):
        if not _check_topic("/odom", "odom (bridge)", tf_prof):
            fail = True
    if fail:
        print(f"[nav] PRE-FLIGHT FAILED — bridge={bridge_proc.pid} tf={tf_proc.pid} left running")
        raise BringupFailure(1)

    print("[nav] depth -> /scan_raw -> /scan (restamped for TF)...")
    scan_proc = _popen(depth_scan_cmd(cam_frame), crossover, "/tmp/nav_scan.log")
    started.append(scan_proc)
    scanfix_proc = _popen(python_cmd(_script("scan_stamp_fix.py")), bridge, "/tmp/nav_scan_fix.log")
    started.append(scanfix_proc)
    time.sleep(SLEEP_AFTER_SCAN_START_S)
    print("[nav] waiting for restamped /scan...")
    # Same wait as nav_bringup.sh: TF_PROF (crossover, or bridge_both if that probe failed).
    scan_wait = subprocess.run(
        [
            "timeout", str(SCAN_WAIT_TIMEOUT_S), "bash", "-c",
            'until ros2 topic hz /scan 2>/dev/null | grep -q "average rate"; do sleep 1; done',
        ],
        env=_env(tf_prof),
        check=False,
    )
    if scan_wait.returncode != 0:
        print("  [FAIL] /scan not publishing — see /tmp/nav_scan_fix.log /tmp/nav_scan.log")
        try:
            lines = Path("/tmp/nav_scan_fix.log").read_text(encoding="utf-8", errors="replace").splitlines()
            if lines:
                print("\n".join(lines[-40:]))
        except OSError:
            pass
        raise BringupFailure(1)
    print(f"  [OK]   /scan live (pid={scanfix_proc.pid})")

    print(f"[nav] RTAB-Map LOCALIZATION (keep {database}, no incremental mapping)...")
    rtab_proc = _popen(rtabmap_nav_cmd(database), crossover, "/tmp/nav_rtabmap.log")
    started.append(rtab_proc)
    time.sleep(SLEEP_AFTER_RTABMAP_S)

    print("[nav] waiting for /rtabmap/map (rclpy probe; ros2cli daemon/list flaky on crossover)...")
    wait_map = _script("wait_map_probe.py")
    if not _wait_map(wait_map, "/rtabmap/map", MAP_WAIT_TIMEOUT_S, crossover):
        print("  [FAIL] /rtabmap/map not published — is DB valid? See /tmp/nav_rtabmap.log")
        raise BringupFailure(1)
    print("  [OK]   /rtabmap/map present")

    print("[nav] relaying /rtabmap/map -> /map for Nav2...")
    relay_proc = _popen(python_cmd(_script("map_to_nav.py")), crossover, "/tmp/nav_map_relay.log")
    started.append(relay_proc)
    time.sleep(SLEEP_AFTER_MAP_RELAY_S)
    if not _wait_map(wait_map, "/map", MAP_RELAY_WAIT_TIMEOUT_S, crossover):
        print("  [FAIL] /map relay did not come up — see /tmp/nav_map_relay.log")
        raise BringupFailure(1)
    print(f"  [OK]   /map present (relay pid={relay_proc.pid})")

    print("[nav] relaying /initialpose -> /rtabmap/initialpose (for RViz 2D Pose Estimate)...")
    pose_proc = _popen(
        python_cmd(_script("initialpose_to_rtab.py")), crossover, "/tmp/nav_pose_relay.log"
    )
    started.append(pose_proc)
    time.sleep(SLEEP_AFTER_POSE_RELAY_S)

    pids = " ".join(
        f"{label}={proc.pid}"
        for label, proc in (
            ("bridge", bridge_proc),
            ("odomtf", odom_proc),
            ("tf", tf_proc),
            ("scan", scan_proc),
            ("scanfix", scanfix_proc),
            ("rtab", rtab_proc),
            ("maprelay", relay_proc),
            ("pose", pose_proc),
        )
    )
    print("[nav] launching Nav2 (RPP controller, static+/scan costmaps)...")
    print(f"      {pids}")
    print("      Tip: Fixed Frame=map. UNCHECK Amcl Particle Swarm (unused).")
    print("      Use TF axes / RobotModel as truth. 2D Pose Estimate -> wait 3s -> Nav2 Goal.")
    print("      Named goals: bash scripts/go_to_named.sh kitchen   (see config/named_locations.yaml)")

    nav2 = _popen(nav2_cmd(params), crossover, None, own_session=False)
    try:
        nav2.wait()
    finally:
        if _TERM["flag"]:
            return
        print("[nav] Nav2 exited. Cleaning helpers...")
        _term(started)
        _nav_cleanup_pkill()
        if nav2.poll() is None:
            nav2.terminate()


def _image_topic_ok(topic: str, profile: str) -> bool:
    """mapping_bringup.sh check_image_topic: several QoS profiles, echo not hz."""
    qos_sets = (
        ["--qos-profile", "sensor_data"],
        ["--qos-reliability", "reliable", "--qos-durability", "volatile"],
        ["--qos-reliability", "reliable", "--qos-durability", "transient_local"],
        ["--qos-reliability", "best_effort", "--qos-durability", "volatile"],
    )
    for _ in range(5):
        for qos in qos_sets:
            proc = subprocess.run(
                ["timeout", "6", "ros2", "topic", "echo", topic, "--once", "--no-arr", "--no-str", *qos],
                env=_env(profile),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            text = proc.stdout or ""
            if any(token in text for token in ("header:", "stamp:", "height:", "width:")):
                return True
        time.sleep(1)
    return False


def run_mapping(cfg: dict[str, str]) -> None:
    """Fresh or extended RTAB-Map. delete_db_on_start matches mapping_bringup.sh."""
    previous = _install_term_handler()
    started: list[subprocess.Popen] = []
    try:
        _run_mapping(cfg, started)
    except BringupFailure:
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
        if _TERM["flag"]:
            print("[map] SIGTERM: leaving mapping processes running")


def _run_mapping(cfg: dict[str, str], started: list[subprocess.Popen]) -> None:
    database = cfg["database_path"]
    bridge_yaml = cfg["bridge_yaml"]
    crossover = cfg["crossover_profile"]
    bridge = cfg["bridge_profile"]
    for path, label in (
        (bridge_yaml, "bridge yaml"),
        (crossover, "crossover DDS profile"),
        (bridge, "bridge DDS profile"),
    ):
        if not os.path.isfile(path):
            print(f"[map] ERROR: missing {label}: {path}")
            raise BringupFailure(1)

    _pkill(MAPPING_PKILL, MAPPING_PKILL_SLEEP_S)

    print("[map] starting domain_bridge (Create3 d5 -> d0; odom+tf_static+cmd_vel, no /tf)...")
    bridge_proc = _popen(domain_bridge_cmd(bridge_yaml), bridge, "/tmp/bridge.log")
    started.append(bridge_proc)
    time.sleep(SLEEP_AFTER_BRIDGE_S)

    print("[map] publishing TF from /odom (avoids lagged Create3 /tf)...")
    odom_proc = _popen(python_cmd(_script("odom_to_tf.py")), bridge, "/tmp/map_odom_tf.log")
    started.append(odom_proc)
    time.sleep(SLEEP_AFTER_ODOM_S)

    print(f"[map] starting static TF base_link -> {MAPPING_STATIC_TF['child_frame_id']}...")
    tf_proc = _popen(
        static_tf_cmd(
            MAPPING_STATIC_TF["x"],
            MAPPING_STATIC_TF["y"],
            MAPPING_STATIC_TF["z"],
            MAPPING_STATIC_TF["roll"],
            MAPPING_STATIC_TF["pitch"],
            MAPPING_STATIC_TF["yaw"],
            MAPPING_STATIC_TF["child_frame_id"],
        ),
        crossover,
        "/tmp/static_tf.log",
    )
    started.append(tf_proc)
    time.sleep(SLEEP_AFTER_STATIC_TF_MAPPING_S)

    optical = "camera_color_optical_frame"
    print(f"[map] waiting for TF base_link -> {optical}...")
    if not _wait_tf(crossover, optical):
        print("  [FAIL] TF tree NOT connected.")
        print("         - is the camera running? (frames must be camera_* i.e. no tf_prefix)")
        print("         - check /tmp/static_tf.log")
        print(f"         bridge={bridge_proc.pid} static_tf={tf_proc.pid} left running.")
        raise BringupFailure(1)

    print("[map] pre-flight input checks...")
    fail = False
    if _image_topic_ok(COLOR_TOPIC, crossover):
        print("  [OK]   camera color")
    else:
        print(f"  [FAIL] camera color ({COLOR_TOPIC}) — topic listed but no samples (restart Nano camera / align_depth:=true)")
        fail = True
    if _image_topic_ok(DEPTH_TOPIC, crossover):
        print("  [OK]   aligned depth")
    else:
        print(f"  [FAIL] aligned depth ({DEPTH_TOPIC}) — topic listed but no samples (restart Nano camera / align_depth:=true)")
        fail = True
    odom_echo = subprocess.run(
        ["timeout", "10", "ros2", "topic", "echo", "/odom", "--once", "--no-arr", "--no-str"],
        env=_env(bridge),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    odom_text = odom_echo.stdout or ""
    if any(token in odom_text for token in ("header:", "child_frame_id:")) or _topic_hz("/odom", bridge, 10):
        print("  [OK]   odom (bridge)")
    else:
        print("  [FAIL] odom (bridge)")
        fail = True
    if fail:
        print("[map] PRE-FLIGHT FAILED")
        print("  On Nano restart camera with align_depth:=true enable_sync:=true")
        print("  Then: ros2 topic echo /d435i/d435i/aligned_depth_to_color/image_raw --once --qos-profile sensor_data")
        print("  odom? -> Create3 domain5+undock; /tmp/bridge.log")
        print(f"  bridge={bridge_proc.pid} static_tf={tf_proc.pid} left running.")
        raise BringupFailure(1)

    print(f"[map] all inputs live. Launching RTAB-Map (bridge={bridge_proc.pid} tf={tf_proc.pid})...")
    rtab = _popen(rtabmap_mapping_cmd(database), crossover, None, own_session=False)
    try:
        rtab.wait()
    finally:
        if _TERM["flag"]:
            return
        # mapping_bringup.sh kills the bridge and the static TF only.
        # odom_to_tf is left running (the next mapping start pkills it).
        print(f"[map] RTAB-Map exited. kill bridge/tf: kill {bridge_proc.pid} {tf_proc.pid}")
        _term([bridge_proc, tf_proc])
        if rtab.poll() is None:
            rtab.terminate()


def cfg_from_launch_context(context) -> dict[str, str]:
    keys = (
        "workspace",
        "database_path",
        "params_file",
        "bridge_yaml",
        "crossover_profile",
        "bridge_profile",
        "camera_tf_env",
    )
    cfg = {}
    for key in keys:
        value = context.launch_configurations.get(key, "")
        cfg[key] = str(value)
    return cfg


def _exit_failed(kind: str, exc: BringupFailure) -> None:
    code = exc.code if isinstance(exc.code, int) else 1
    print(f"[{kind}] stopped ({code})", file=sys.stderr)
    # Do not ask the launch system to tear children down: a failed gate in the
    # shell leaves the processes it already started. os._exit skips flush.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def nav_main(context, *args, **kwargs):
    _line_buffer()
    try:
        run_nav(cfg_from_launch_context(context))
    except BringupFailure as exc:
        _exit_failed("nav", exc)
    return []


def mapping_main(context, *args, **kwargs):
    _line_buffer()
    try:
        run_mapping(cfg_from_launch_context(context))
    except BringupFailure as exc:
        _exit_failed("map", exc)
    return []
