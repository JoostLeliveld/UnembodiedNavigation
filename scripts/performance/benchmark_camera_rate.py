#!/usr/bin/env python3
"""Measure the delivered camera frame rate of a world, and refuse to guess.

WHY THIS EXISTS. The camera->belief update rate is the binding constraint on how
long a development iteration takes, and it is set by how fast Gazebo renders, not
by the detector. Measuring it by hand produced a ranking of world variants that
REVERSED on re-measurement, because a leaked gz server from an earlier run was
still holding the single render thread. Every number in that ranking described
the contention rather than the world.

So this script fails loudly instead of returning a number it cannot stand behind:

  * it refuses to start while any gz server is running;
  * it repeats each world and reports the spread, not one sample;
  * it declares two worlds indistinguishable when their spreads overlap;
  * it verifies the server is gone afterwards and says so if it is not.

Usage:

    python3 scripts/performance/benchmark_camera_rate.py \
        --world warehouse_v2.world.sdf --world warehouse_v2_fast.world.sdf \
        --repeats 3

Each run launches the world headless with all five cameras bridged, waits for
the render pipeline to settle, and samples `ros2 topic hz` on one camera.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: Seconds of warm-up before sampling. The first frames after launch arrive while
#: ogre2 is still uploading meshes and textures, so sampling immediately reports a
#: rate the world never actually sustains.
SETTLE_S = 30.0
SAMPLE_S = 30.0
TOPIC = "/external_camera_c/image_raw"


def gz_server_pids() -> list[int]:
    """PIDs of running gz servers.

    Matched on the executable name, never with `pkill -f`: a pattern match on the
    full command line also matches the shell this script was launched from, and
    an agent's own tooling.
    """
    out = subprocess.run(["ps", "-eo", "pid,comm"], capture_output=True, text=True).stdout
    pids = []
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) == 2 and parts[1] == "ruby":
            pids.append(int(parts[0]))
    return pids


def stop_gz(timeout_s: float = 25.0) -> bool:
    """SIGINT, wait, then SIGKILL. Returns True if the server is gone."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        pids = gz_server_pids()
        if not pids:
            return True
        for pid in pids:
            subprocess.run(["kill", "-INT", str(pid)], capture_output=True)
        time.sleep(1.0)
    for pid in gz_server_pids():
        subprocess.run(["kill", "-9", str(pid)], capture_output=True)
    time.sleep(2.0)
    return not gz_server_pids()


def measure_once(world: str, log_dir: Path, tag: str, nvidia_offload: bool) -> float:
    """One launch-measure-teardown cycle. Returns the delivered rate in Hz."""
    if gz_server_pids():
        raise SystemExit(
            "REFUSING TO MEASURE: a gz server is already running. A leaked server "
            "shares the one render thread and silently halves the result. Stop it "
            "first (kill by PID from `ps -eo pid,comm`, matching comm == ruby)."
        )

    log = log_dir / f"{tag}.launch.log"
    cmd = [
        "ros2", "launch", "sim", "bringup_sim.launch.py",
        f"world:={world}", "world_name:=warehouse_v2", "headless:=true",
        f"nvidia_offload:={'true' if nvidia_offload else 'false'}",
        "reset_world:=false", "use_lidar:=false", "bridge_scan:=false",
        "bridge_contacts:=true",
        "bridge_camera_a:=true", "bridge_camera_b:=true", "bridge_camera_c:=true",
        "bridge_camera_d:=true", "bridge_camera_e:=true",
        "spawn_x:=10.6", "spawn_y:=8.6", "spawn_z:=0.01", "spawn_yaw:=-1.5708",
    ]
    with log.open("w") as fh:
        launch = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                  cwd=str(REPO))
    try:
        time.sleep(SETTLE_S)
        # `ros2 topic hz` runs until interrupted, so the sample window is
        # enforced with `timeout` and a clean exit is not expected.
        hz = subprocess.run(
            ["timeout", str(int(SAMPLE_S)), "ros2", "topic", "hz", TOPIC,
             "--window", "60"],
            capture_output=True, text=True, timeout=SAMPLE_S + 30,
            cwd=str(REPO))
        rates = [float(m) for m in re.findall(r"average rate: ([\d.]+)", hz.stdout)]
        (log_dir / f"{tag}.hz.txt").write_text(hz.stdout)
        if not rates:
            raise SystemExit(
                f"NO FRAMES on {TOPIC} for world {world}. The world loaded but "
                f"published nothing; see {log}. This is a failure, not a slow world."
            )
        return rates[-1]
    except subprocess.TimeoutExpired:
        raise SystemExit(f"`ros2 topic hz` did not return for {world}; see {log}")
    finally:
        launch.send_signal(2)
        try:
            launch.wait(timeout=10)
        except subprocess.TimeoutExpired:
            launch.kill()
        if not stop_gz():
            raise SystemExit(
                "gz server survived teardown. Later measurements would be "
                "contended; refusing to continue."
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", action="append", required=True,
                    help="world SDF filename under sim/gazebo_worlds/worlds "
                         "(repeat the flag to compare worlds)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument(
        "--nvidia-offload", choices=("true", "false"), default="true",
        help="select the same Gazebo PRIME-offload launch path used in campaigns",
    )
    ap.add_argument("--out", default=None,
                    help="directory for logs and the JSON result")
    a = ap.parse_args()

    if shutil.which("ros2") is None:
        raise SystemExit("ros2 not on PATH: source the ROS and workspace setup first")

    out = Path(a.out) if a.out else REPO / "logs/performance/camera_rate"
    out.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[float]] = {w: [] for w in a.world}
    # Interleave worlds rather than finishing one before starting the next, so a
    # machine that warms up or picks up background load over the session cannot
    # be mistaken for a difference between worlds.
    for rep in range(a.repeats):
        for world in a.world:
            tag = f"{Path(world).stem}_r{rep}"
            rate = measure_once(
                world, out, tag, nvidia_offload=(a.nvidia_offload == "true"))
            results[world].append(rate)
            print(f"  {world:34s} rep {rep}: {rate:5.2f} Hz", flush=True)

    print()
    summary = {}
    for world, rates in results.items():
        lo, hi = min(rates), max(rates)
        summary[world] = {
            "rates_hz": rates, "median_hz": statistics.median(rates),
            "min_hz": lo, "max_hz": hi,
        }
        print(f"{world:34s} median {statistics.median(rates):5.2f} Hz  "
              f"range {lo:.2f}-{hi:.2f}")

    # A difference is only reported when the per-world spreads do not overlap.
    worlds = list(results)
    if len(worlds) > 1:
        print()
        for i in range(len(worlds)):
            for j in range(i + 1, len(worlds)):
                a_w, b_w = worlds[i], worlds[j]
                a_r, b_r = results[a_w], results[b_w]
                if min(a_r) > max(b_r) or min(b_r) > max(a_r):
                    fast, slow = ((a_w, b_w) if statistics.median(a_r)
                                  > statistics.median(b_r) else (b_w, a_w))
                    ratio = (statistics.median(results[fast])
                             / statistics.median(results[slow]))
                    print(f"SEPARATED: {fast} is {ratio:.2f}x {slow}")
                else:
                    print(f"NOT SEPARATED: {a_w} vs {b_w} — spreads overlap, "
                          f"so this run does not rank them")

    (out / "camera_rate.json").write_text(json.dumps({
        "topic": TOPIC, "settle_s": SETTLE_S, "sample_s": SAMPLE_S,
        "repeats": a.repeats, "nvidia_offload": a.nvidia_offload,
        "results": summary,
    }, indent=2))
    print(f"\nwrote {out / 'camera_rate.json'}")


if __name__ == "__main__":
    main()
