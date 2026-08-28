"""How much of a camera's error repeats, and therefore cannot be averaged away.

**No search, no fitting, no threshold chosen against an outcome.** Averaging N readings
divides the part of the error that is fresh each frame by N, and leaves the part that repeats
untouched. So average the residuals over runs of N consecutive readings and watch the spread:

    fresh noise            spread falls as 1 / sqrt(N)
    error that repeats     spread does not fall at all

Where the curve stops falling IS the repeating part, in centimetres. That number is a
measurement of the sensor, not a parameter of the filter, and it is the level below which a
belief built from these readings may not claim to know the position -- because no number of
further looks removes it.

Residuals are scored against the truth at the instant each camera saw the robot.

**One row per reading, or this measures the logger.** The camera manager decides at 20 Hz
against a 5 Hz detector, so `fusion_observations.csv` carries each detection about four
times, 50 ms apart -- closer together than MAX_GAP_S, so they never break a run. Averaging
a value with three copies of itself reduces the spread by nothing, so windows of N <= 4
could not fall no matter what the sensor did: the published curve read 1.64 / 1.64 / 1.64
for camera A at N = 1, 2, 4, flat by construction. Deduplicating drops the measured floor
for two of five cameras (B 1.58 -> 1.33 cm, D 3.39 -> 2.56 cm) and leaves the others
unchanged, so a genuine repeating component does survive -- but it is smaller than the
number that was published, and that number is what the common-mode covariance constant was
set from.

    python3 experiments/fusion_on_fixed_routes/repeating_error.py

Writes ``logs/studies/fusion_on_fixed_routes/replay/repeating_error.json``.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import aligned as A  # noqa: E402

DRIVES = REPO / "logs/studies/fusion_on_fixed_routes"
OUT = DRIVES / "replay"
#: a run of consecutive readings is broken by a gap longer than this, so a "window" is
#: always a genuine uninterrupted look rather than two looks either side of an outage
MAX_GAP_S = 0.6
WINDOWS = (1, 2, 4, 8, 16, 32, 64)


def residuals(run: Path):
    """Every admitted reading's miss, ONCE per detection, per camera, in time order."""

    per_camera, instants = {}, {}
    for entry in A.readings(run, admitted_only=True, dedupe=True):
        per_camera.setdefault(entry["camera"], []).append(
            (entry["obs_stamp"], entry["error"]))
        instants.setdefault(round(entry["obs_stamp"], 6), []).append(entry["error"])
    # The mean of the cameras' misses at one detector ROUND -- grouped by capture time,
    # not by manager decision, so one round contributes one point rather than four.
    fused = [(stamp, np.mean(instants[stamp], axis=0)) for stamp in sorted(instants)]
    for camera in per_camera:
        per_camera[camera].sort(key=lambda r: r[0])
    return per_camera, fused



def averaged_spread(series, window: int):
    """RMS of the residual averaged over runs of ``window`` consecutive readings."""

    means = []
    run = []
    previous = None
    for stamp, error in series:
        if previous is not None and stamp - previous > MAX_GAP_S:
            run = []
        previous = stamp
        run.append(error)
        if len(run) == window:
            means.append(np.mean(run, axis=0))
            run = []
    if len(means) < 8:
        return None
    means = np.asarray(means)
    return float(np.sqrt(np.mean(np.sum(means ** 2, axis=1))))


def main():
    runs = sorted({p.parent for g in DRIVES.glob("drives_commissioning_v*")
                   for p in g.rglob("fusion_observations.csv")})
    runs = [r for r in runs if "obs_stamp" in open(r / "fusion_observations.csv").readline()]
    pooled = {}
    report = {}
    for run in runs:
        label = f"{run.parents[3].name}/{run.parents[2].name}"
        per_camera, fused = residuals(run)
        series = dict(per_camera)
        series["the fused answer"] = fused
        report[label] = {}
        for name, data in series.items():
            curve = {n: averaged_spread(data, n) for n in WINDOWS}
            curve = {n: v for n, v in curve.items() if v is not None}
            report[label][name] = {str(n): v * 100 for n, v in curve.items()}
            pooled.setdefault(name, []).append(curve)

    print("Averaging N consecutive readings. Fresh noise falls as 1/sqrt(N); what repeats "
          "does not.\n")
    header = "  ".join(f"N={n:<3}" for n in WINDOWS)
    print(f"  {'camera':18} {header}   {'if it were all fresh':>21}")
    floors = {}
    for name, curves in pooled.items():
        row, first = [], None
        for n in WINDOWS:
            values = [c[n] for c in curves if n in c]
            if not values:
                row.append("   -  ")
                continue
            value = float(np.median(values)) * 100
            if first is None:
                first = value
            row.append(f"{value:5.2f} ")
        longest = max(n for n in WINDOWS if any(n in c for c in curves))
        plateau = float(np.median([c[longest] for c in curves if longest in c])) * 100
        ideal = first / math.sqrt(longest) if first else float("nan")
        floors[name] = plateau
        print(f"  {name:18} {'  '.join(row)}  {ideal:8.2f} cm at N={longest}")
    print("\n  Last column: where the curve would be at that N if every reading's error were "
          "fresh.\n  The gap between it and the measured value is the part that repeats.\n")

    for name, plateau in floors.items():
        print(f"  {name:18} error that repeats: {plateau:5.2f} cm")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "repeating_error.json").write_text(json.dumps(
        dict(max_gap_s=MAX_GAP_S, windows=list(WINDOWS), per_drive=report,
             repeating_error_cm=floors), indent=1))
    print(f"\nwrote {OUT / 'repeating_error.json'}")


if __name__ == "__main__":
    main()
