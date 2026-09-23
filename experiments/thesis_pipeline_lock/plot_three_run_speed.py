#!/usr/bin/env python3
"""Plot robot speed for three frozen runs and mark correction gaps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.fusion_on_fixed_routes.aligned import (  # noqa: E402
    assimilations,
    mission_interval,
    rows,
    validate_run_ledger,
)


def finite(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return np.nan


def speed_series(run: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples: dict[float, tuple[float, float]] = {}
    for row in rows(run):
        t, x, y = (finite(row, key) for key in ("gt_stamp", "gt_x", "gt_y"))
        if np.isfinite([t, x, y]).all():
            samples[t] = (x, y)
    stamps = np.array(sorted(samples))
    xy = np.array([samples[t] for t in stamps])
    dt = np.diff(stamps)
    speed = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt
    return stamps[:-1], stamps[1:], speed


def longest_gap(run: Path) -> tuple[float, float]:
    ledger = validate_run_ledger(run)
    start, stop = mission_interval(run)
    accepted = sorted(
        a["apply_stamp"] for a in assimilations(run)
        if start <= a["apply_stamp"] <= stop
        and a["source_batch_id"] in ledger.accepted_update_ids
    )
    stamps = np.array([start, *accepted, stop])
    index = int(np.argmax(np.diff(stamps)))
    return float(stamps[index]), float(stamps[index + 1])


def rolling_mean(time: np.ndarray, values: np.ndarray, window_s: float = 0.5) -> np.ndarray:
    output = np.empty_like(values)
    for i, stamp in enumerate(time):
        keep = np.abs(time - stamp) <= window_s / 2
        output[i] = np.mean(values[keep])
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs=3, type=int, default=[91500, 91502, 91504])
    args = parser.parse_args()

    selected = []
    for result in json.loads(args.results.read_text()):
        if (result["condition"] == "spatial_removal"
                and result["task"] == "thesis09_parallel_aisles_west"
                and result["seed"] in args.seeds):
            selected.append(result)
    selected.sort(key=lambda result: args.seeds.index(result["seed"]))
    if len(selected) != 3:
        raise SystemExit(f"expected three runs, found {len(selected)}")

    fig, axes = plt.subplots(3, 1, figsize=(9.2, 7.2), sharex=True, sharey=True,
                             constrained_layout=True)
    summaries = []
    for ax, result in zip(axes, selected):
        run = REPO / result["run"]
        start, stop = mission_interval(run)
        left, right, speed = speed_series(run)
        keep = (right > start) & (left < stop)
        left, right, speed = left[keep], right[keep], speed[keep]
        time = (left + right) / 2 - start
        gap_start, gap_stop = longest_gap(run)
        gap_start -= start
        gap_stop -= start

        ax.plot(time, speed, color="#9ecae1", lw=0.7, alpha=0.65,
                label="Ground-truth speed")
        ax.plot(time, rolling_mean(time, speed), color="#08519c", lw=1.7,
                label="0.5 s moving average")
        ax.axhline(1.0, color="#555555", ls="--", lw=0.9, label="1.0 m/s limit")
        ax.axvspan(gap_start, gap_stop, color="#d62728", alpha=0.13,
                   label="Longest correction gap")

        duration = gap_stop - gap_start
        overlap = np.maximum(
            0.0,
            np.minimum(right, gap_stop + start) - np.maximum(left, gap_start + start),
        )
        distance = float(np.sum(speed * overlap))
        average = distance / duration
        ax.text(
            0.99, 0.90,
            f"seed {result['seed']}   gap {duration:.2f} s   "
            f"distance {distance:.2f} m   mean {average:.2f} m/s",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                  "alpha": 0.88, "edgecolor": "#bbbbbb"},
        )
        ax.set_ylabel("Speed [m/s]")
        ax.grid(alpha=0.25)
        summaries.append({
            "seed": result["seed"],
            "gap_s": duration,
            "gap_distance_m": distance,
            "gap_mean_speed_mps": average,
            "mission_mean_speed_mps": float(np.sum(
                speed * np.maximum(0.0, np.minimum(right, stop) - np.maximum(left, start))
            ) / (stop - start)),
            "mission_peak_speed_mps": float(np.max(speed)),
        })

    axes[0].legend(ncol=4, loc="lower left", fontsize=7.5)
    axes[-1].set_xlabel("Mission time [s]")
    axes[-1].set_xlim(left=0)
    axes[-1].set_ylim(-0.03, 1.10)
    fig.suptitle("Robot speed in three spatial-removal runs  |  west task", fontsize=12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    fig.savefig(args.output.with_suffix(".pdf"))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
