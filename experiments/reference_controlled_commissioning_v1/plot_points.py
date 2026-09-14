#!/usr/bin/env python3
"""Plot every commissioning opportunity and every admitted raw residual as points."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/reference_commissioning_points_mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np


CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")
COLORS = {"miss": "#77828b", "refused": "#d28a2c", "admitted": "#238e83"}


def _read(paths: list[Path]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    required = {
        "drive_id", "camera_id", "reference_x", "reference_y", "yolo_hit",
        "sensor_gate_admitted", "sensor_gate_reason", "raw_projected_x",
        "raw_projected_y",
    }
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path} is missing columns: {sorted(missing)}")
            records.extend(reader)
    return records


def _flag(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _finite_xy(row: dict[str, str], x_key: str, y_key: str) -> tuple[float, float] | None:
    try:
        point = float(row[x_key]), float(row[y_key])
    except (TypeError, ValueError):
        return None
    return point if np.all(np.isfinite(point)) else None


def _axes(title: str):
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.6), constrained_layout=True)
    fig.suptitle(title, fontsize=11)
    return fig, axes.ravel()


def _finish(ax, points: list[tuple[float, float]], title: str) -> None:
    if points:
        xy = np.asarray(points)
        margin = 0.7
        ax.set_xlim(float(xy[:, 0].min()) - margin, float(xy[:, 0].max()) + margin)
        ax.set_ylim(float(xy[:, 1].min()) - margin, float(xy[:, 1].max()) + margin)
    ax.set(title=title, xlabel="World x [m]", ylabel="World y [m]", aspect="equal")
    ax.grid(alpha=0.12, linewidth=0.4)


def _save(fig, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_opportunities(rows: list[dict[str, str]], output: Path) -> None:
    fig, axes = _axes("Every reference-controlled commissioning opportunity")
    for index, camera_id in enumerate(CAMERAS):
        ax = axes[index]
        members = [row for row in rows if row["camera_id"] == camera_id]
        all_points = []
        for outcome in ("miss", "refused", "admitted"):
            points = []
            for row in members:
                reference = _finite_xy(row, "reference_x", "reference_y")
                if reference is None:
                    continue
                actual = (
                    "miss" if not _flag(row["yolo_hit"])
                    else "admitted" if _flag(row["sensor_gate_admitted"])
                    else "refused"
                )
                if actual == outcome:
                    points.append(reference)
                    all_points.append(reference)
            if points:
                xy = np.asarray(points)
                ax.scatter(
                    xy[:, 0], xy[:, 1], s=9,
                    marker="x" if outcome == "miss" else "o",
                    facecolors="none" if outcome == "refused" else COLORS[outcome],
                    edgecolors=COLORS[outcome], linewidths=0.55, alpha=0.72,
                    label=outcome,
                )
        _finish(ax, all_points, f"Camera {camera_id[-1]}")
        if index == 0:
            ax.legend(frameon=False, fontsize=8)
    axes[5].axis("off")
    axes[5].text(
        0.02, 0.95,
        f"{len(rows)} opportunities from {len({row['drive_id'] for row in rows})} complete drives.\n"
        "No interpolation or filled support patches.",
        va="top", fontsize=9,
    )
    _save(fig, output, "commissioning_opportunity_points")


def plot_raw_residuals(rows: list[dict[str, str]], output: Path) -> None:
    fig, axes = _axes("Raw camera-reading residuals at admitted commissioning points")
    for index, camera_id in enumerate(CAMERAS):
        ax = axes[index]
        members = [
            row for row in rows
            if row["camera_id"] == camera_id and _flag(row["sensor_gate_admitted"])
        ]
        segments = []
        references = []
        for row in members:
            reference = _finite_xy(row, "reference_x", "reference_y")
            raw = _finite_xy(row, "raw_projected_x", "raw_projected_y")
            if reference is not None:
                references.append(reference)
            if reference is not None and raw is not None:
                segments.append((reference, raw))
        ax.add_collection(LineCollection(segments, colors="#2a78d6", linewidths=0.45, alpha=0.5))
        if references:
            xy = np.asarray(references)
            ax.scatter(xy[:, 0], xy[:, 1], s=4, color="#22282c", alpha=0.45)
        _finish(ax, references, f"Camera {camera_id[-1]}")
    axes[5].axis("off")
    axes[5].text(0.02, 0.95, "Line: reference position to raw projected reading", va="top", fontsize=9)
    _save(fig, output, "commissioning_raw_residual_points")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tables", type=Path, nargs="+", help="camera_opportunities.csv files")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = _read([path.resolve() for path in args.tables])
    args.output.mkdir(parents=True, exist_ok=True)
    plot_opportunities(rows, args.output)
    plot_raw_residuals(rows, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
