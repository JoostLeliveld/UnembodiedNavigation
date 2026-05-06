#!/usr/bin/env python3
"""Plot plan-sample evolution for a single run."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import warnings
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_individual_model_selection_runs import (
    GOAL_SUCCESS_RADIUS_M,
    _draw_world,
    _load_csv_rows,
    _load_gp,
    _load_json,
    _mean_csv_value,
    _pf,
)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "na"


def _load_plans(path: Path) -> list[tuple[float, np.ndarray]]:
    grouped: dict[float, list[tuple[int, float, float]]] = defaultdict(list)
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            grouped[float(row["plan_stamp"])].append(
                (int(row["point_idx"]), float(row["x"]), float(row["y"]))
            )
    plans: list[tuple[float, np.ndarray]] = []
    for stamp in sorted(grouped):
        pts = np.asarray([(x, y) for _, x, y in sorted(grouped[stamp])], dtype=float)
        if pts.size:
            plans.append((stamp, pts))
    return plans


def _plot_run(run_dir: Path, gp, out_dir: Path, formats: set[str]) -> Path:
    manifest = _load_json(run_dir / "run_manifest.json")
    summary = _load_json(run_dir / "run_summary.json")
    plans = _load_plans(run_dir / "plan_samples.csv")
    rows = _load_csv_rows(run_dir / "experiment.csv")
    truth = np.asarray(
        [
            (_pf(r, "truth_x"), _pf(r, "truth_y"))
            for r in rows
            if _pf(r, "truth_available") > 0.5 and math.isfinite(_pf(r, "truth_x")) and math.isfinite(_pf(r, "truth_y"))
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(7.6, 6.6), constrained_layout=True)
    im = _draw_world(ax, gp, manifest)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("nominal raw-GP rho_plan")

    if len(rows) > 0:
        goal_x = _mean_csv_value(rows[:10], "goal_x")
        goal_y = _mean_csv_value(rows[:10], "goal_y")
        if math.isfinite(goal_x) and math.isfinite(goal_y):
            ax.scatter([goal_x], [goal_y], marker="*", s=260, facecolor="#ef4444",
                       edgecolors="black", linewidths=1.0, zorder=20)
            ax.add_patch(plt.Circle((goal_x, goal_y), GOAL_SUCCESS_RADIUS_M, fill=False,
                                    edgecolor="#ef4444", linestyle="--", linewidth=1.4, zorder=19))
    if truth.size:
        ax.plot(truth[:, 0], truth[:, 1], color="#111827", linewidth=2.8, zorder=18, label="truth")
        ax.scatter([truth[0, 0]], [truth[0, 1]], s=110, facecolor="#22c55e", edgecolors="black", linewidths=1.0, zorder=20)

    cmap = plt.get_cmap("viridis")
    n = max(len(plans) - 1, 1)
    for i, (stamp, pts) in enumerate(plans):
        frac = i / n
        color = cmap(frac)
        lw = 0.9 + 1.3 * frac
        alpha = 0.18 + 0.62 * frac
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=lw, alpha=alpha, zorder=10)
        if i in {0, len(plans)//2, len(plans)-1}:
            ax.scatter([pts[-1, 0]], [pts[-1, 1]], color=color, s=18, zorder=12)

    if plans:
        ax.plot([], [], color=cmap(0.1), linewidth=1.0, alpha=0.4, label="early plans")
        ax.plot([], [], color=cmap(0.9), linewidth=2.0, alpha=0.8, label="late plans")

    ax.set_title(
        f"{run_dir.parent.parent.name} | {run_dir.parent.name} | {summary.get('completion_reason', '')}",
        fontsize=12,
    )
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.92)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_base = out_dir / _safe_name(f"{run_dir.parent.parent.name}_{run_dir.parent.name}_plan_evolution")
    for fmt in formats:
        fig.savefig(out_base.with_suffix(f".{fmt}"), dpi=190 if fmt == "png" else None)
    plt.close(fig)
    return out_base


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--reference-gp", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--formats", default="png,pdf")
    args = parser.parse_args()

    gp = _load_gp(Path(args.reference_gp).expanduser())
    out_dir = Path(args.out_dir).expanduser()
    formats = {s.strip().lower() for s in args.formats.split(",") if s.strip()} or {"png"}
    for run_dir_str in args.run_dir:
        _plot_run(Path(run_dir_str).expanduser(), gp, out_dir, formats)
    print(f"Wrote plan-evolution plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
