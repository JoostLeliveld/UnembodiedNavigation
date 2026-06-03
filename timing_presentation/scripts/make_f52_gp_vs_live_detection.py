#!/usr/bin/env python3
"""F52: compare AWS GP reliability with live YOLO detections in F50."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import yaml


ROOT = Path("/home/joostleliveld/Thesis/UnembodiedNavigation")
LOG = ROOT / "logs/visibility_comparison/f50_b1_tight_local_goal_3seed_v1"
OUT = ROOT / "timing_presentation/figures/F52"
GP = ROOT / "logs/visibility_comparison/aws_gp_v5/yolo_score_raw_gp.npz"
WORLD_PROFILES = ROOT / "src/experiments/config/world_profiles.yaml"
WORLD = "warehouse_aws.world.sdf"
TASK = "F31_b1_apron_a3_mid"
C = {"C1": "#2563eb", "C2": "#dc2626"}
RACKS = [
    ("R4L", 1.725, 2.275, -0.8, 1.25),
    ("R4U", 1.725, 2.275, 2.2, 4.25),
    ("R5L", 3.875, 4.425, -0.8, 1.25),
    ("R5U", 3.875, 4.425, 2.2, 4.25),
]


def load_gp():
    d = np.load(GP, allow_pickle=True)
    return d["xs"], d["ys"], d["P_conservative_plan_map"]


def gp_at(xs, ys, p, x, y):
    xi = np.clip(np.searchsorted(xs, x), 1, len(xs) - 1)
    yi = np.clip(np.searchsorted(ys, y), 1, len(ys) - 1)
    xi = np.where(np.abs(xs[xi] - x) < np.abs(xs[xi - 1] - x), xi, xi - 1)
    yi = np.where(np.abs(ys[yi] - y) < np.abs(ys[yi - 1] - y), yi, yi - 1)
    return p[yi, xi]


def load_regions():
    d = yaml.safe_load(WORLD_PROFILES.read_text())
    return d["worlds"][WORLD].get("known_2d_regions", [])


def load_run(cond, seed):
    root = LOG / TASK / cond / f"seed{seed}"
    exps = sorted(root.glob("experiment_*"))
    if not exps:
        return None
    d = exps[-1]
    exp = pd.read_csv(d / "experiment.csv")
    perc = pd.read_csv(d / "perception.csv") if (d / "perception.csv").exists() else pd.DataFrame()
    summ = json.loads((d / "run_summary.json").read_text()) if (d / "run_summary.json").exists() else {}
    return d, exp, perc, summ


def draw_scene(ax, xs, ys, p):
    ax.contourf(xs, ys, p, levels=np.linspace(0, 1, 16), cmap="RdYlGn", alpha=0.35)
    for r in load_regions():
        if "non_driveable" not in str(r.get("type", "")):
            continue
        x, y = float(r["xmin"]), float(r["ymin"])
        w, h = float(r["xmax"]) - x, float(r["ymax"]) - y
        ax.add_patch(Rectangle((x, y), w, h, fc="#fca5a5", ec="#dc2626", alpha=0.25, lw=0.8))
    for nm, xmin, xmax, ymin, ymax in RACKS:
        ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, ec="#111827", lw=1.5))
    ax.scatter([3.3], [-1.0], c="#16a34a", s=80, edgecolors="white", zorder=5)
    ax.scatter([1.0], [1.75], c="#111827", s=130, marker="*", zorder=5)
    ax.set_xlim(-0.5, 5.0)
    ax.set_ylim(-3.2, 4.8)
    ax.set_aspect("equal", "box")
    ax.grid(alpha=0.15)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    xs, ys, p = load_gp()
    runs = {(cond, seed): load_run(cond, seed) for cond in ("C1", "C2") for seed in (0, 1, 2)}

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.1, 0.9])
    fig.suptitle("F52 - GP reliability versus live YOLO detections in F50", fontsize=16, fontweight="bold")

    for col, cond in enumerate(("C1", "C2")):
        ax = fig.add_subplot(gs[0, col])
        draw_scene(ax, xs, ys, p)
        for seed in (0, 1, 2):
            run = runs[(cond, seed)]
            if run is None:
                continue
            _, exp, perc, summ = run
            if len(exp.dropna(subset=["truth_x", "truth_y"])):
                e = exp.dropna(subset=["truth_x", "truth_y"])
                ax.plot(e["truth_x"], e["truth_y"], color=C[cond], lw=1.8, alpha=0.45 + 0.15 * seed, label=f"s{seed}")
            if not perc.empty and {"true_x", "true_y", "detected"}.issubset(perc.columns):
                q = perc.dropna(subset=["true_x", "true_y"])
                miss = q[q["detected"].astype(int) == 0]
                hit = q[q["detected"].astype(int) == 1]
                ax.scatter(hit["true_x"], hit["true_y"], c=C[cond], s=10, alpha=0.25)
                ax.scatter(miss["true_x"], miss["true_y"], c="#111827", s=18, marker="x", alpha=0.75)
        ax.set_title(f"{cond}: paths, YOLO hits, YOLO misses (x)")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 2])
    rows = []
    for cond in ("C1", "C2"):
        for seed in (0, 1, 2):
            run = runs[(cond, seed)]
            if run is None:
                continue
            _, exp, perc, summ = run
            if perc.empty or "detected" not in perc:
                continue
            q = perc.dropna(subset=["true_x", "true_y"]).copy()
            if q.empty:
                continue
            q["gp"] = gp_at(xs, ys, p, q["true_x"].to_numpy(), q["true_y"].to_numpy())
            det = q["detected"].astype(float)
            rows.append([
                f"{cond}s{seed}",
                summ.get("completion_reason", "-"),
                det.mean(),
                q["gp"].mean(),
                q.loc[det == 0, "gp"].mean() if (det == 0).any() else np.nan,
                len(q),
            ])
    ax.axis("off")
    table_rows = [
        [name, reason, f"{det:.2f}", f"{gpmean:.2f}", "-" if np.isnan(gpmiss) else f"{gpmiss:.2f}", str(n)]
        for name, reason, det, gpmean, gpmiss, n in rows
    ]
    tbl = ax.table(
        cellText=table_rows,
        colLabels=["run", "reason", "det rate", "mean GP", "GP on misses", "N"],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.4)
    ax.set_title("Live detector vs GP summary")

    ax = fig.add_subplot(gs[1, :])
    for cond in ("C1", "C2"):
        for seed in (0, 1, 2):
            run = runs[(cond, seed)]
            if run is None:
                continue
            _, exp, perc, summ = run
            if perc.empty or not {"diag_stamp", "detected", "true_x", "true_y"}.issubset(perc.columns):
                continue
            q = perc.dropna(subset=["true_x", "true_y"]).copy()
            if q.empty:
                continue
            q["gp"] = gp_at(xs, ys, p, q["true_x"].to_numpy(), q["true_y"].to_numpy())
            t0 = summ.get("first_cmd_stamp", q["diag_stamp"].iloc[0])
            t = q["diag_stamp"] - float(t0)
            alpha = 0.45 + 0.15 * seed
            ax.plot(t, q["gp"], color=C[cond], alpha=alpha, lw=1.6, label=f"{cond}s{seed} GP")
            miss = q["detected"].astype(int) == 0
            if miss.any():
                ax.scatter(t[miss], np.full(miss.sum(), -0.05 if cond == "C1" else -0.12), c=C[cond], marker="x", s=18, alpha=alpha)
    ax.axhline(0.2, color="#6b7280", ls="--", lw=1, label="low GP")
    ax.set_ylim(-0.2, 1.05)
    ax.set_xlabel("t after first command [s]")
    ax.set_ylabel("GP P_conservative at true robot position")
    ax.set_title("GP reliability along true path; x markers below axis are missed YOLO detections")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, ncol=4)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png = OUT / "F52_gp_vs_live_detection.png"
    pdf = OUT / "F52_gp_vs_live_detection.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)

    note = OUT / "F52_gp_vs_live_detection.md"
    note.write_text(
        "# F52 - GP Versus Live Detection\n\n"
        f"- Figure: `{png}`\n"
        f"- PDF: `{pdf}`\n\n"
        "This diagnostic checks whether the planner-facing GP reliability agrees with the live YOLO detector during F50. "
        "A mismatch matters because C2 can only execute the intended visibility-aware behavior if the route that looks reliable to the GP is also observable by the runtime detector.\n\n"
        "Interpretation: runs that miss detections in regions with high GP reliability indicate stale or miscalibrated reliability evidence, detector/rendering mismatch, or a route passing through a viewpoint where the offline capture was too optimistic. Those cases should be fixed before tuning ambiguity weights further.\n"
    )
    print(png)
    print(pdf)
    print(note)


if __name__ == "__main__":
    main()
