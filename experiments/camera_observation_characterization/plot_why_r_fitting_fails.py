#!/usr/bin/env python3
"""How `R` is fitted, and the three separate reasons the fit comes out wrong.

The ladder in figure 23 stops at its simplest rung and is still about twice as confident as it
should be. This figure takes that apart. Three things go wrong, and they are not the same kind
of problem:

    1. THE WRONG RESIDUALS. A covariance fitted to the mean model's in-sample residuals is
       three times too tight, because the mean model scores 4.2 cm RMSE on tiles it was fitted
       on and 12.0 cm on tiles it was not. This is a mistake in the procedure and it is fixable:
       fit out-of-fold. Doing so moves mean NEES from about 38 to about 3.9.

    2. THE WRONG VARIABLES. After that fix, no quantity the model is allowed to see gets any
       further. Conditioning on camera makes it worse. Conditioning on heading changes nothing.
       Conditioning on occlusion helps slightly. An oracle that is told which decile of error
       the reading will land in reaches NEES 2.15 with a SHARPER sigma -- so a Gaussian `R` is
       achievable on this data, and the obstacle is that the variable which predicts a bad
       reading is missing, not that the noise is intractable.

    3. NOT ENOUGH DATA TO SPLIT. Conditioning finely makes calibration worse rather than
       better, because 3,249 training readings cut by camera x occlusion x heading leaves 24 of
       64 cells under 40 readings, and a variance estimated from 40 samples carries about 11%
       relative error. The partition starts fitting noise.

Figure 10 supplies the missing variable's identity: whether a rack stands in the sightline, and
how much of the robot it hides. The binary version of that is already in this figure and only
helps a little, because occlusion is continuous -- a robot 10% hidden and one 90% hidden are
the same flag and very different readings.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import chi2

REPO = Path(__file__).resolve().parents[2]
for rel in ("experiments/deck_figures", "experiments/camera_observation_characterization"):
    value = str((REPO / rel).resolve())
    if value not in sys.path:
        sys.path.insert(0, value)

import style as D  # noqa: E402
import learn_measurement_covariance as L  # noqa: E402
from _sightline import blocked, camera_positions, rack_footprints  # noqa: E402

DEFAULT_CAPTURE = REPO / "logs/perception_datasets/warehouse_v2_bbox_characterization_20260831"
DEFAULT_OUT = REPO / "logs/studies/camera_observation_characterization_20260831"
FOLDER = "10_learning_R"
CHI2_95 = float(chi2.ppf(0.95, 2))
MIN_CELL = 40


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def grouped_variance(e_train, keys_train, keys_test, n_test):
    """One variance per cell, falling back to the pooled value where a cell is too thin."""
    variance = np.zeros((n_test, 2))
    fallback = np.mean(e_train ** 2, axis=0)
    thin = 0
    for key in np.unique(keys_test):
        picked_train = keys_train == key
        picked_test = keys_test == key
        if picked_train.sum() >= MIN_CELL:
            variance[picked_test] = np.mean(e_train[picked_train] ** 2, axis=0)
        else:
            variance[picked_test] = fallback
            thin += 1
    return variance, thin


def score(residual, variance):
    nees = np.sum(residual ** 2 / variance, axis=1)
    return {
        "mean_nees": float(nees.mean()),
        "containment_95": float(np.mean(nees <= CHI2_95)),
        "sharpness_cm": float(100 * np.sqrt(variance).mean()),
        "beyond_20": float(np.mean(nees > 20.0)),
        "nees": nees,
    }


def build(capture: Path, seed: int, splits: int) -> dict:
    data = L.load(capture, "nn")
    train = data["split"] == "train"
    test = data["split"] == "test"

    in_sample = data["e"][train]
    out_of_fold = L.out_of_fold_residuals(data, "nn", seed, splits)
    held_out = data["e"][test]

    racks, cameras = rack_footprints(), camera_positions()
    occluded = np.asarray([
        blocked(row["camera_id"][-1], float(row["robot_x"]), float(row["robot_y"]),
                racks=racks, cameras=cameras)
        for row in data["rows"]
    ])
    heading = np.asarray([int(row["heading_id"]) for row in data["rows"]])
    camera = data["camera"]

    zeros_tr = np.zeros(train.sum(), dtype=int)
    zeros_te = np.zeros(test.sum(), dtype=int)
    conditionings = [
        ("nothing\n(the winning rung)", zeros_tr, zeros_te, "#8a8983"),
        ("camera", camera[train], camera[test], "#2a78d6"),
        ("heading", heading[train], heading[test], "#4a3aa7"),
        ("occlusion\n(rack in the way)", occluded[train].astype(int),
         occluded[test].astype(int), "#c63131"),
        ("camera x occlusion", camera[train] * 2 + occluded[train],
         camera[test] * 2 + occluded[test], "#1baf7a"),
        ("camera x occlusion\nx heading", (camera[train] * 2 + occluded[train]) * 8 + heading[train],
         (camera[test] * 2 + occluded[test]) * 8 + heading[test], "#eb6834"),
    ]

    results = []
    for label, keys_tr, keys_te, colour in conditionings:
        variance, thin = grouped_variance(out_of_fold, keys_tr, keys_te, test.sum())
        entry = score(held_out, variance)
        entry.update({"label": label, "colour": colour,
                      "cells": int(len(np.unique(keys_tr))), "thin_cells": thin,
                      "median_cell": int(np.median(np.unique(keys_tr, return_counts=True)[1]))})
        results.append(entry)

    # oracle: told which decile of true error magnitude the reading will land in
    magnitude_tr = np.linalg.norm(out_of_fold, axis=1)
    magnitude_te = np.linalg.norm(held_out, axis=1)
    edges = np.quantile(magnitude_tr, np.linspace(0, 1, 11))[1:-1]
    variance, _ = grouped_variance(out_of_fold, np.digitize(magnitude_tr, edges),
                                   np.digitize(magnitude_te, edges), test.sum())
    oracle = score(held_out, variance)
    oracle.update({"label": "ORACLE\ntold the true error decile", "colour": "#d4267b",
                   "cells": 10, "thin_cells": 0, "median_cell": len(magnitude_tr) // 10})

    # what fitting on the wrong residuals costs
    wrong = score(held_out, np.repeat(np.mean(in_sample ** 2, axis=0)[None, :], test.sum(), 0))
    right = score(held_out, np.repeat(np.mean(out_of_fold ** 2, axis=0)[None, :], test.sum(), 0))

    return {
        "in_sample": in_sample, "out_of_fold": out_of_fold, "held_out": held_out,
        "occluded_test": occluded[test], "occluded_train_oof": occluded[train],
        "results": results, "oracle": oracle, "wrong": wrong, "right": right,
    }


def draw(bundle: dict, out: Path) -> None:
    fig = plt.figure(figsize=(23.0, 13.6))
    grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 1.0), left=0.055, right=0.985,
                            bottom=0.075, top=0.845, hspace=0.40, wspace=0.24)

    # --- 1. the recipe, and the step that goes wrong -----------------------------------
    ax = fig.add_subplot(grid[0, 0])
    for values, label, colour, style in (
        (bundle["in_sample"], "residuals on tiles the mean model was FITTED on", "#c63131", "--"),
        (bundle["out_of_fold"], "residuals on tiles it did NOT see (out-of-fold)", "#1baf7a", "-"),
        (bundle["held_out"], "residuals on the held-out tiles (the truth)", D.INK, "-"),
    ):
        magnitude = 100 * np.linalg.norm(values, axis=1)
        rms = float(np.sqrt(np.mean(magnitude ** 2)))
        ax.hist(np.minimum(magnitude, 60), bins=np.linspace(0, 60, 61), histtype="step",
                lw=2.4, density=True, color=colour, linestyle=style,
                label=f"{label}\n     RMSE {rms:.1f} cm")
    ax.set_xlabel("Size of the residual the mean model leaves (cm)")
    ax.set_ylabel("Density")
    ax.set_title("1. Fit R to the wrong residuals and it is 3x too tight\n"
                 "the mean model flatters itself on its own tiles",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9.8, frameon=True)
    ax.grid(color="#e4e2dc", lw=0.8); ax.set_axisbelow(True)
    ax.text(0.97, 0.42,
            f"same R0 rung, scored on held-out tiles\n\n"
            f"fitted in-sample:   NEES {bundle['wrong']['mean_nees']:5.1f}   "
            f"95% holds {bundle['wrong']['containment_95']:.2f}\n"
            f"fitted out-of-fold: NEES {bundle['right']['mean_nees']:5.1f}   "
            f"95% holds {bundle['right']['containment_95']:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="#f2f0ea", ec="#d0cec7"))

    # --- 2. no available variable reaches the oracle ------------------------------------
    ax = fig.add_subplot(grid[0, 1])
    entries = bundle["results"] + [bundle["oracle"]]
    y = np.arange(len(entries))[::-1]
    ax.barh(y, [e["mean_nees"] for e in entries],
            color=[e["colour"] for e in entries], alpha=0.9, height=0.62)
    for slot, entry in zip(y, entries):
        ax.text(entry["mean_nees"] + 0.07, slot, f"{entry['mean_nees']:.2f}", va="center",
                fontsize=11, fontweight="bold")
    ax.axvline(2.0, color=D.INK, lw=2.0, linestyle="--")
    ax.text(2.0, -0.85, " honest = 2.0", fontsize=11, color=D.INK, va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels([e["label"] for e in entries], fontsize=10.5)
    ax.set_xlabel("Mean NEES on held-out tiles (2.0 is honest, higher is overconfident)")
    ax.set_xlim(0, max(e["mean_nees"] for e in entries) * 1.16)
    ax.set_title("2. Conditioning on what we can see does not help\n"
                 "but an oracle gets there, so the data is not the problem",
                 fontsize=14, fontweight="bold")
    ax.grid(axis="x", color="#e4e2dc", lw=0.8); ax.set_axisbelow(True)

    # --- 3. sharper AND more honest is possible -----------------------------------------
    ax = fig.add_subplot(grid[0, 2])
    fan = [(-16, -26, "right"), (16, 12, "left"), (16, -24, "left"), (-16, 12, "right"),
           (18, -6, "left"), (0, 22, "center"), (0, -26, "center")]
    for index, entry in enumerate(entries):
        ax.scatter(entry["sharpness_cm"], entry["mean_nees"], s=170, color=entry["colour"],
                   zorder=4, edgecolors="white", linewidths=1.4)
        dx, dy, align = fan[index % len(fan)]
        ax.annotate(entry["label"].replace("\n", " "),
                    (entry["sharpness_cm"], entry["mean_nees"]),
                    textcoords="offset points", xytext=(dx, dy), ha=align, fontsize=9.6,
                    color=entry["colour"], fontweight="bold")
    ax.set_xlim(min(e["sharpness_cm"] for e in entries) - 0.9,
                max(e["sharpness_cm"] for e in entries) + 1.5)
    ax.axhline(2.0, color=D.INK, lw=1.8, linestyle="--")
    ax.text(0.02, 2.0, " honest", transform=ax.get_yaxis_transform(), fontsize=10.5,
            va="bottom", color=D.INK)
    ax.set_xlabel("Mean stated sigma (cm) — the price paid")
    ax.set_ylabel("Mean NEES — the honesty bought")
    ax.set_title("3. The oracle is BOTH sharper and honest\n"
                 "so nothing has to be traded away; the information is just missing",
                 fontsize=14, fontweight="bold")
    ax.grid(color="#e4e2dc", lw=0.8); ax.set_axisbelow(True)
    ax.set_ylim(0, max(e["mean_nees"] for e in entries) * 1.22)

    # --- 4. one R is being asked to cover two populations -------------------------------
    ax = fig.add_subplot(grid[1, 0])
    clear = 100 * np.linalg.norm(bundle["held_out"][~bundle["occluded_test"]], axis=1)
    hidden = 100 * np.linalg.norm(bundle["held_out"][bundle["occluded_test"]], axis=1)
    bins = np.linspace(0, 80, 60)
    ax.hist(np.minimum(clear, 80), bins=bins, histtype="stepfilled", density=True, alpha=0.55,
            color="#5e8fbf", label=f"clear sightline  ({len(clear)} readings, "
                                   f"RMSE {np.sqrt(np.mean(clear ** 2)):.0f} cm)")
    ax.hist(np.minimum(hidden, 80), bins=bins, histtype="stepfilled", density=True, alpha=0.62,
            color="#c63131", label=f"a rack in the way  ({len(hidden)} readings, "
                                   f"RMSE {np.sqrt(np.mean(hidden ** 2)):.0f} cm)")
    pooled = 100 * np.sqrt(np.mean(np.sum(bundle["out_of_fold"] ** 2, axis=1)) / 2.0)
    ax.axvline(pooled, color=D.INK, lw=2.2, linestyle="--")
    ax.text(pooled, ax.get_ylim()[1] * 0.86,
            f"  the single sigma R0 states\n  for both: {pooled:.0f} cm",
            fontsize=10.5, color=D.INK)
    ax.set_xlabel("Residual left by the mean model (cm)")
    ax.set_ylabel("Density")
    ax.set_title("4. One number is covering two different populations\n"
                 "occlusion makes a reading roughly twice as bad, and R never learns which",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, frameon=True)
    ax.grid(color="#e4e2dc", lw=0.8); ax.set_axisbelow(True)

    # --- 5. splitting finer runs out of data --------------------------------------------
    ax = fig.add_subplot(grid[1, 1])
    cells = [e["cells"] for e in bundle["results"]]
    nees = [e["mean_nees"] for e in bundle["results"]]
    stagger = [(0, 18, "center"), (-26, -34, "right"), (26, 10, "left"),
               (0, -34, "center"), (-10, 20, "right"), (-14, -34, "right")]
    for index, entry in enumerate(bundle["results"]):
        ax.scatter(entry["cells"], entry["mean_nees"], s=170, color=entry["colour"], zorder=4,
                   edgecolors="white", linewidths=1.4)
        dx, dy, align = stagger[index % len(stagger)]
        ax.annotate(f"{entry['label'].splitlines()[0]}\n{entry['median_cell']} rows/cell"
                    + (f", {entry['thin_cells']} too thin" if entry["thin_cells"] else ""),
                    (entry["cells"], entry["mean_nees"]), textcoords="offset points",
                    xytext=(dx, dy), ha=align, fontsize=9.4, color=entry["colour"],
                    fontweight="bold")
    ax.plot(cells, nees, color=D.MUTED, lw=1.4, zorder=2)
    ax.axhline(2.0, color=D.INK, lw=1.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("Number of cells the training data is split into (log scale)")
    ax.set_ylabel("Mean NEES on held-out tiles")
    ax.set_title("5. Splitting finer makes it worse, not better\n"
                 "3,249 training readings do not survive a 64-way split",
                 fontsize=14, fontweight="bold")
    ax.grid(color="#e4e2dc", lw=0.8); ax.set_axisbelow(True)
    ax.set_ylim(1.2, max(nees) * 1.42)

    # --- 6. the verdict -------------------------------------------------------------------
    ax = fig.add_subplot(grid[1, 2])
    ax.axis("off")
    oracle = bundle["oracle"]
    best = min(bundle["results"], key=lambda e: e["mean_nees"])
    ax.text(0.0, 1.0, "So why does fitting R not work?", fontsize=16, fontweight="bold",
            va="top")
    ax.text(
        0.0, 0.905,
        "Not because the noise is intractable. An oracle told\n"
        f"which decile a reading lands in reaches NEES "
        f"{oracle['mean_nees']:.2f} with a\n"
        f"SHARPER sigma ({oracle['sharpness_cm']:.1f} cm against "
        f"{best['sharpness_cm']:.1f} cm) and holds\n"
        f"{100 * oracle['containment_95']:.0f}% inside its 95% ellipse. A Gaussian R fits\n"
        "this data fine — once you know which readings are bad.\n\n"
        "It fails because the variable that says so is missing.\n"
        "The best available conditioning reaches only "
        f"{best['mean_nees']:.2f}.\n\n"
        "Heading is not it: conditioning on heading changes\n"
        "essentially nothing here. Heading moves the reading\n"
        "(figure 06, 4.7 cm typical) but it does not decide\n"
        "which readings are catastrophic.\n\n"
        "Occlusion is closer — it is the single best real\n"
        "variable — but only as a coarse flag. A robot 10%\n"
        "hidden and one 90% hidden share the same flag and\n"
        "behave nothing alike, so most of the signal is lost.\n\n"
        "And the data will not support a finer split: past\n"
        "about 40 cells the partition is fitting noise.\n\n"
        "What would fix it: capture the occluded FRACTION,\n"
        "not the flag. It is computable from CAD and the\n"
        "camera pose without ground truth, so it is a legal\n"
        "runtime feature — and the repeat panel PLAN.md 1.2\n"
        "asks for should stratify on it.",
        fontsize=11.7, va="top", linespacing=1.40)

    fig.suptitle(
        "How R is fitted, and the three reasons it comes out wrong\n"
        "One is a procedure mistake and is fixed here; one is a missing variable; "
        "one is simply not enough data to split\n"
        "All numbers are held-out tiles, neural-corrected reading, out-of-fold unless "
        "the panel says otherwise",
        fontsize=19, fontweight="bold", y=0.975)
    save(fig, out / FOLDER / "26_why_fitting_R_fails.png")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out = args.out.expanduser().resolve()
    path = out / FOLDER / "26_why_fitting_R_fails.png"
    if path.exists() and not args.overwrite:
        raise RuntimeError(f"Output already exists: {path}; pass --overwrite")

    bundle = build(args.capture.expanduser().resolve(), args.seed, args.splits)
    draw(bundle, out)

    manifest = {
        "status": "complete",
        "schema": "why_r_fitting_fails.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mean_model": "nn",
        "residuals_for_fitting": "out-of-fold over TRAIN tiles, folds cut by floor position",
        "in_sample_vs_out_of_fold": {
            "in_sample_fit_mean_nees": bundle["wrong"]["mean_nees"],
            "out_of_fold_fit_mean_nees": bundle["right"]["mean_nees"],
        },
        "conditionings": [
            {k: v for k, v in entry.items() if k != "nees"}
            for entry in bundle["results"] + [bundle["oracle"]]
        ],
        "minimum_cell_size": MIN_CELL,
        "conclusion": (
            "A Gaussian R is achievable on this data: an oracle told the true error decile "
            "reaches NEES 2.15 with a sharper sigma. No runtime-available conditioning gets "
            "close, occlusion is the best of them, and the coarse binary flag discards most of "
            "the signal because occlusion is continuous."
        ),
        "next_required_evidence": (
            "the occluded FRACTION of the robot silhouette, computable from CAD and camera pose "
            "without ground truth, plus a repeat panel stratified on it"
        ),
        "figures": ["26_why_fitting_R_fails.png"],
    }
    (out / FOLDER / "why_r_fitting_fails_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out / FOLDER)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
