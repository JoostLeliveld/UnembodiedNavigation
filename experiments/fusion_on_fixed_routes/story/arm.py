#!/usr/bin/env python3
"""One arm's storyline, from its own drive. Three figures, no comparison with any other arm.

    python3 experiments/fusion_on_fixed_routes/story/arm.py F4

  02_the_drive        where it drove and how far its belief was from the truth
  03_claim_vs_truth   the error against what it claimed to know, and the honesty number
  04_worst_moment     the single worst moment, in the place it happened

Beat 1 -- what this arm's rule does -- is the shared mechanism figure
logs/studies/fusion_on_fixed_routes/00_hull_observation/03_three_rules_one_moment.png,
which draws all four rules on one real moment rather than six near-identical pictures.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "deck_figures"))
sys.path.insert(0, str(HERE.parents[1]))
import aligned as A  # noqa: E402
import style as D                                          # noqa: E402
from score import FOLDER, TASKS, _latest_run, _route_polyline, score, story_dir  # noqa: E402

ARM_TITLE = {
    "F1": "F1 — the single best camera",
    "F2": "F2 — weighted by distance and viewing angle",
    "F3": "F3 — precisions add",
    "F4": "F4 — the network as one sensor",
    "O1": "O1 — the box bottom-centre taken as the robot",
    "O2": "O2 — the box bottom-centre plus a fixed offset",
}


def load(arm, task=TASKS[0]):
    run = _latest_run(arm, task)
    rows = list(csv.DictReader(open(run / "experiment.csv")))

    def col(key):
        out = []
        for r in rows:
            try:
                out.append(float(r[key]))
            except (KeyError, TypeError, ValueError):
                out.append(math.nan)
        return np.array(out)

    cams_used = np.array([r.get("fusion_cameras", "") for r in rows], dtype=object)
    cands = np.array([r.get("fusion_candidates", "") for r in rows], dtype=object)
    d = {k: col(k) for k in ("stamp", "gt_x", "gt_y", "gt_available",
                             "planner_belief_x", "planner_belief_y", "planner_cov_x",
                             "planner_cov_xy", "planner_cov_y", "state_available",
                             "state_stamp", "odom_map_gt_drift_m")}
    # Errors come from the shared loader, scored against the truth at each estimate's own
    # stamp. Reading `belief_error_gt_m` out of a pre-2026-08-28 drive pairs the belief
    # with the truth one 10 Hz log row later, which adds a fixed ~2.2 cm of robot travel
    # to every point on every figure built from it.
    d["belief_error_gt_m"] = A.aligned_error_cm(run, "belief", rows)["aligned_cm"] / 100.0
    d["state_error_gt_m"] = A.aligned_error_cm(run, "state", rows)["aligned_cm"] / 100.0
    keep = (d["gt_available"] == 1.0) & np.isfinite(d["belief_error_gt_m"])
    d = {k: v[keep] for k, v in d.items()}
    d["cameras_used"] = cams_used[keep]
    d["cameras_available"] = cands[keep]
    gt = np.column_stack([d["gt_x"], d["gt_y"]])
    travelled = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(gt, axis=0), axis=1))])
    stated = np.sqrt((d["planner_cov_x"] + d["planner_cov_y"]) / 2.0) * 100.0
    return run, rows, d, gt, travelled, d["belief_error_gt_m"] * 100.0, stated


def main() -> int:
    args = [a for a in sys.argv[1:]]
    task = next((a.split("=", 1)[1] for a in args if a.startswith("--task=")), TASKS[0])
    positional = [a for a in args if not a.startswith("--")]
    arm = (positional[0] if positional else "F4").upper()
    if arm not in FOLDER:
        raise SystemExit(f"unknown arm {arm!r}")
    out = story_dir(task, arm)
    out.mkdir(parents=True, exist_ok=True)
    run, rows, d, gt, travelled, err_cm, stated_cm = load(arm, task)
    numbers = score(arm, task)
    poly = _route_polyline(task)
    lay = D.layout()
    title = ARM_TITLE[arm]
    route_name = task.replace("fusion_", "").replace("_", " ")

    # ---------------- 02 the drive ----------------
    fig = plt.figure(figsize=(15.4, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.06, 1.0], height_ratios=[1.0, 1.0])
    ax = fig.add_subplot(gs[:, 0])
    D.draw_warehouse(ax, lay)
    ax.plot(poly[:, 0], poly[:, 1], color=D.MUTED, lw=2.0, ls=(0, (5, 4)), zorder=5,
            label="the route it was told to drive")
    sc = ax.scatter(gt[:, 0], gt[:, 1], c=err_cm, cmap="magma_r", s=16, zorder=7,
                    vmin=0, vmax=max(8.0, float(np.percentile(err_cm, 98))))
    ax.plot(gt[0, 0], gt[0, 1], "o", ms=14, color=D.INK, zorder=9)
    ax.plot(gt[-1, 0], gt[-1, 1], "*", ms=24, color=D.INK, zorder=9)
    bar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.01)
    bar.set_label("how far the robot's belief was from the truth (cm)", fontsize=11.5)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.01), fontsize=12, frameon=False)
    ax.set_title(f"{title}: where it drove — {route_name}", loc="left", fontsize=17,
                 color=D.INK, pad=8)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(travelled, err_cm, color=D.BAD, lw=2.0, label="error of the belief")
    ax.plot(travelled, stated_cm, color=D.ROBOT, lw=2.0,
            label="what it claimed to know (1σ)")
    ax.fill_between(travelled, 0, stated_cm, color=D.ROBOT, alpha=0.12, lw=0)
    ax.set_xlabel("distance driven (m)", fontsize=12)
    ax.set_ylabel("centimetres", fontsize=12)
    ax.grid(True, color="#eeede8", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=11.5, loc="upper left")
    ax.set_title("Error against the claim, along the way", loc="left", fontsize=14.5,
                 color=D.INK)

    ax = fig.add_subplot(gs[1, 1])
    # 711 corrections drawn as lines is a solid block. What matters is the GAPS: how long the
    # robot went uncorrected, which is what lets the belief drift.
    since = np.zeros(len(travelled))
    last = d["stamp"][0]
    for i, (t, st) in enumerate(zip(d["stamp"], d["state_stamp"])):
        if math.isfinite(st) and st > last:
            last = st
        since[i] = max(t - last, 0.0)
    ax.fill_between(travelled, 0, since, color=D.BAD, alpha=0.22, lw=0)
    ax.plot(travelled, since, color=D.BAD, lw=2.0)
    ax.set_xlabel("distance driven (m)", fontsize=12)
    ax.set_ylabel("seconds since the last\ncamera correction", fontsize=12)
    ax.grid(True, color="#eeede8", lw=0.6); ax.set_axisbelow(True)
    ax.set_xlim(travelled.min(), travelled.max())
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.set_title(f"How long it went uncorrected "
                 f"({numbers['corrections']['received']} corrections, worst gap "
                 f"{numbers['corrections']['longest_gap_s']} s)",
                 loc="left", fontsize=14.5, color=D.INK)

    e, h, dr = numbers["belief_error_cm"], numbers["honesty"], numbers["driving"]
    fig.text(0.006, -0.02,
             f"One drive, seed 0, {numbers['completion']} after {numbers['duration_s']} s.  "
             f"Belief error median {e['median']} cm, 95th {e['p95']} cm, worst {e['worst']} cm "
             f"over {e['n_samples']} samples.\n"
             f"Ground truth forms these errors and scores them; it is never an input to the "
             f"filter, the manager or the planner.  Odometry alone had drifted "
             f"{numbers['odometry']['final_odom_vs_truth_drift_m']:.2f} m from the truth by the end.\n"
             f"Drove {dr['ground_truth_path_m']} m and stayed within "
             f"{dr['max_offset_from_commanded_route_m']} m of the commanded route.  "
             f"ONE drive, one seed: not a variance claim.",
             fontsize=11.5, color=D.INK2, va="top", linespacing=1.5)
    fig.savefig(out / "02_the_drive.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ---------------- 03 claim vs truth ----------------
    # The honest question is not "is the error bigger than an averaged sigma" but "is the
    # truth inside the ellipse this arm actually stated" -- an ellipse can be wide one way and
    # tight the other, which is exactly what a single camera produces. So the curve is the
    # Mahalanobis distance in units of the arm's own 95% ellipse: 1.0 is the honesty line.
    resid = np.column_stack([d["gt_x"] - d["planner_belief_x"],
                             d["gt_y"] - d["planner_belief_y"]])
    units = np.full(len(resid), np.nan)
    for i in range(len(resid)):
        cov = np.array([[d["planner_cov_x"][i], d["planner_cov_xy"][i]],
                        [d["planner_cov_xy"][i], d["planner_cov_y"][i]]])
        if not np.all(np.isfinite(cov)) or np.linalg.det(cov) <= 0.0:
            continue
        units[i] = math.sqrt(float(resid[i] @ np.linalg.solve(cov, resid[i])) / 5.991)
    ok = np.isfinite(units)

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.8), constrained_layout=True)
    ax = axes[0]
    ax.axhspan(0, 1, color=D.GOOD, alpha=0.10, lw=0)
    ax.plot(travelled[ok], units[ok], color=D.BAD, lw=1.8)
    ax.axhline(1.0, color=D.INK, lw=1.6, ls=(0, (5, 3)))
    ax.text(0.99, 0.965, "honest below this line — the truth inside its own 95% ellipse",
            transform=ax.transAxes, ha="right", va="top", fontsize=11.5, color=D.INK)
    ax.set_xlabel("distance driven (m)", fontsize=12)
    ax.set_ylabel("how many of its own 95% ellipses\nthe truth was away", fontsize=12)
    ax.grid(True, color="#eeede8", lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("Was the truth where it said the truth would be?", loc="left",
                 fontsize=15, color=D.INK)

    ax = axes[1]
    ax.plot(travelled, err_cm, color=D.BAD, lw=2.0, label="error of the belief")
    ax.plot(travelled, stated_cm, color=D.ROBOT, lw=2.0, label="its stated 1σ (averaged)")
    ax.set_xlabel("distance driven (m)", fontsize=12)
    ax.set_ylabel("centimetres", fontsize=12)
    ax.grid(True, color="#eeede8", lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=11.5)
    ax.set_title(f"Sharpness beside it: median error {np.median(err_cm):.1f} cm, "
                 f"mean claim {np.mean(stated_cm):.1f} cm", loc="left", fontsize=15,
                 color=D.INK)

    inside = h["truth_inside_stated_95pct_ellipse"]
    fig.suptitle(f"{title}: was it honest?", x=0.005, ha="left", fontsize=19, color=D.INK)
    fig.text(0.005, -0.04,
             f"The truth fell inside its stated 95% ellipse "
             f"{'' if inside is None else f'{inside*100:.0f}% of the time'} — honest is about "
             f"95%; below that is overconfident, far above means a padded ellipse.\n"
             f"An averaged sigma can look generous while the ellipse is tight in one "
             f"direction, which is why the left panel uses the arm's own ellipse and not an "
             f"average.\nOne drive, seed 0.",
             fontsize=11.5, color=D.INK2, va="top", linespacing=1.5)
    fig.savefig(out / "03_claim_vs_truth.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ---------------- 04 worst moment ----------------
    k = int(np.argmax(err_cm))
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.2), constrained_layout=True)
    ax = axes[0]
    D.draw_warehouse(ax, lay, camera_labels=True)
    ax.plot(gt[:, 0], gt[:, 1], color=D.MUTED, lw=2.0, zorder=6)
    ax.plot(gt[k, 0], gt[k, 1], "*", ms=26, color=D.BAD, zorder=9)
    ax.set_title(f"Where the worst moment happened, {travelled[k]:.1f} m in",
                 loc="left", fontsize=15, color=D.INK)
    ax = axes[1]
    span = max(err_cm[k] * 1.8, 12.0)
    cov = np.array([[d["planner_cov_x"][k], d["planner_cov_xy"][k]],
                    [d["planner_cov_xy"][k], d["planner_cov_y"][k]]])
    w, V = np.linalg.eigh(cov)
    t = np.linspace(0, 2 * math.pi, 200)
    ell = (V @ (np.sqrt(np.maximum(w, 0))[:, None] * np.array([np.cos(t), np.sin(t)]))).T * 100.0
    bx, by = d["planner_belief_x"][k], d["planner_belief_y"][k]
    rel = (np.array([bx, by]) - gt[k]) * 100.0
    ax.plot(ell[:, 0] + rel[0], ell[:, 1] + rel[1], color=D.ROBOT, lw=2.4)
    ax.plot(1.96 * ell[:, 0] + rel[0], 1.96 * ell[:, 1] + rel[1], color=D.ROBOT, lw=1.4,
            ls=(0, (4, 3)))
    ax.annotate("", xy=tuple(rel), xytext=(0, 0), zorder=5,
                arrowprops=dict(arrowstyle="-|>", color=D.BAD, lw=2.6, shrinkA=8, shrinkB=6))
    ax.plot(*rel, "o", ms=12, color=D.ROBOT, mec="white", mew=1.6, zorder=6)
    ax.plot(0, 0, "*", ms=24, color=D.INK, mec="white", mew=1.2, zorder=7)
    ax.text(0.02, 0.03, "★  the truth", transform=ax.transAxes, fontsize=12.5, color=D.INK)
    ax.set_xlim(-span, span); ax.set_ylim(-span, span); ax.set_aspect("equal")
    ax.grid(True, color="#f2f1ec", lw=0.7); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    used = str(d["cameras_used"][k] or "none")
    avail = str(d["cameras_available"][k] or "none")
    ax.set_title(f"It was {err_cm[k]:.1f} cm out, claiming ±{stated_cm[k]:.1f} cm",
                 loc="left", fontsize=15, color=D.INK)
    ax.set_xlabel(f"centimetres from the truth   ·   using camera {used} "
                  f"(available: {avail})", fontsize=12)
    fig.suptitle(f"{title}: the worst moment of the drive", x=0.005, ha="left",
                 fontsize=19, color=D.INK)
    fig.text(0.005, -0.035,
             f"Solid ellipse is its stated 1σ at that instant, dashed is its 95% claim.  "
             f"The worst single moment of a drive is a tail, not a summary: read it with the "
             f"median ({e['median']} cm) and the 95th ({e['p95']} cm).",
             fontsize=11.5, color=D.INK2, va="top", linespacing=1.5)
    fig.savefig(out / "04_worst_moment.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    print(f"{task}/{arm}: wrote 3 figures to {out.relative_to(D.REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
