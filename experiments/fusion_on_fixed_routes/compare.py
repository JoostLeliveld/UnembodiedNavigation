#!/usr/bin/env python3
"""The cross-arm figures. Built LAST, from folders that already have their own storylines.

    python3 experiments/fusion_on_fixed_routes/compare.py

  01_error_and_claim_vs_cameras  the plot the experiment exists for: error and CLAIMED
                                 uncertainty against how many cameras were contributing
  02_the_six_arms                median and 95th error per arm with the honesty number
                                 beside it -- sharpness and honesty never apart
  03_what_the_box_meant          arms F4, O1, O2: what the observation model costs a filter

Refuses to run until every arm has a drive, because a partial comparison invites reading a
missing arm as a bad one. --partial overrides that for a working look.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "deck_figures"))
sys.path.insert(0, str(HERE.parent))
import style as D                                        # noqa: E402
import aligned as A  # noqa: E402
from score import FOLDER, TASKS, showcase_run, _selected_runs, score, story_dir   # noqa: E402
sys.path.insert(0, str(HERE.parent / "story"))
from fusion_examples import draw_moment, load as load_moments   # noqa: E402

STORY_ROOT = D.REPO / "logs/studies/fusion_on_fixed_routes"
LABEL = {"F1": "single best\ncamera", "F2": "distance and\nangle weights",
         "F3": "precisions\nadd", "F4": "joint network\nestimator",
         "O1": "raw box as\nthe robot", "O2": "box plus a\nfixed offset"}
COLOUR = {"F1": D.MUTED, "F2": D.OLD, "F3": D.BAD, "F4": D.GOOD,
          "O1": "#b06a3b", "O2": "#7a6a3b"}
BOX_NAME = {"F4": "hull — predict the box", "O1": "the box bottom-centre\nIS the robot",
            "O2": "the box plus one\nfixed offset"}


def per_arm(task):
    out = {}
    for arm in FOLDER:
        try:
            runs = _selected_runs(arm, task)
        except SystemExit:
            continue
        belief_run_median, stated_run_median = [], []
        corr_err, corr_stated, corr_cams, corr_run = [], [], [], []
        belief_err, belief_stated, belief_cams, belief_run = [], [], [], []
        for run_idx, run in enumerate(runs):
            rows = list(csv.DictReader(open(run / "experiment.csv")))
            if not rows:
                continue

            def col(key):
                vals = []
                for row in rows:
                    try:
                        vals.append(float(row[key]))
                    except (KeyError, TypeError, ValueError):
                        vals.append(math.nan)
                return np.array(vals)

            aligned = A.aligned_error_cm(run, "belief", rows)["aligned_cm"]
            sigma = np.sqrt((col("planner_cov_x") + col("planner_cov_y")) / 2.0) * 100.0
            keep = np.isfinite(aligned) & np.isfinite(sigma)
            if keep.any():
                belief_run_median.append(float(np.median(aligned[keep])))
                stated_run_median.append(float(np.median(sigma[keep])))

            for event in A.fused_answers(run):
                covariance = event["fused_cov"]
                if not (math.isfinite(event["error_cm"]) and np.isfinite(covariance).all()):
                    continue
                corr_err.append(event["error_cm"])
                corr_stated.append(float(np.sqrt(np.trace(covariance) / 2.0) * 100.0))
                corr_cams.append(event["n_candidates"])
                corr_run.append(run_idx)
            for event in A.belief_at_fusion_events(run, rows):
                belief_err.append(event["error_cm"])
                belief_stated.append(event["stated_sigma_cm"])
                belief_cams.append(event["n_candidates"])
                belief_run.append(run_idx)

        if not belief_run_median:
            continue
        out[arm] = {
            "numbers": score(arm, task),
            "err": np.asarray(belief_run_median),
            "stated": np.asarray(stated_run_median),
            "corr_err": np.asarray(corr_err), "corr_stated": np.asarray(corr_stated),
            "corr_cams": np.asarray(corr_cams), "corr_run": np.asarray(corr_run),
            "belief_err": np.asarray(belief_err),
            "belief_stated": np.asarray(belief_stated),
            "belief_cams": np.asarray(belief_cams),
            "belief_run": np.asarray(belief_run),
        }
    return out


def main() -> int:
    args = sys.argv[1:]
    # Accept "--task NAME" as well as "--task=NAME", and refuse anything unrecognised.
    # Silently ignoring an argument here meant asking for one route and being handed
    # another route's figures, with the same filenames and no warning.
    task = TASKS[0]
    rest = []
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith("--task="):
            task = a.split("=", 1)[1]
        elif a == "--task":
            if i + 1 >= len(args):
                raise SystemExit("--task needs a route name")
            task = args[i + 1]; i += 1
        elif a == "--partial":
            rest.append(a)
        else:
            raise SystemExit(f"unknown argument {a!r}; use --task=<route> and --partial")
        i += 1
    if task not in TASKS:
        raise SystemExit(f"unknown route {task!r}; choose one of {', '.join(TASKS)}")
    OUT = STORY_ROOT / task / "compare"
    OUT.mkdir(parents=True, exist_ok=True)
    route = task.replace("fusion_", "").replace("_", " ")
    arms = per_arm(task)
    missing = [a for a in FOLDER if a not in arms]
    if missing and "--partial" not in sys.argv:
        raise SystemExit(f"no drive yet for {missing}; a partial comparison reads a missing "
                         "arm as a bad one. Pass --partial for a working look.")
    ids = [a for a in FOLDER if a in arms]

    # ---------------- 01 error and claim against camera count ----------------
    bins = [1, 2, 3, 4]
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 10.4), constrained_layout=True)
    panels = [
        (axes[0][0], "corr_err", "how far the CORRECTION was from the truth"),
        (axes[0][1], "corr_stated", "how precise the correction CLAIMED to be"),
        (axes[1][0], "belief_err", "how far the robot's BELIEF was from the truth"),
        (axes[1][1], "belief_stated", "how precise the belief CLAIMED to be"),
    ]
    for ax, key, title in panels:
        for a in ids:
            prefix = "corr" if key.startswith("corr_") else "belief"
            cams, values = arms[a][f"{prefix}_cams"], arms[a][key]
            run_ids = arms[a][f"{prefix}_run"]
            xs, ys, lows, highs = [], [], [], []
            for b in bins:
                run_medians = []
                for run_id in np.unique(run_ids):
                    sel = (cams == b) & (run_ids == run_id) & np.isfinite(values)
                    if sel.sum() >= 5:
                        run_medians.append(float(np.median(values[sel])))
                if len(run_medians) < 3:
                    continue
                xs.append(b)
                ys.append(float(np.median(run_medians)))
                lows.append(float(min(run_medians)))
                highs.append(float(max(run_medians)))
            if xs:
                ax.plot(xs, ys, "-o", color=COLOUR[a], lw=2.2, ms=8, label=a)
                ax.fill_between(xs, lows, highs, color=COLOUR[a], alpha=0.10)
        ax.set_xlabel("cameras available at that correction", fontsize=12)
        ax.set_ylabel("centimetres", fontsize=12)
        ax.set_xticks(bins)
        ax.grid(True, color="#eeede8", lw=0.7); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_title(title, loc="left", fontsize=14.5, color=D.INK)
    axes[0][0].legend(frameon=False, fontsize=11.5, ncol=3)
    fig.suptitle(f"Does the claim shrink faster than the error? — {route}",
                 x=0.005, ha="left", fontsize=19, color=D.INK)
    fig.text(0.005, -0.025,
             "Top row: the correction the camera network published — this is where the fusion "
             "rules differ, by construction.\n"
             "Bottom row: what the robot's own filter ended up believing, which also carries "
             "odometry between corrections and so hides much of that difference.\n"
             "The axis is how many cameras were AVAILABLE, not how many each rule chose to use, "
             "so the arms are read against the same thing. Bins with fewer than 5 samples are "
             "dropped. Points are medians of per-run medians across five paired seeds; "
             "shading is the run range.",
             fontsize=11.5, color=D.INK2, va="top", linespacing=1.5)
    fig.savefig(OUT / "01_error_and_claim_vs_cameras.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ---------------- 02 the six arms ----------------
    fig, ax = plt.subplots(figsize=(13.6, 6.4), constrained_layout=True)
    x = np.arange(len(ids))
    med = np.array([float(np.median(arms[a]["err"])) for a in ids])
    p95 = np.array([float(np.percentile(arms[a]["err"], 95)) for a in ids])
    claim = np.array([float(np.median(arms[a]["stated"])) for a in ids])
    ax.bar(x - 0.21, med, 0.4, color=[COLOUR[a] for a in ids], label="median error")
    ax.bar(x - 0.21, p95 - med, 0.4, bottom=med, color=[COLOUR[a] for a in ids], alpha=0.35,
           label="up to the 95th percentile")
    ax.bar(x + 0.21, claim, 0.4, facecolor="none", edgecolor=D.ROBOT, lw=2.2, hatch="///",
           label="what it claimed to know (median 1σ)")
    labels = []
    for a in ids:
        inside = arms[a]["numbers"]["honesty"]["truth_inside_stated_95pct_ellipse"]
        note = "—" if inside is None else f"{inside*100:.0f}%"
        labels.append(f"{a}\n{LABEL[a]}\ntruth inside its 95%: {note}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11.5)
    ax.set_ylim(0, max(p95.max(), claim.max()) * 1.12)
    ax.set_ylabel("centimetres  (lower is better)", fontsize=12.5)
    ax.grid(True, axis="y", color="#eeede8", lw=0.7); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=11.5, ncol=3, loc="upper left")
    ax.set_title(f"Six ways of using the same cameras — {route}",
                 loc="left", fontsize=19, color=D.INK)
    fig.text(0.005, -0.03,
             "Five paired seeds per arm on the frozen route. Bars summarize per-run medians; "
             "error against ground truth; the hatched bar is what that arm claimed to know.\n"
             "Honest means the truth falls inside the stated 95% ellipse about 95% of the "
             "time: much less is overconfident, much more is a padded ellipse.\n"
             "The transparent extension shows the 95th percentile across the five run-level "
             "median errors, not across thousands of correlated log rows.",
             fontsize=11.5, color=D.INK2, va="top", linespacing=1.5)
    fig.savefig(OUT / "02_the_six_arms.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ---------------- 03 what the box meant ----------------
    box_ids = [a for a in ("F4", "O1", "O2") if a in arms]
    if len(box_ids) >= 2:
        fig, ax = plt.subplots(figsize=(11.6, 5.8), constrained_layout=True)
        for i, a in enumerate(box_ids):
            err = arms[a]["err"]
            parts = ax.violinplot(err, positions=[i], widths=0.7, showextrema=False)
            for body in parts["bodies"]:
                body.set_facecolor(COLOUR[a]); body.set_alpha(0.35)
            ax.plot([i - 0.2, i + 0.2], [np.median(err)] * 2, color=COLOUR[a], lw=3)
            ax.text(i, float(np.percentile(err, 98)) * 1.04, f"median {np.median(err):.1f} cm",
                    ha="center", fontsize=12, color=D.INK)
        ax.set_xticks(range(len(box_ids)))
        ax.set_xticklabels([f"{a}\n{BOX_NAME[a]}" for a in box_ids], fontsize=12)
        ax.set_ylabel("belief error (cm)", fontsize=12.5)
        ax.grid(True, axis="y", color="#eeede8", lw=0.7); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_title("What the detector's box was taken to mean, measured in a filter",
                     loc="left", fontsize=17, color=D.INK)
        fig.text(0.005, -0.05,
                 "Same joint network estimator, same route: only the meaning "
                 "of the box differs.\nCommissioning measured that gap at 24-36 cm on single "
                 "sightings; this is what it costs a filter that fuses many of them.\n"
                 "Each violin contains five per-run median errors (paired seeds).",
                 fontsize=11.5, color=D.INK2, va="top", linespacing=1.5)
        fig.savefig(OUT / "03_what_the_box_meant.png", dpi=170, bbox_inches="tight")
        plt.close(fig)

    # ---------------- 04 the same places, six ways ----------------
    # Same route, so "the same place" is well defined even though the arms drove it at their
    # own pace and with their own beliefs. For each place, each arm's correction nearest to it.
    # Places are taken along THIS route rather than named by hand, so the figure means the
    # same thing on a 30 m corridor and on the 63 m out-and-back.
    from score import _route_polyline
    poly = _route_polyline(task)
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    PLACES = []
    for frac in (0.2, 0.45, 0.7, 0.92):
        d = frac * cum[-1]
        i = int(np.searchsorted(cum, d))
        i = min(max(i, 1), len(poly) - 1)
        t = (d - cum[i - 1]) / max(seg[i - 1], 1e-9)
        pt = poly[i - 1] + t * (poly[i] - poly[i - 1])
        PLACES.append((float(pt[0]), float(pt[1]), f"{d:.0f} m in"))
    moments_by_arm = {}
    for a in ids:
        try:
            _run, moments = load_moments(a, task)
        except SystemExit:
            continue
        moments_by_arm[a] = moments
    if moments_by_arm:
        rows_n = len(moments_by_arm)
        fig, axes = plt.subplots(rows_n, len(PLACES),
                                 figsize=(3.5 * len(PLACES) + 1.2, 3.3 * rows_n + 1.0),
                                 constrained_layout=True)
        axes = np.atleast_2d(axes)
        for r, a in enumerate(moments_by_arm):
            for c, (px, py, place) in enumerate(PLACES):
                ax = axes[r][c]
                target = np.array([px, py])
                near = min(moments_by_arm[a],
                           key=lambda m: float(np.linalg.norm(np.asarray(m["gt"]) - target)))
                draw_moment(ax, "" if r else place, near)
                ax.set_xlabel(ax.get_xlabel(), fontsize=10)
                if c == 0:
                    ax.set_ylabel(f"{a}\n{LABEL[a]}", fontsize=12, color=COLOUR[a],
                                  fontweight="bold")
        fig.suptitle(f"The same four places, handled six ways — {route}",
                     x=0.004, y=1.012, ha="left", fontsize=19, color=D.INK)
        fig.text(0.004, -0.012,
                 "Each panel is one correction: every camera's own answer (dots used, crosses "
                 "not used), what that arm's rule produced (black diamond), and the truth "
                 "(green star).\n"
                 "Rows are arms, columns are places. Each arm drove the same route at its own "
                 "pace, so these are the corrections nearest each place, not the same instant.\n"
                 "One drive per arm, seed 0.",
                 fontsize=11.5, color=D.INK2, va="top", linespacing=1.5)
        fig.savefig(OUT / "04_the_same_places_six_ways.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    for a in ids:
        n = arms[a]["numbers"]
        print(f"  {a}: median {np.median(arms[a]['err']):5.2f} cm  p95 "
              f"{np.percentile(arms[a]['err'], 95):5.2f} cm  claimed "
              f"{np.mean(arms[a]['stated']):5.2f} cm  inside 95% "
              f"{n['honesty']['truth_inside_stated_95pct_ellipse']}  {n['completion']}")
    print(f"{task}: wrote cross-arm figures to {OUT.relative_to(D.REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
