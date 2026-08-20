#!/usr/bin/env python3
"""Figures for the availability-aware planning paper.

Follows ``supervisor_comparison/FIGURE_CONTRACT.md``: one canvas, fixed [0, 1]
reliability scale, camera colours, and — the part that matters — every title
states the finding and every axis says which direction is good. A reader who has
not seen the code should be able to read each panel alone.

Run:
    python3 experiments/availability_paper/make_figures.py
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common as C  # noqa: E402
import render_all as base  # noqa: E402

E1 = C.OUT_ROOT / "e1_availability_calibration"
E2 = C.OUT_ROOT / "e2_availability_vs_accuracy"
E3 = C.OUT_ROOT / "e3_route_discrimination"
FIGURES = C.OUT_ROOT / "figures"

SURVEY_HATCH = "///"
DPI = 170


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise RuntimeError(f"missing input: {path}. Run the experiment that writes it first.")
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _order() -> list[C.Source]:
    """Arms in ladder order: floor, geometry, deployable, learned."""

    keys = ["constant", "distance", "fov_range", "cad_reference", "mono_depth", "gp", "hybrid"]
    return [C.SOURCE_BY_KEY[k] for k in keys]


def _wrap(label: str, width: int = 13) -> str:
    words = label.split()
    lines, current = [], ""
    for word in words:
        if len(current) + len(word) + 1 > width and current:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    lines.append(current)
    return "\n".join(lines)


def fig_calibration() -> None:
    """E1: which estimator predicts whether a detection arrives."""

    summary = {r["source"]: r for r in read_csv(E1 / "e1_summary.csv") if r["variant"] == "linked"}
    paired = read_csv(E1 / "e1_paired.csv")
    vs_cad = {
        (r["source"], r["metric"]): r
        for r in paired
        if r["reference"] == "cad_reference"
    }

    arms = _order()
    labels = [_wrap(s.label) for s in arms]
    brier = [float(summary[s.key]["brier_mean"]) for s in arms]
    brier_sd = [float(summary[s.key]["brier_std"]) for s in arms]
    auroc = [float(summary[s.key]["auroc_mean"]) for s in arms]
    auroc_sd = [float(summary[s.key]["auroc_std"]) for s in arms]

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.4))
    x = np.arange(len(arms))

    for ax, values, errors, name, good in (
        (axes[0], brier, brier_sd, "Brier score", "lower"),
        (axes[1], auroc, auroc_sd, "AUROC", "higher"),
    ):
        for i, source in enumerate(arms):
            ax.bar(
                x[i],
                values[i],
                yerr=errors[i],
                capsize=4,
                color=source.color,
                edgecolor="#222222",
                hatch=SURVEY_HATCH if source.needs_surveyed_model else None,
                linewidth=0.9,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylabel(f"{name} on held-out ground  ({good} is better)", fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    axes[1].axhline(0.5, color="#b0271f", ls="--", lw=1.2)
    axes[1].text(
        -0.35, 0.512, "0.5 = no better than guessing",
        ha="left", fontsize=9, color="#b0271f",
    )
    axes[1].set_ylim(0.4, 1.0)

    # Mark the arms that are statistically indistinguishable from the CAD raycast.
    for i, source in enumerate(arms):
        row = vs_cad.get((source.key, "brier"))
        if row is None:
            continue
        p = float(row["sign_test_p_two_sided"])
        if p > 0.05:
            axes[0].annotate(
                f"ties CAD\n(p = {p:.2f})",
                xy=(x[i], brier[i]),
                xytext=(x[i], brier[i] + 0.055),
                ha="center",
                fontsize=9,
                color="#1a6b1a",
                arrowprops=dict(arrowstyle="-", color="#1a6b1a", lw=1.0),
            )

    fig.suptitle(
        "Monocular depth predicts whether a camera will see the robot as well as a surveyed CAD model;\n"
        "a Gaussian process without geometry still ranks poses, but its probabilities are no better than a constant",
        fontsize=14,
        weight="bold",
    )
    fig.text(
        0.5,
        0.015,
        "8,808 real detector outcomes, four cameras, warehouse_full_4cam. Leave-one-spatial-block-out: "
        "six contiguous blocks, so held-out ground is never adjacent to training ground.\n"
        "Bars are means over 24 camera-folds, whiskers one standard deviation. "
        f"Hatched ({SURVEY_HATCH}) = the method needs a surveyed 3-D model of the warehouse. "
        "EXPLORATORY offline prediction — no navigation claim.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.90))
    out = FIGURES / "01_availability_calibration.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


def fig_calibration_curves(bins: int = 10) -> None:
    """E1: predicted versus observed detection frequency."""

    ece = {
        r["source"]: float(r["ece_mean"])
        for r in read_csv(E1 / "e1_summary.csv")
        if r["variant"] == "linked"
    }
    rows = read_csv(E1 / "e1_predictions.csv")
    by_source: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append((float(r["p_linked"]), float(r["hit"])))

    fig, ax = plt.subplots(figsize=(10.4, 8.0))
    ax.plot([0, 1], [0, 1], color="#444444", ls="--", lw=1.3, label="perfectly calibrated", zorder=1)

    edges = np.linspace(0.0, 1.0, bins + 1)
    for source in _order():
        data = by_source.get(source.key)
        if not data:
            continue
        p = np.asarray([d[0] for d in data])
        y = np.asarray([d[1] for d in data])
        centres, observed = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
            if int(m.sum()) < 25:  # too few events to plot a frequency honestly
                continue
            centres.append(float(np.mean(p[m])))
            observed.append(float(np.mean(y[m])))
        ax.plot(
            centres,
            observed,
            marker="o",
            ms=5,
            lw=2.0,
            color=source.color,
            label=(
                f"{source.label}"
                f"{' (needs survey)' if source.needs_surveyed_model else ''}"
                f" — calibration error {ece.get(source.key, float('nan')):.3f}"
            ),
            zorder=3,
        )

    ax.set_xlabel("Predicted probability that a usable detection arrives", fontsize=11)
    ax.set_ylabel("Observed fraction that actually did", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title(
        "No availability model here is sharply calibrated, but the geometry-free\n"
        "Gaussian process is confidently wrong in both directions",
        fontsize=12.5,
        weight="bold",
    )
    fig.text(
        0.5,
        0.015,
        "Held-out predictions only. Above the diagonal = the model is pessimistic; below = it promises "
        "observations that do not arrive.\nCurves pool four cameras and six spatial blocks, each with its own "
        "fitted link, so the wobble overstates any single camera's disorder — the calibration error in the legend "
        "is the number to compare.\nBins with fewer than 25 events are omitted rather than plotted as noise. "
        "EXPLORATORY offline prediction.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    out = FIGURES / "02_calibration_curves.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def fig_availability_vs_accuracy() -> None:
    """E2: the two fields are not the same field."""

    corr = read_csv(E2 / "e2_correlations.csv")
    disagree = read_csv(E2 / "e2_camera_disagreement.csv")[0]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2))

    cameras = [r["camera"] for r in corr]
    x = np.arange(len(cameras))
    width = 0.38
    rho_avail_err = [abs(float(r["spearman_availability_vs_error"])) for r in corr]
    rho_cad_avail = [abs(float(r["spearman_cad_vs_availability"])) for r in corr]
    rho_cad_err = [abs(float(r["spearman_cad_vs_error"])) for r in corr]

    axes[0].bar(x - width / 2, rho_cad_avail, width, color="#2a6f97",
                label="geometry predicts WHETHER a detection arrives")
    axes[0].bar(x + width / 2, rho_cad_err, width, color="#c9772f",
                label="the same geometry predicts HOW ACCURATE it is")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([c.replace("camera_", "camera ") for c in cameras])
    axes[0].set_ylabel("Rank correlation with the geometric visibility field\n(higher = better predicted)", fontsize=10)
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=9, loc="upper right")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].set_axisbelow(True)
    axes[0].set_title("One geometric field explains availability, not accuracy", fontsize=12, weight="bold")

    axes[1].bar(x, rho_avail_err, 0.5, color="#6a4c93")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([c.replace("camera_", "camera ") for c in cameras])
    axes[1].set_ylabel(
        "|rank correlation| between availability and conditional error\n(1 = one field would do; 0 = two separate fields)",
        fontsize=10,
    )
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_axisbelow(True)
    axes[1].set_title(
        "Knowing a camera often sees you says little about how well", fontsize=12, weight="bold"
    )

    fraction = float(disagree["fraction_argmax_availability_ne_argmin_error"]) * 100.0
    penalty_cm = float(disagree["median_error_penalty_following_availability_m"]) * 100.0
    axes[1].text(
        0.5,
        0.94,
        f"At {fraction:.0f}% of the {disagree['n_positions_compared']} positions seen by two or more cameras,\n"
        f"the camera most likely to see the robot is NOT the most accurate one.\n"
        f"Choosing by availability costs a median {penalty_cm:.1f} cm there.",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f4f0fa", edgecolor="#6a4c93"),
    )

    fig.suptitle(
        "Availability and conditional accuracy must be estimated separately: one scalar trust score cannot carry both",
        fontsize=13.5,
        weight="bold",
    )
    fig.text(
        0.5,
        0.015,
        "2026-08-07 commissioning grid, warehouse_full_4cam: 942 positions x 8 headings x 4 cameras, "
        "current zero-parameter floor IPM.\nAvailability = detections / attempted headings at a position. "
        "Conditional accuracy = mean projection error over that position's detections (ground truth used "
        "offline only). EXPLORATORY — no navigation or safety reading.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    out = FIGURES / "03_availability_vs_accuracy.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


def fig_route_discrimination(task: str = "mc_blind_L") -> None:
    """E3: the field changes the route, and the endpoint that shows it."""

    rows = read_csv(E3 / "e3_routes.csv")
    routes_path = E3 / "e3_selected_routes.json"
    selected = json.loads(routes_path.read_text())["routes"] if routes_path.is_file() else {}

    apparatus = C.build_apparatus()
    sys.path.insert(0, str(HERE / "e3_route_discrimination"))
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "e3_runner", HERE / "e3_route_discrimination/run_experiment.py"
    )
    assert spec and spec.loader
    e3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e3)
    fused_reference = e3.fused_field(apparatus.fields["cad_reference"])

    # Three columns: the corridor is ~3 m wide and ~17 m long, so the map panel is
    # unavoidably tall and thin and cannot host a legend. The route detail gets its
    # own panel instead of covering the field.
    fig = plt.figure(figsize=(16.5, 7.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.60, 0.78, 1.25], wspace=0.55)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_key = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[0, 2])
    ax_key.axis("off")

    base.draw_field(
        ax_map,
        fused_reference,
        apparatus.xs,
        apparatus.ys,
        apparatus.driveable,
        apparatus.prisms,
        title="",
    )

    # draw_field calls draw_cameras internally. Those labels sit at the wall
    # cameras, outside this zoom, and are drawn unclipped — clip them so they do
    # not float outside the panel.
    for artist in list(ax_map.texts):
        artist.set_clip_on(True)

    # e3_selected_routes.json is keyed camera-subset -> task -> source. This panel
    # shows the full four-camera network; the density sweep has its own figure.
    task_routes = selected.get("four", {}).get(task, {})
    show = [
        ("availability_blind", "#111111", "-", "ignores availability"),
        ("cad_reference", "#d94b4b", "--", "plans with surveyed CAD geometry"),
        ("mono_depth", "#f0a020", ":", "plans with the cameras' own depth"),
    ]
    all_pts: list[np.ndarray] = []
    key_entries: list[tuple] = []
    for key, colour, style, note in show:
        pts = task_routes.get(key)
        if not pts:
            continue
        arr = np.asarray(pts, dtype=float)
        all_pts.append(arr)
        row = next(
            (r for r in rows if r["task"] == task and r["source"] == key
             and r.get("cameras", "four") == "four"),
            None,
        )
        ax_map.plot(arr[:, 0], arr[:, 1], color=colour, lw=3.2, ls=style, alpha=0.95, zorder=6)
        key_entries.append((colour, style, note, row))
    if all_pts:
        first = all_pts[0]
        ax_map.plot(first[0, 0], first[0, 1], "o", color="#1a7f37", ms=11, zorder=8)
        ax_map.plot(first[-1, 0], first[-1, 1], "*", color="#b0271f", ms=17, zorder=8)

        # Zoom to where the routes actually are. At full-warehouse scale a 0.16 m
        # reroute is invisible, and an unreadable panel is worse than no panel.
        stack = np.vstack(all_pts)
        pad = 1.6
        ax_map.set_xlim(float(stack[:, 0].min()) - pad, float(stack[:, 0].max()) + pad)
        ax_map.set_ylim(float(stack[:, 1].min()) - pad, float(stack[:, 1].max()) + pad)

    # Route key, as its own panel.
    ax_key.text(0.0, 0.98, f"Task {task}", fontsize=11, weight="bold", va="top",
                transform=ax_key.transAxes)
    ax_key.text(0.0, 0.925, "green circle = start,  red star = goal",
                fontsize=8.5, va="top", color="#444444", transform=ax_key.transAxes)
    y_cursor = 0.85
    for colour, style, note, row in key_entries:
        ax_key.plot([0.0, 0.13], [y_cursor, y_cursor], color=colour, lw=3.2, ls=style,
                    transform=ax_key.transAxes, clip_on=False)
        ax_key.text(0.17, y_cursor, note, fontsize=9.5, va="center", weight="bold",
                    color=colour, transform=ax_key.transAxes)
        if row:
            ax_key.text(
                0.17, y_cursor - 0.055,
                f"{float(row['length_m']):.2f} m path\n"
                f"{float(row['longest_unobserved_run_s']):.1f} s longest unobserved stretch\n"
                f"{float(row['empirical_hit_rate']) * 100:.0f}% of real detections nearby were hits",
                fontsize=8.5, va="top", color="#333333", transform=ax_key.transAxes,
            )
        y_cursor -= 0.235
    ax_map.set_title("Zoomed to the routes", fontsize=11, weight="bold", pad=8)

    arms = ["availability_blind"] + [s.key for s in _order() if s.key != "constant"]
    labels, gaps, colours, hatches = [], [], [], []
    for key in arms:
        subset = [
            r for r in rows
            if r["source"] == key and r["longest_unobserved_run_s"] != ""
            and r.get("cameras", "four") == "four"
        ]
        if not subset:
            continue
        labels.append(_wrap(subset[0]["label"], 20))
        gaps.append(float(np.mean([float(r["longest_unobserved_run_s"]) for r in subset])))
        if key == "availability_blind":
            colours.append("#111111")
            hatches.append(None)
        else:
            source = C.SOURCE_BY_KEY[key]
            colours.append(source.color)
            hatches.append(SURVEY_HATCH if source.needs_surveyed_model else None)

    y = np.arange(len(labels))
    for i in range(len(labels)):
        ax_bar.barh(y[i], gaps[i], color=colours[i], edgecolor="#222222", hatch=hatches[i], linewidth=0.9)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(labels, fontsize=9)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel(
        "Longest continuous stretch with no camera likely to see the robot, seconds  (shorter is better)",
        fontsize=10,
    )
    ax_bar.grid(axis="x", alpha=0.25)
    ax_bar.set_axisbelow(True)
    blind_gap = gaps[0] if gaps else float("nan")
    ax_bar.axvline(blind_gap, color="#111111", ls=":", lw=1.4)
    for i, value in enumerate(gaps):
        ax_bar.text(value + 0.05, y[i], f"{value:.1f} s", va="center", fontsize=9)
    ax_bar.set_xlim(0, max(gaps) * 1.18 if gaps else 1.0)
    ax_bar.set_title(
        "A correct availability model cuts the blind stretch by three quarters;\n"
        "a weak one changes nothing at all  (mean over the four declared tasks)",
        fontsize=11.5,
        weight="bold",
        pad=10,
    )

    fig.suptitle(
        "Availability-aware routing trades a few centimetres of path for far less time unobserved",
        fontsize=13.5,
        weight="bold",
    )
    fig.text(
        0.5,
        0.015,
        "Left: fused probability that at least one of four cameras returns a usable detection, from the CAD raycast "
        f"reference — dark = unobserved, bright = well seen; grey = shelves and structure. Hatched ({SURVEY_HATCH}) = "
        "needs a surveyed 3-D model of the warehouse.\nTerminal uncertainty is deliberately not plotted: the belief "
        "re-converges once the robot leaves a blackspot, so it hides this gap entirely. EXPLORATORY offline route "
        "study on the frozen field — no closed-loop navigation claim.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.01, 0.17, 0.99, 0.90))
    out = FIGURES / "04_route_discrimination.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


#: Camera subsets in decreasing density, with labels a reader can understand
#: without knowing which letter is on which wall.
DENSITY_ORDER = (
    ("four", "4 cameras"),
    ("three", "3 cameras"),
    ("two_opposite", "2 cameras\nopposite walls"),
    ("two_same_side", "2 cameras\nsame wall"),
    ("one", "1 camera"),
)


def fig_density() -> None:
    """E3: how much availability modelling buys, as coverage gets sparser.

    This is the scope condition. With four cameras and noisy-OR fusion there is
    little for the planner to avoid, so a single four-camera number understates the
    method; a sparse network is also the more realistic deployment.
    """

    rows = read_csv(E3 / "e3_routes.csv")
    if not any(r.get("cameras") for r in rows):
        print("skipping density figure: e3_routes.csv has no 'cameras' column yet")
        return

    def mean_of(subset: str, source: str, column: str) -> float:
        vals = [
            float(r[column])
            for r in rows
            if r["cameras"] == subset and r["source"] == source and r[column] != ""
        ]
        return float(np.mean(vals)) if vals else float("nan")

    present = [(k, lbl) for k, lbl in DENSITY_ORDER if any(r["cameras"] == k for r in rows)]
    if not present:
        print("skipping density figure: no recognised camera subsets in e3_routes.csv")
        return

    labels = [lbl for _, lbl in present]
    x = np.arange(len(present))
    arms = (
        ("availability_blind", "ignores availability", "#111111"),
        ("cad_reference", "plans with surveyed CAD geometry", "#d94b4b"),
        ("mono_depth", "plans with the cameras' own depth", "#d89000"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.4))
    width = 0.26
    for i, (key, label, colour) in enumerate(arms):
        vals = [mean_of(k, key, "longest_unobserved_run_s") for k, _ in present]
        axes[0].bar(
            x + (i - 1) * width, vals, width, color=colour, edgecolor="#222222",
            linewidth=0.8, label=label,
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel(
        "Longest stretch with no camera likely to see the robot, seconds\n(shorter is better)",
        fontsize=10,
    )
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].set_axisbelow(True)
    axes[0].set_title("As coverage thins, the unobserved stretch grows", fontsize=12, weight="bold")

    # What the modelling actually saves, which is the quantity the paper claims.
    for key, label, colour in arms[1:]:
        saved = [
            mean_of(k, "availability_blind", "longest_unobserved_run_s")
            - mean_of(k, key, "longest_unobserved_run_s")
            for k, _ in present
        ]
        axes[1].plot(x, saved, marker="o", ms=7, lw=2.4, color=colour, label=label)
    axes[1].axhline(0.0, color="#444444", lw=1.2, ls="--")
    axes[1].text(
        0.02, 0.02, "below this line = modelling availability made the route worse",
        transform=axes[1].transAxes, fontsize=9, color="#444444",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel(
        "Seconds of unobserved driving avoided\nby planning with the field (higher = more benefit)",
        fontsize=10,
    )
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.25)
    axes[1].set_axisbelow(True)
    axes[1].set_title(
        "…but how much modelling helps depends on placement, not just count",
        fontsize=12,
        weight="bold",
    )

    # Name the non-monotonicity rather than letting a line imply a trend that is not
    # there: two cameras on the same wall give the SMALLEST benefit of any
    # configuration despite a worse blind baseline than two on opposite walls.
    same_side_idx = next((i for i, (k, _) in enumerate(present) if k == "two_same_side"), None)
    if same_side_idx is not None:
        saved_same = mean_of("two_same_side", "availability_blind", "longest_unobserved_run_s") - mean_of(
            "two_same_side", "cad_reference", "longest_unobserved_run_s"
        )
        axes[1].annotate(
            "same-wall views overlap,\nso there is little to choose between them",
            xy=(same_side_idx, saved_same),
            xytext=(same_side_idx - 0.15, saved_same + 0.75),
            ha="center",
            fontsize=8.5,
            color="#444444",
            arrowprops=dict(arrowstyle="->", color="#444444", lw=1.0),
        )

    fig.suptitle(
        "Availability modelling pays most where views are complementary, not simply where cameras are few",
        fontsize=13.5,
        weight="bold",
    )
    fig.text(
        0.5,
        0.015,
        "Mean over the four declared tasks, on the frozen fields, warehouse_full_4cam. The two-camera pairs separate "
        "placement from count: opposite walls versus the same wall.\nThis axis was added AFTER the four-camera result "
        "came out nearly null, and is reported as a scope condition found in the data, not as a planned comparison. "
        "EXPLORATORY offline route study — no navigation claim.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.91))
    out = FIGURES / "05_camera_density.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")



E5 = C.OUT_ROOT / "e5_offline_efe_solve"
#: Cost breakdown of the SELECTED plan in the stopped closed-loop campaign
#: (mc_blind_L / C2 / seed0, global_plan_meta.json). Hard-coded here only as a
#: figure annotation; the authoritative copy is in that run directory.
CLOSED_LOOP_COSTS = {"risk": 511682.74, "obstacle": 1017054.90, "ambiguity": 337.58}


def fig_objective_imbalance() -> None:
    """E5: the fields carry the signal; the runtime objective cannot act on it."""

    solve = read_csv(E5 / "e5_offline_efe_solve.csv")
    disc = read_csv(E5 / "route_field_discrimination.csv")

    fig, axes = plt.subplots(1, 2, figsize=(15.6, 6.6))

    # --- left: does the FIELD contain the signal? -------------------------------
    tasks = ["mc_blind_L", "route_tall_shadow_west", "mc_m2_w2e_traverse"]
    fields = ["C2_operational_gp", "C4_depth_plus_gp", "C3_mono_depth"]
    colours = {"C2_operational_gp": "#2a9d58", "C4_depth_plus_gp": "#d89000", "C3_mono_depth": "#7b53b5"}
    names = {"C2_operational_gp": "operational GP (needs survey)",
             "C4_depth_plus_gp": "monocular depth + GP (no survey)",
             "C3_mono_depth": "monocular depth alone (no survey)"}
    x = np.arange(len(tasks)); width = 0.26
    for i, f in enumerate(fields):
        vals = []
        for t in tasks:
            r = next((r for r in disc if r["task"] == t and r["field"] == f), None)
            vals.append(float(r["ratio"]) if r else np.nan)
        bars = axes[0].bar(x + (i - 1) * width, vals, width, color=colours[f],
                           edgecolor="#222222", linewidth=0.9,
                           hatch=SURVEY_HATCH if f == "C2_operational_gp" else None, label=names[f])
        for b, v in zip(bars, vals):
            axes[0].text(b.get_x() + b.get_width() / 2, v * 1.10, f"{v:.0f}x" if v >= 2 else "1x",
                         ha="center", fontsize=8.5)
    axes[0].axhline(1.0, color="#b0271f", ls="--", lw=1.2)
    axes[0].set_ylim(0.8, 900)
    axes[0].text(0.015, 0.93, "dashed line at 1x = the field sees no difference between the two routes",
                 transform=axes[0].transAxes, fontsize=8.5, color="#b0271f")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([t.replace("_", "\n") for t in tasks], fontsize=9)
    axes[0].set_ylabel("How much better the detour looks than the short route\n"
                       "(ratio of minimum availability; higher = clearer signal)", fontsize=10)
    axes[0].legend(fontsize=8.5, loc="upper right")
    axes[0].grid(axis="y", alpha=0.25); axes[0].set_axisbelow(True)
    axes[0].set_title("The field carries the signal — and a survey-free field carries it best",
                      fontsize=12, weight="bold")

    # --- right: can the OBJECTIVE act on it? ------------------------------------
    arms = ["C1_blind", "C2_operational_gp", "C3_mono_depth", "C4_depth_plus_gp"]
    arm_lbl = {"C1_blind": "blind", "C2_operational_gp": "operational\nGP",
               "C3_mono_depth": "monocular\ndepth", "C4_depth_plus_gp": "depth\n+ GP"}
    risk = [float(next(r for r in solve if r["task"] == "mc_blind_L" and r["arm"] == a)["risk"]) for a in arms]
    amb = [float(next(r for r in solve if r["task"] == "mc_blind_L" and r["arm"] == a)["ambiguity"]) for a in arms]
    xa = np.arange(len(arms))
    axes[1].bar(xa - 0.19, risk, 0.38, color="#37474f", edgecolor="#222222", label="goal-reaching risk")
    axes[1].bar(xa + 0.19, amb, 0.38, color="#00a6a6", edgecolor="#222222", label="availability (ambiguity)")
    for i, (r, a) in enumerate(zip(risk, amb)):
        axes[1].text(xa[i] - 0.19, r * 1.08, f"{r:,.0f}", ha="center", fontsize=8.5)
        axes[1].text(xa[i] + 0.19, a * 1.08, f"{a:.0f}", ha="center", fontsize=8.5)
    axes[1].set_yscale("log")
    axes[1].set_ylim(100, 40000)
    axes[1].set_xticks(xa); axes[1].set_xticklabels([arm_lbl[a] for a in arms], fontsize=9)
    axes[1].set_ylabel("Contribution to the planner's objective, log scale", fontsize=10)
    axes[1].legend(fontsize=9, loc="upper right")
    axes[1].grid(axis="y", alpha=0.25); axes[1].set_axisbelow(True)
    axes[1].set_title("…but availability is a rounding error in the objective,\nso every arm picks the same route",
                      fontsize=12, weight="bold")
    cl = CLOSED_LOOP_COSTS
    axes[1].text(0.02, 0.82,
                 f"In the closed-loop runs the gap was wider still:\n"
                 f"risk {cl['risk']:,.0f} + obstacle {cl['obstacle']:,.0f} vs ambiguity {cl['ambiguity']:.0f}\n"
                 f"— availability was 0.02 % of the objective.",
                 transform=axes[1].transAxes, fontsize=8.5,
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#eef4f4", edgecolor="#00a6a6"))

    fig.suptitle("Availability modelling fails at the planner, not at the estimator",
                 fontsize=13.5, weight="bold")
    fig.text(0.5, 0.015,
             "Left: minimum fused availability along the two routes seeded to the planner, per field, on the frozen "
             "four-camera grid.\nRight: runtime EFE objective terms for the selected plan, offline solve on mc_blind_L. "
             "All four arms selected the availability-blind route on every task.\n"
             "The visibility weighting is frozen method and was NOT adjusted — doing so would manufacture the effect. "
             "EXPLORATORY offline study.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.10, 1, 0.91))
    out = FIGURES / "06_objective_imbalance.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")



CL_ROOT = C.REPO / "logs/visibility_comparison/e4_availability_closed_loop_v1"
#: Closed-loop arms actually run. C3 is the explicit Bernoulli observation model,
#: NOT the monocular-depth field — the depth arms were offline only, and mislabelling
#: them here would claim a closed-loop result that does not exist.
CL_ARMS = {
    "C1": ("constant-$R$, availability-blind", "#d62728"),
    "C2": ("visibility-aware ($R/p$)", "#1f4fd8"),
    "C3": ("visibility-aware (explicit hit/miss)", "#00a6a6"),
}

FIELD_PANELS = [
    ("A0 constant baseline", None, "no spatial availability estimator", False),
    ("A1 operational GP", C.REPO / "logs/visibility_comparison/spawn_grid_20260727/fused_planner_four_camera.npz",
     "GP on a geometric day-zero prior", True),
    ("A2 monocular depth", C.OUT_ROOT / "mono_depth_planner_v1/fused_planner_four_camera.npz",
     "camera RGB + calibration + drivable map", False),
    ("A3 monocular depth + GP", C.OUT_ROOT / "depth_gp_planner_v1/fused_planner_four_camera.npz",
     "GP residual on the depth prior", False),
]


def _seeded_routes(task: str) -> dict:
    route_file = E3 / "e3_selected_routes.json"
    routes = json.loads(route_file.read_text())["routes"]["four"][task]
    return {
        name: np.asarray(points, dtype=float)
        for name, points in routes.items()
        if name in {"availability_blind", "cad_reference"}
    }


def _draw_r_plan_field(ax, sigma_px: np.ndarray, apparatus, title: str):
    """Draw the scalar isotropic R_plan map on the registered fixed px scale."""

    shown = np.where(apparatus.driveable, sigma_px, np.nan)
    image = ax.imshow(
        shown,
        origin="lower",
        extent=(apparatus.xs[0], apparatus.xs[-1], apparatus.ys[0], apparatus.ys[-1]),
        cmap="magma",
        vmin=C.R_VISIBLE_UV_PX,
        vmax=C.R_MISS_UV_PX,
        aspect="equal",
        zorder=1,
    )
    base.draw_geometry(ax, apparatus.prisms)
    base.draw_cameras(ax)
    for label in ax.texts:
        label.set_clip_on(True)
    ax.set_xlim(apparatus.xs[0], apparatus.xs[-1])
    ax.set_ylim(apparatus.ys[0], apparatus.ys[-1])
    ax.set_title(title, fontsize=10, weight="bold")
    ax.set_xlabel("x [m]", fontsize=9)
    ax.tick_params(labelsize=8)
    return image


def _load_field_panel_values(apparatus) -> list[np.ndarray]:
    values = []
    for _name, path, _inputs, _survey in FIELD_PANELS:
        if path is None:
            values.append(np.full_like(apparatus.driveable, 0.5, dtype=float))
        else:
            values.append(np.asarray(np.load(path)["P_conservative_plan_map"], dtype=float))
    return values


def fig_field_maps(task: str = "mc_blind_L") -> None:
    """Availability-estimator outputs only; no planner-model labels or R maps."""

    apparatus = C.build_apparatus()
    fields = _load_field_panel_values(apparatus)
    disc = read_csv(C.OUT_ROOT / "e5_offline_efe_solve/route_field_discrimination.csv")
    routes = _seeded_routes(task)
    key_map = {"A1 operational GP": "C2_operational_gp",
               "A2 monocular depth": "C3_mono_depth",
               "A3 monocular depth + GP": "C4_depth_plus_gp"}

    fig, axes = plt.subplots(1, 4, figsize=(19.0, 6.2), constrained_layout=False)
    fig.subplots_adjust(left=0.045, right=0.925, top=0.84, bottom=0.16, wspace=0.16)
    image = None
    for column, (ax, panel, field) in enumerate(zip(axes, FIELD_PANELS, fields)):
        name, _path, inputs, survey = panel
        image = base.draw_field(ax, field, apparatus.xs, apparatus.ys, apparatus.driveable,
                                apparatus.prisms, title="")
        for artist in list(ax.texts):
            artist.set_clip_on(True)
        ax.plot(routes["availability_blind"][:, 0], routes["availability_blind"][:, 1],
                color="#111111", lw=2.6, label="short route")
        ax.plot(routes["cad_reference"][:, 0], routes["cad_reference"][:, 1],
                color="#ffffff", lw=2.6, ls="--", label="detour")
        row = next((r for r in disc if r["task"] == task and r["field"] == key_map.get(name)), None)
        sub = inputs if row is None else (
            f"{inputs}\nshort {float(row['blind_min']):.3f} vs detour {float(row['detour_min']):.3f}"
            f"  ({float(row['ratio']):.0f}x)")
        ax.set_title(f"{name}{'  [needs survey]' if survey else ''}\n{sub}", fontsize=10, weight="bold")
        ax.set_xlabel("x [m]", fontsize=9)
        ax.set_ylabel("y [m]" if column == 0 else "", fontsize=9)
    axes[0].legend(fontsize=8.5, loc="lower left", framealpha=0.9)
    cax = fig.add_axes([0.94, 0.255, 0.012, 0.47])
    fig.colorbar(image, cax=cax, label=r"availability $p_{use}$")
    fig.suptitle("Availability estimators A0–A3 — outputs only", fontsize=13.5, weight="bold")
    fig.text(
        0.5, 0.035,
        f"Task {task}. Each map estimates the fused probability that at least one camera returns a usable detection. "
        "These A-labels identify field sources, not planner conditions.\n"
        "Fixed 0–1 colour scale; grey = shelves and structure. EXPLORATORY offline fields.",
        ha="center", fontsize=9,
    )
    out = FIGURES / "07_field_maps.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


def fig_folded_r_maps() -> None:
    """Diagnostic: pass each availability source through the historical R adapter."""

    apparatus = C.build_apparatus()
    fields = _load_field_panel_values(apparatus)
    fig, axes = plt.subplots(1, 4, figsize=(19.0, 5.8), constrained_layout=False)
    fig.subplots_adjust(left=0.045, right=0.925, top=0.82, bottom=0.18, wspace=0.16)
    image = None
    for column, (ax, panel, field) in enumerate(zip(axes, FIELD_PANELS, fields)):
        name = panel[0]
        if column == 0:
            sigma = np.full_like(field, C.R_VISIBLE_UV_PX)
            subtitle = r"actual availability-blind constant $R$"
        else:
            sigma = C.folded_planner_sigma_px(field)
            subtitle = r"diagnostic old fold $R_{plan}(p_{use})$"
        image = _draw_r_plan_field(ax, sigma, apparatus, f"{name}\n{subtitle}")
        ax.set_ylabel("y [m]" if column == 0 else "", fontsize=9)
    cax = fig.add_axes([0.94, 0.22, 0.012, 0.50])
    fig.colorbar(image, cax=cax, label=r"planner per-axis $1\sigma_R$ [px]")
    fig.suptitle("Old folded covariance maps — diagnostic conversion of A0–A3", fontsize=13.5, weight="bold")
    fig.text(
        0.5, 0.035,
        r"Precision blend with registered 2.5 px visible and 40 px miss endpoints. A1–A3 answer only: "
        r"‘what if this field were fed into the old adapter?’" "\n"
        r"They are not the explicit C3 planner model, not geometry-based $R_{cond}$, and not measured camera error. TESTING VIEW.",
        ha="center", fontsize=9,
    )
    out = FIGURES / "09_folded_R_maps_diagnostic.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


def fig_planner_observation_models() -> None:
    """Actual registered C1/C2/C3 planner semantics, separate from field sources."""

    apparatus = C.build_apparatus()
    gp = np.asarray(np.load(
        C.REPO / "logs/visibility_comparison/spawn_grid_20260727/fused_planner_four_camera.npz"
    )["P_conservative_plan_map"], dtype=float)
    unused = np.full_like(gp, 0.5)
    p_fields = (unused, gp, gp)
    column_titles = (
        "C1 planner\navailability-blind",
        "C2 planner\nfolded covariance",
        "C3 planner\nexplicit Bernoulli hit/miss",
    )
    r_fields = (
        np.full_like(gp, C.R_VISIBLE_UV_PX),
        C.folded_planner_sigma_px(gp),
        np.full_like(gp, C.R_VISIBLE_UV_PX),
    )
    r_titles = (
        r"constant $R$",
        r"$R_{plan}(p_{use})$",
        r"conditional $R_{cond}$ on a hit",
    )

    fig, axes = plt.subplots(2, 3, figsize=(16.2, 10.2), constrained_layout=False)
    fig.subplots_adjust(left=0.055, right=0.92, top=0.84, bottom=0.13, wspace=0.18, hspace=0.33)
    p_image = None
    r_image = None
    for column in range(3):
        p_ax = axes[0, column]
        p_image = base.draw_field(p_ax, p_fields[column], apparatus.xs, apparatus.ys,
                                  apparatus.driveable, apparatus.prisms, title="")
        for artist in list(p_ax.texts):
            artist.set_clip_on(True)
        p_ax.set_title(column_titles[column] + "\n" + (
            r"$p_{use}$ not consumed" if column == 0 else r"same operational-GP $p_{use}$ input"
        ), fontsize=10.5, weight="bold")
        p_ax.set_ylabel("availability input\ny [m]" if column == 0 else "", fontsize=9)
        if column == 0:
            p_ax.text(0.5, 0.08, "NOT USED", transform=p_ax.transAxes, ha="center", fontsize=12,
                      weight="bold", bbox=dict(facecolor="white", edgecolor="#555555", alpha=0.9))

        r_ax = axes[1, column]
        r_image = _draw_r_plan_field(r_ax, r_fields[column], apparatus, r_titles[column])
        r_ax.set_ylabel("measurement model\ny [m]" if column == 0 else "", fontsize=9)
        if column == 2:
            r_ax.text(
                0.5, 0.08,
                r"hit: update with $R_{cond}$" "\n" r"miss: no camera update",
                transform=r_ax.transAxes, ha="center", fontsize=9.5, weight="bold",
                bbox=dict(facecolor="white", edgecolor="#00a6a6", alpha=0.92),
            )
    p_cax = fig.add_axes([0.94, 0.53, 0.012, 0.27])
    fig.colorbar(p_image, cax=p_cax, label=r"availability $p_{use}$")
    r_cax = fig.add_axes([0.94, 0.18, 0.012, 0.27])
    fig.colorbar(r_image, cax=r_cax, label=r"per-axis $1\sigma_R$ [px]")
    fig.suptitle("Planner observation models C1–C3 — how availability is actually used", fontsize=14, weight="bold")
    fig.text(
        0.5, 0.025,
        r"C1 ignores $p_{use}$. C2 converts it into a spatial planner covariance." "\n"
        r"C3 uses it directly as a hit probability: $E[P^+]=p_{use}P_{hit}+(1-p_{use})P^-$." "\n"
        r"Registered testing configuration: isotropic $R_{cond}=(2.5\,px)^2I$. Geometry-conditioned $R_{cond}(x,y)$ "
        r"is a separate future test, not yet commissioned. TESTING VIEW; E4 stopped after 12/45 runs "
        r"because all conditions selected the same global route.",
        ha="center", fontsize=8.7,
    )
    out = FIGURES / "10_planner_observation_models.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


def fig_p_maps_in_planning() -> None:
    """Show p_use on the persisted route and its distinct C1/C2/C3 use."""

    apparatus = C.build_apparatus()
    gp = np.asarray(np.load(
        C.REPO / "logs/visibility_comparison/spawn_grid_20260727/fused_planner_four_camera.npz"
    )["P_conservative_plan_map"], dtype=float)
    plan_path = (
        C.REPO
        / "logs/visibility_comparison/e4_availability_closed_loop_v1/mc_blind_L/C3/seed0"
        / "experiment_20260818_100342/global_plan.csv"
    )
    plan_rows = read_csv(plan_path)
    plan = np.asarray([[float(row["x"]), float(row["y"])] for row in plan_rows], dtype=float)
    distance = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(plan, axis=0), axis=1))])
    p_along = np.asarray(C.sample_field_at(gp, apparatus.xs, apparatus.ys, plan), dtype=float)
    sigma_along = C.folded_planner_sigma_px(p_along)

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.0), constrained_layout=False)
    fig.subplots_adjust(left=0.065, right=0.93, top=0.84, bottom=0.13, wspace=0.32, hspace=0.36)
    fig.suptitle(r"How the planner uses the $p_{use}$ map along the route it actually solved", fontsize=15, weight="bold")

    columns = (
        ("C1 availability-blind", np.full_like(gp, 0.5), r"$p_{use}$ is not consumed"),
        ("C2 folded covariance", gp, r"$p_{use}\rightarrow R_{plan}(p_{use})$"),
        ("C3 explicit hit/miss", gp, r"$p_{use}\rightarrow$ branch probabilities"),
    )
    image = None
    for column, (name, field, use) in enumerate(columns):
        ax = axes[0, column]
        image = base.draw_field(ax, field, apparatus.xs, apparatus.ys, apparatus.driveable,
                                apparatus.prisms, title="")
        for artist in list(ax.texts):
            artist.set_clip_on(True)
        # Black underlay plus white centre keeps the exact, unshifted plan visible
        # on both low- and high-probability cells.
        ax.plot(plan[:, 0], plan[:, 1], color="#111111", lw=5.0, zorder=8)
        ax.plot(plan[:, 0], plan[:, 1], color="white", lw=2.2, zorder=9,
                label="persisted global plan")
        ax.plot(plan[0, 0], plan[0, 1], "o", color="#1A7F37", ms=8, zorder=10)
        ax.plot(plan[-1, 0], plan[-1, 1], "*", color="#B0271F", ms=12, zorder=10)
        ax.set_title(f"{name}\n{use}", fontsize=11, weight="bold")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]" if column == 0 else "")
        if column == 0:
            ax.text(0.5, 0.07, "MAP IGNORED", transform=ax.transAxes, ha="center",
                    fontsize=10.5, weight="bold",
                    bbox=dict(facecolor="white", edgecolor="#555555", alpha=0.92))
        if column == 2:
            ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    cax = fig.add_axes([0.945, 0.535, 0.012, 0.265])
    fig.colorbar(image, cax=cax, label=r"fused availability $p_{use}$")

    ax = axes[1, 0]
    ax.axis("off")
    ax.text(
        0.5, 0.58,
        r"No spatial probability enters the planner" "\n\n"
        r"$R=(2.5\,\mathrm{px})^2 I$ everywhere" "\n\n"
        "The same observation model is assumed\nat every route position.",
        ha="center", va="center", fontsize=13, linespacing=1.25,
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#F8FAFC", edgecolor="#98A2B3"),
    )
    ax.set_title("C1 calculation", fontsize=11, weight="bold")

    ax = axes[1, 1]
    p_line = ax.plot(distance, p_along, color="#1F4FD8", lw=2.4, label=r"map $p_{use}$")[0]
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("distance along persisted plan [m]")
    ax.set_ylabel(r"$p_{use}$", color="#1F4FD8")
    ax.tick_params(axis="y", colors="#1F4FD8")
    ax.grid(alpha=0.25)
    twin = ax.twinx()
    r_line = twin.plot(distance, sigma_along, color="#A23B72", lw=2.4,
                       label=r"folded $1\sigma_R$")[0]
    twin.set_ylabel(r"planner $1\sigma_R$ [px]", color="#A23B72", labelpad=4)
    twin.tick_params(axis="y", colors="#A23B72")
    ax.legend([p_line, r_line], [p_line.get_label(), r_line.get_label()], fontsize=8, loc="upper right")
    ax.set_title(r"C2: low $p_{use}$ becomes larger $R_{plan}$", fontsize=11, weight="bold")

    ax = axes[1, 2]
    ax.fill_between(distance, 0.0, p_along, color="#00A6A6", alpha=0.55,
                    label=r"hit branch: $p_{use}$")
    ax.fill_between(distance, p_along, 1.0, color="#D0D5DD", alpha=0.85,
                    label=r"miss branch: $1-p_{use}$")
    ax.plot(distance, p_along, color="#087F8C", lw=2.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("distance along persisted plan [m]")
    ax.set_ylabel("branch probability", labelpad=8)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(r"C3: $p_{use}$ weights hit versus no-update", fontsize=11, weight="bold")
    ax.text(
        0.03, 0.05,
        r"hit uses $R_{cond}=(2.5\,\mathrm{px})^2I$" "\n" r"miss uses no camera update",
        transform=ax.transAxes, fontsize=8.5,
        bbox=dict(facecolor="white", edgecolor="#00A6A6", alpha=0.9),
    )

    fig.text(
        0.5, 0.025,
        "The white line is an actual persisted global_plan.csv, not a Dijkstra route or executed trajectory. "
        "C2 and C3 receive the same operational-GP probability map and solved the same coordinates.\n"
        "Testing visualization from the incomplete mc_blind_L campaign; no navigation-performance or accuracy claim.",
        ha="center", fontsize=8.8, color="#475467",
    )
    out = FIGURES / "12_p_maps_in_planning.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


def fig_closed_loop_trajectories(task: str = "mc_blind_L") -> None:
    """Driven paths from the real campaign, per arm, over the field the aware arms used."""

    import glob as _glob
    sys.path.insert(0, str(C.REPO / "scripts/geometry_visibility"))
    import campaign_metrics as CM

    apparatus = C.build_apparatus()
    field = np.asarray(np.load(
        C.REPO / "logs/visibility_comparison/spawn_grid_20260727/fused_planner_four_camera.npz"
    )["P_conservative_plan_map"], dtype=float)

    runs = {}
    for arm in CL_ARMS:
        for d in sorted(_glob.glob(str(CL_ROOT / task / arm / "seed*/experiment_*"))):
            summary = Path(d) / "run_summary.json"
            if not summary.is_file():
                continue
            s = json.loads(summary.read_text())
            r = CM.load_run(str(Path(d) / "experiment.csv"))
            xy = np.column_stack([r["truth_x"], r["truth_y"]])
            xy = xy[np.isfinite(xy).all(axis=1)]
            runs.setdefault(arm, []).append((xy, s))
    if not runs:
        print("skipping trajectory figure: no completed runs"); return

    fig, ax = plt.subplots(figsize=(11.0, 8.8))
    base.draw_field(ax, field, apparatus.xs, apparatus.ys, apparatus.driveable,
                    apparatus.prisms, title="")
    for artist in list(ax.texts):
        artist.set_clip_on(True)

    caption = []
    for arm, (label, colour) in CL_ARMS.items():
        entries = runs.get(arm, [])
        if not entries:
            continue
        goals = sum(1 for _, s in entries if s["completion_reason"] == "goal_reached")
        colls = sum(1 for _, s in entries if s.get("collision_any"))
        for i, (xy, s) in enumerate(entries):
            ax.plot(xy[:, 0], xy[:, 1], color=colour, lw=2.0, alpha=0.75,
                    label=label if i == 0 else None, zorder=6)
            if s.get("collision_any"):
                ax.plot(xy[-1, 0], xy[-1, 1], "x", color=colour, ms=13, mew=3, zorder=9)
        caption.append(f"{arm} {goals}/{len(entries)} goal, {colls} coll")

    first = next(iter(runs.values()))[0][0]
    ax.plot(first[0, 0], first[0, 1], "o", color="#1a7f37", ms=12, zorder=10, label="start")
    ax.legend(fontsize=8.5, loc="lower left", framealpha=0.92)
    ax.set_title(f"{task}   —   " + "  |  ".join(caption), fontsize=11, weight="bold")

    fig.suptitle("Every arm drove the same route: the availability model never changed the path",
                 fontsize=13, weight="bold")
    fig.text(0.5, 0.015,
             "Ground-truth paths from the real four-camera Gazebo campaign, one line per seed,\n"
             "over the fused availability field the aware arms consumed. x = collision.\n"
             "Campaign stopped after 12 of 45 runs, once every arm was seen selecting the\n"
             "availability-blind route (see figure 06). EXPLORATORY — no navigation claim.",
             ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    out = FIGURES / "08_closed_loop_trajectories.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None, help="subset: calib curves e2 e3 density")
    args = parser.parse_args()

    jobs = {
        "calib": fig_calibration,
        "curves": fig_calibration_curves,
        "e2": fig_availability_vs_accuracy,
        "e3": fig_route_discrimination,
        "density": fig_density,
        "e5": fig_objective_imbalance,
        "maps": fig_field_maps,
        "rmaps": fig_folded_r_maps,
        "models": fig_planner_observation_models,
        "pplanning": fig_p_maps_in_planning,
        "traj": fig_closed_loop_trajectories,
    }
    for name, fn in jobs.items():
        if args.only and name not in args.only:
            continue
        fn()

    provenance = {
        "study": C.STUDY_NAME,
        "figures_from": {
            "01_availability_calibration.png": ["e1_summary.csv", "e1_paired.csv"],
            "02_calibration_curves.png": ["e1_predictions.csv"],
            "03_availability_vs_accuracy.png": ["e2_correlations.csv", "e2_camera_disagreement.csv"],
            "04_route_discrimination.png": ["e3_routes.csv", "e3_selected_routes.json"],
            "05_camera_density.png": ["e3_routes.csv"],
            "06_objective_imbalance.png": ["e5_offline_efe_solve.csv", "route_field_discrimination.csv"],
            "07_field_maps.png": [
                "fused availability artifacts for estimator sources A0/A1/A2/A3",
                "e5_offline_efe_solve/route_field_discrimination.csv",
            ],
            "08_closed_loop_trajectories.png": [
                "e4_availability_closed_loop_v1/mc_blind_L frozen run inventory",
                "canonical experiment.csv planner_belief/gt columns via campaign_metrics.py",
                "operational GP fused availability artifact used by E4 C2/C3",
            ],
            "09_folded_R_maps_diagnostic.png": [
                "fused availability artifacts for estimator sources A0/A1/A2/A3",
                "e4_closed_loop/campaign.yaml observation endpoints",
            ],
            "10_planner_observation_models.png": [
                "operational GP fused availability artifact used by E4 C2/C3",
                "e4_closed_loop/campaign.yaml observation endpoints",
            ],
            "11_offline_map_vs_solved_routes.png": [
                "e6_offline_map_plan_audit/summary.json",
                "e5_offline_efe_solve/e5_offline_efe_solve.csv",
                "frozen persisted global_plan.csv run inventory",
            ],
            "12_p_maps_in_planning.png": [
                "operational GP fused availability artifact used by E4 C2/C3",
                "mc_blind_L/C3/seed0 persisted global_plan.csv",
                "e4_closed_loop/campaign.yaml observation endpoints",
            ],
        },
        "folded_R_map_contract": {
            "metric_object": "planner_facing_image_space_observation_covariance",
            "mapping": "isotropic precision blend of visible and miss endpoints",
            "r_visible_uv_sigma_px": C.R_VISIBLE_UV_PX,
            "r_miss_uv_sigma_px": C.R_MISS_UV_PX,
            "reference": "none; analytic planner input",
            "evaluation_only_inputs": [],
            "restriction": "not camera-measurement error and not the future geometry-based conditional R_cond",
        },
        "label_contract": {
            "A0_A3": "availability estimator/source labels used only in field figures",
            "C1_C3": "registered planner observation-model conditions used only in E4 figures",
            "legacy_directories": "C1_blind/C2_operational_gp/C3_mono_depth/C4_depth_plus_gp retained for path compatibility",
        },
        "status": "EXPLORATORY — offline prediction and route evidence; no navigation claim",
    }
    C.write_json(FIGURES / "provenance.json", provenance)
    print(f"wrote {FIGURES}/provenance.json")


if __name__ == "__main__":
    main()
