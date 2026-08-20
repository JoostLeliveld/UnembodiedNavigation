#!/usr/bin/env python3
"""Figures for the reconfiguration holdout.

Each figure is meant to be read by somebody who has not read the code, so the title
states what was found rather than which variable was plotted, and every axis says what
the number means and which direction is better.

    python3 experiments/reconfiguration_holdout/make_figures.py            # all
    python3 experiments/reconfiguration_holdout/make_figures.py --only 1 3  # some

Writes PDF and PNG into logs/studies/reconfiguration_holdout/figures/.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "experiments/dynamic_world_oracle"))

import common as C  # noqa: E402
import oracle as ora  # noqa: E402

FIGDIR = C.OUT_ROOT / "figures"
RESULTS = C.OUT_ROOT / "e1_reconfiguration_holdout"
LAYOUT = C.OUT_ROOT / "layout/layout_selected.json"

#: One colour per arm, stable across every figure so a reader can follow an arm.
ARM_COLOUR = {
    "constant": "#8a8a8a",
    "distance": "#5875a4",
    "fov_range": "#00a6a6",
    "cad_l0": "#d94b4b",
    "cad_env": "#f2a0a0",
    "mono_depth": "#d89000",
    "gp": "#7b53b5",
    "hybrid": "#2a9d58",
}
ENV_LABEL = {
    "L0": "nominal\nwarehouse",
    "L1": "racks\nrestocked",
    "L0_lit": "nominal layout,\nlighting changed",
    "L1_lit": "restocked and\nlighting changed",
}
PLAIN = {
    "constant": "how often it fires on average",
    "distance": "range to camera only",
    "fov_range": "framing and range, no obstacles",
    "cad_l0": "surveyed model of the old warehouse",
    "cad_env": "surveyed model, re-surveyed",
    "mono_depth": "depth from the camera's own image",
    "gp": "learned from past detections",
    "hybrid": "image depth + learned correction",
}


def save(fig, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {name}")


def _read(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------
# Figure 1 — the environments
# --------------------------------------------------------------------------

def fig1_environments() -> None:
    layout = json.loads(LAYOUT.read_text(encoding="utf-8"))
    restocked = {s["name"] for s in layout["restocked_segments"]}
    sys.path.insert(0, str(HERE))
    import choose_layout as CLay

    segs = CLay.rack_segments(C.WORLDS / "warehouse_full_4cam.world.sdf")
    lanes = CLay.lanes()
    grid = C.floor_grid()
    xs, ys = C.working_grid()

    scene0 = ora.OracleScene.from_world(C.WORLDS / "warehouse_full_4cam.world.sdf",
                                        list(C.CAMERAS))
    scene1 = ora.OracleScene.from_world(C.WORLDS / "warehouse_full_4cam_recfg.world.sdf",
                                        list(C.CAMERAS))
    cover = {}
    for tag, scene in (("L0", scene0), ("L1", scene1)):
        g = ora.visibility_grids(scene.cameras, grid, scene.static_prisms, (),
                                 target_height_m=C.TARGET_HEIGHT_M)
        cover[tag] = np.stack([gg == ora.VISIBLE for gg in g.values()]).sum(axis=0)

    drive = CLay.driveable_mask(grid, lanes)
    extent = [xs[0], xs[-1], ys[0], ys[-1]]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8))
    fig.subplots_adjust(bottom=0.28, wspace=0.22)
    for ax, tag, title in zip(
        axes[:2], ("L0", "L1"),
        ("Nominal warehouse", "After restocking 12 of 27 rack rows"),
    ):
        m = np.where(drive, cover[tag], np.nan)
        im = ax.imshow(m, origin="lower", extent=extent, cmap="viridis",
                       vmin=0, vmax=4, interpolation="nearest")
        for s in segs:
            hot = s["name"] in restocked
            ax.add_patch(Rectangle(
                (s["xmin"], s["ymin"]), s["xmax"] - s["xmin"], s["ymax"] - s["ymin"],
                facecolor=("#c94f0f" if hot else "#bbbbbb"),
                edgecolor="#333333", linewidth=0.4, alpha=0.95, zorder=3))
        for cx, cy in ((-6, -10), (-6, 10), (6, -10), (6, 10)):
            ax.plot(cx, cy, marker="v", ms=9, color="#111111", zorder=5, clip_on=False)
        n_dark = int(np.sum(drive & (cover[tag] == 0)))
        ax.set_title(f"{title}\n{n_dark} driveable cells seen by no camera", fontsize=10)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)" if tag == "L0" else "")
        ax.set_xlim(xs[0], xs[-1])
        ax.set_ylim(ys[0] - 1.2, ys[-1] + 1.2)
    # Horizontal colourbar under the two maps it belongs to: a vertical one between
    # panels collides with the third panel's axis, and the label is a sentence.
    cax = fig.add_axes([0.135, 0.10, 0.40, 0.035])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 1, 2, 3, 4])
    cb.set_label("how many of the four cameras can see a robot standing here "
                 "(0 = nobody can)", fontsize=8.5)

    ax = axes[2]
    lost = drive & (cover["L0"] > 0) & (cover["L1"] == 0)
    thinned = drive & (cover["L1"] < cover["L0"]) & (cover["L1"] > 0)
    ax.imshow(np.where(drive, 0.0, np.nan), origin="lower", extent=extent,
              cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    ax.imshow(np.where(thinned, 1.0, np.nan), origin="lower", extent=extent,
              cmap="YlOrBr", vmin=0, vmax=1.4, interpolation="nearest")
    ax.imshow(np.where(lost, 1.0, np.nan), origin="lower", extent=extent,
              cmap="Reds", vmin=0, vmax=1.2, interpolation="nearest")
    for s in segs:
        if s["name"] in restocked:
            ax.add_patch(Rectangle(
                (s["xmin"], s["ymin"]), s["xmax"] - s["xmin"], s["ymax"] - s["ymin"],
                facecolor="#c94f0f", edgecolor="#333333", linewidth=0.4, zorder=3))
    ax.set_title(f"What the restock cost the network\n"
                 f"{int(lost.sum())} cells went dark, {int(thinned.sum())} lost a camera",
                 fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(ys[0] - 1.2, ys[-1] + 1.2)
    handles = [Rectangle((0, 0), 1, 1, facecolor="#c94f0f", edgecolor="#333"),
               Rectangle((0, 0), 1, 1, facecolor="#d94b4b"),
               Rectangle((0, 0), 1, 1, facecolor="#e8b96a")]
    ax.legend(handles, ["restocked rack row", "no camera left", "fewer cameras"],
              fontsize=7.5, loc="lower left", framealpha=0.9)

    fig.suptitle("Restocking the racks removes sight-lines without touching a single aisle: "
                 "the driveable network is identical in both layouts",
                 fontsize=11, y=1.02)
    save(fig, "fig1_environments")


# --------------------------------------------------------------------------
# Figure 2 — what is recomputed and what is frozen
# --------------------------------------------------------------------------

def fig2_pipeline() -> None:
    """The schematic the whole paper turns on: which boxes see the new warehouse."""
    RECOMP, FROZEN, INPUT, OUT = "#2a9d58", "#b06a2c", "#dfe6ec", "#33475b"
    fig, ax = plt.subplots(figsize=(3.45, 1.62))
    ax.set_xlim(-0.05, 10.05); ax.set_ylim(-0.75, 4.1); ax.axis("off")

    def box(x, y, w, h, text, fc, tc="#111111", fs=3.3):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor="#333333",
                               linewidth=0.45, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, zorder=3, linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=1,
                    arrowprops=dict(arrowstyle="-|>", color="#444444", lw=0.55,
                                    shrinkA=1, shrinkB=1))

    box(0.02, 2.55, 1.62, 0.8, "the camera's\nown RGB frame", INPUT)
    box(0.02, 1.45, 1.62, 0.8, "calibration +\n2-D lane map", INPUT)
    box(0.02, 0.35, 1.62, 0.8, "past detector\noutcomes", INPUT)

    box(1.86, 2.55, 2.02, 0.8, "monocular depth\n(recomputed)", RECOMP, "#ffffff")
    box(1.86, 1.45, 2.02, 0.8, "floor anchor\n(commissioned once)", FROZEN, "#ffffff")
    box(4.10, 2.55, 1.90, 0.8, "sight-line\nray-cast", RECOMP, "#ffffff")
    box(4.10, 0.35, 1.90, 0.8, "GP residual\n(frozen)", FROZEN, "#ffffff")

    box(6.22, 1.45, 1.86, 0.8, "calibration link\n(frozen)", FROZEN, "#ffffff")
    box(8.30, 1.45, 1.68, 0.8, "chance a\ndetection arrives", OUT, "#ffffff")

    arrow(1.64, 2.95, 1.86, 2.95)
    arrow(1.64, 1.85, 1.86, 1.85)
    arrow(1.64, 0.75, 4.10, 0.75)
    arrow(2.87, 2.25, 2.87, 2.55)   # anchor -> depth (scale it)
    arrow(3.88, 2.95, 4.10, 2.95)
    arrow(6.00, 2.95, 6.75, 2.25)
    arrow(6.00, 0.75, 6.75, 1.45)
    arrow(8.08, 1.85, 8.30, 1.85)

    handles = [Rectangle((0, 0), 1, 1, facecolor=RECOMP),
               Rectangle((0, 0), 1, 1, facecolor=FROZEN),
               Rectangle((0, 0), 1, 1, facecolor=INPUT, edgecolor="#333")]
    ax.legend(handles,
              ["recomputed from the warehouse as it is now",
               "fitted in the nominal warehouse and never updated",
               "deployment inputs"],
              fontsize=3.9, loc="upper center", bbox_to_anchor=(0.5, 0.10),
              ncol=1, frameon=False, handlelength=1.1, handleheight=0.9,
              labelspacing=0.28)
    ax.set_title("Only two boxes see the reconfigured warehouse. Everything else was\n"
                 "fitted before the racks were restocked and is applied unchanged.",
                 fontsize=5.0)
    save(fig, "fig2_pipeline")


# --------------------------------------------------------------------------
# Figure 3 — the fields, before and after
# --------------------------------------------------------------------------

def fig3_fields() -> None:
    """What each estimator thinks changed, against what actually changed.

    Plotting the fields themselves does not work: the geometric ones are nearly binary,
    so six near-identical black-and-cream maps hide the very thing the figure is for.
    Plotting the DIFFERENCE between each estimator's reconfigured and nominal field
    shows it directly -- a frozen arm is exactly flat by construction, and a recomputed
    arm is not.
    """
    xs, ys = C.working_grid()
    extent = [xs[0], xs[-1], ys[0], ys[-1]]
    work = C.OUT_ROOT / "work/fields"
    cam = "external_camera"

    try:
        mono0 = C.mono_depth_field("L0")[cam]
        mono1 = C.mono_depth_field("L1")[cam]
    except RuntimeError as exc:
        print(f"[fig3] skipped: {exc}")
        return

    cad0 = C.cad_field("warehouse_full_4cam")[cam]
    cad1 = C.cad_field("warehouse_full_4cam_recfg")[cam]
    panels = [("what actually changed\n(fresh survey minus old survey)", cad1 - cad0,
               "#333333"),
              ("depth from the camera's own image\nRECOMPUTED", mono1 - mono0,
               ARM_COLOUR["mono_depth"])]
    hy0, hy1 = work / "hybrid_L0.npz", work / "hybrid_L1.npz"
    if hy0.is_file() and hy1.is_file():
        panels.append(("image depth + learned correction\nprior recomputed, residual frozen",
                       np.load(hy1)[f"{cam}__field"] - np.load(hy0)[f"{cam}__field"],
                       ARM_COLOUR["hybrid"]))
    panels.append(("learned from past detections\nFROZEN -- cannot change",
                   np.zeros_like(mono0), ARM_COLOUR["gp"]))
    panels.append(("surveyed model of the old warehouse\nFROZEN -- cannot change",
                   np.zeros_like(mono0), ARM_COLOUR["cad_l0"]))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(2.75 * n, 3.15))
    fig.subplots_adjust(bottom=0.24, top=0.78, wspace=0.10)
    for ax, (title, field, colour) in zip(np.atleast_1d(axes), panels):
        im = ax.imshow(field, origin="lower", extent=extent, cmap="RdBu",
                       vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=8, color=colour)
        ax.set_xticks([]); ax.set_yticks([])
        moved = float(np.mean(np.abs(field) > 0.25))
        ax.set_xlabel(f"{100 * moved:.1f}% of cells moved", fontsize=8)
    cax = fig.add_axes([0.30, 0.055, 0.40, 0.045])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-1, 0, 1])
    cb.ax.set_xticklabels(["says camera A lost this ground", "no change",
                           "says it gained ground"], fontsize=8)
    fig.suptitle("Only the estimators that re-read the camera's image register the "
                 "restock at all. The two frozen fields are flat by construction.",
                 fontsize=10.5, y=1.02)
    save(fig, "fig3_fields")


# --------------------------------------------------------------------------
# Figure 4 — degradation
# --------------------------------------------------------------------------

def fig4_degradation() -> None:
    """Skill lost, and the ranking inversion that goes with it.

    Plots SKILL rather than raw Brier.  The restock lowers the detection base rate from
    0.313 to 0.270, and a rarer positive class makes Brier smaller for free -- the
    constant-prevalence arm alone "improves" by 0.044 while knowing nothing.  Dividing
    by each unit's own climatology removes that, so a bar here is a change in what the
    estimator knew.
    """
    rows = _read(RESULTS / "e1_degradation.csv")
    summary = _read(RESULTS / "e1_summary.csv")
    if not rows or not summary:
        print("[fig4] skipped: no results yet")
        return
    envs = [e for e in ("L1", "L0_lit", "L1_lit")
            if any(r["environment"] == e for r in rows)]
    arms = list(dict.fromkeys(r["arm"] for r in rows))

    fig, axes = plt.subplots(1, 1 + len(envs),
                             figsize=(4.7 * (1 + len(envs)), 4.4))
    axes = np.atleast_1d(axes)

    for ax, env in zip(axes[:len(envs)], envs):
        sub = {r["arm"]: r for r in rows if r["environment"] == env}
        keys = [a for a in arms if a in sub]
        vals = np.array([float(sub[a]["skill_lost"]) for a in keys])
        los = np.array([float(sub[a]["skill_ci95_low"]) for a in keys])
        his = np.array([float(sub[a]["skill_ci95_high"]) for a in keys])
        ypos = np.arange(len(keys))
        ax.barh(ypos, vals, color=[ARM_COLOUR[a] for a in keys], height=0.62)
        ax.errorbar(vals, ypos, xerr=[vals - los, his - vals], fmt="none",
                    ecolor="#222222", elinewidth=1.0, capsize=2.5)
        ax.axvline(0.0, color="#000000", lw=0.9)
        ax.set_yticks(ypos)
        ax.set_yticklabels([PLAIN[a] for a in keys], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("skill lost after the change\n(right = the estimator got worse)",
                      fontsize=8.5)
        ax.set_title(ENV_LABEL[env].replace("\n", " "), fontsize=10)

    ax = axes[-1]
    look = {(r["arm"], r["environment"]): r for r in summary}
    env = envs[0]
    keys = [a for a in arms if (a, "L0") in look and (a, env) in look and a != "constant"]
    ends = []
    for a in keys:
        y0 = float(look[(a, "L0")]["brier_skill_mean"])
        y1 = float(look[(a, env)]["brier_skill_mean"])
        ax.plot([0, 1], [y0, y1], marker="o", ms=6, lw=2.0, color=ARM_COLOUR[a])
        ends.append((y1, a))
    # The whole point of this panel is that the lines cross, so their end labels land
    # on top of each other.  Push them apart vertically, keeping the order, and draw a
    # leader line back to the point each one belongs to.
    ends.sort()
    span = max(e[0] for e in ends) - min(e[0] for e in ends)
    gap = max(0.035, span / 14.0)
    placed = []
    for y1, a in ends:
        y = y1 if not placed else max(y1, placed[-1][0] + gap)
        placed.append((y, y1, a))
    for y, y1, a in placed:
        ax.annotate(PLAIN[a], xy=(1.05, y), fontsize=7.8, va="center",
                    color=ARM_COLOUR[a])
        if abs(y - y1) > 1e-6:
            ax.plot([1.005, 1.045], [y1, y], lw=0.6, color=ARM_COLOUR[a], alpha=0.7)
    ax.set_xlim(-0.06, 1.9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["nominal\nwarehouse", "racks\nrestocked"], fontsize=8.5)
    ax.set_ylabel("skill at predicting whether a detection arrives\n"
                  "(1 = perfect, 0 = no better than the base rate)", fontsize=8.5)
    ax.set_title("The ranking inverts", fontsize=10)
    ax.grid(axis="y", lw=0.4, alpha=0.4)

    fig.suptitle("The frozen learned field is the only estimator that loses real skill; "
                 "in the changed warehouse the survey-free adaptive field is the best one.",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    save(fig, "fig4_degradation")


# --------------------------------------------------------------------------
# Figure 5 — camera density
# --------------------------------------------------------------------------

def fig5_density() -> None:
    rows = _read(RESULTS / "e1_density.csv")
    if not rows:
        print("[fig5] skipped: no e1_density.csv yet")
        return
    order = ["1", "2_same_wall", "2_opposite", "3", "4"]
    labels = {"1": "1 camera", "2_same_wall": "2, same wall",
              "2_opposite": "2, opposite walls", "3": "3 cameras", "4": "4 cameras"}
    envs = list(dict.fromkeys(r["environment"] for r in rows))
    fig, axes = plt.subplots(1, len(envs), figsize=(4.4 * len(envs), 4.0), sharey=True)
    for ax, env in zip(np.atleast_1d(axes), envs):
        for arm in dict.fromkeys(r["arm"] for r in rows):
            xs_, ys_ = [], []
            for i, subset in enumerate(order):
                m = [r for r in rows if r["environment"] == env
                     and r["arm"] == arm and r["subset"] == subset]
                if m:
                    xs_.append(i)
                    ys_.append(float(m[0]["brier"]))
            if xs_:
                ax.plot(xs_, ys_, marker="o", ms=4, lw=1.5,
                        color=ARM_COLOUR.get(arm, "#444"), label=PLAIN.get(arm, arm))
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([labels[o] for o in order], fontsize=7.5, rotation=20)
        ax.set_xlabel("how many cameras the network has", fontsize=8.5)
        ax.set_title(ENV_LABEL[env].replace("\n", " "), fontsize=10)
    np.atleast_1d(axes)[0].set_ylabel("prediction error of 'will any camera see me'\n"
                                      "(Brier score, lower is better)", fontsize=8.5)
    # Legend below the panels: inside the right panel it sits on top of exactly the
    # crossover the figure exists to show.
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="upper center",
               bbox_to_anchor=(0.5, 0.045), ncol=3, frameon=False)
    fig.subplots_adjust(bottom=0.34)
    fig.suptitle("The inversion is not an artefact of one network size: the learned field "
                 "leads everywhere before the restock\nand the survey-free adaptive field "
                 "matches or beats it everywhere after it",
                 fontsize=10, y=1.03)
    save(fig, "fig5_density")


FIGURES = {1: fig1_environments, 2: fig2_pipeline, 3: fig3_fields,
           4: fig4_degradation, 5: fig5_density}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", type=int, default=None)
    args = ap.parse_args(argv)
    for key in sorted(FIGURES):
        if args.only and key not in args.only:
            continue
        FIGURES[key]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
