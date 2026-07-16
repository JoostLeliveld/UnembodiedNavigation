#!/usr/bin/env python3
"""Render evidence-aware protocol figures for the four-camera walkthrough.

These graphics intentionally describe the data and evaluation contract.  They
are not substitutions for the measured plots that D0–D6 will eventually
produce.  Current world/layout figures are linked from the main showcase.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/four_camera_walkthrough_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Ellipse


HERE = Path(__file__).resolve().parent

INK = "#17212f"
MUTED = "#5d6878"
BLUE = "#2f80ed"
GREEN = "#219a5b"
PURPLE = "#8d53c7"
ORANGE = "#ed8a25"
RED = "#c81919"
PALE_BLUE = "#e9f3ff"
PALE_GREEN = "#e9f7ee"
PALE_PURPLE = "#f2ecfb"
PALE_ORANGE = "#fff3e6"
PALE_GRAY = "#f7f9fc"


def canvas(title: str, subtitle: str):
    figure, axis = plt.subplots(figsize=(13.2, 7.15), dpi=170)
    figure.patch.set_facecolor("white")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.text(0.035, 0.945, title, fontsize=20, weight="bold", color=INK, va="top")
    axis.text(0.035, 0.895, subtitle, fontsize=10.8, color=MUTED, va="top")
    axis.plot([0.035, 0.12], [0.865, 0.865], color=RED, linewidth=3.1, solid_capstyle="butt")
    return figure, axis


def box(axis, x: float, y: float, width: float, height: float, title: str, body: str, color: str, pale: str, *, title_size: float = 11.6):
    patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.012,rounding_size=0.018", linewidth=1.1, edgecolor=color, facecolor=pale)
    axis.add_patch(patch)
    axis.text(x + 0.022, y + height - 0.052, title, fontsize=title_size, weight="bold", color=color, va="top")
    if body:
        axis.text(x + 0.022, y + height - 0.105, body, fontsize=8.9, color=INK, va="top", wrap=True, linespacing=1.35)


def arrow(axis, x1: float, y1: float, x2: float, y2: float, color: str = MUTED):
    axis.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, linewidth=1.4, color=color))


def banner(axis, text: str):
    patch = FancyBboxPatch((0.18, 0.06), 0.64, 0.045, boxstyle="round,pad=0.008,rounding_size=0.012", linewidth=0.8, edgecolor="#f4d2ab", facecolor=PALE_ORANGE)
    axis.add_patch(patch)
    axis.text(0.5, 0.083, text, fontsize=8.9, weight="bold", color=ORANGE, ha="center", va="center")


def save(figure, relative: str) -> None:
    output = HERE / relative
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"wrote {output}")


def collection_record_protocol() -> None:
    figure, axis = canvas(
        "Commissioning collects evidence with its spatial uncertainty attached",
        "The operational log is rich enough to learn and evaluate without leaking ground truth into a live decision.",
    )
    box(axis, 0.05, 0.39, 0.22, 0.35, "1  Execute validated routes", "A–D single-source regions\nAdjacent-camera corridors\nBoth directions, speeds, offsets\nOne gap crossing", BLUE, PALE_BLUE)
    box(axis, 0.38, 0.39, 0.25, 0.35, "2  Keep one operational record", "camera ID + timestamp\npixel pose + detector quality\nframe age\nestimated pose ŝₜ + covariance Σₛ,ₜ\ndetection OR miss", PURPLE, PALE_PURPLE)
    box(axis, 0.74, 0.39, 0.21, 0.35, "3  Hold evaluation apart", "Ground truth scores later.\nOperational records drive\nthe GP and manager.\nHeld-out routes test\ngeneralisation.", GREEN, PALE_GREEN)
    arrow(axis, 0.28, 0.565, 0.365, 0.565)
    arrow(axis, 0.64, 0.565, 0.725, 0.565)
    axis.text(0.05, 0.27, "Executed-route figure to generate next", fontsize=12.5, weight="bold", color=INK)
    axis.plot([0.08, 0.23, 0.34, 0.47, 0.61], [0.17, 0.21, 0.14, 0.22, 0.16], color=BLUE, linewidth=2.2)
    for x, y, a, b in [(0.12, 0.19, 0.04, 0.018), (0.28, 0.17, 0.06, 0.025), (0.46, 0.20, 0.045, 0.018), (0.58, 0.17, 0.07, 0.022)]:
        axis.add_patch(Ellipse((x, y), a, b, edgecolor=PURPLE, facecolor="none", linewidth=1.3))
    axis.text(0.66, 0.19, "Executed paths + repeated passes\n+ pose-covariance ellipses\n+ per-camera detections/misses", fontsize=9.4, color=MUTED, va="center")
    banner(axis, "PROTOCOL / DATA PENDING  ·  Do not replace this with an invented route or fabricated observation coverage.")
    save(figure, "03_uncertainty_aware_collection/figures/collection_record_protocol.png")


def per_camera_gp_learning_protocol() -> None:
    figure, axis = canvas(
        "Each camera earns its own posterior reliability GP",
        "Separate source histories prevent a blind spot in one view from becoming a false conclusion about another.",
    )
    cameras = [("A", BLUE, PALE_BLUE), ("B", GREEN, PALE_GREEN), ("C", PURPLE, PALE_PURPLE), ("D", ORANGE, PALE_ORANGE)]
    for index, (name, color, pale) in enumerate(cameras):
        x = 0.055 + index * 0.235
        box(axis, x, 0.47, 0.20, 0.28, f"Camera {name}", "calibration prior\n+ its own D0/D1 records\n→ posterior mean μᶜ(s)\n  and uncertainty σᶜ(s)", color, pale, title_size=12.2)
        axis.text(x + 0.10, 0.38, f"held-out score for {name}", fontsize=8.1, color=MUTED, ha="center")
        axis.plot([x + 0.04, x + 0.16], [0.32, 0.32], color=color, linewidth=4.2, solid_capstyle="round")
        for dot_x, dot_y in [(x + 0.055, 0.32), (x + 0.10, 0.34), (x + 0.145, 0.30)]:
            axis.scatter(dot_x, dot_y, s=26, color=color, edgecolors=INK, linewidths=0.45)
    axis.text(0.055, 0.23, "Final figure set per camera", fontsize=12.5, weight="bold", color=INK)
    labels = [("Observed records", BLUE), ("Posterior mean", GREEN), ("Posterior uncertainty", PURPLE), ("Held-out calibration", ORANGE)]
    for index, (label, color) in enumerate(labels):
        x = 0.065 + index * 0.235
        patch = FancyBboxPatch((x, 0.13), 0.19, 0.065, boxstyle="round,pad=0.008,rounding_size=0.014", linewidth=0.9, edgecolor=color, facecolor="white")
        axis.add_patch(patch)
        axis.text(x + 0.095, 0.162, label, fontsize=8.4, color=color, weight="bold", ha="center", va="center")
    banner(axis, "PROTOCOL / DATA PENDING  ·  Current A–D maps are day-zero priors, not fitted GP results.")
    save(figure, "04_per_camera_gp_learning/figures/per_camera_gp_learning_protocol.png")


def overlap_gate_protocol() -> None:
    figure, axis = canvas(
        "Geometric overlap is a test surface—not automatic permission to fuse",
        "The system can select a source first; combination is allowed only after synchronized evidence passes the overlap gate.",
    )
    stages = [
        ("1  Geometry", "Where ≥2 calibrated\nviews could see robot.\nCurrent: 42.2% overlap.", BLUE, PALE_BLUE),
        ("2  D2 pairs", "Synchronized projected\nobservations in adjacent\ncamera corridors.", PURPLE, PALE_PURPLE),
        ("3  Gate", "≥30 held-out pairs / edge\n≤10% spatial outliers\nbias + p90 disagreement", ORANGE, PALE_ORANGE),
        ("4  Action", "Select credible source.\nCombine only after\nconsistency + covariance.", GREEN, PALE_GREEN),
    ]
    for index, (title, body, color, pale) in enumerate(stages):
        x = 0.045 + index * 0.24
        box(axis, x, 0.47, 0.205, 0.28, title, body, color, pale)
        if index < 3:
            arrow(axis, x + 0.208, 0.61, x + 0.232, 0.61)
    axis.text(0.05, 0.24, "Evidence visuals generated after D2", fontsize=12.5, weight="bold", color=INK)
    for index, (title, detail) in enumerate([("Overlap graph", "tested physical pairs"), ("Sync timeline", "pair count + time offset"), ("Disagreement map", "where views diverge")]):
        x = 0.06 + index * 0.31
        box(axis, x, 0.15, 0.27, 0.075, title, "", MUTED, PALE_GRAY, title_size=9.5)
    banner(axis, "GEOMETRY CURRENT / D2 DATA PENDING  ·  Availability overlap is not a validated fused measurement.")
    save(figure, "05_overlap_selection_and_combination/figures/overlap_gate_protocol.png")


def closed_loop_evaluation_protocol() -> None:
    figure, axis = canvas(
        "The final claim is operational and matched—not a hand-picked trajectory",
        "Use identical routes, seeds, and world conditions to test whether source-aware observations improve continuity and safety.",
    )
    box(axis, 0.05, 0.46, 0.22, 0.30, "Matched inputs", "same world\nsame route and seed\nsame robot/planner\ncontrolled camera degradation", BLUE, PALE_BLUE)
    arrow(axis, 0.285, 0.61, 0.36, 0.61)
    box(axis, 0.38, 0.46, 0.22, 0.30, "Policy comparison", "fixed source\nscore-only\nreliability-aware selection\nconservative combination\n(if D2 passes)", PURPLE, PALE_PURPLE)
    arrow(axis, 0.615, 0.61, 0.69, 0.61)
    box(axis, 0.71, 0.46, 0.24, 0.30, "Report distributions", "goal / breach / recovery\nbelief error + covariance\nsource switches + gaps\nNIS / NEES around handovers", GREEN, PALE_GREEN)
    axis.text(0.05, 0.25, "Required final visual board", fontsize=12.5, weight="bold", color=INK)
    board = ["Route montage", "Belief timeline", "Decision trace", "Outcome table"]
    for index, title in enumerate(board):
        x = 0.05 + index * 0.235
        box(axis, x, 0.15, 0.20, 0.075, title, "", RED if index == 3 else MUTED, PALE_GRAY, title_size=9.3)
    banner(axis, "PROTOCOL / DATA PENDING  ·  Shadow replay precedes opt-in live correction; one attractive run is never sufficient evidence.")
    save(figure, "06_closed_loop_evaluation/figures/closed_loop_evaluation_protocol.png")


def main() -> None:
    collection_record_protocol()
    per_camera_gp_learning_protocol()
    overlap_gate_protocol()
    closed_loop_evaluation_protocol()


if __name__ == "__main__":
    main()
