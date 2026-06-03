#!/usr/bin/env python3
"""Create an AWS-world problem setup figure without replacing the compact one.

This figure is intentionally a problem-statement visual, not a results or
method-validation plot. It shows the warehouse geometry, the external camera,
the task, and the qualitative belief-space problem: when the robot enters a
region with weak external-camera updates, predicted position uncertainty grows.

It does not show a GP reliability map, a planner cost map, C1/C2 routes, or
campaign outcomes. Those belong in method/results figures after the observation
model has been introduced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon, Rectangle
import numpy as np
import pandas as pd
import yaml


REPO = Path(__file__).resolve().parents[2]
THESIS = REPO.parent / "thesis-report"

WORLD = "warehouse_aws.world.sdf"
TASK = "F31_b1_apron_a3_mid"

DEFAULT_IMAGE = REPO / "logs/perception_datasets/aws_simseg_v2/images/train/000095.jpg"
DEFAULT_PROFILE = REPO / "src/experiments/config/world_profiles.yaml"
DEFAULT_TASKS = REPO / "src/experiments/config/tasks.yaml"
DEFAULT_COV_RUN = REPO / "logs/visibility_comparison/paper_final_v1/F31_b1_apron_a3_mid/C1/seed3/experiment_20260603_091605"
DEFAULT_OUT = THESIS / "figures/problem_setup_aws.pdf"
DEFAULT_PREVIEW = REPO / "logs/paper_figures/problem_setup_aws.png"


COL = {
    "drive": "#cae8c8",
    "drive_edge": "#1b9850",
    "non": "#f2b8b5",
    "non_edge": "#d73027",
    "rack": "#f2cf23",
    "rack_edge": "#0b6f8a",
    "weak": "#f7d95a",
    "camera": "#222222",
    "start": "#1f78b4",
    "goal": "#18a558",
    "belief": "#111111",
    "truth": "#2b78c2",
    "cov": "#d95f02",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--world-profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--cov-run", type=Path, default=DEFAULT_COV_RUN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--preview", type=Path, default=DEFAULT_PREVIEW)
    return parser.parse_args()


def load_task(tasks_path: Path) -> tuple[dict, dict]:
    raw = yaml.safe_load(tasks_path.read_text(encoding="utf-8"))
    for task in raw["tasks"][WORLD]:
        if task["name"] == TASK:
            return task["start"], task["goal"]
    raise RuntimeError(f"{TASK} not found in {tasks_path}")


def load_profile(profile_path: Path) -> dict:
    return yaml.safe_load(profile_path.read_text(encoding="utf-8"))["worlds"][WORLD]


def load_covariance_trace(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "experiment.csv"
    if not path.exists():
        raise RuntimeError(f"Missing experiment.csv for covariance trace: {path}")
    df = pd.read_csv(path)
    required = ["planner_belief_x", "planner_belief_y", "planner_cov_x", "planner_cov_xy", "planner_cov_y"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing covariance columns in {path}: {missing}")
    out = df.dropna(subset=required).copy()
    if out.empty:
        raise RuntimeError(f"No valid covariance rows in {path}")
    majors = []
    for _, r in out.iterrows():
        S = np.array(
            [[float(r.planner_cov_x), float(r.planner_cov_xy)], [float(r.planner_cov_xy), float(r.planner_cov_y)]],
            dtype=float,
        )
        majors.append(float(2.0 * np.sqrt(max(np.linalg.eigvalsh(S)[-1], 0.0))))
    out["cov_2sigma_major_m"] = majors
    return out


def draw_regions(ax, profile: dict, *, labels: bool = False) -> None:
    ax.set_facecolor("#f7f6f2")
    for region in profile.get("known_2d_regions", []):
        name = str(region.get("name", ""))
        typ = str(region.get("type", ""))
        x0 = float(region["xmin"])
        y0 = float(region["ymin"])
        w = float(region["xmax"]) - x0
        h = float(region["ymax"]) - y0
        if typ == "traversable":
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    w,
                    h,
                    facecolor=COL["drive"],
                    edgecolor=COL["drive_edge"],
                    lw=0.85,
                    alpha=0.48,
                    zorder=1,
                )
            )
        elif typ == "non_driveable_staging":
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    w,
                    h,
                    facecolor=COL["non"],
                    edgecolor=COL["non_edge"],
                    lw=0.7,
                    alpha=0.45,
                    hatch="///",
                    zorder=3,
                )
            )
            if labels and ("shipping" in name or "receiving" in name or "staging" in name):
                clean = name.replace("wall_side_", "").replace("_", "\n")
                ax.text(
                    x0 + w / 2,
                    y0 + h / 2,
                    clean,
                    fontsize=5.5,
                    ha="center",
                    va="center",
                    color="#6b1c16",
                    zorder=4,
                    clip_on=True,
                )


def draw_racks(ax) -> None:
    rack_xs = [-4.05, -2.00, 0.05, 2.00, 4.15]
    rack_w = 0.55
    segments = [(-0.82, 1.20), (2.20, 4.25)]
    for i, x in enumerate(rack_xs, start=1):
        for y0, y1 in segments:
            ax.add_patch(
                Rectangle(
                    (x - rack_w / 2, y0),
                    rack_w,
                    y1 - y0,
                    facecolor=COL["rack"],
                    edgecolor=COL["rack_edge"],
                    lw=0.9,
                    zorder=5,
                )
            )
        ax.text(x, 4.43, f"R{i}", fontsize=7, ha="center", va="bottom", fontweight="bold", color="#333333", zorder=6, clip_on=True)
    for label, x in [("A3", 1.075), ("A4", 3.125)]:
        ax.text(x, 0.12, label, fontsize=8, ha="center", va="center", color="#333333", fontweight="bold", zorder=6, clip_on=True)


def weak_observability_patch() -> Polygon:
    # Qualitative line-of-sight shadow for the wall-mounted external camera.
    # This is deliberately not a GP/reliability field.
    pts = np.array(
        [
            [2.35, -0.78],
            [3.88, -0.20],
            [3.95, 2.55],
            [2.32, 2.35],
            [1.72, 1.15],
            [1.78, -0.55],
        ]
    )
    return Polygon(
        pts,
        closed=True,
        facecolor=COL["weak"],
        edgecolor="#bf8c00",
        lw=0.9,
        linestyle=(0, (5, 3)),
        alpha=0.34,
        zorder=4,
    )


def style_map(ax, title: str, *, zoom: bool = False) -> None:
    ax.set_title(title, fontsize=10.2, fontweight="bold", pad=6)
    if zoom:
        ax.set_xlim(0.35, 4.35)
        ax.set_ylim(-1.65, 3.15)
    else:
        ax.set_xlim(-5.55, 5.55)
        ax.set_ylim(-5.25, 5.15)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]", fontsize=8.5)
    ax.set_ylabel("y [m]", fontsize=8.5)
    if zoom:
        ax.set_xticks([0.5, 1.5, 2.5, 3.5, 4.0])
        ax.set_yticks([-1.0, 0.0, 1.0, 2.0, 3.0])
    else:
        ax.set_xticks([-5, -3, -1, 1, 3, 5])
        ax.set_yticks([-5, -3, -1, 1, 3, 5])
    ax.grid(True, color="#cccccc", lw=0.35, alpha=0.5, zorder=0)
    ax.tick_params(labelsize=7.5, length=2)


def camera_pose_from_profile(profile: dict) -> tuple[np.ndarray, np.ndarray]:
    meta = profile.get("metadata", {})
    cam = meta.get("camera_position")
    look = meta.get("camera_look_at")
    if cam is not None and look is not None:
        return np.asarray(cam, dtype=float), np.asarray(look, dtype=float)
    # Fallback to the accepted AWS v5 camera pose.
    return np.array([0.0, -4.9, 5.8]), np.array([0.0, -0.48, 0.0])


def draw_markers(ax, start: dict, goal: dict, profile: dict, *, task_labels: bool = True) -> None:
    sx, sy = float(start["x"]), float(start["y"])
    gx, gy = float(goal["x"]), float(goal["y"])
    cam, look = camera_pose_from_profile(profile)

    ax.scatter([sx], [sy], s=52, color=COL["start"], edgecolor="black", lw=0.6, zorder=12)
    ax.scatter([gx], [gy], s=86, marker="*", color=COL["goal"], edgecolor="black", lw=0.7, zorder=12)
    if task_labels:
        ax.text(sx + 0.12, sy - 0.20, "start\nvisible pick pose", fontsize=6.3, ha="left", va="top", zorder=13, clip_on=True)
        ax.text(gx + 0.12, gy + 0.18, "goal\nvisible aisle target", fontsize=6.3, ha="left", va="bottom", zorder=13, clip_on=True)

    ax.scatter([cam[0]], [cam[1]], marker="^", s=54, color=COL["camera"], zorder=12, clip_on=True)
    ax.annotate(
        "",
        xy=(look[0], look[1]),
        xytext=(cam[0], cam[1]),
        arrowprops={"arrowstyle": "->", "color": COL["camera"], "lw": 1.1, "alpha": 0.85},
        zorder=11,
    )
    ax.text(cam[0] + 0.16, cam[1] + 0.10, "fixed\ncamera", fontsize=6.2, ha="left", va="bottom", zorder=13, clip_on=True)


def draw_camera_view(ax, image_path: Path) -> None:
    ax.set_title("(a) external-camera setting", fontsize=10.2, fontweight="bold", pad=6)
    img = mpimg.imread(image_path)
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        0.03,
        0.07,
        "warehouse scene observed by a fixed camera",
        transform=ax.transAxes,
        fontsize=7.5,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 3},
    )


def ellipse_angle_deg(vec: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(vec[1], vec[0])))


def draw_belief_ellipse(ax, xy: np.ndarray, S: np.ndarray, alpha: float = 0.34) -> None:
    vals, vecs = np.linalg.eigh(S)
    vals = np.maximum(vals, 0.0)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    major = 2.0 * np.sqrt(vals[0])
    minor = 2.0 * np.sqrt(vals[1])
    angle = ellipse_angle_deg(vecs[:, 0])
    ax.add_patch(
        Ellipse(
            xy,
            width=major,
            height=minor,
            angle=angle,
            facecolor=COL["cov"],
            edgecolor="#7f2704",
            lw=0.9,
            alpha=alpha,
            zorder=14,
        )
    )


def draw_setup_panel(ax, profile: dict, start: dict, goal: dict) -> None:
    style_map(ax, "(b) task geometry and sensing challenge", zoom=True)
    draw_regions(ax, profile, labels=True)
    ax.add_patch(weak_observability_patch())
    draw_racks(ax)
    draw_markers(ax, start, goal, profile)
    ax.text(
        2.95,
        2.67,
        "reduced camera-update\nobservability",
        fontsize=6.6,
        color="#7a5500",
        ha="center",
        va="bottom",
        zorder=15,
    )
    ax.add_patch(
        FancyArrowPatch(
            (3.55, -4.55),
            (3.25, -0.35),
            arrowstyle="->",
            mutation_scale=9,
            color="#7a5500",
            lw=0.9,
            alpha=0.65,
            zorder=10,
        )
    )


def draw_uncertainty_panel(ax, profile: dict, start: dict, goal: dict, cov_trace: pd.DataFrame) -> None:
    style_map(ax, "(c) logged covariance grows when updates weaken", zoom=True)
    draw_regions(ax, profile)
    ax.add_patch(weak_observability_patch())
    draw_racks(ax)
    draw_markers(ax, start, goal, profile, task_labels=False)

    trace = cov_trace[
        (cov_trace["planner_belief_x"] >= 0.35)
        & (cov_trace["planner_belief_x"] <= 4.35)
        & (cov_trace["planner_belief_y"] >= -1.65)
        & (cov_trace["planner_belief_y"] <= 3.15)
    ].copy()
    ax.plot(
        trace["planner_belief_x"],
        trace["planner_belief_y"],
        color=COL["truth"],
        lw=1.8,
        zorder=13,
        label="logged belief mean",
    )

    if len(trace) >= 8:
        rows = [trace.iloc[int(round(q * (len(trace) - 1)))] for q in np.linspace(0.05, 0.92, 6)]
        rows.append(trace.loc[trace["cov_2sigma_major_m"].idxmax()])
    else:
        rows = list(trace.itertuples(index=False))
    used = set()
    for r in rows:
        key = (round(float(r.planner_belief_x), 2), round(float(r.planner_belief_y), 2))
        if key in used:
            continue
        used.add(key)
        S = np.array(
            [[float(r.planner_cov_x), float(r.planner_cov_xy)], [float(r.planner_cov_xy), float(r.planner_cov_y)]],
            dtype=float,
        )
        draw_belief_ellipse(ax, np.array([float(r.planner_belief_x), float(r.planner_belief_y)]), S)

    ax.annotate(
        "logged covariance\nincreases in weak-\nupdate section",
        xy=(3.05, 2.50),
        xytext=(1.10, 2.70),
        fontsize=6.6,
        color="#7f2704",
        arrowprops={"arrowstyle": "->", "lw": 0.9, "color": "#7f2704"},
        ha="left",
        va="center",
        zorder=16,
    )
    ax.annotate(
        "update recovers",
        xy=(1.03, 1.75),
        xytext=(0.58, 0.72),
        fontsize=6.5,
        color="#245c2c",
        arrowprops={"arrowstyle": "->", "lw": 0.9, "color": "#245c2c"},
        ha="left",
        va="center",
        zorder=16,
    )


def legend_handles() -> list:
    return [
        Rectangle((0, 0), 1, 1, facecolor=COL["drive"], edgecolor=COL["drive_edge"], alpha=0.48, label="known driveable floor"),
        Rectangle((0, 0), 1, 1, facecolor=COL["non"], edgecolor=COL["non_edge"], alpha=0.45, hatch="///", label="known forbidden/staging zone"),
        Rectangle((0, 0), 1, 1, facecolor=COL["rack"], edgecolor=COL["rack_edge"], label="rack geometry"),
        Rectangle((0, 0), 1, 1, facecolor=COL["weak"], edgecolor="#bf8c00", alpha=0.34, label="reduced camera-update observability"),
        Line2D([0], [0], color=COL["truth"], lw=1.8, label="logged belief mean"),
        Ellipse((0, 0), 0.24, 0.12, facecolor=COL["cov"], edgecolor="#7f2704", alpha=0.34, label="logged 2-sigma covariance"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=COL["camera"], markeredgecolor=COL["camera"], markersize=6, label="fixed camera"),
    ]


def main() -> int:
    args = parse_args()
    profile = load_profile(args.world_profile)
    start, goal = load_task(args.tasks)
    cov_trace = load_covariance_trace(args.cov_run)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(14.0, 5.25), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    draw_camera_view(ax0, args.image)
    draw_setup_panel(ax1, profile, start, goal)
    draw_uncertainty_panel(ax2, profile, start, goal, cov_trace)

    fig.legend(handles=legend_handles(), loc="lower center", ncol=4, fontsize=7.2, frameon=False, bbox_to_anchor=(0.5, -0.03))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.preview.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.preview, dpi=240, bbox_inches="tight")
    plt.close(fig)

    caption = (
        "AWS-world problem setup. A fixed external camera observes a robot in a "
        "warehouse aisle task. The green regions are known driveable floor and "
        "the red hatched regions are known forbidden/staging zones. The yellow "
        "region denotes reduced external-camera update observability caused by "
        "the warehouse geometry; it is not a GP map, reward, or traversability "
        "constraint. The right panel uses logged planner-belief covariance from "
        "an existing AWS run to show the core belief-space problem: when camera "
        "updates weaken, position covariance grows until more reliable "
        "observations become available again."
    )
    caption_path = args.preview.with_name("problem_setup_aws_caption.txt")
    caption_path.write_text(caption + "\n", encoding="utf-8")

    provenance = {
        "figure": "problem_setup_aws",
        "world": WORLD,
        "task": TASK,
        "image": str(args.image),
        "profile": str(args.world_profile),
        "covariance_run": str(args.cov_run),
        "covariance_source": "planner_belief_x/y and planner_cov_x/xy/y from experiment.csv",
        "max_logged_2sigma_major_m": float(cov_trace["cov_2sigma_major_m"].max()),
        "notes": [
            "This figure is a problem-statement visual and does not replace problem_setup.pdf.",
            "It intentionally omits the GP/reliability map and C1/C2 route evidence.",
            "The yellow region is a qualitative reduced-observability region, not a cost or traversability layer.",
            "Belief ellipses are logged 2-sigma position covariance from an existing AWS run.",
        ],
    }
    provenance_path = args.preview.with_name("problem_setup_aws_provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2, allow_nan=False), encoding="utf-8")

    print(f"wrote {args.out}")
    print(f"wrote {args.preview}")
    print(f"wrote {caption_path}")
    print(f"wrote {provenance_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
