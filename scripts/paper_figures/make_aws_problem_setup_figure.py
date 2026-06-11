#!/usr/bin/env python3
"""Create the AWS-world problem setup figure.

This figure follows the same grammar as ``problem_setup.pdf``: one camera-view
panel and two quiet top-down belief-space snapshots. It is a problem-statement
visual, not a results figure. It intentionally omits C1/C2 labels, GP maps,
cost fields, route-choice outcomes, and dense diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Polygon, Rectangle
import numpy as np
import pandas as pd
import yaml


REPO = Path(__file__).resolve().parents[2]
THESIS = REPO.parent / "thesis-report"

WORLD = "warehouse_aws.world.sdf"
TASK = "F31_b1_apron_a3_mid"

# Panel (a): current-world (camera z=4.8/y=-5.5) external-camera frame, robot near the
# F31_b1 start (3.3,-1.0); copied to a stable paper-input path from the v7 capture so the
# figure does not depend on archived capture data. (Old aws_simseg_v2 image was removed.)
DEFAULT_IMAGE = REPO / "logs/paper_figures/inputs/problem_setup_panel_a_aws.jpg"
DEFAULT_PROFILE = REPO / "src/experiments/config/world_profiles.yaml"
DEFAULT_TASKS = REPO / "src/experiments/config/tasks.yaml"
# Snapshot panels (b,c): the ORIGINAL constant-R (C1) rollout used by the first
# problem_setup figure. The source run was moved to _archive_nonpaper during cleanup, so its
# two needed CSVs (experiment.csv, plan_samples.csv) were copied to a stable paper-input dir
# — the figure no longer depends on the archive. Panel (a) is the separate current-world view.
DEFAULT_COV_RUN = REPO / "logs/paper_figures/inputs/problem_setup_cov_run_2026-06-03"
DEFAULT_OUT = THESIS / "figures/problem_setup_aws.pdf"
DEFAULT_PREVIEW = REPO / "logs/paper_figures/problem_setup_aws.png"


COL = {
    "floor": "#f8f8f8",
    "grid": "#d6d6d6",
    "rack": "#bdbdbd",
    "rack_edge": "#1f1f1f",
    "weak": "#f3d48a",
    "camera": "#222222",
    "start": "#16a34a",
    "goal": "#e41a1c",
    "truth": "#222222",
    "belief": "#7b3294",
    "horizon": "#d94b41",
    "cov": "#7b3294",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--world-profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--cov-run", type=Path, default=DEFAULT_COV_RUN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--preview", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument(
        "--split",
        action="store_true",
        help="Emit two figures instead of the combined 3-panel PDF: "
        "problem_setup_camera.pdf (panel a, for the Introduction) and "
        "problem_setup_snapshots.pdf (panels b+c, for the Problem Statement).",
    )
    return parser.parse_args()


def load_task(tasks_path: Path) -> tuple[dict, dict]:
    raw = yaml.safe_load(tasks_path.read_text(encoding="utf-8"))
    for task in raw["tasks"][WORLD]:
        if task["name"] == TASK:
            return task["start"], task["goal"]
    raise RuntimeError(f"{TASK} not found in {tasks_path}")


def load_profile(profile_path: Path) -> dict:
    return yaml.safe_load(profile_path.read_text(encoding="utf-8"))["worlds"][WORLD]


def load_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    exp_path = run_dir / "experiment.csv"
    plan_path = run_dir / "plan_samples.csv"
    if not exp_path.exists():
        raise RuntimeError(f"Missing experiment.csv: {exp_path}")
    if not plan_path.exists():
        raise RuntimeError(f"Missing plan_samples.csv: {plan_path}")

    exp = pd.read_csv(exp_path)
    required = [
        "stamp",
        "truth_x",
        "truth_y",
        "state_x",
        "state_y",
        "state_cov_xx",
        "state_cov_xy",
        "state_cov_yy",
    ]
    missing = [c for c in required if c not in exp.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in {exp_path}: {missing}")
    exp = exp.dropna(subset=required).copy().reset_index(drop=True)
    if exp.empty:
        raise RuntimeError(f"No valid rows in {exp_path}")

    plan = pd.read_csv(plan_path)
    missing = [c for c in ["point_idx", "x", "y"] if c not in plan.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in {plan_path}: {missing}")
    plan = plan.dropna(subset=["point_idx", "x", "y"]).copy()
    plan = plan.sort_values("point_idx").reset_index(drop=True)
    return exp, plan


def camera_pose_from_profile(profile: dict) -> tuple[np.ndarray, np.ndarray]:
    meta = profile.get("metadata", {})
    cam = meta.get("camera_position")
    look = meta.get("camera_look_at")
    if cam is not None and look is not None:
        return np.asarray(cam, dtype=float), np.asarray(look, dtype=float)
    return np.array([0.0, -4.9, 5.8]), np.array([0.0, -0.48, 0.0])


def covariance_matrix(row: pd.Series) -> np.ndarray:
    return np.array(
        [
            [float(row.state_cov_xx), float(row.state_cov_xy)],
            [float(row.state_cov_xy), float(row.state_cov_yy)],
        ],
        dtype=float,
    )


def sigma_major(row: pd.Series, sigma: float = 3.0) -> float:
    vals = np.linalg.eigvalsh(covariance_matrix(row))
    return float(sigma * math.sqrt(max(float(vals[-1]), 0.0)))


def first_motion_row(exp: pd.DataFrame, threshold_m: float = 0.12) -> int:
    x0 = float(exp.iloc[0].truth_x)
    y0 = float(exp.iloc[0].truth_y)
    disp = np.hypot(exp["truth_x"].to_numpy() - x0, exp["truth_y"].to_numpy() - y0)
    idx = np.flatnonzero(disp >= threshold_m)
    return int(idx[0]) if idx.size else 0


def weak_region_row(exp: pd.DataFrame) -> int:
    # Pick a logged covariance snapshot while the robot is in the A4-side
    # weak-update corridor. This keeps the ellipse real without plotting the
    # whole jagged failure trace.
    weak = exp[
        (exp["truth_x"] > 2.0)
        & (exp["truth_x"] < 3.6)
        & (exp["truth_y"] > -0.4)
        & (exp["truth_y"] < 1.5)
    ].copy()
    if weak.empty:
        return min(first_motion_row(exp) + 40, len(exp) - 1)
    majors = [sigma_major(r, sigma=3.0) for _, r in weak.iterrows()]
    return int(weak.index[int(np.argmax(majors))])


def plan_segment(plan: pd.DataFrame, xy: tuple[float, float], count: int = 22) -> np.ndarray:
    pts = plan[["x", "y"]].to_numpy(dtype=float)
    if len(pts) < 2:
        return pts
    dist = np.hypot(pts[:, 0] - xy[0], pts[:, 1] - xy[1])
    start = int(np.argmin(dist))
    end = min(start + count, len(pts))
    return pts[start:end]


def weak_update_patch() -> Polygon:
    pts = np.array(
        [
            [2.35, -0.72],
            [3.90, -0.20],
            [3.96, 2.55],
            [2.35, 2.35],
            [1.72, 1.15],
            [1.80, -0.54],
        ],
        dtype=float,
    )
    return Polygon(pts, closed=True, facecolor=COL["weak"], edgecolor="none", alpha=0.25, zorder=1)


def draw_racks(ax) -> None:
    rack_w = 0.55
    # Only the zoom-relevant rack rows are drawn. Using neutral gray mirrors the
    # compact problem figure and keeps this as a formulation visual.
    for x in [0.05, 2.00, 4.15]:
        for y0, y1 in [(-0.82, 1.20), (2.20, 4.25)]:
            ax.add_patch(
                Rectangle(
                    (x - rack_w / 2, y0),
                    rack_w,
                    y1 - y0,
                    facecolor=COL["rack"],
                    edgecolor=COL["rack_edge"],
                    linewidth=0.9,
                    zorder=5,
                )
            )
    ax.text(1.05, 0.10, "A3", fontsize=8.5, ha="center", va="center", color="#333333")
    ax.text(3.10, 0.10, "A4", fontsize=8.5, ha="center", va="center", color="#333333")


def style_topdown(ax, title: str) -> None:
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.45, 4.45)
    ax.set_ylim(-1.65, 2.85)
    ax.set_xlabel("position x [m]", fontsize=9)
    ax.set_ylabel("position y [m]", fontsize=9)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_yticks([-1, 0, 1, 2])
    ax.grid(True, color=COL["grid"], linewidth=0.45, zorder=0)
    ax.tick_params(labelsize=8, length=2)
    ax.set_facecolor(COL["floor"])


def draw_camera_marker(ax, profile: dict) -> None:
    cam, look = camera_pose_from_profile(profile)
    marker_y = -1.47
    ax.scatter([cam[0]], [marker_y], marker="<", s=45, color=COL["camera"], zorder=12)
    ax.text(cam[0] + 0.15, marker_y - 0.03, "fixed camera", fontsize=7.5, ha="left", va="top")
    ax.plot(
        [cam[0], look[0] + 1.4],
        [marker_y, 0.2],
        color=COL["camera"],
        linewidth=0.65,
        linestyle=(0, (2, 3)),
        alpha=0.45,
        zorder=2,
    )


def draw_markers(ax, start: dict, goal: dict) -> None:
    sx, sy = float(start["x"]), float(start["y"])
    gx, gy = float(goal["x"]), float(goal["y"])
    ax.scatter([sx], [sy], s=46, color=COL["start"], edgecolor="black", linewidth=0.6, zorder=13)
    ax.scatter([gx], [gy], s=52, color=COL["goal"], edgecolor="black", linewidth=0.6, zorder=13)
    ax.text(sx - 0.08, sy + 0.18, "start", fontsize=8, ha="right", va="bottom")
    ax.text(gx + 0.10, gy + 0.16, "goal", fontsize=8, ha="left", va="bottom")


def draw_covariance(ax, xy: tuple[float, float], cov: np.ndarray) -> None:
    vals, vecs = np.linalg.eigh(np.asarray(cov, dtype=float))
    vals = np.maximum(vals, 1e-9)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = math.degrees(math.atan2(float(vecs[1, 0]), float(vecs[0, 0])))
    # Matplotlib ellipse width/height are full diameters: 2*sigma*sqrt(lambda).
    width, height = 2.0 * 3.0 * np.sqrt(vals)
    ax.add_patch(
        Ellipse(
            xy,
            width=float(width),
            height=float(height),
            angle=angle,
            facecolor=COL["cov"],
            edgecolor=COL["cov"],
            alpha=0.26,
            linewidth=1.0,
            zorder=9,
        )
    )


def draw_snapshot_panel(
    ax,
    profile: dict,
    start: dict,
    goal: dict,
    exp: pd.DataFrame,
    plan: pd.DataFrame,
    row_idx: int,
    title: str,
    *,
    annotate: bool,
) -> None:
    style_topdown(ax, title)
    ax.add_patch(weak_update_patch())
    draw_racks(ax)
    draw_camera_marker(ax, profile)
    draw_markers(ax, start, goal)

    row = exp.iloc[row_idx]
    current_xy = (float(row.state_x), float(row.state_y))
    horizon = plan_segment(plan, current_xy)
    if len(horizon) >= 2:
        ax.plot(horizon[:, 0], horizon[:, 1], color=COL["horizon"], linewidth=1.45, zorder=8)

    hist = exp.iloc[: row_idx + 1]
    ax.plot(hist["truth_x"], hist["truth_y"], color=COL["truth"], linewidth=1.55, zorder=10)
    ax.plot(
        hist["state_x"],
        hist["state_y"],
        color=COL["belief"],
        linewidth=1.3,
        linestyle=(0, (4, 2)),
        zorder=10,
    )
    ax.scatter([float(row.truth_x)], [float(row.truth_y)], color=COL["truth"], s=30, zorder=12)
    ax.scatter([float(row.state_x)], [float(row.state_y)], color=COL["belief"], s=34, zorder=12)
    draw_covariance(ax, current_xy, covariance_matrix(row))

    if annotate:
        ax.text(3.08, 2.06, "reduced\nupdates", fontsize=8.5, color="#8a5b0b", ha="center")
        ax.annotate(
            "predicted\nuncertainty",
            xy=current_xy,
            xytext=(float(row.state_x) - 0.58, float(row.state_y) + 0.88),
            fontsize=8.5,
            color=COL["belief"],
            ha="center",
            arrowprops={"arrowstyle": "->", "linewidth": 0.8, "color": COL["belief"]},
        )


def draw_camera_view(ax, image_path: Path) -> None:
    ax.set_title("(a) external-camera warehouse setup", fontsize=10, pad=6)
    img = mpimg.imread(image_path)
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        0.03,
        0.07,
        "External-camera / Gazebo view",
        transform=ax.transAxes,
        fontsize=8,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 3},
    )


def legend_handles() -> list:
    return [
        Line2D([0], [0], color=COL["truth"], linewidth=1.6, label="truth path"),
        Line2D([0], [0], color=COL["belief"], linewidth=1.3, linestyle=(0, (4, 2)), label="belief mean"),
        Line2D([0], [0], color=COL["horizon"], linewidth=1.45, label="current horizon"),
        Ellipse((0, 0), 0.18, 0.10, facecolor=COL["cov"], edgecolor=COL["cov"], alpha=0.26, label=r"3$\sigma$ posterior covariance"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COL["start"], markeredgecolor="black", markersize=7, label="start"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COL["goal"], markeredgecolor="black", markersize=7, label="goal"),
        Rectangle((0, 0), 1, 1, facecolor=COL["weak"], edgecolor="none", alpha=0.25, label="reduced camera-update reliability"),
    ]


def main() -> int:
    args = parse_args()
    profile = load_profile(args.world_profile)
    start, goal = load_task(args.tasks)
    exp, plan = load_run(args.cov_run)

    early_idx = first_motion_row(exp)
    late_idx = weak_region_row(exp)
    if "plan_stamp" in plan.columns and plan["plan_stamp"].notna().any():
        time_reference = float(plan["plan_stamp"].dropna().min())
    else:
        time_reference = float(exp.iloc[0].stamp)
    early_time = max(0.0, float(exp.iloc[early_idx].stamp - time_reference))
    late_time = max(0.0, float(exp.iloc[late_idx].stamp - time_reference))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    if args.split:
        # Panel (a) standalone -> Introduction figure.
        cam_out = args.out.with_name("problem_setup_camera.pdf")
        cam_prev = args.preview.with_name("problem_setup_camera.png")
        fig_cam, ax_cam = plt.subplots(1, 1, figsize=(5.2, 4.9), constrained_layout=True)
        draw_camera_view(ax_cam, args.image)
        cam_out.parent.mkdir(parents=True, exist_ok=True)
        cam_prev.parent.mkdir(parents=True, exist_ok=True)
        fig_cam.savefig(cam_out, bbox_inches="tight")
        fig_cam.savefig(cam_prev, dpi=260, bbox_inches="tight")
        plt.close(fig_cam)

        # Panels (b)+(c) -> Problem Statement figure. Lettering is kept as (b),(c) so the
        # camera view (panel a, moved to the Introduction) and these two panels still read
        # as one conceptual figure across the two sections.
        snap_out = args.out.with_name("problem_setup_snapshots.pdf")
        snap_prev = args.preview.with_name("problem_setup_snapshots.png")
        fig_s, axes_s = plt.subplots(1, 2, figsize=(9.4, 4.9), constrained_layout=True)
        draw_snapshot_panel(
            axes_s[0], profile, start, goal, exp, plan, early_idx,
            f"(b) initial rollout\n$t={early_time:.1f}\\,$s", annotate=False,
        )
        draw_snapshot_panel(
            axes_s[1], profile, start, goal, exp, plan, late_idx,
            f"(c) near reduced camera-update reliability\n$t={late_time:.1f}\\,$s", annotate=True,
        )
        fig_s.legend(
            handles=legend_handles(), loc="lower center", ncol=7, fontsize=8,
            frameon=False, bbox_to_anchor=(0.5, -0.02),
        )
        snap_out.parent.mkdir(parents=True, exist_ok=True)
        fig_s.savefig(snap_out, bbox_inches="tight")
        fig_s.savefig(snap_prev, dpi=260, bbox_inches="tight")
        plt.close(fig_s)

        print(f"wrote {cam_out}")
        print(f"wrote {snap_out}")
        print(f"wrote {cam_prev}")
        print(f"wrote {snap_prev}")
        return 0

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.8, 4.9),
        gridspec_kw={"width_ratios": [1.05, 1.0, 1.0]},
        constrained_layout=True,
    )

    draw_camera_view(axes[0], args.image)
    draw_snapshot_panel(
        axes[1],
        profile,
        start,
        goal,
        exp,
        plan,
        early_idx,
        f"(b) initial rollout\n$t={early_time:.1f}\\,$s",
        annotate=False,
    )
    draw_snapshot_panel(
        axes[2],
        profile,
        start,
        goal,
        exp,
        plan,
        late_idx,
        f"(c) near reduced camera-update reliability\n$t={late_time:.1f}\\,$s",
        annotate=True,
    )

    fig.legend(
        handles=legend_handles(),
        loc="lower center",
        ncol=7,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.preview.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.preview, dpi=260, bbox_inches="tight")
    plt.close(fig)

    caption = (
        "AWS-world problem setup. A fixed external camera observes a warehouse "
        "aisle task. The pale region denotes reduced camera-update reliability "
        "in the external-camera localization pathway. It is a problem-statement "
        "annotation, not a GP map, reward, cost field, or traversability layer. "
        "Panels (b) and (c) use the same zoomed A3/A4 coordinate frame. The "
        "truth path, belief mean, current horizon, and 3-sigma posterior "
        "covariance are drawn from an existing logged AWS run."
    )
    caption_path = args.preview.with_name("problem_setup_aws_caption.txt")
    caption_path.write_text(caption + "\n", encoding="utf-8")

    provenance = {
        "figure": "problem_setup_aws",
        "world": WORLD,
        "task": TASK,
        "image": str(args.image),
        "profile": str(args.world_profile),
        "run": str(args.cov_run),
        "covariance_source": "state_x/y and state_cov_xx/xy/yy from experiment.csv",
        "plan_source": "plan_samples.csv from the same run",
        "early_snapshot": {
            "row": int(early_idx),
            "stamp": float(exp.iloc[early_idx].stamp),
            "relative_s": early_time,
            "sigma3_major_m": sigma_major(exp.iloc[early_idx], sigma=3.0),
        },
        "late_snapshot": {
            "row": int(late_idx),
            "stamp": float(exp.iloc[late_idx].stamp),
            "relative_s": late_time,
            "sigma3_major_m": sigma_major(exp.iloc[late_idx], sigma=3.0),
        },
        "time_reference_stamp": time_reference,
        "notes": [
            "The compact problem_setup.pdf is not modified.",
            "This is a problem-statement visual, not a C1/C2 results plot.",
            "The weak-update polygon is qualitative and not a GP/reliability map.",
            "The camera glyph in top-down panels indicates the south-wall viewing direction in the zoomed crop.",
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
