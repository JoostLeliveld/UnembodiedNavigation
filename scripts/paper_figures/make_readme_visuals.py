#!/usr/bin/env python3
"""Generate README-facing visual explanations from packaged artifacts.

This script is intentionally docs-only. It does not run ROS, Gazebo, YOLO, or
any planner. The visuals are built from checked-in presentation assets, paper
artifacts, GP grids, and campaign CSV/JSON bundles so the top-level module
READMEs can be regenerated without private model weights.
"""

from __future__ import annotations

import csv
import json
import math
import os
import warnings
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-unav")
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle


REPO = Path(__file__).resolve().parents[2]
MIDTERM_ASSETS = REPO.parent / "midterm_presentation" / "assets"

GP_ARTIFACT = REPO / "paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz"
CAMPAIGN_CONFIG = REPO / "scripts/visibility_comparison/warehouse_visibility_campaign.yaml"
WEST_PAIR = REPO / "paper_artifacts/figures/paired_mechanism_west_current_data"
FULL_CAMPAIGN_LOG = REPO / "docs/paper_vs_current/current/data/paired_mechanism_taskA_lowlat/campaign_log.json"

R_VISIBLE_UV = 2.5
R_MISS_UV = 40.0
MIN_PROB = 1.0e-4

TUE_RED = "#c8193c"
TUE_BLUE = "#0066a2"
DARK = "#1f2933"
MUTED = "#64748b"
LIGHT_BG = "#f7f9fc"
CARD_BG = "#ffffff"
GRID = "#d7dee8"
RACK = "#f2cf23"
RACK_EDGE = "#0b6f8a"
DRIVE = "#22c55e"
C1 = "#d73027"
C2 = "#1f78b4"


def ensure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def maybe_image(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        return mpimg.imread(path)
    except Exception:
        return None


def save(fig: plt.Figure, path: Path, *, dpi: int = 180) -> None:
    ensure(path)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {rel(path)}")


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 13.0,
            "axes.labelsize": 10.5,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_gp() -> dict[str, np.ndarray | str]:
    with np.load(GP_ARTIFACT, allow_pickle=False) as data:
        return {
            "xs": np.asarray(data["xs"], dtype=float),
            "ys": np.asarray(data["ys"], dtype=float),
            "x_train": np.asarray(data["X_train"], dtype=float),
            "p_train": np.asarray(data["p_train"], dtype=float),
            "p_mean": np.asarray(data["P_mean_map"], dtype=float),
            "p_plan": np.asarray(data["P_conservative_plan_map"], dtype=float),
            "f_std": np.asarray(data["F_std_map"], dtype=float),
            "geometry_json": str(np.asarray(data["geometry_json"]).reshape(-1)[0]),
        }


def covariance_std(p_plan: np.ndarray) -> np.ndarray:
    p_eff = np.clip(np.asarray(p_plan, dtype=float), MIN_PROB, 1.0 - MIN_PROB)
    visible_var = R_VISIBLE_UV**2
    miss_var = R_MISS_UV**2
    var = 1.0 / np.maximum(p_eff / visible_var + (1.0 - p_eff) / miss_var, 1.0e-12)
    return np.sqrt(var)


def gp_extent(gp: dict[str, np.ndarray | str]) -> tuple[float, float, float, float]:
    xs = np.asarray(gp["xs"], dtype=float)
    ys = np.asarray(gp["ys"], dtype=float)
    return float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])


def draw_racks(ax: plt.Axes, gp: dict[str, np.ndarray | str], *, alpha: float = 0.95) -> None:
    try:
        geometry = json.loads(str(gp["geometry_json"]))
    except json.JSONDecodeError:
        geometry = {"prisms": []}
    for prism in geometry.get("prisms", []):
        x0 = float(prism["xmin"])
        x1 = float(prism["xmax"])
        y0 = float(prism["ymin"])
        y1 = float(prism["ymax"])
        ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                facecolor=RACK,
                edgecolor=RACK_EDGE,
                lw=0.45,
                alpha=alpha,
                zorder=6,
            )
        )


def draw_driveable(ax: plt.Axes, *, lw: float = 1.3, alpha: float = 0.72) -> None:
    if not CAMPAIGN_CONFIG.is_file():
        return
    cfg = yaml.safe_load(CAMPAIGN_CONFIG.read_text(encoding="utf-8"))
    raw = cfg.get("driveable_geometry_json")
    if not raw:
        return
    for prism in json.loads(raw).get("prisms", []):
        x0 = float(prism["xmin"])
        x1 = float(prism["xmax"])
        y0 = float(prism["ymin"])
        y1 = float(prism["ymax"])
        ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                facecolor="none",
                edgecolor=DRIVE,
                lw=lw,
                alpha=alpha,
                zorder=5,
            )
        )


def style_map_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, pad=7)
    ax.set_xlim(-5.55, 5.55)
    ax.set_ylim(-5.05, 5.05)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, color=GRID, lw=0.35, alpha=0.55, zorder=1)
    ax.tick_params(labelsize=9)


def read_csv_points(path: Path, x_names: Iterable[str], y_names: Iterable[str]) -> np.ndarray:
    if not path.is_file():
        return np.zeros((0, 2), dtype=float)
    xs: list[float] = []
    ys: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            x = first_float(row, x_names)
            y = first_float(row, y_names)
            if math.isfinite(x) and math.isfinite(y):
                xs.append(x)
                ys.append(y)
    if not xs:
        return np.zeros((0, 2), dtype=float)
    return np.column_stack([xs, ys])


def first_float(row: dict[str, str], names: Iterable[str], default: float = math.nan) -> float:
    for name in names:
        if name not in row:
            continue
        try:
            value = float(row[name])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return default


def route_distance(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0)
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def query_gp_trust(gp: dict[str, np.ndarray | str], points: np.ndarray) -> np.ndarray:
    xs = np.asarray(gp["xs"], dtype=float)
    ys = np.asarray(gp["ys"], dtype=float)
    p_plan = np.asarray(gp["p_plan"], dtype=float)
    out: list[float] = []
    for x, y in points:
        ix = int(np.clip(np.searchsorted(xs, float(x)), 0, len(xs) - 1))
        iy = int(np.clip(np.searchsorted(ys, float(y)), 0, len(ys) - 1))
        out.append(float(p_plan[iy, ix]))
    return np.asarray(out, dtype=float)


def read_series(path: Path, names: Iterable[str]) -> np.ndarray:
    out: list[float] = []
    if not path.is_file():
        return np.zeros(0)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out.append(first_float(row, names))
    return np.asarray(out, dtype=float)


def load_full_campaign() -> dict:
    if FULL_CAMPAIGN_LOG.is_file():
        return json.loads(FULL_CAMPAIGN_LOG.read_text(encoding="utf-8"))
    return {}


def draw_title(fig: plt.Figure, title: str, subtitle: str | None = None) -> None:
    fig.text(0.035, 0.965, title, ha="left", va="top", fontsize=19, fontweight="bold", color=DARK)
    if subtitle:
        fig.text(0.035, 0.925, subtitle, ha="left", va="top", fontsize=10.8, color=MUTED)
    fig.add_artist(Line2D([0.035, 0.965], [0.895, 0.895], color=TUE_RED, lw=2.6, transform=fig.transFigure))


def add_card(ax: plt.Axes, xy: tuple[float, float], wh: tuple[float, float], title: str, text: str, color: str) -> None:
    x, y = xy
    w, h = wh
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=CARD_BG,
            edgecolor="#d9e1ec",
            lw=1.2,
            zorder=2,
        )
    )
    ax.add_patch(Rectangle((x, y + h - 0.03), w, 0.03, facecolor=color, edgecolor="none", zorder=3))
    ax.text(x + 0.025, y + h - 0.07, title, ha="left", va="top", fontsize=12.2, fontweight="bold", color=DARK, zorder=4)
    ax.text(x + 0.025, y + h - 0.145, text, ha="left", va="top", fontsize=9.6, color="#334155", linespacing=1.28, zorder=4)


def make_contribution_map() -> None:
    fig, ax = plt.subplots(figsize=(14.5, 8.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(LIGHT_BG)
    draw_title(
        fig,
        "External-camera warehouse navigation: contribution map",
        "The project is modular: each block turns one messy camera-navigation problem into a planner-readable signal.",
    )

    cards = [
        ("Warehouse setup", "Fixed external camera,\nracks, occlusions, and\nmatched route tasks.", "#475569"),
        ("YOLO perception", "Detect the robot and\nexport the bottom-centre\nimage point plus score.", TUE_RED),
        ("BEV localization", "Project image evidence\nto the ground plane and\napply affine calibration.", "#f59e0b"),
        ("Belief model", "Predict with odometry;\ncorrect camera x,y when\nfresh detections arrive.", "#10b981"),
        ("GP reliability", "Learn where the camera\nis trustworthy from\nspatial detector data.", TUE_BLUE),
        ("Planning", "Scale R_plan, evaluate\nambiguity, and avoid\nno-go geometry.", "#7c3aed"),
        ("Experiments", "Compare C1 and C2 under\nmatched tasks, seeds,\nand metrics.", "#0f766e"),
    ]
    x0s = np.linspace(0.045, 0.795, len(cards))
    y = 0.47
    w = 0.112
    h = 0.27
    for idx, (title, text, color) in enumerate(cards):
        add_card(ax, (float(x0s[idx]), y), (w, h), title, text, color)
        if idx < len(cards) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x0s[idx] + w + 0.01, y + h / 2),
                    (x0s[idx + 1] - 0.01, y + h / 2),
                    arrowstyle="-|>",
                    mutation_scale=15,
                    lw=1.7,
                    color="#94a3b8",
                    zorder=1,
                )
            )

    thumbnails = [
        MIDTERM_ASSETS / "slide01/2_external_global_view.png",
        MIDTERM_ASSETS / "slide02/camera_system_pipeline.png",
        MIDTERM_ASSETS / "slide05/gp_pipeline_summary.png",
        MIDTERM_ASSETS / "slide06/efe_rollout_compare.png",
        MIDTERM_ASSETS / "slide09/campaign_summary.png",
    ]
    thumb_positions = [(0.05, 0.13), (0.24, 0.13), (0.43, 0.13), (0.62, 0.13), (0.81, 0.13)]
    thumb_titles = ["real setup", "camera signal", "reliability field", "route choice", "campaign result"]
    for path, (x, ty), title in zip(thumbnails, thumb_positions, thumb_titles):
        ax.add_patch(Rectangle((x, ty), 0.14, 0.18, facecolor="white", edgecolor="#d9e1ec", lw=1.1, zorder=1))
        img = maybe_image(path)
        if img is not None:
            ax.imshow(img, extent=(x + 0.006, x + 0.134, ty + 0.037, ty + 0.168), zorder=2, aspect="auto")
        ax.text(x + 0.07, ty + 0.015, title, ha="center", va="bottom", fontsize=9.4, color=MUTED, zorder=3)

    ax.text(
        0.50,
        0.82,
        "Core claim: the GP does not learn R online and does not reward visibility directly. "
        "It predicts trust, and trust scales the planner's observation covariance R_plan.",
        ha="center",
        va="center",
        fontsize=12.2,
        color=DARK,
        fontweight="bold",
    )
    save(fig, REPO / "docs/media/contribution_map.png")


def make_yolo_bottom_center() -> None:
    img_path = REPO / "paper_artifacts/figures/inputs/loc_pathway_frame_v7b.jpg"
    img = maybe_image(img_path)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), gridspec_kw={"width_ratios": [1.65, 1.0]})
    fig.subplots_adjust(wspace=0.12)
    ax = axes[0]
    ax.set_title("Runtime camera observation")
    if img is not None:
        ax.imshow(img)
        h, w = img.shape[:2]
    else:
        h, w = 720, 1280
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
    u, v = 159.2, 345.8
    box_w, box_h = 92.0, 108.0
    x0 = max(0.0, u - box_w / 2)
    y0 = max(0.0, v - box_h)
    ax.add_patch(Rectangle((x0, y0), box_w, box_h, fill=False, edgecolor=TUE_RED, lw=2.6))
    ax.plot(u, v, marker="o", ms=10, color=TUE_BLUE, mec="white", mew=1.4)
    ax.annotate(
        "bottom-centre pixel",
        xy=(u, v),
        xytext=(u + 135, v - 70),
        arrowprops={"arrowstyle": "->", "lw": 2.0, "color": TUE_BLUE},
        fontsize=12,
        color=TUE_BLUE,
        fontweight="bold",
    )
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")

    ax = axes[1]
    ax.set_title("What the planner receives")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(Rectangle((0.18, 0.20), 0.44, 0.60, fill=False, edgecolor=TUE_RED, lw=2.8))
    ax.plot(0.40, 0.20, "o", ms=12, color=TUE_BLUE, mec="white", mew=1.4)
    ax.add_patch(FancyArrowPatch((0.40, 0.18), (0.40, 0.04), arrowstyle="-|>", mutation_scale=22, lw=2.0, color=TUE_BLUE))
    ax.text(0.40, 0.88, "YOLO box", ha="center", va="center", fontsize=13, color=TUE_RED, fontweight="bold")
    ax.text(0.40, 0.01, "selected (u, v)", ha="center", va="bottom", fontsize=12, color=TUE_BLUE, fontweight="bold")
    ax.text(
        0.03,
        0.48,
        "Only image x,y evidence is exported.\nHeading is handled downstream by\nodometry and belief propagation.",
        ha="left",
        va="center",
        fontsize=11,
        color=DARK,
        linespacing=1.35,
    )
    save(fig, REPO / "yolo/demos/images/bottom_centre_01.png")


def make_image_to_bev(gp: dict[str, np.ndarray | str]) -> None:
    img = maybe_image(REPO / "paper_artifacts/figures/inputs/loc_pathway_frame_v7b.jpg")
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.2), gridspec_kw={"width_ratios": [1.35, 1.0]})
    fig.subplots_adjust(wspace=0.18)
    ax = axes[0]
    ax.set_title("Image-space detection")
    if img is not None:
        ax.imshow(img)
        h, w = img.shape[:2]
    else:
        h, w = 720, 1280
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
    u, v = 159.2, 345.8
    ax.plot(u, v, marker="o", ms=10, color=TUE_RED, mec="white", mew=1.4)
    ax.annotate(
        "selected pixel",
        xy=(u, v),
        xytext=(u + 150, v - 80),
        arrowprops={"arrowstyle": "->", "lw": 1.9, "color": TUE_RED},
        fontsize=11.5,
        color=TUE_RED,
        fontweight="bold",
    )
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")

    ax = axes[1]
    xs = np.asarray(gp["xs"], dtype=float)
    ys = np.asarray(gp["ys"], dtype=float)
    ax.imshow(
        np.asarray(gp["p_plan"], dtype=float),
        origin="lower",
        extent=(xs[0], xs[-1], ys[0], ys[-1]),
        cmap="viridis",
        vmin=0.0,
        vmax=0.9,
        alpha=0.72,
        zorder=0,
    )
    draw_driveable(ax)
    draw_racks(ax, gp)
    world_point = np.array([-4.61, -1.67])
    ax.scatter([world_point[0]], [world_point[1]], s=130, color=TUE_RED, edgecolor="white", lw=1.4, zorder=9)
    ax.annotate(
        "projected BEV point",
        xy=world_point,
        xytext=(-3.95, -2.65),
        arrowprops={"arrowstyle": "->", "lw": 1.9, "color": TUE_RED},
        color=TUE_RED,
        fontsize=11.5,
        fontweight="bold",
    )
    style_map_axis(ax, "Ground-plane estimate")
    fig.text(
        0.50,
        0.02,
        "The camera measurement is a ground-contact point after homography and affine calibration, not a full robot pose.",
        ha="center",
        fontsize=11.5,
        color=DARK,
    )
    save(fig, REPO / "estimation/demos/images/image_to_bev_01.png")


def read_affine() -> tuple[np.ndarray, np.ndarray]:
    default = "0.996142,-0.002705,-0.002021,-0.002609,0.991001,0.066112"
    raw = default
    if CAMPAIGN_CONFIG.is_file():
        cfg = yaml.safe_load(CAMPAIGN_CONFIG.read_text(encoding="utf-8"))
        raw = str(cfg.get("bev_affine_calibration", default))
    vals = [float(item) for item in raw.split(",")]
    a = np.array([[vals[0], vals[1]], [vals[3], vals[4]]], dtype=float)
    b = np.array([vals[2], vals[5]], dtype=float)
    return a, b


def make_affine_calibration(gp: dict[str, np.ndarray | str]) -> None:
    a, b = read_affine()
    gx, gy = np.meshgrid(np.linspace(-4.9, 4.9, 8), np.linspace(-3.2, 3.8, 6))
    truth = np.column_stack([gx.ravel(), gy.ravel()])
    raw = np.linalg.solve(a, (truth - b).T).T
    corrected = raw @ a.T + b
    before = raw - truth
    after = corrected - truth

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.1), sharex=True, sharey=True)
    for ax, residual, title, color, scale in [
        (axes[0], before, "Raw homography residual", TUE_RED, 1.0),
        (axes[1], after, "After affine calibration", TUE_BLUE, 1.0),
    ]:
        ax.imshow(
            np.asarray(gp["p_plan"], dtype=float),
            origin="lower",
            extent=gp_extent(gp),
            cmap="Greys",
            alpha=0.22,
            zorder=0,
        )
        draw_driveable(ax, lw=1.0, alpha=0.45)
        draw_racks(ax, gp, alpha=0.65)
        ax.quiver(
            truth[:, 0],
            truth[:, 1],
            residual[:, 0],
            residual[:, 1],
            angles="xy",
            scale_units="xy",
            scale=scale,
            color=color,
            width=0.004,
            zorder=9,
        )
        ax.scatter(truth[:, 0], truth[:, 1], s=14, color=DARK, zorder=10)
        style_map_axis(ax, title)
    before_rmse = float(np.sqrt(np.mean(np.sum(before**2, axis=1))))
    after_rmse = float(np.sqrt(np.mean(np.sum(after**2, axis=1))))
    fig.text(
        0.50,
        0.02,
        f"Campaign affine map: x' = A x + b. Synthetic grid RMSE {before_rmse:.3f} m before, {after_rmse:.3f} m after.",
        ha="center",
        fontsize=11.5,
        color=DARK,
    )
    save(fig, REPO / "estimation/demos/images/affine_calibration_before_after.png")


def make_induced_covariance(gp: dict[str, np.ndarray | str]) -> None:
    p_plan = np.asarray(gp["p_plan"], dtype=float)
    std = covariance_std(p_plan)
    fig, axes = plt.subplots(1, 2, figsize=(13.7, 5.9))
    for ax, data, title, cmap, label, vmin, vmax in [
        (axes[0], p_plan, "GP planner trust", "viridis", "trust", 0.0, 0.9),
        (axes[1], std, "Induced R_plan standard deviation", "magma", "pixels", R_VISIBLE_UV, R_MISS_UV),
    ]:
        im = ax.imshow(data, origin="lower", extent=gp_extent(gp), cmap=cmap, vmin=vmin, vmax=vmax, zorder=0)
        draw_driveable(ax, lw=0.9, alpha=0.50)
        draw_racks(ax, gp)
        style_map_axis(ax, title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label=label)
    fig.text(
        0.50,
        0.02,
        "Precision blend: 1/R_plan = trust/R_visible + (1 - trust)/R_miss. The GP predicts trust; it does not learn R online.",
        ha="center",
        fontsize=11.3,
        color=DARK,
    )
    save(fig, REPO / "gp/demos/images/induced_covariance.png")


def make_r_plan_ellipses(gp: dict[str, np.ndarray | str]) -> None:
    p_plan = np.asarray(gp["p_plan"], dtype=float)
    std = covariance_std(p_plan)
    xs = np.asarray(gp["xs"], dtype=float)
    ys = np.asarray(gp["ys"], dtype=float)

    fig, ax = plt.subplots(figsize=(8.2, 7.4))
    im = ax.imshow(std, origin="lower", extent=gp_extent(gp), cmap="magma", vmin=R_VISIBLE_UV, vmax=R_MISS_UV, zorder=0)
    draw_driveable(ax, lw=1.0, alpha=0.50)
    draw_racks(ax, gp)
    samples = [(-4.6, -1.7), (-2.6, 2.8), (0.9, 0.25), (3.6, 3.6), (4.9, -3.0)]
    for x, y in samples:
        ix = int(np.clip(np.searchsorted(xs, x), 0, len(xs) - 1))
        iy = int(np.clip(np.searchsorted(ys, y), 0, len(ys) - 1))
        value = float(std[iy, ix])
        radius = 0.08 + 0.34 * (value - R_VISIBLE_UV) / (R_MISS_UV - R_VISIBLE_UV)
        ax.add_patch(Ellipse((x, y), radius, radius, facecolor="white", edgecolor=TUE_BLUE, lw=2.0, alpha=0.92, zorder=10))
        ax.text(x, y - radius * 0.85 - 0.10, f"{value:.0f}px", ha="center", va="top", fontsize=9.5, color=DARK, zorder=11)
    style_map_axis(ax, "R_plan across the warehouse")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="std in image pixels")
    fig.text(
        0.50,
        0.02,
        "In the locked model R_plan is diagonal with equal u/v variance, so the glyphs are circular ellipses whose size changes with trust.",
        ha="center",
        fontsize=10.7,
        color=DARK,
    )
    save(fig, REPO / "gp/demos/images/r_plan_map_and_ellipses.png")


def make_paired_route_choice(gp: dict[str, np.ndarray | str]) -> None:
    c1 = read_csv_points(WEST_PAIR / "C1/experiment.csv", ["gt_x", "truth_x"], ["gt_y", "truth_y"])
    c2 = read_csv_points(WEST_PAIR / "C2/experiment.csv", ["gt_x", "truth_x"], ["gt_y", "truth_y"])

    fig, ax = plt.subplots(figsize=(9.3, 7.3))
    im = ax.imshow(
        np.asarray(gp["p_plan"], dtype=float),
        origin="lower",
        extent=gp_extent(gp),
        cmap="viridis",
        vmin=0.0,
        vmax=0.9,
        alpha=0.86,
        zorder=0,
    )
    draw_driveable(ax, lw=1.2, alpha=0.72)
    draw_racks(ax, gp)
    if len(c1):
        ax.plot(c1[:, 0], c1[:, 1], color=C1, lw=2.8, label="C1 constant R", zorder=9)
        ax.scatter(c1[-1, 0], c1[-1, 1], marker="x", s=150, lw=3.2, color=C1, zorder=11)
    if len(c2):
        ax.plot(c2[:, 0], c2[:, 1], color=C2, lw=2.8, label="C2 GP-scaled R_plan", zorder=9)
        ax.scatter(c2[-1, 0], c2[-1, 1], marker="*", s=170, color=C2, edgecolor="white", lw=0.8, zorder=11)
    if len(c1):
        ax.scatter(c1[0, 0], c1[0, 1], s=85, color="#16a34a", edgecolor="white", lw=1.0, zorder=12, label="start")
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.95)
    style_map_axis(ax, "Matched west-route pair: covariance changes route behavior")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="planner trust")
    fig.text(
        0.50,
        0.02,
        "Same task, seed, map, and tracker. C2 avoids the camera-poor shortcut because low trust inflates future observation covariance.",
        ha="center",
        fontsize=10.8,
        color=DARK,
    )
    save(fig, REPO / "planning/demos/images/paired_route_choice.png")


def make_covariance_along_route(gp: dict[str, np.ndarray | str]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10.3, 7.2), sharex=True)
    for cond, color, label in [("C1", C1, "C1 constant R"), ("C2", C2, "C2 GP-scaled R_plan")]:
        csv_path = WEST_PAIR / cond / "experiment.csv"
        points = read_csv_points(csv_path, ["gt_x", "truth_x"], ["gt_y", "truth_y"])
        if len(points) == 0:
            continue
        dist = route_distance(points)
        r_std = read_series(csv_path, ["r_plan_u_std"])
        p_vis = read_series(csv_path, ["p_vis_plan_eff", "p_vis_plan"])
        if len(r_std) != len(dist) or not np.isfinite(r_std).any():
            if cond == "C1":
                r_std = np.full(len(dist), R_VISIBLE_UV, dtype=float)
            else:
                r_std = covariance_std(query_gp_trust(gp, points))
        if len(p_vis) != len(dist) or not np.isfinite(p_vis).any():
            p_vis = query_gp_trust(gp, points)
        n = min(len(dist), len(r_std), len(p_vis))
        dist = dist[:n]
        r_std = r_std[:n]
        p_vis = p_vis[:n]
        good_r = np.isfinite(dist) & np.isfinite(r_std)
        good_p = np.isfinite(dist) & np.isfinite(p_vis)
        axes[0].plot(dist[good_r], r_std[good_r], color=color, lw=2.2, label=label)
        axes[1].plot(dist[good_p], p_vis[good_p], color=color, lw=2.2, label=label)
    axes[0].set_ylabel("R_plan std [px]")
    axes[0].set_title("Observation covariance seen during the same route")
    axes[0].grid(True, color=GRID, lw=0.45)
    axes[0].legend(loc="upper left")
    axes[1].set_ylabel("GP trust along path")
    axes[1].set_xlabel("executed distance [m]")
    axes[1].set_ylim(-0.04, 1.04)
    axes[1].grid(True, color=GRID, lw=0.45)
    fig.text(
        0.50,
        0.02,
        "Top: C1 keeps R constant; C2 scales R_plan from GP trust. Bottom: background GP trust along each path, shown for context.",
        ha="center",
        fontsize=10.8,
        color=DARK,
    )
    save(fig, REPO / "planning/demos/images/covariance_along_route.png")


def outcome_counts() -> tuple[dict[str, dict[str, int]], dict[str, dict[str, dict[str, int]]]]:
    campaign = load_full_campaign()
    counts = {"C1": {"clean": 0, "collision": 0, "near": 0, "invalid": 0}, "C2": {"clean": 0, "collision": 0, "near": 0, "invalid": 0}}
    per_task: dict[str, dict[str, dict[str, int]]] = {}
    for entry in campaign.values():
        cond = str(entry.get("condition", ""))
        task = str(entry.get("task", ""))
        if cond not in counts:
            continue
        per_task.setdefault(task, {"C1": {"clean": 0, "collision": 0}, "C2": {"clean": 0, "collision": 0}})
        outcome = str(entry.get("outcome", ""))
        completion = str(entry.get("completion_reason", ""))
        crashed = bool(entry.get("crashed", False)) or outcome == "collision" or completion == "collision"
        reached = bool(entry.get("goal_reached", False)) or outcome == "goal_reached" or completion.startswith("goal_reached")
        invalid = "invalid" in outcome or "invalid" in completion
        near = "near" in outcome or "near" in completion
        if reached and not crashed and not invalid:
            counts[cond]["clean"] += 1
            per_task[task][cond]["clean"] += 1
        elif crashed:
            counts[cond]["collision"] += 1
            per_task[task][cond]["collision"] += 1
        elif near:
            counts[cond]["near"] += 1
        elif invalid:
            counts[cond]["invalid"] += 1
        else:
            counts[cond]["invalid"] += 1
    return counts, per_task


def make_outcome_counts() -> None:
    counts, _ = outcome_counts()
    labels = ["clean", "collision", "near", "invalid"]
    colors = ["#16a34a", "#dc2626", "#f59e0b", "#64748b"]
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    x = np.arange(2)
    bottoms = np.zeros(2)
    for label, color in zip(labels, colors):
        vals = np.array([counts["C1"][label], counts["C2"][label]], dtype=float)
        ax.bar(x, vals, bottom=bottoms, width=0.58, color=color, label=label)
        for xi, bottom, val in zip(x, bottoms, vals):
            if val > 0:
                ax.text(xi, bottom + val / 2, f"{int(val)}", ha="center", va="center", color="white", fontweight="bold")
        bottoms += vals
    ax.set_xticks(x, ["C1 constant R", "C2 GP-scaled R_plan"])
    ax.set_ylabel("runs")
    ax.set_ylim(0, max(20, float(np.max(bottoms))) + 1.5)
    ax.set_title("Current honest campaign outcome counts")
    ax.legend(loc="upper right", frameon=False, ncol=4)
    ax.grid(True, axis="y", color=GRID, lw=0.45)
    fig.text(0.50, 0.02, "Current 40-run surface: four routes x five seeds x two conditions.", ha="center", fontsize=10.8, color=DARK)
    save(fig, REPO / "experiments/demos/images/outcome_counts_by_condition.png")


def make_campaign_table() -> None:
    counts, per_task = outcome_counts()
    tasks = [
        "route_apron_to_a3_mid",
        "route_apron_to_a2_mid",
        "route_west_to_a1_upper",
        "control_west_to_a1_low",
    ]
    rows = []
    for task in tasks:
        data = per_task.get(task, {"C1": {"clean": 0, "collision": 0}, "C2": {"clean": 0, "collision": 0}})
        rows.append(
            [
                task.replace("_", " "),
                f"{data['C1']['clean']}/5 clean, {data['C1']['collision']} coll.",
                f"{data['C2']['clean']}/5 clean, {data['C2']['collision']} coll.",
            ]
        )
    rows.append(["total", f"{counts['C1']['clean']}/20 clean, {counts['C1']['collision']} coll.", f"{counts['C2']['clean']}/20 clean, {counts['C2']['collision']} coll."])

    fig, ax = plt.subplots(figsize=(11.8, 4.8))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["route", "C1 constant R", "C2 GP-scaled R_plan"],
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.44, 0.28, 0.28],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d9e1ec")
        if row == 0:
            cell.set_facecolor(DARK)
            cell.set_text_props(color="white", fontweight="bold")
        elif row == len(rows):
            cell.set_facecolor("#e8f2fb")
            cell.set_text_props(fontweight="bold")
        elif col == 2:
            cell.set_facecolor("#eef7ff")
        elif col == 1:
            cell.set_facecolor("#fff5f5")
        else:
            cell.set_facecolor("white")
    ax.set_title("Current campaign result table", fontsize=16, fontweight="bold", color=DARK, pad=18)
    save(fig, REPO / "experiments/demos/images/campaign_result_table.png")


def main() -> int:
    set_style()
    gp = load_gp()
    make_contribution_map()
    make_yolo_bottom_center()
    make_image_to_bev(gp)
    make_affine_calibration(gp)
    make_induced_covariance(gp)
    make_r_plan_ellipses(gp)
    make_paired_route_choice(gp)
    make_covariance_along_route(gp)
    make_outcome_counts()
    make_campaign_table()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
