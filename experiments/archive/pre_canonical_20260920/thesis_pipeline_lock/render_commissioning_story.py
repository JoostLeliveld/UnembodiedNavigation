#!/usr/bin/env python3
"""Render the thesis commissioning story from spatially held-out artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/thesis_commissioning_story_mpl")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import numpy as np

from reliability.commissioned_availability import CommissionedAvailabilityModel
from reliability.projection import camera_model_from_world
from unav_common.occlusion_geometry import parse_collision_scene_from_world


REPO = Path(__file__).resolve().parents[2]
CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")
CAMERA_INCLUDES = dict(zip(CAMERAS, (
    "external_camera", "external_camera_b", "external_camera_c",
    "external_camera_d", "external_camera_e",
)))
WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf"
CAPTURE = REPO / "experiments/thesis_pipeline_lock/generated/camera_capture_poses_v3.json"
ADMISSIONS = REPO / (
    "logs/thesis_final_pipeline_v1/stage06_detector_gate/"
    "gate_selection_v1/admission_records.csv"
)
MLP_RUN = REPO / "logs/thesis_final_pipeline_v1/stage07_correction_covariance/run_mlp_v1"
AVAILABILITY_RUN = REPO / "logs/thesis_final_pipeline_v1/stage08_availability/run_v2"
COLORS = {
    "hull": "#657789",
    "mlp": "#2a78d6",
    "uncalibrated": "#d79c36",
    "calibrated": "#238e83",
    "accent": "#8d67a7",
}
CHI2_95 = 5.99146454711


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig, output: Path, stem: str) -> None:
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def scene_prisms():
    return parse_collision_scene_from_world(
        str(WORLD),
        model_names=("warehouse_shell", "warehouse_v2_occluders"),
        robot_z_range=(0.0, 0.55),
    ).prisms


def decorate_map(ax, prisms, *, title: str | None = None) -> None:
    for prism in prisms:
        ax.add_patch(Rectangle(
            (prism.xmin, prism.ymin), prism.xmax - prism.xmin,
            prism.ymax - prism.ymin, facecolor="#d9dfe2", edgecolor="#a7b0b5",
            linewidth=0.25, alpha=0.75, zorder=0,
        ))
    ax.set(xlim=(-12.5, 12.5), ylim=(-10.5, 10.5), aspect="equal",
           xlabel="World x [m]", ylabel="World y [m]")
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.10, linewidth=0.4)


def bias_field(output: Path, prisms) -> dict:
    data = rows(MLP_RUN / "oof_predictions.csv")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in data:
        grouped[(row["camera_id"], row["position_id"])].append(row)

    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.65), constrained_layout=True)
    axes = axes.ravel()
    magnification = 10.0
    for index, camera in enumerate(CAMERAS):
        ax = axes[index]
        decorate_map(ax, prisms, title=f"Camera {camera[-1]}")
        aggregates = []
        for (camera_id, _position), members in grouped.items():
            if camera_id != camera:
                continue
            truth = np.mean([[float(r["truth_x"]), float(r["truth_y"])] for r in members], axis=0)
            hull = np.mean([[float(r["C1_visual_hull_x"]), float(r["C1_visual_hull_y"])] for r in members], axis=0)
            mlp = np.mean([[float(r["C3_mlp_residual_x"]), float(r["C3_mlp_residual_y"])] for r in members], axis=0)
            aggregates.append((truth, mlp - hull))
        xy = np.asarray([value[0] for value in aggregates])
        correction = np.asarray([value[1] for value in aggregates])
        magnitude_cm = 100.0 * np.linalg.norm(correction, axis=1)
        arrows = ax.quiver(
            xy[:, 0], xy[:, 1], magnification * correction[:, 0],
            magnification * correction[:, 1], magnitude_cm, cmap="viridis",
            clim=(0, 12), angles="xy", scale_units="xy", scale=1,
            width=0.005, headwidth=3.0, zorder=3,
        )
        camera_model = camera_model_from_world(WORLD, include_name=CAMERA_INCLUDES[camera])
        ax.scatter(camera_model.cam_pos[0], camera_model.cam_pos[1], marker="*",
                   s=70, color="#22282c", edgecolor="white", linewidth=0.5, zorder=4)
        if index == 4:
            fig.colorbar(arrows, ax=axes[:5].tolist(), shrink=0.72, pad=0.015,
                         label="Mean learned correction magnitude [cm]")

    ax = axes[5]
    for key, label, color in (
        ("C1_visual_hull_error_m", "Analytic hull", COLORS["hull"]),
        ("C3_mlp_residual_error_m", "Hull + MLP", COLORS["mlp"]),
    ):
        values = np.sort(np.asarray([100.0 * float(row[key]) for row in data]))
        ax.plot(values, np.arange(1, len(values) + 1) / len(values),
                lw=1.6, color=color, label=label)
    ax.set(xlabel="Spatial OOF position error [cm]", ylabel="Empirical CDF",
           xlim=(0, 18), ylim=(0, 1.01), title="Held-out correction effect")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, loc="lower right")
    ax.text(
        0.04, 0.95,
        "Arrows are 10x longer\nthan their world magnitude.",
        transform=ax.transAxes, ha="left", va="top", fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    fig.suptitle("Commissioned MLP correction field and spatially held-out effect", fontsize=11)
    save(fig, output, "stage07_mlp_bias_field")
    report = json.loads((MLP_RUN / "correction_report.json").read_text())
    return {
        "observations": len(data),
        "hull": report["candidate_reports"]["C1_visual_hull"]["pooled"],
        "mlp": report["candidate_reports"]["C3_mlp_residual"]["pooled"],
        "arrow_magnification": magnification,
    }


def r_learning(output: Path, prisms) -> dict:
    oof = rows(MLP_RUN / "oof_predictions.csv")
    admissions = {
        (row["pose_id"], row["camera_id"]): row for row in rows(ADMISSIONS)
        if row["admitted"] == "1"
    }
    report = json.loads((MLP_RUN / "covariance_report.json").read_text())
    areas = np.asarray([
        math.pi * CHI2_95 * math.sqrt(max(
            float(row["selected_R_xx_m2"]) * float(row["selected_R_yy_m2"])
            - float(row["selected_R_xy_m2"]) ** 2, 0.0
        )) for row in oof
    ])
    widths = np.asarray([
        float(admissions[(row["pose_id"], row["camera_id"])]["best_box_x1"])
        - float(admissions[(row["pose_id"], row["camera_id"])]["best_box_x0"])
        for row in oof
    ])
    position_area: dict[str, list[float]] = defaultdict(list)
    position_xy = {}
    for row, area in zip(oof, areas):
        position_area[row["position_id"]].append(float(area))
        position_xy[row["position_id"]] = (float(row["truth_x"]), float(row["truth_y"]))

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.35), constrained_layout=True)
    ax = axes[0]
    decorate_map(ax, prisms, title="Learned residual-uncertainty field")
    ids = sorted(position_area)
    xy = np.asarray([position_xy[key] for key in ids])
    equivalent_radius_cm = np.asarray([
        100.0 * math.sqrt(np.mean(position_area[key]) / math.pi) for key in ids
    ])
    scatter = ax.scatter(xy[:, 0], xy[:, 1], c=equivalent_radius_cm,
                         cmap="magma", s=19, edgecolor="white", linewidth=0.25, zorder=3)
    fig.colorbar(scatter, ax=ax, pad=0.02, label="Mean 95% ellipse-equivalent radius [cm]")

    ax = axes[1]
    nominal = np.asarray([50, 90, 95, 99], dtype=float)
    for candidate, label, color, marker in (
        ("R2", "Width-conditioned", COLORS["uncalibrated"], "o"),
        ("R2C", "Nested calibrated", COLORS["calibrated"], "s"),
    ):
        empirical = 100.0 * np.asarray([
            report["candidates"][candidate]["containment"][str(int(level))]
            for level in nominal
        ])
        ax.plot(nominal, empirical, marker=marker, color=color, lw=1.4, label=label)
    ax.plot([45, 100], [45, 100], "k--", lw=0.9, label="Ideal")
    ax.set(xlabel="Nominal ellipse containment [%]",
           ylabel="Spatial OOF containment [%]", xlim=(47, 100), ylim=(47, 100),
           title="What calibration changes")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=7.5)

    ax = axes[2]
    ax.scatter(widths, 1e4 * areas, s=6, alpha=0.10, color=COLORS["mlp"], rasterized=True)
    edges = np.quantile(widths, np.linspace(0, 1, 13))
    centres, medians = [], []
    for low, high in zip(edges[:-1], edges[1:]):
        use = (widths >= low) & (widths <= high)
        if np.any(use):
            centres.append(float(np.median(widths[use])))
            medians.append(float(np.median(1e4 * areas[use])))
    ax.plot(centres, medians, "o-", color="#20272c", ms=3.5, lw=1.2,
            label="12-bin median")
    ax.set(xlabel="Detected robot width [px]", ylabel="95% ellipse area [cm$^2$]",
           title="$R$ is conditional on visible scale")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    save(fig, output, "stage07_R_learning")
    return {
        "selected": report["selected"],
        "data_driven_selection": report.get("data_driven_selection"),
        "calibration_scale": report["deployment_R2C_scale"],
        "selected_scores": report["candidates"][report["selected"]],
    }


def availability_field(output: Path, prisms) -> dict:
    model_path = AVAILABILITY_RUN / "availability_model.json"
    model = CommissionedAvailabilityModel(model_path, WORLD)
    cameras = {
        camera: camera_model_from_world(WORLD, include_name=CAMERA_INCLUDES[camera])
        for camera in CAMERAS
    }
    capture = json.loads(CAPTURE.read_text())
    xs0 = [float(pose["x"]) for pose in capture["poses"]]
    ys0 = [float(pose["y"]) for pose in capture["poses"]]
    xs = np.linspace(min(xs0), max(xs0), 96)
    ys = np.linspace(min(ys0), max(ys0), 80)
    headings = np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
    fields = np.empty((len(CAMERAS), len(ys), len(xs)), dtype=float)
    for ci, camera_id in enumerate(CAMERAS):
        camera = cameras[camera_id]
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                fields[ci, iy, ix] = np.mean([
                    model.probability(camera_id, float(x), float(y), float(yaw), camera)
                    for yaw in headings
                ])
    network = np.max(fields, axis=0)
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.65), constrained_layout=True)
    axes = axes.ravel()
    image = None
    for index, camera in enumerate(CAMERAS):
        ax = axes[index]
        decorate_map(ax, prisms, title=f"Camera {camera[-1]}")
        image = ax.imshow(fields[index], origin="lower",
                          extent=(xs[0], xs[-1], ys[0], ys[-1]),
                          vmin=0, vmax=1, cmap="viridis", alpha=0.88, zorder=1)
        ax.contour(xs, ys, fields[index], levels=[0.5], colors="white",
                   linewidths=0.8, zorder=2)
        ax.scatter(cameras[camera].cam_pos[0], cameras[camera].cam_pos[1],
                   marker="*", s=65, color="#20272c", edgecolor="white",
                   linewidth=0.5, zorder=4)
    ax = axes[5]
    decorate_map(ax, prisms, title="Camera network: best available view")
    image = ax.imshow(network, origin="lower", extent=(xs[0], xs[-1], ys[0], ys[-1]),
                      vmin=0, vmax=1, cmap="viridis", alpha=0.88, zorder=1)
    ax.contour(xs, ys, network, levels=[0.5], colors="white", linewidths=1.0, zorder=2)
    fig.colorbar(image, ax=axes.tolist(), shrink=0.76, pad=0.012,
                 label="Heading-averaged usable-observation probability")
    fig.suptitle("Commissioned availability field $q_c(x,y,\psi)$", fontsize=11)
    save(fig, output, "stage08_availability_field")
    return {
        "grid": [len(xs), len(ys)],
        "heading_samples": len(headings),
        "network_fraction_above_0_5": float(np.mean(network >= 0.5)),
    }


def availability_validation(output: Path) -> dict:
    oof = rows(AVAILABILITY_RUN / "oof_predictions.csv")
    curve = rows(AVAILABILITY_RUN / "commissioning_size_curve.csv")
    report = json.loads((AVAILABILITY_RUN / "availability_report.json").read_text())
    truth = np.asarray([float(row["available"]) for row in oof])
    probability = np.asarray([float(row["Q1_geometry_logistic"]) for row in oof])
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0), constrained_layout=True)
    edges = np.linspace(0, 1, 11)
    for low, high in zip(edges[:-1], edges[1:]):
        use = (probability >= low) & (probability < high if high < 1 else probability <= high)
        if np.any(use):
            axes[0].scatter(np.mean(probability[use]), np.mean(truth[use]),
                            s=14 + 2.0 * math.sqrt(np.sum(use)), color=COLORS["calibrated"])
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.9)
    axes[0].set(xlabel="Predicted availability", ylabel="Observed usable fraction",
                xlim=(-0.02, 1.02), ylim=(-0.02, 1.02),
                title="Spatially held-out calibration")
    sizes = np.asarray([float(row["requested_positions"]) for row in curve])
    brier = np.asarray([float(row["block_macro_brier"]) for row in curve])
    axes[1].plot(sizes, brier, "o-", color=COLORS["calibrated"], lw=1.4, ms=4)
    axes[1].set(xlabel="Commissioning positions", ylabel="Block-macro Brier score",
                xticks=sizes, title="How much commissioning data is needed?")
    for ax in axes:
        ax.grid(alpha=0.18)
    save(fig, output, "stage08_availability_validation")
    selected = report["candidates"][report["selected"]]
    return {
        "selected": report["selected"],
        "block_macro_brier": selected["block_macro_brier"],
        "ece_10": selected["pooled_ece_10"],
        "calibration": selected["calibration"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    expected = [
        output / f"{stem}.{suffix}"
        for stem in (
            "stage07_mlp_bias_field", "stage07_R_learning",
            "stage08_availability_field", "stage08_availability_validation",
        )
        for suffix in ("pdf", "svg", "png")
    ]
    if any(path.exists() for path in expected):
        raise FileExistsError("refusing to overwrite commissioning-story figures")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.2,
        "pdf.fonttype": 42, "svg.fonttype": "none", "axes.linewidth": 0.8,
    })
    prisms = scene_prisms()
    report = {
        "schema": "thesis_commissioning_story_figures.v1",
        "status": "complete",
        "claim_boundary": "commissioning-fit spatial OOF only; no final-audit or navigation data",
        "bias_field": bias_field(output, prisms),
        "R_learning": r_learning(output, prisms),
        "availability_field": availability_field(output, prisms),
        "availability_validation": availability_validation(output),
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
