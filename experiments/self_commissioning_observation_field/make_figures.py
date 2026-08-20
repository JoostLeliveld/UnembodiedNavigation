#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-self-commissioning-field")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

import field_common as C
import planner_ablation as P


def availability_apparatus():
    directory = C.REPO / "experiments/availability_paper"
    name = "availability_common_for_self_commissioning_figure"
    spec = importlib.util.spec_from_file_location(name, directory / "common.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.build_apparatus()


def warehouse_outline(ax, driveable: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> None:
    ax.contour(xs, ys, driveable.astype(float), levels=[0.5], colors="#30343b", linewidths=0.55)
    ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(ys[0], ys[-1]); ax.set_aspect("equal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")


def covariance_ellipse(ax, centre: np.ndarray, covariance: np.ndarray, **kwargs) -> None:
    values, vectors = np.linalg.eigh(P.nearest_psd(covariance))
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    angle = math.degrees(math.atan2(vectors[1, 0], vectors[0, 0]))
    scale = math.sqrt(5.9914645471)
    ax.add_patch(Ellipse(
        centre,
        2.0 * scale * math.sqrt(values[0]),
        2.0 * scale * math.sqrt(values[1]),
        angle=angle,
        fill=False,
        **kwargs,
    ))


def camera_uncertainty_map(field: P.ObservationField, camera: str, xs: np.ndarray, ys: np.ndarray, stride: int = 4):
    xq, yq = xs[::stride], ys[::stride]
    sigma = np.full((len(yq), len(xq)), np.nan)
    for iy, y in enumerate(yq):
        for ix, x in enumerate(xq):
            covariance = field.ground_covariance(camera, np.asarray([x, y]), "full_field")
            if covariance is not None and np.isfinite(covariance).all():
                sigma[iy, ix] = math.sqrt(max(np.linalg.eigvalsh(covariance)))
    return xq, yq, sigma


def make_per_camera_fields(out: Path, p: dict[str, np.ndarray], field: P.ObservationField, driveable: np.ndarray) -> None:
    xs, ys = np.asarray(p["xs"], float), np.asarray(p["ys"], float)
    fig, axes = plt.subplots(2, 4, figsize=(14.2, 6.0), constrained_layout=True)
    probability_image = uncertainty_image = None
    for camera_index, camera in enumerate(C.CAMERAS):
        ax = axes[0, camera_index]
        probability_image = ax.imshow(
            p[f"P_{camera}_map"], origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]],
            cmap="viridis", vmin=0.0, vmax=1.0, aspect="equal",
        )
        warehouse_outline(ax, driveable, xs, ys)
        ax.set_title(f"{camera[-1]}: $p_{{use}}(x,c)$")
        if camera_index:
            ax.set_ylabel("")

        ax = axes[1, camera_index]
        xq, yq, sigma = camera_uncertainty_map(field, camera, xs, ys)
        uncertainty_image = ax.imshow(
            sigma, origin="lower", extent=[xq[0], xq[-1], yq[0], yq[-1]],
            cmap="magma", vmin=0.0, vmax=0.20, aspect="equal",
        )
        warehouse_outline(ax, driveable, xs, ys)
        ax.set_title(f"{camera[-1]}: induced ground $\\sigma_{{max}}$")
        if camera_index:
            ax.set_ylabel("")
        # Sparse true metric 95% ellipses. No visual multiplier is applied.
        for x in np.linspace(xs[18], xs[-19], 5):
            for y in np.linspace(ys[16], ys[-17], 4):
                point = np.asarray([x, y])
                covariance = field.ground_covariance(camera, point, "full_field")
                if covariance is not None and np.isfinite(covariance).all():
                    covariance_ellipse(ax, point, covariance, edgecolor="white", linewidth=0.55, alpha=0.85)
    fig.colorbar(probability_image, ax=axes[0, :], shrink=0.80, label="$p_{use}$")
    fig.colorbar(uncertainty_image, ax=axes[1, :], shrink=0.80, label="major-axis $1\\sigma$ [m]")
    fig.suptitle(
        "Self-commissioned four-camera observation field — bottom-centre detector (95% ellipses at true scale)",
        fontsize=12,
    )
    for extension in ("png", "pdf"):
        fig.savefig(out / f"per_camera_p_and_R_fields.{extension}", dpi=220)
    plt.close(fig)


def route_row(ablation: dict) -> dict:
    return max(ablation["tasks"], key=lambda item: item["p_to_full_route_separation_m"])


def make_route_and_gate(out: Path, p: dict[str, np.ndarray], driveable: np.ndarray, summary: dict, ablation: dict) -> None:
    xs, ys = np.asarray(p["xs"], float), np.asarray(p["ys"], float)
    fused = C.fused_p(p)
    row = route_row(ablation)
    p_route = row["p_only_selected"]
    full_route = row["full_field_selected"]
    p_path, full_path = np.asarray(p_route["path"], float), np.asarray(full_route["path"], float)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), constrained_layout=True)
    image = axes[0].imshow(
        fused, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]],
        cmap="viridis", vmin=0.0, vmax=1.0, aspect="equal",
    )
    warehouse_outline(axes[0], driveable, xs, ys)
    axes[0].plot(p_path[:, 0], p_path[:, 1], "--", color="#f4a261", linewidth=2.0, label="$p_{use}$ only")
    axes[0].plot(full_path[:, 0], full_path[:, 1], "-", color="#e63946", linewidth=2.2, label="complete field")
    axes[0].scatter(p_path[0, 0], p_path[0, 1], s=28, color="black", zorder=5)
    axes[0].scatter(p_path[-1, 0], p_path[-1, 1], s=48, color="black", marker="*", zorder=5)
    axes[0].legend(frameon=True, fontsize=8)
    axes[0].set_title(f"(a) route decision: {row['task']}")
    fig.colorbar(image, ax=axes[0], fraction=0.046, label="four-camera $p_{use}$")

    ax = axes[1]
    for route, color, label in ((p_route, "#f4a261", "$p_{use}$-only route"), (full_route, "#e63946", "complete-field route")):
        trace = np.asarray(route["full_field_belief"]["trace_profile_m2"], float)
        distance = np.arange(len(trace)) * C.SPEED_MPS * C.DT_S
        ax.plot(distance, 100.0 * np.sqrt(trace / 2.0), color=color, linewidth=2.0, label=label)
    ax.set(xlabel="distance along route [m]", ylabel="RMS belief sigma [cm]", title="(b) exact 16-subset belief")
    ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    models = ["per-mode NIW", "spatial Bayesian field"]
    nll = [summary["models"][name]["mean_nll"] for name in ("constant", "spatial")]
    coverage = [100.0 * summary["models"][name]["coverage_95"] for name in ("constant", "spatial")]
    x = np.arange(2)
    bars = ax.bar(x - 0.18, nll, width=0.36, color="#457b9d", label="held-out NLL")
    ax.set_xticks(x, models, rotation=10)
    ax.set_ylabel("Gaussian NLL [nat]", color="#1d3557")
    ax.tick_params(axis="y", labelcolor="#1d3557"); ax.grid(axis="y", alpha=0.20)
    twin = ax.twinx()
    twin.bar(x + 0.18, coverage, width=0.36, color="#2a9d8f", label="95% coverage")
    twin.axhline(95.0, color="black", linestyle="--", linewidth=1.0)
    twin.set_ylabel("empirical 95% coverage [%]", color="#206a5d"); twin.tick_params(axis="y", labelcolor="#206a5d")
    ax.set_title(f"(c) spatial gate: {summary['spatial_candidate']['gate']}")
    ax.legend([bars, twin.patches[0]], ["held-out NLL", "95% coverage"], frameon=False, fontsize=8, loc="lower left")

    fig.suptitle(
        "A3 geometry prior + learned availability, bias and conditional covariance",
        fontsize=12,
    )
    for extension in ("png", "pdf"):
        fig.savefig(out / f"planning_ablation_and_spatial_gate.{extension}", dpi=220)
    plt.close(fig)


def main() -> None:
    summary_path = C.OUT / "commissioned/summary.json"
    ablation_path = C.OUT / "planning/ablation.json"
    if not summary_path.is_file() or not ablation_path.is_file():
        raise RuntimeError("run commission_field.py and planner_ablation.py first")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    ablation = json.loads(ablation_path.read_text(encoding="utf-8"))
    p = C.load_p(); field = P.ObservationField(); apparatus = availability_apparatus()
    out = C.OUT / "figures"; out.mkdir(parents=True, exist_ok=True)
    make_per_camera_fields(out, p, field, np.asarray(apparatus.driveable, bool))
    make_route_and_gate(out, p, np.asarray(apparatus.driveable, bool), summary, ablation)

    meeting = C.REPO / "logs/studies/availability_paper/figures"
    meeting.mkdir(parents=True, exist_ok=True)
    copies = {
        "per_camera_p_and_R_fields.png": "15_self_commissioned_per_camera_p_and_R.png",
        "planning_ablation_and_spatial_gate.png": "16_self_commissioned_planning_ablation.png",
    }
    for source, target in copies.items():
        shutil.copyfile(out / source, meeting / target)
    provenance = {
        "inputs": [str(summary_path.relative_to(C.REPO)), str(ablation_path.relative_to(C.REPO))],
        "selected_model": summary["selected_model"],
        "ellipse_scale": "true metric 95% ellipse; no display multiplier",
        "meeting_copies": copies,
    }
    C.write_json(out / "provenance.json", provenance)
    print(out)


if __name__ == "__main__":
    main()
