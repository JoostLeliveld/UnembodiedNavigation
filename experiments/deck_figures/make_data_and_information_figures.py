#!/usr/bin/env python3
"""Build the final data-partition and expected-information thesis figures."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src/planning"), str(ROOT / "src/unav_common")]
from planning.core.camera_network import CameraNetworkModel  # noqa: E402
from style import layout  # noqa: E402
PAPER_FIGURES = ROOT.parent / "papers" / "Thesis" / "figures"
POSES = ROOT / "logs/thesis_final_pipeline_v1/recapture_v5/capture_poses_v5.json"
PLANNING_DIR = ROOT / "logs/thesis_final_pipeline_v1/planning_precision"
PLANNING_ARTIFACT = PLANNING_DIR / "m2_planning_precision.npz"
PLANNING_MODELS = (
    ("m0_planning_precision.npz", r"(a) global $R_0$"),
    ("m1_planning_precision.npz", r"(b) camera-specific $R_1$"),
    ("m2_planning_precision.npz", r"(c) spatial $R_2$"),
)

INK = "#1d2530"
MUTED = "#68727d"
ROLE_COLOURS = {
    "D_mu": "#2f6f9f",
    "D_R": "#dd8a2e",
    "D_dev": "#55a868",
    "final_audit": "#9a66ad",
}
ROLE_LABELS = {
    "D_mu": r"$D_\mu$: fit correction",
    "D_R": r"$D_R$: fit covariance",
    "D_dev": r"$D_{\mathrm{dev}}$: select parameters",
    "final_audit": r"$D_{\mathrm{eval}}$: held-out evaluation",
}
CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")
def load_positions() -> list[dict]:
    poses = json.loads(POSES.read_text(encoding="utf-8"))
    unique: dict[int, dict] = {}
    for pose in poses:
        position_id = int(pose["position_id"])
        previous = unique.setdefault(position_id, pose)
        if previous["stratum"] != pose["stratum"]:
            raise RuntimeError(f"position {position_id} crosses data roles")
    if len(unique) != 2569:
        raise RuntimeError(f"expected 2569 positions, found {len(unique)}")
    return [unique[key] for key in sorted(unique)]


def data_role_figure() -> None:
    positions = load_positions()
    counts = {role: sum(p["stratum"] == role for p in positions) for role in ROLE_COLOURS}

    fig, ax = plt.subplots(figsize=(7.16, 4.15), constrained_layout=True)
    for role in ROLE_COLOURS:
        selected = [p for p in positions if p["stratum"] == role]
        ax.scatter([p["x"] for p in selected], [p["y"] for p in selected], s=7.5,
                   color=ROLE_COLOURS[role], alpha=0.82, linewidths=0,
                   label=f"{ROLE_LABELS[role]} ({counts[role]:,})")
    ax.set(xlim=(-11.4, 11.4), ylim=(-9.4, 9.4), aspect="equal",
           xlabel="east (m)", ylabel="north (m)", title="Position-level data partition")
    ax.grid(color="#d9dee3", lw=0.45, alpha=0.65)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2,
              frameon=False, fontsize=9, handletextpad=0.35, columnspacing=1.0)

    save(fig, "data_collection_roles")


def planner_field_construction_figure() -> None:
    """Compare the network information field of the three covariance models."""
    fields, xs, ys = [], None, None
    for filename, title in PLANNING_MODELS:
        with np.load(PLANNING_DIR / filename, allow_pickle=False) as archive:
            xs = np.asarray(archive["xs"], dtype=float)
            ys = np.asarray(archive["ys"], dtype=float)
            precision = np.asarray(archive["matched_precision_m2_inv"], dtype=float)
        network = precision.sum(axis=0)
        fields.append((0.5 * np.trace(network, axis1=-2, axis2=-1), title))

    # Cap at the 90th percentile of the spatial field.  The 99th percentile
    # pushes the two constant models into one indistinguishable colour; this
    # keeps R_0 and R_1 separable and saturates the brightest R_2 cells, which
    # the extended colourbar marks.
    vmax = float(np.percentile(fields[-1][0], 90))

    warehouse = layout()
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.30), constrained_layout=True,
                             sharex=True, sharey=True)
    image = None
    for ax, (values, title) in zip(axes, fields, strict=True):
        image = ax.pcolormesh(xs, ys, values, shading="nearest", cmap="viridis",
                              vmin=0.0, vmax=vmax, rasterized=True)
        for zone in warehouse.zones:
            ax.add_patch(Rectangle(
                (zone.xmin, zone.ymin), zone.xmax - zone.xmin,
                zone.ymax - zone.ymin, facecolor="#d9d7cf", edgecolor="#74736e",
                lw=0.4, alpha=0.30, zorder=4))
        for camera in warehouse.cameras:
            ax.plot(camera.x, camera.y, marker="o", ms=3.2, color="white",
                    mec="#111111", mew=0.6, zorder=7)
        ax.set(xlim=(-10.8, 10.8), ylim=(-8.8, 8.8), aspect="equal")
        ax.set_title(title, fontsize=8.4, pad=3.0)
        ax.set_xlabel("east (m)", fontsize=8.0)
        ax.tick_params(labelsize=7.2)
    axes[0].set_ylabel("north (m)", fontsize=8.0)
    bar = fig.colorbar(image, ax=list(axes), shrink=0.92, pad=0.012, aspect=18,
                       extend="max")
    bar.set_label(r"$\frac{1}{2}\mathrm{tr}\sum_i\Lambda_{m,i}(p)$ (m$^{-2}$)",
                  fontsize=8.0)
    bar.ax.tick_params(labelsize=7.2)
    save(fig, "planner_facing_field")


def planning_information_figure() -> None:
    intact_model = CameraNetworkModel(PLANNING_ARTIFACT)
    removed_model = CameraNetworkModel(
        PLANNING_ARTIFACT, cameras=tuple(c for c in CAMERAS if c != "camera_B"))
    # Query through the exact planner interface on a 0.6 m display grid.  The
    # deployed artifact remains 0.2 m; this only limits figure-generation cost.
    xs, ys = intact_model.xs[::3], intact_model.ys[::3]
    gx, gy = np.meshgrid(xs, ys)
    query = np.column_stack((gx.ravel(), gy.ravel()))

    prior = np.diag([0.10**2, 0.10**2, np.deg2rad(10.0)**2])
    intact_sd_cm = np.empty(len(query))
    removed_sd_cm = np.empty(len(query))
    for index, (x, y) in enumerate(query):
        state = np.asarray([x, y, 0.0])
        intact_posterior, _ = intact_model.expected_belief(
            state, prior, opportunities=1)
        removed_posterior, _ = removed_model.expected_belief(
            state, prior, opportunities=1)
        intact_sd_cm[index] = 100.0 * np.sqrt(0.5 * np.trace(intact_posterior[:2, :2]))
        removed_sd_cm[index] = 100.0 * np.sqrt(0.5 * np.trace(removed_posterior[:2, :2]))
    increase_cm = removed_sd_cm - intact_sd_cm

    posterior_min = float(np.percentile(np.concatenate((intact_sd_cm, removed_sd_cm)), 1))
    posterior_max = float(np.percentile(np.concatenate((intact_sd_cm, removed_sd_cm)), 99))

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.25), constrained_layout=True,
                             sharex=True, sharey=True)
    panels = [
        (intact_sd_cm, r"(a) Intact five-camera network", (posterior_min, posterior_max), "viridis_r",
         "equivalent position SD (cm)"),
        (removed_sd_cm, r"(b) Camera B removed", (posterior_min, posterior_max), "viridis_r",
         "equivalent position SD (cm)"),
        (increase_cm, r"(c) Increase caused by removal", None, "magma",
         "removed minus intact SD (cm)"),
    ]
    for ax, (values, title, scale, palette, colourbar_label) in zip(
            axes.ravel(), panels, strict=True):
        kwargs = {}
        if isinstance(scale, tuple):
            kwargs.update(vmin=scale[0], vmax=scale[1])
        elif scale is None:
            kwargs.update(vmin=0, vmax=float(np.percentile(values, 99)))
        image = ax.pcolormesh(xs, ys, values.reshape(gx.shape), shading="nearest",
                              cmap=palette, rasterized=True, **kwargs)
        ax.set(title=title, xlim=(-10.8, 10.8), ylim=(-8.8, 8.8), aspect="equal")
        ax.grid(color="white", alpha=0.18, lw=0.35)
        fig.colorbar(image, ax=ax, shrink=0.78, pad=0.015, label=colourbar_label)
    for ax in axes:
        ax.set_xlabel("east (m)")
    axes[0].set_ylabel("north (m)")
    fig.suptitle("Predicted position uncertainty", color=INK,
                 fontsize=13.2, fontweight="bold")
    save(fig, "planning_information_fields")


def save(fig: plt.Figure, stem: str) -> None:
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = PAPER_FIGURES / f"{stem}.{suffix}"
        fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.03)
        print(f"wrote {path}")
    plt.close(fig)


def main() -> None:
    data_role_figure()
    planner_field_construction_figure()
    planning_information_figure()


if __name__ == "__main__":
    main()
