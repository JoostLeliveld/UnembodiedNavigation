#!/usr/bin/env python3
"""Show the old GP-induced spatial R as true-scale circles in the warehouse.

This reproduces the visual language of the old AWS GP pipeline and makes its
missing implication explicit: the GP did not produce one fixed R. It produced a
spatial scalar reliability field which the old metric adapter mapped to an
isotropic, spatially varying measurement covariance.

The final panel draws 2-sigma circles in metres at their native scale. There is
no plot-only magnification.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle
from matplotlib.colors import Normalize
import numpy as np


HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src" / "reliability"))
sys.path.insert(0, str(REPO / "src" / "unav_common"))

from reliability.projection import camera_model_from_world  # noqa: E402


GP_ARTIFACT = REPO / "paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz"
WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_aws.world.sdf"
OUT = REPO / "logs/studies/bayesian_filter_showcase/meeting_r_spatial"

# Old metric GP-to-R adapter endpoints (ReplayConfig defaults).
SIGMA_GOOD_M = 0.04
SIGMA_BAD_M = 0.40
N_SIGMA = 2.0

# Expected future figure: geometry-only conditional covariance plus an honest
# visibility hit/miss branch. Both are explicit illustrative assumptions for the
# old AWS mount, not commissioned future constants.
CONDITIONAL_SIGMA_UV_PX = 2.5
PROJECTION_PLANE_Z_M = 0.05
REPRESENTATIVE_PRIOR_SIGMA_M = 0.20

COL = {
    "rack": "#F2CF23",
    "rack_edge": "#0B6F8A",
    "camera": "#344054",
    "ink": "#172033",
    "muted": "#667085",
}

RACK_XS = (-4.05, -2.00, 0.05, 2.00, 4.15)
RACK_W = 0.55
SPLIT_SEGMENTS = ((-0.82, 1.20), (2.20, 4.25))
PLOT_XLIM = (-5.55, 5.55)
PLOT_YLIM = (-5.05, 5.05)


def _load_gp() -> dict[str, np.ndarray]:
    with np.load(GP_ARTIFACT, allow_pickle=False) as data:
        return {
            "xs": np.asarray(data["xs"], dtype=float),
            "ys": np.asarray(data["ys"], dtype=float),
            "X_train": np.asarray(data["X_train"], dtype=float),
            "p_train": np.asarray(data["p_train"], dtype=float),
            "P_plan": np.asarray(data["P_conservative_plan_map"], dtype=float),
            "camera_pos": np.asarray(data["camera_pos"], dtype=float),
        }


def _rack_segments() -> list[tuple[float, float, float, float]]:
    segments: list[tuple[float, float, float, float]] = []
    for x in RACK_XS:
        ys = ((-0.82, 4.25),) if abs(x + 4.05) < 1e-9 else SPLIT_SEGMENTS
        for y0, y1 in ys:
            segments.append((x - RACK_W / 2.0, x + RACK_W / 2.0, y0, y1))
    return segments


def _inside_rack(x: float, y: float, margin_m: float = 0.0) -> bool:
    return any(
        x0 - margin_m <= x <= x1 + margin_m and y0 - margin_m <= y <= y1 + margin_m
        for x0, x1, y0, y1 in _rack_segments()
    )


def _draw_racks(ax: plt.Axes) -> None:
    for x0, x1, y0, y1 in _rack_segments():
        ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                facecolor=COL["rack"],
                edgecolor=COL["rack_edge"],
                linewidth=0.65,
                zorder=9,
            )
        )


def _draw_camera(ax: plt.Axes, position: np.ndarray) -> None:
    x, y = float(position[0]), float(position[1])
    ax.add_patch(
        Polygon(
            [[x, y + 0.23], [x - 0.23, y - 0.18], [x + 0.23, y - 0.18]],
            closed=True,
            facecolor=COL["camera"],
            edgecolor="white",
            linewidth=0.8,
            zorder=12,
            clip_on=False,
        )
    )


def _style_axis(ax: plt.Axes, title: str, *, show_ylabel: bool) -> None:
    ax.set_title(title, fontsize=12.0, fontweight="bold", pad=6)
    ax.set_xlim(*PLOT_XLIM)
    ax.set_ylim(*PLOT_YLIM)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$ [m]", fontsize=10.5)
    if show_ylabel:
        ax.set_ylabel(r"$y$ [m]", fontsize=10.5)
    else:
        ax.tick_params(labelleft=False)
    ax.set_xticks([-5, -3, -1, 1, 3, 5])
    ax.set_yticks([-5, -3, -1, 1, 3, 5])
    ax.grid(True, color="#D0D5DD", linewidth=0.38, alpha=0.55, zorder=1)
    ax.tick_params(labelsize=9.0, length=2)


def _metric_sigma_from_gp(reliability: np.ndarray | float) -> np.ndarray:
    """Old scalar adapter: linear interpolation in metric variance."""
    p = np.clip(np.asarray(reliability, dtype=float), 0.0, 1.0)
    variance = p * SIGMA_GOOD_M**2 + (1.0 - p) * SIGMA_BAD_M**2
    return np.sqrt(variance)


def _nearest_field_value(xs: np.ndarray, ys: np.ndarray, field: np.ndarray, x: float, y: float) -> float:
    ix = int(np.argmin(np.abs(xs - x)))
    iy = int(np.argmin(np.abs(ys - y)))
    return float(field[iy, ix])


def _circle_locations(gp: dict[str, np.ndarray]) -> list[tuple[float, float, float, float]]:
    records: list[tuple[float, float, float, float]] = []
    # Regular evaluation lattice: these are field probes, not training observations.
    for y in np.arange(-4.50, 4.51, 1.5):
        for x in np.arange(-5.0, 5.01, 1.5):
            if _inside_rack(float(x), float(y), margin_m=0.08):
                continue
            p = _nearest_field_value(gp["xs"], gp["ys"], gp["P_plan"], float(x), float(y))
            sigma = float(_metric_sigma_from_gp(p))
            records.append((float(x), float(y), p, sigma))
    return records


def make_figure() -> tuple[plt.Figure, dict[str, object]]:
    gp = _load_gp()
    extent = (float(gp["xs"][0]), float(gp["xs"][-1]), float(gp["ys"][0]), float(gp["ys"][-1]))
    sigma_map = _metric_sigma_from_gp(gp["P_plan"])
    circles = _circle_locations(gp)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 5.7), constrained_layout=False)
    fig.subplots_adjust(left=0.052, right=0.94, top=0.84, bottom=0.18, wspace=0.27)
    fig.suptitle(
        r"The old GP already made measurement covariance $R(\mathbf{p})$ spatial",
        fontsize=16.0,
        fontweight="bold",
        y=0.975,
        color=COL["ink"],
    )

    ax = axes[0]
    _draw_racks(ax)
    keep = np.asarray(
        [not _inside_rack(float(point[0]), float(point[1]), margin_m=0.01) for point in gp["X_train"]]
    )
    sample_plot = ax.scatter(
        gp["X_train"][keep, 0],
        gp["X_train"][keep, 1],
        c=gp["p_train"][keep],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        s=17,
        edgecolor="black",
        linewidth=0.12,
        zorder=10,
    )
    fig.colorbar(sample_plot, ax=ax, fraction=0.045, pad=0.025, label="YOLO score")
    _draw_camera(ax, gp["camera_pos"][:2])
    _style_axis(ax, "(a) YOLO-score samples", show_ylabel=True)

    ax = axes[1]
    reliability_plot = ax.imshow(
        gp["P_plan"],
        extent=extent,
        origin="lower",
        cmap="viridis",
        vmin=0.0,
        vmax=0.9,
        aspect="equal",
        zorder=0,
    )
    _draw_racks(ax)
    _draw_camera(ax, gp["camera_pos"][:2])
    fig.colorbar(reliability_plot, ax=ax, fraction=0.045, pad=0.025, label=r"$\rho_{\mathrm{plan}}(\mathbf{p})$")
    _style_axis(ax, r"(b) GP reliability $\rho_{\mathrm{plan}}(\mathbf{p})$", show_ylabel=False)

    ax = axes[2]
    cmap = plt.get_cmap("magma")
    normalizer = Normalize(vmin=SIGMA_GOOD_M, vmax=SIGMA_BAD_M)
    for x, y, _, sigma in circles:
        colour = cmap(normalizer(sigma))
        ax.add_patch(
            Circle(
                (x, y),
                radius=N_SIGMA * sigma,
                facecolor=colour,
                edgecolor="#53266A",
                linewidth=0.55,
                alpha=0.58,
                zorder=5,
            )
        )
        ax.plot(x, y, marker=".", markersize=2.0, color="#2B1539", zorder=7)
    _draw_racks(ax)
    _draw_camera(ax, gp["camera_pos"][:2])
    sigma_mappable = plt.cm.ScalarMappable(norm=normalizer, cmap=cmap)
    sigma_mappable.set_array([])
    fig.colorbar(sigma_mappable, ax=ax, fraction=0.045, pad=0.025, label=r"stated $1\sigma_R$ [m]")
    _style_axis(ax, r"(c) induced $R(\mathbf{p})=\sigma_R^2(\mathbf{p})I$", show_ylabel=False)
    ax.text(
        0.03,
        0.035,
        r"circle radius $=2\sigma_R$ — true map scale"
        "\nspatial magnitude; isotropic shape",
        transform=ax.transAxes,
        fontsize=8.7,
        color="#7A245C",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#D7A4C7", alpha=0.94),
        zorder=13,
    )

    handles = [
        Rectangle((0, 0), 1, 1, facecolor=COL["rack"], edgecolor=COL["rack_edge"], label="rack geometry"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#440154", markeredgecolor="black", markersize=4.5, label="training sample"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=COL["camera"], markeredgecolor=COL["camera"], markersize=6.5, label="external camera"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, 0.035))
    fig.text(
        0.5,
        0.008,
        r"Old metric adapter: $\sigma_R^2(\mathbf{p})=\rho_{plan}(\mathbf{p})(0.04^2)+"
        r"[1-\rho_{plan}(\mathbf{p})](0.40^2)$ m$^2$.  No visual magnification.",
        ha="center",
        fontsize=8.7,
        color=COL["muted"],
    )

    details = {
        "n_training_samples": int(len(gp["X_train"])),
        "n_circle_probes": int(len(circles)),
        "rho_plan_min": float(np.min(gp["P_plan"])),
        "rho_plan_max": float(np.max(gp["P_plan"])),
        "sigma_R_min_m": float(np.min(sigma_map)),
        "sigma_R_max_m": float(np.max(sigma_map)),
        "circle_probes": [
            {"x_m": x, "y_m": y, "rho_plan": p, "sigma_R_m": sigma}
            for x, y, p, sigma in circles
        ],
    }
    return fig, details


def _projection_jacobian(model, u: float, v: float, step_px: float = 0.5) -> np.ndarray:
    jac = np.zeros((2, 2), dtype=float)
    for axis in (0, 1):
        du = step_px if axis == 0 else 0.0
        dv = step_px if axis == 1 else 0.0
        plus = model.pixel_to_world_at_z(u + du, v + dv, PROJECTION_PLANE_Z_M)
        minus = model.pixel_to_world_at_z(u - du, v - dv, PROJECTION_PLANE_Z_M)
        if plus is None or minus is None:
            raise ValueError("pixel perturbation does not intersect the projection plane")
        jac[:, axis] = (np.asarray(plus[:2]) - np.asarray(minus[:2])) / (2.0 * step_px)
    return jac


def _conditional_geometry_covariance(model, x: float, y: float) -> np.ndarray | None:
    u, v, visible = model.world_to_pixel(x, y, PROJECTION_PLANE_Z_M)
    if not visible:
        return None
    try:
        jac = _projection_jacobian(model, float(u), float(v))
    except ValueError:
        return None
    r_uv = np.diag([CONDITIONAL_SIGMA_UV_PX**2, CONDITIONAL_SIGMA_UV_PX**2])
    cov = jac @ r_uv @ jac.T
    return 0.5 * (cov + cov.T)


def _honest_expected_posterior(
    prior: np.ndarray, conditional_covariance: np.ndarray, p_use: float
) -> tuple[np.ndarray, np.ndarray]:
    # H=I for the ground-plane xy update. This information-form expression is
    # algebraically identical to the Kalman/Joseph posterior for SPD matrices.
    hit = np.linalg.inv(np.linalg.inv(prior) + np.linalg.inv(conditional_covariance))
    expected = float(p_use) * hit + (1.0 - float(p_use)) * prior
    return 0.5 * (hit + hit.T), 0.5 * (expected + expected.T)


def _covariance_ellipse(
    centre: tuple[float, float], cov: np.ndarray, *, colour, alpha: float
) -> Ellipse:
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 1e-12)
    vectors = vectors[:, order]
    width, height = 2.0 * N_SIGMA * np.sqrt(values)
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    return Ellipse(
        centre,
        float(width),
        float(height),
        angle=float(angle),
        facecolor=colour,
        edgecolor=colour,
        linewidth=0.7,
        alpha=alpha,
        zorder=5,
    )


def _expected_probe_records(gp: dict[str, np.ndarray]) -> list[dict[str, object]]:
    model = camera_model_from_world(WORLD, include_name="external_camera")
    prior = np.diag([REPRESENTATIVE_PRIOR_SIGMA_M**2] * 2)
    records: list[dict[str, object]] = []
    for y in np.arange(-4.50, 4.51, 1.5):
        for x in np.arange(-5.0, 5.01, 1.5):
            x_f, y_f = float(x), float(y)
            if _inside_rack(x_f, y_f, margin_m=0.08):
                continue
            conditional = _conditional_geometry_covariance(model, x_f, y_f)
            if conditional is None:
                continue
            p_use = _nearest_field_value(gp["xs"], gp["ys"], gp["P_plan"], x_f, y_f)
            hit, expected = _honest_expected_posterior(prior, conditional, p_use)
            records.append(
                {
                    "x_m": x_f,
                    "y_m": y_f,
                    "p_use": p_use,
                    "R_hit_xy_m2": conditional,
                    "P_hit_xy_m2": hit,
                    "P_expected_xy_m2": expected,
                }
            )
    return records


def _major_sigma(covariance: np.ndarray) -> float:
    return float(np.sqrt(np.max(np.linalg.eigvalsh(covariance))))


def make_expected_figure() -> tuple[plt.Figure, dict[str, object]]:
    gp = _load_gp()
    records = _expected_probe_records(gp)
    extent = (float(gp["xs"][0]), float(gp["xs"][-1]), float(gp["ys"][0]), float(gp["ys"][-1]))
    r_sigmas = np.asarray([_major_sigma(row["R_hit_xy_m2"]) for row in records])
    p_sigmas = np.asarray([_major_sigma(row["P_expected_xy_m2"]) for row in records])

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 5.7), constrained_layout=False)
    fig.subplots_adjust(left=0.052, right=0.94, top=0.84, bottom=0.18, wspace=0.27)
    fig.suptitle(
        "Expected future localization uncertainty: visibility decides whether; geometry decides shape",
        fontsize=15.0,
        fontweight="bold",
        y=0.975,
        color=COL["ink"],
    )

    ax = axes[0]
    visibility_plot = ax.imshow(
        gp["P_plan"],
        extent=extent,
        origin="lower",
        cmap="viridis",
        vmin=0.0,
        vmax=0.9,
        aspect="equal",
        zorder=0,
    )
    _draw_racks(ax)
    _draw_camera(ax, gp["camera_pos"][:2])
    fig.colorbar(visibility_plot, ax=ax, fraction=0.045, pad=0.025, label=r"$p_{\mathrm{use}}(\mathbf{p})$")
    _style_axis(ax, r"(a) visibility $p_{\mathrm{use}}(\mathbf{p})$", show_ylabel=True)

    ax = axes[1]
    r_norm = Normalize(vmin=float(np.min(r_sigmas)), vmax=float(np.max(r_sigmas)))
    r_cmap = plt.get_cmap("Blues")
    for row, sigma in zip(records, r_sigmas):
        colour = r_cmap(r_norm(sigma))
        centre = (float(row["x_m"]), float(row["y_m"]))
        ax.add_patch(_covariance_ellipse(centre, row["R_hit_xy_m2"], colour=colour, alpha=0.72))
        ax.plot(*centre, marker=".", markersize=2.0, color="#08306B", zorder=7)
    _draw_racks(ax)
    _draw_camera(ax, gp["camera_pos"][:2])
    r_mappable = plt.cm.ScalarMappable(norm=r_norm, cmap=r_cmap)
    r_mappable.set_array([])
    fig.colorbar(r_mappable, ax=ax, fraction=0.045, pad=0.025, label=r"major $1\sigma$ of $R_{hit,xy}$ [m]")
    _style_axis(ax, r"(b) geometry $R_{\mathrm{hit},xy}=J R_{uv}J^\top$", show_ylabel=False)
    ax.text(
        0.03,
        0.035,
        r"conditional on a usable detection; $\sigma_{uv}=2.5$ px"
        "\ngeometry-only placeholder, native scale",
        transform=ax.transAxes,
        fontsize=8.2,
        color="#08519C",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#9ECAE1", alpha=0.94),
        zorder=13,
    )

    ax = axes[2]
    p_norm = Normalize(vmin=float(np.min(p_sigmas)), vmax=float(np.max(p_sigmas)))
    p_cmap = plt.get_cmap("magma")
    for row, sigma in zip(records, p_sigmas):
        colour = p_cmap(p_norm(sigma))
        centre = (float(row["x_m"]), float(row["y_m"]))
        ax.add_patch(_covariance_ellipse(centre, row["P_expected_xy_m2"], colour=colour, alpha=0.62))
        ax.plot(*centre, marker=".", markersize=2.0, color="#2B1539", zorder=7)
    _draw_racks(ax)
    _draw_camera(ax, gp["camera_pos"][:2])
    p_mappable = plt.cm.ScalarMappable(norm=p_norm, cmap=p_cmap)
    p_mappable.set_array([])
    fig.colorbar(p_mappable, ax=ax, fraction=0.045, pad=0.025, label=r"major $1\sigma$ of $E[P^+]$ [m]")
    _style_axis(ax, r"(c) expected belief $E[P^+]$", show_ylabel=False)
    ax.text(
        0.03,
        0.035,
        r"$E[P^+]=p_{use}P_{hit}+(1-p_{use})P^-$"
        f"\nrepresentative prior: $1\\sigma={REPRESENTATIVE_PRIOR_SIGMA_M:.2f}$ m; native scale",
        transform=ax.transAxes,
        fontsize=8.2,
        color="#7A245C",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#D7A4C7", alpha=0.94),
        zorder=13,
    )

    handles = [
        Rectangle((0, 0), 1, 1, facecolor=COL["rack"], edgecolor=COL["rack_edge"], label="rack geometry"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=COL["camera"], markeredgecolor=COL["camera"], markersize=6.5, label="external camera"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, 0.035))
    fig.text(
        0.5,
        0.008,
        "All outlines are 2σ at native map scale. Visibility is a hit/miss probability; a miss gives no camera update.",
        ha="center",
        fontsize=8.7,
        color=COL["muted"],
    )

    return fig, {
        "n_probe_locations": len(records),
        "representative_prior_sigma_m": REPRESENTATIVE_PRIOR_SIGMA_M,
        "conditional_sigma_uv_px": CONDITIONAL_SIGMA_UV_PX,
        "R_hit_major_sigma_range_m": [float(np.min(r_sigmas)), float(np.max(r_sigmas))],
        "P_expected_major_sigma_range_m": [float(np.min(p_sigmas)), float(np.max(p_sigmas))],
        "probes": [
            {
                "x_m": row["x_m"],
                "y_m": row["y_m"],
                "p_use": row["p_use"],
                "R_hit_xy_m2": np.asarray(row["R_hit_xy_m2"]).tolist(),
                "P_hit_xy_m2": np.asarray(row["P_hit_xy_m2"]).tolist(),
                "P_expected_xy_m2": np.asarray(row["P_expected_xy_m2"]).tolist(),
            }
            for row in records
        ],
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    figure, details = make_figure()
    stem = "old_gp_spatial_r_circles"
    for extension in ("png", "pdf"):
        figure.savefig(OUT / f"{stem}.{extension}", dpi=220)
    plt.close(figure)

    summary = {
        "figure": stem,
        "metric_object": "camera_measurement",
        "metric_name": "old_gp_induced_metric_measurement_covariance_R_xy",
        "frame": "world_xy",
        "world": "warehouse_aws.world.sdf",
        "gp_artifact": str(GP_ARTIFACT.relative_to(REPO)),
        "projection_runtime": "historical_gp_metric_adapter",
        "status": "historical",
        "experimental_unit": "spatial_field_probe",
        "online_inputs": ["frozen GP reliability field", "query position"],
        "evaluation_only_inputs": [],
        "mapping": "variance_linear: rho*0.04^2 + (1-rho)*0.40^2 m^2",
        "ellipse_convention": "2_sigma circle at native map scale",
        "display_magnification": 1.0,
        "restrictions": [
            "isotropic spatial GP-to-R adapter",
            "historical planner/manager concept; not current measurement calibration",
            "R is camera-measurement covariance, not filter-belief covariance P",
            "circle lattice is a deterministic visualization probe grid, not detections",
        ],
        **details,
    }
    (OUT / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT / f'{stem}.png'}")
    print(f"wrote {OUT / f'{stem}.pdf'}")
    print(f"wrote {OUT / f'{stem}_summary.json'}")

    expected_figure, expected_details = make_expected_figure()
    expected_stem = "expected_geometry_visibility"
    for extension in ("png", "pdf"):
        expected_figure.savefig(OUT / f"{expected_stem}.{extension}", dpi=220)
    plt.close(expected_figure)

    expected_summary = {
        "figure": expected_stem,
        "metric_object": "filter_belief",
        "metric_name": "analytic_expected_one_step_posterior_covariance_E_P_plus_xy",
        "reference": "none; analytic planning prediction",
        "frame": "world_xy",
        "world": "warehouse_aws.world.sdf",
        "visibility_source": str(GP_ARTIFACT.relative_to(REPO)),
        "visibility_interpretation": "historical conservative GP reliability used as p_use proxy",
        "conditional_projection_runtime": "geometry_only_aws_mount_placeholder",
        "status": "illustrative_not_calibrated_not_deployed",
        "experimental_unit": "spatial_field_probe",
        "online_inputs": [
            "visibility probability p_use",
            "representative prior covariance P_minus",
            "camera projection geometry",
            "placeholder conditional pixel covariance R_uv",
        ],
        "evaluation_only_inputs": [],
        "posterior_expectation": "p_use*P_hit + (1-p_use)*P_minus; miss gives no camera update",
        "ellipse_convention": "2_sigma ellipse at native map scale",
        "display_magnification": 1.0,
        "restrictions": [
            "historical GP reliability is only a proxy for future detection usability probability",
            "conditional 2.5 px image noise is a visible-endpoint placeholder, not commissioned AWS calibration",
            "geometry-only conditional covariance omits yaw marginal, residual calibration, extrinsic/timing uncertainty, and sequential correlation",
            "expected posterior covariance depends on the explicitly stated representative prior and is not a unique warehouse map",
            "analytic planning prediction, not observed localization accuracy or a current result",
            "no ground truth is used by the online construction",
        ],
        **expected_details,
    }
    (OUT / f"{expected_stem}_summary.json").write_text(
        json.dumps(expected_summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {OUT / f'{expected_stem}.png'}")
    print(f"wrote {OUT / f'{expected_stem}.pdf'}")
    print(f"wrote {OUT / f'{expected_stem}_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
