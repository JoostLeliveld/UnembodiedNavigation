#!/usr/bin/env python3
"""Assemble deterministic fusion and state-update supervisor comparisons.

The source figures are outputs of repository-native analysis scripts. This
renderer copies them into the supervisor package under a stable naming contract,
builds two contact sheets, and records SHA-256 provenance. It does not generate,
retouch, or infer image content with an AI model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import render_all as base


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

FUSION = {
    "01_camera_availability_inputs.png": REPO / "logs/studies/fused_observation_model/exp1_availability_fusion/fig_a1_spatial_availability.png",
    "02_selection_policy_map.png": REPO / "logs/studies/achievable_precision_map/exp1_precision_vs_coverage/fig_m1_precision_vs_coverage.png",
    "03_availability_fusion_calibration.png": REPO / "logs/studies/fused_observation_model/exp1_availability_fusion/fig_a3_fusion_calibration.png",
    "04_measurement_fusion_evidence.png": REPO / "logs/studies/external_camera_bias_model/exp1_residual_characterization/fig_b5_fusion_simulation.png",
    "05_dynamic_occlusion_update.png": REPO / "logs/studies/multicamera_fusion_extension/dynamic_occlusion_regression/dynamic_occlusion_showcase.png",
    "06_route_consequence.png": REPO / "logs/studies/multicamera_fusion_extension/health_adaptive_reroute/fig_health_adaptive_reroute.png",
}

STATE_UPDATE = {
    "01_predict_update_loop.png": REPO / "logs/studies/bayesian_filter_showcase/demo_how_the_filter_works/fig_a2_predict_update_loop.png",
    "02_expected_hit_miss_vs_blend.png": REPO / "logs/studies/efe_hit_miss_mixture/exp1_mixture_vs_blend/fig_e1_mixture_vs_blend.png",
    "03_single_R_error_sweep.png": REPO / "logs/studies/planner_covariance_branching/exp1_scaled_vs_branch/fig_p1_relative_trace_error.png",
    "04_failure_region.png": REPO / "logs/studies/planner_covariance_branching/exp1_scaled_vs_branch/fig_p3_acceptable_region.png",
    "05_calibration_floor_update.png": REPO / "logs/studies/bayesian_filter_showcase/demo_how_the_filter_works/fig_a6_the_promise_kept_or_broken.png",
    "06_prior_dependent_equivalent_R.png": REPO / "logs/studies/planner_covariance_branching/exp1_scaled_vs_branch/fig_p2_equivalent_covariance.png",
}

HISTORICAL_V2_FLOOR_M = {
    "camera_A": 0.0071,
    "camera_B": 0.0123,
    "camera_C": 0.0768,
    "camera_D": 0.0328,
}
DETECTION_RATE_HZ = 3.0
Q_RATE_M2_PER_S = 0.04**2 * 0.3
MIN_USABLE_P = 0.02


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_set(mapping: dict[str, Path], destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name, source in mapping.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        target = destination / name
        shutil.copy2(source, target)
        outputs.append(target)
    return outputs


def contact_sheet(
    mapping: dict[str, Path],
    copied_dir: Path,
    output: Path,
    title: str,
    subtitles: tuple[str, ...],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(20, 11.5), constrained_layout=True)
    for ax, ((name, _), subtitle) in zip(axes.flat, zip(mapping.items(), subtitles)):
        ax.imshow(plt.imread(copied_dir / name))
        ax.set_title(subtitle, fontsize=13, weight="bold")
        ax.axis("off")
    fig.suptitle(title, fontsize=21, weight="bold")
    fig.text(
        0.5,
        0.006,
        "DETERMINISTIC REPOSITORY FIGURES — panel-specific evidence status is retained; no AI-generated imagery",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def route_context():
    with np.load(HERE / "generated_data/method_fields.npz", allow_pickle=False) as data:
        xs = np.asarray(data["xs"], dtype=float)
        ys = np.asarray(data["ys"], dtype=float)
        driveable = np.asarray(data["driveable_mask"], dtype=bool)
    prisms = tuple(
        base.parse_occlusion_scene_from_world(
            str(base.WORLD),
            model_name="warehouse_rack_occluders",
            geometry_tags=("collision",),
        ).prisms
    )
    return xs, ys, driveable, prisms, base.tasks()


def render_selection_route_grid(output: Path) -> None:
    """Compare two defensible single-camera management policies on common routes."""

    xs, ys, driveable, prisms, tasks = route_context()
    with np.load(base.GP_FUSED, allow_pickle=False) as data:
        per_camera = {
            camera_id: np.asarray(data[f"P_camera_{camera_id[-1]}_map"], dtype=float)
            for camera_id in base.CAMERA_POSES
        }
    p_stack = np.stack([per_camera[c] for c in base.CAMERA_POSES])
    sigma_stack = []
    for camera_id in base.CAMERA_POSES:
        p = per_camera[camera_id]
        sigma = np.sqrt(
            HISTORICAL_V2_FLOOR_M[camera_id] ** 2
            + Q_RATE_M2_PER_S / (DETECTION_RATE_HZ * np.maximum(p, MIN_USABLE_P))
        )
        sigma_stack.append(np.where(p >= MIN_USABLE_P, sigma, np.inf))
    sigma_stack = np.stack(sigma_stack)
    max_p_index = np.argmax(p_stack, axis=0)
    sigma_follow_max_p = np.take_along_axis(
        sigma_stack, max_p_index[None, ...], axis=0
    )[0]
    sigma_best = np.min(sigma_stack, axis=0)

    def precision_utility(sigma):
        # Common declared visualization transform: 2 cm -> 1, 12 cm -> 0.
        return np.clip((0.12 - sigma) / 0.10, 0.0, 1.0)

    policies = (
        ("maximum availability selector", precision_utility(sigma_follow_max_p), "#3167b1"),
        ("minimum achievable-sigma selector", precision_utility(sigma_best), "#d97706"),
    )
    route_keys = ("R1", "R2", "R3", "R6")
    fig, axes = plt.subplots(2, 4, figsize=(20, 9.6), sharex=True, sharey=True)
    for row, (label, field, color) in enumerate(policies):
        for col, route_key in enumerate(route_keys):
            ax = axes[row, col]
            base.draw_field(
                ax,
                field,
                xs,
                ys,
                driveable,
                prisms,
                title=f"{label}\n{route_key}: {base.route_title(route_key)}",
            )
            for text_artist in list(ax.texts):
                if text_artist.get_text() in {"A", "B", "C", "D"}:
                    text_artist.remove()
            base.draw_routes(
                ax,
                field,
                driveable,
                xs,
                ys,
                base.route_spec(tasks, route_key),
                color,
            )
    fig.suptitle(
        "Camera selection — same GP availability, different definition of useful precision",
        fontsize=17,
        weight="bold",
    )
    fig.legend(
        handles=[
            Line2D([], [], color="#3167b1", lw=2.6, label="max-availability route"),
            Line2D([], [], color="#d97706", lw=2.6, label="min-sigma route"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.025),
    )
    fig.text(
        0.5,
        0.004,
        "EXPLORATORY OFFLINE ROUTES — field source, grid and cost fixed; utility=(12 cm-sigma)/10 cm clipped to [0,1]",
        ha="center",
        fontsize=8.5,
        color="#4b5563",
    )
    fig.subplots_adjust(
        left=0.045, right=0.99, top=0.86, bottom=0.13, wspace=0.10, hspace=0.36
    )
    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_state_update_route_grid(output: Path) -> None:
    """Show route consequences of three planner-facing availability mappings."""

    xs, ys, driveable, prisms, tasks = route_context()
    with np.load(base.GP_FUSED, allow_pickle=False) as data:
        p_use = np.asarray(data["P_conservative_plan_map"], dtype=float)

    prior_var = 0.30**2
    r_hit = 0.05**2
    r_miss = 1.20**2
    p_hit = prior_var * r_hit / (prior_var + r_hit)
    branch_post = p_use * p_hit + (1.0 - p_use) * prior_var
    # Algebraically equivalent to a Kalman update with R_hit / p, but stable at p=0.
    scaled_post = prior_var * r_hit / (r_hit + prior_var * p_use)
    blended_precision = p_use / r_hit + (1.0 - p_use) / r_miss
    blended_r = 1.0 / blended_precision
    blended_post = prior_var * blended_r / (prior_var + blended_r)

    def information_credit(posterior):
        return np.clip((prior_var - posterior) / (prior_var - p_hit), 0.0, 1.0)

    rules = (
        ("deployed precision blend", information_credit(blended_post), "#c45a00"),
        ("R / p shortcut", information_credit(scaled_post), "#8b5a3c"),
        ("explicit hit/miss branch", information_credit(branch_post), "#087e8b"),
    )
    route_keys = ("R1", "R2", "R3", "R6")
    fig, axes = plt.subplots(3, 4, figsize=(20, 13.5), sharex=True, sharey=True)
    for row, (label, field, color) in enumerate(rules):
        for col, route_key in enumerate(route_keys):
            ax = axes[row, col]
            base.draw_field(
                ax,
                field,
                xs,
                ys,
                driveable,
                prisms,
                title=f"{label}\n{route_key}: {base.route_title(route_key)}",
            )
            for text_artist in list(ax.texts):
                if text_artist.get_text() in {"A", "B", "C", "D"}:
                    text_artist.remove()
            base.draw_routes(
                ax,
                field,
                driveable,
                xs,
                ys,
                base.route_spec(tasks, route_key),
                color,
            )
    fig.suptitle(
        "State-update rule — same frozen GP field, different planning credit for a possible observation",
        fontsize=17,
        weight="bold",
    )
    fig.text(
        0.5,
        0.008,
        "EXPLORATORY OFFLINE ROUTES — fixed source and cost; prior sigma=0.30 m, conditional sigma=0.05 m, Rmiss sigma=1.20 m; not navigation evidence",
        ha="center",
        fontsize=8.5,
        color="#4b5563",
    )
    fig.subplots_adjust(
        left=0.045, right=0.99, top=0.90, bottom=0.06, wspace=0.12, hspace=0.40
    )
    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    fusion_dir = HERE / "08_fusion_comparison/figures"
    update_dir = HERE / "09_state_update_comparison/figures"
    fusion_outputs = copy_set(FUSION, fusion_dir)
    update_outputs = copy_set(STATE_UPDATE, update_dir)

    fusion_sheet = fusion_dir / "all_fusion_panels.png"
    contact_sheet(
        FUSION,
        fusion_dir,
        fusion_sheet,
        "Fusion comparison — field aggregation, camera management, and measurement fusion",
        (
            "1. Per-camera availability inputs",
            "2. Camera selection map",
            "3. Availability-fusion calibration",
            "4. Simultaneous measurement fusion",
            "5. Runtime trust update",
            "6. Route consequence",
        ),
    )
    state_sheet = update_dir / "all_state_update_panels.png"
    contact_sheet(
        STATE_UPDATE,
        update_dir,
        state_sheet,
        "State-update comparison — what a hit, miss, and covariance promise mean",
        (
            "1. Predict / update loop",
            "2. Honest hit/miss expectation",
            "3. Shortcut error across conditions",
            "4. Where shortcuts fail",
            "5. Persistent-error floor",
            "6. Prior-dependent equivalent R",
        ),
    )

    fusion_routes = fusion_dir / "07_selection_route_grid.png"
    state_routes = update_dir / "07_exploratory_route_grid.png"
    render_selection_route_grid(fusion_routes)
    render_state_update_route_grid(state_routes)

    inputs = list(FUSION.values()) + list(STATE_UPDATE.values())
    outputs = fusion_outputs + update_outputs + [
        fusion_sheet,
        state_sheet,
        fusion_routes,
        state_routes,
    ]
    payload = {
        "status": "mixed_locked_evidence_and_labelled_mechanism_figures",
        "renderer": str(Path(__file__).relative_to(REPO)),
        "separation_rule": (
            "availability-field fusion, runtime measurement fusion, and planner "
            "state-update algebra are distinct experimental axes"
        ),
        "inputs": {str(path.relative_to(REPO)): sha256(path) for path in inputs},
        "outputs": {str(path.relative_to(HERE)): sha256(path) for path in outputs},
    }
    manifest = HERE / "generated_data/decision_layer_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"assembled {len(outputs)} fusion/state-update PNG figures")
    print(f"manifest: {manifest.relative_to(REPO)}")


if __name__ == "__main__":
    main()
