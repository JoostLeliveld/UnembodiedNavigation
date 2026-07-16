#!/usr/bin/env python3
"""Build per-chapter media views: research_story/<chapter>/{figures,videos}/ as
relative symlinks into the two canonical media homes (paper_artifacts/ = locked,
logs/ = regenerable). Idempotent: re-run after adding media; it recreates all
links and reports dangling targets. Never copies files.

Curation lives here — a chapter's view is the set of files an examiner should
see for that chapter, not everything a study ever produced.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # UnembodiedNavigation/
RS = ROOT / "research_story"

PA = "paper_artifacts/figures"
CS = f"{PA}/current_surface"
PS = f"{PA}/paper_snapshot"
OA = "logs/studies/optionA_commissioning"
GV = "logs/studies/geometry_visibility_prior"
FC = "logs/studies/multicamera_commissioning_bigwarehouse/four_camera_showcase"

FIGS: dict[str, list[str]] = {
    "00_problem_and_existing_baseline": [
        f"{PA}/paired_mechanism_taskA_current.pdf",
        f"{PA}/paired_mechanism_west_current.pdf",
        f"{PA}/paired_mechanism_taskA_lowlat.pdf",
        f"{PA}/paired_mechanism_a2mid_lowlat.pdf",
        f"{PA}/paired_mechanism_west_lowlat.pdf",
        f"{PA}/paired_mechanism_control_lowlat.pdf",
        f"{PA}/paired_mechanism_taskA_PAPER.pdf",
        f"{PA}/gp_pipeline_aws.png",
        f"{PA}/localization_pathway.png",
        f"{PA}/yolo_training_clarification.png",
        f"{PA}/explainers/contribution_map.png",
        f"{PA}/explainers/system_architecture.svg",
        f"{CS}/robustness_spread_current.png",
        f"{PS}/robustness_spread.png",
        f"{CS}/gp_pipeline_current.png",
    ],
    "01_operational_belief_and_logging": [
        f"{OA}/exp5_trajectory_smoothing/fig1_example_run.png",
        f"{OA}/exp5_trajectory_smoothing/fig2_calibration.png",
        f"{OA}/exp5_trajectory_smoothing/fig3_L0_L3_maps.png",
    ],
    "02_trust_target_and_calibration": [
        f"{OA}/exp0_confidence_audit/fig1_confidence_vs_error.png",
        f"{OA}/exp0_confidence_audit/fig2_gate_interaction.png",
        f"{OA}/exp0_confidence_audit/fig3_conditioning.png",
        f"{OA}/exp0_confidence_audit/fig4_spatial.png",
    ],
    "03_uncertain_input_gp": [
        f"{OA}/exp1_synthetic_gp/fig1_setup_and_gate.png",
        f"{OA}/exp1_synthetic_gp/fig2_prediction_maps.png",
        f"{OA}/exp1_synthetic_gp/fig3_metrics.png",
        f"{OA}/SHOWCASE.png",
    ],
    "05_trust_to_rplan": [
        f"{OA}/exp2_operational_mapping/fig1_commissioning_data.png",
        f"{OA}/exp2_operational_mapping/fig2_maps.png",
        f"{OA}/exp2_operational_mapping/fig3_heldout_metrics.png",
        f"{OA}/exp2_operational_mapping/fig4_semisynthetic_sweep.png",
        f"{GV}/warehouse_aws_v0/figures/07_geometry_r_plan_std.png",
        f"{GV}/warehouse_aws_v0/figures/08_r_plan_matrix_examples.png",
        f"{GV}/demo/rplan_nis_calibration.png",
    ],
    "06_original_warehouse_navigation": [
        f"{OA}/exp7_planner_replay/fig1_interface_and_paths.png",
        f"{OA}/exp7_planner_replay/fig2_replay_profiles.png",
        f"{OA}/exp6_stress_test/fig1_inflation_curves.png",
        f"{OA}/exp6_stress_test/fig2_stale_vs_current.png",
    ],
    "07_weak_priors_and_geometry": [
        *[f"{GV}/warehouse_aws_v0/figures/{n}" for n in (
            "01_geometry_overlay.png", "02_height_map.png", "03_fov_mask.png",
            "04_clearance_map.png", "05_raw_visibility.png",
            "06_visibility_components.png", "09_current_yolo_gp_vs_geometry_prior.png",
            "10_fusion_posterior.png")],
        f"{GV}/calibrated_prior_v1/calibrated_prior.png",
        f"{OA}/exp34_init_budget/fig1_priors.png",
        f"{OA}/exp34_init_budget/fig2_budget_curves.png",
        *[f"{GV}/demo/{n}" for n in (
            "sensed_height_prior.png", "depth_occlusion_prior.png",
            "depth_realism.png", "depth_source_comparison.png",
            "diagnose_prior.png", "hard_evidence.png", "stage3_validation.png",
            "gp_update_final.png", "whatif_layout_change.png")],
    ],
    # Only current warehouse_full_4cam material is linked here. Retired-testbed
    # media remains outside the curated chapter views.
    "08_large_warehouse_scaling": [
        f"{FC}/dayzero_reliability_atlas.png",
        f"{FC}/best_camera_and_reliability.png",
        f"{FC}/live_four_camera_montage.png",
    ],
    "09_multicamera_handover_fusion": [
        f"{FC}/overlap_handover_corridor.png",
    ],
}

VIDEOS: dict[str, list[str]] = {
    "00_problem_and_existing_baseline": [
        *[f"{CS}/paired_mechanism_{n}.gif" for n in (
            "west_current", "taskA_current", "taskA_seed2_honest",
            "taskA_lowlat", "a2mid_lowlat", "west_lowlat", "control_lowlat")],
        f"{PS}/paired_mechanism_taskA_PAPER.gif",
    ],
    "07_weak_priors_and_geometry": [
        f"{GV}/demo/gp_online_update.gif",
        f"{GV}/demo/stereo_online_showcase.gif",
    ],
}


def build(mapping: dict[str, list[str]], subdir: str) -> tuple[int, list[str]]:
    made, missing = 0, []
    for chapter, targets in mapping.items():
        view = RS / chapter / subdir
        view.mkdir(parents=True, exist_ok=True)
        for old in view.iterdir():
            if old.is_symlink():
                old.unlink()
        for t in targets:
            target = ROOT / t
            if not target.exists():
                missing.append(t)
                continue
            link = view / Path(t).name
            rel = os.path.relpath(target, view)
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(rel)
            made += 1
    return made, missing


def main() -> int:
    nf, mf = build(FIGS, "figures")
    nv, mv = build(VIDEOS, "videos")
    print(f"linked {nf} figures, {nv} videos")
    for t in mf + mv:
        print(f"MISSING TARGET: {t}", file=sys.stderr)
    return 1 if (mf or mv) else 0


if __name__ == "__main__":
    raise SystemExit(main())
