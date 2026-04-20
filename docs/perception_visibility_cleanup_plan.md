# Perception-to-Visibility Cleanup Plan

This document records the cleanup performed while switching the repository to the new shared perception-to-visibility comparison backbone.

## Active Story

The active comparison pipeline is now:

1. teleport-sampled raw capture
2. shared perception target extraction
3. GP target construction
4. per-method GP fitting
5. GP value and ambiguity plotting
6. planner runs with explicit `visibility_artifact_path`
7. final comparison report

Canonical generated-output root:

- `logs/visibility_comparison/`

Canonical active scripts:

- `scripts/visibility_comparison/capture_visibility_samples.py`
- `scripts/visibility_comparison/extract_perception_targets.py`
- `scripts/visibility_comparison/compute_area_reference.py`
- `scripts/visibility_comparison/build_gp_targets.py`
- `scripts/visibility_comparison/fit_visibility_gps.py`
- `scripts/visibility_comparison/plot_gp_and_ambiguity_maps.py`
- `scripts/visibility_comparison/plot_planned_paths.py`
- `scripts/visibility_comparison/make_visibility_comparison_report.py`

Canonical active docs:

- `docs/perception_to_visibility_comparison.md`
- `docs/perception_visibility_cleanup_plan.md`

## Deleted Generated Artifacts

The following generated outputs were removed because they belonged to the earlier YOLO/redmask/one-off visibility story and conflicted with the new canonical layout:

- `logs/oob_capture_20260415_visible_side`
- `logs/oob_capture_20260415_visible_side_v2`
- `logs/oob_test_20260415_visible_side_v2`
- `logs/projected_bbox_dataset_20260415_small`
- `logs/projected_bbox_dataset_20260415_train`
- `logs/redmask_seg_dataset_20260415_train`
- `logs/redmask_smoke_gpu_val_preview_20260415`
- `logs/redmask_smoke_subset_preview_epoch3_20260415`
- `logs/redmask_smoke_val_preview_20260415`
- `logs/redmask_smoke_val_preview_epoch2_20260415`
- `logs/redmask_val_subset16_20260415`
- `logs/visibility_capture_source_yolo_soft_20260415`

These were deleted rather than migrated because they did not match the new canonical comparison structure.

## Archived Source

The following older, more monolithic scripts were moved to `archive/visibility_legacy/`:

- `scripts/fit_empirical_visibility_gp.py`
- `scripts/plot_visibility_run.py`
- `scripts/showcase_yolo_gp_performance.py`
- `scripts/generate_docs_figures.py`

Why they were archived:

- they mixed multiple comparison stages into one script
- they made it harder to compare methods cleanly under one schema
- they encoded an older repository story centered on one-off GP fitting and showcase plots

## Kept But Marked Noncanonical

These packaged artifacts were kept for compatibility, but they are no longer the canonical outputs of the comparison framework:

- `src/experiments/data/visibility_gp/*.npz`

Comparison runs in the new framework should pass explicit `visibility_artifact_path` values and should not rely on those packaged defaults.

## What No Longer Belongs To The Active Story

- driving-sweep visibility capture as the primary capture mode
- monolithic “fit and showcase” scripts
- ad hoc YOLO/redmask-generated logs under `logs/`
- packaged GP artifacts as the source of truth
- doc paths that imply one giant offline GP script is still the canonical workflow

## Current Implementation Boundary

This cleanup pass establishes the shared backbone only.

Already implemented:

- shared teleport capture format
- shared perception target schema
- shared GP target schema
- shared GP artifact contract
- shared GP/ambiguity plotting
- shared path plotting
- shared report assembly

Deferred to later method-specific passes:

- red binary target extraction
- red corrected-area target extraction
- YOLO binary target extraction
- YOLO raw-score target extraction
- YOLO calibrated-score target extraction
