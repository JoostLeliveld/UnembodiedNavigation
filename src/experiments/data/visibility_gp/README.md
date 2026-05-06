# `experiments/data/visibility_gp`

This folder stores older packaged empirical GP visibility artifacts that may still be useful for compatibility runs, but they are no longer the canonical outputs of the active comparison framework.

![Example empirical visibility artifact](../../../../docs/figures/visibility_capture_tutorial.png)

## Why This Folder Still Exists

Some launches and older manifests may still point here. The active comparison framework, however, now writes method-specific GP artifacts to:

- `logs/visibility_comparison/current_gp/`

## Current Artifacts

| File | World |
| --- | --- |
| `warehouse_occ_light_empirical_visibility_gp.npz` | primary warehouse benchmark |
| `warehouse_open_shelves_empirical_visibility_gp.npz` | legacy/support artifact, not part of the active single-world comparison surface |

## Legacy Provenance

- `warehouse_occ_light_empirical_visibility_gp.npz` currently points at the fitted artifact produced in:
  - [`../../../../logs/visibility_capture/fit_capture_20260401_144115_20260408_134247`](../../../../logs/visibility_capture/fit_capture_20260401_144115_20260408_134247)
- That artifact is a blob-area legacy baseline for the older fixed-camera simulated stack.

## How They Are Used

- comparison runs in the new framework should pass an explicit `visibility_artifact_path`
- the active comparison scripts live under [`../../../../scripts/visibility_comparison/`](../../../../scripts/visibility_comparison/)
- resolved by [`../../experiments/core/world_profiles.py`](../../experiments/core/world_profiles.py)
- loaded by [`../../../planning/planning/core/visibility_gp_map.py`](../../../planning/planning/core/visibility_gp_map.py)

## Important Caveat

These files are legacy packaged artifacts. The paper pipeline fits the planner artifact from the heading-marginal `yolo_score_raw` target and writes the canonical result to `logs/visibility_comparison/current_gp/yolo_score_raw_gp.npz`. The planner-facing field is `P_conservative_plan_map`. These files should not be presented as the paper artifact or as a general occlusion model.
