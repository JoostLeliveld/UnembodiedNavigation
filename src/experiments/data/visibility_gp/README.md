# `experiments/data/visibility_gp`

This folder stores the fixed empirical GP visibility artifacts used by the planner.

![Example empirical visibility artifact](../../../../docs/figures/visibility_capture_tutorial.png)

## Why This Folder Exists

The GP-aware planner needs a world-specific visibility prior at runtime. These `.npz` files are that prior.

## Current Artifacts

| File | World |
| --- | --- |
| `warehouse_occ_light_empirical_visibility_gp.npz` | primary warehouse benchmark |
| `warehouse_open_shelves_empirical_visibility_gp.npz` | exploratory secondary world |

## Current Default Provenance

- `warehouse_occ_light_empirical_visibility_gp.npz` currently points at the fitted artifact produced in:
  - [`../../../../logs/visibility_capture/fit_capture_20260401_144115_20260408_134247`](../../../../logs/visibility_capture/fit_capture_20260401_144115_20260408_134247)
- That artifact is a blob-area baseline GP for the current fixed-camera simulated stack. It is the active default for `warehouse_occ_light.world.sdf` unless a launch overrides `visibility_artifact_path`.

## How They Are Used

- generated offline by [`../../../../scripts/fit_empirical_visibility_gp.py`](../../../../scripts/fit_empirical_visibility_gp.py) from noisy simulated pose sampling by default, with a retained optional driving-sweep mode
- resolved by [`../../experiments/core/world_profiles.py`](../../experiments/core/world_profiles.py)
- loaded by [`../../../planning/planning/core/visibility_gp_map.py`](../../../planning/planning/core/visibility_gp_map.py)

## Important Caveat

These files represent a learned visibility / detection-success prior for the current simulated camera-detector stack. The current fitter defaults to a normalized blob-area target for first-pass experiments and also supports binary usable detection as an alternate scalar target. These files should not be presented as a general occlusion model.
