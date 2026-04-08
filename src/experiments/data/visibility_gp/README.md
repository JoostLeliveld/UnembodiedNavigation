# `experiments/data/visibility_gp`

This folder stores the fixed empirical GP visibility artifacts used by the planner.

## Why This Folder Exists

The GP-aware planner needs a world-specific visibility prior at runtime. These `.npz` files are that prior.

## Current Artifacts

| File | World |
| --- | --- |
| `warehouse_occ_light_empirical_visibility_gp.npz` | primary warehouse benchmark |
| `warehouse_open_shelves_empirical_visibility_gp.npz` | exploratory secondary world |

## How They Are Used

- generated offline by [`../../../../scripts/fit_empirical_visibility_gp.py`](../../../../scripts/fit_empirical_visibility_gp.py)
- resolved by [`../../experiments/core/world_profiles.py`](../../experiments/core/world_profiles.py)
- loaded by [`../../../planning/planning/core/visibility_gp_map.py`](../../../planning/planning/core/visibility_gp_map.py)

## Important Caveat

These files represent a learned visibility / detection-success prior for the current simulated camera-detector stack. They should not be presented as a general occlusion model.
