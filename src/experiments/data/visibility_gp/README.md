# `experiments/data/visibility_gp`

This folder stores older packaged empirical GP visibility artifacts. They are useful for compatibility runs, but they are not the canonical paper outputs.

## Active Paper Artifact Location

Current paper-facing GP artifacts are generated under:

- `logs/visibility_comparison/current_gp/`

The compact benchmark currently uses:

- `logs/visibility_comparison/current_gp/yolo_score_raw_gp.npz`

## Packaged Artifacts

| File | Status |
| --- | --- |
| `warehouse_occ_light_empirical_visibility_gp.npz` | legacy packaged artifact for compatibility |
| `warehouse_open_shelves_empirical_visibility_gp.npz` | legacy/support artifact |

## Rule

Comparison runs in the paper path must pass `visibility_artifact_path` explicitly. These packaged files should not be presented as the fitted paper artifact unless the campaign config and paper text explicitly say so.
