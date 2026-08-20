# Reuse map

What this study imports rather than reimplements, and the one thing it deliberately
duplicates.

| needed | reused from | why not rewritten |
|---|---|---|
| Camera model, projection | `unav_common.camera_model.ObliqueCameraModel` | the runtime's own model; a second one would drift |
| CAD prisms, ray-cast visibility, floor grid | `experiments/dynamic_world_oracle/oracle.py` | already validated against Gazebo's depth buffer (its acceptance check 8) |
| Obstacle prisms from a model SDF | `oracle.parts_from_model_sdf`, `oracle.place_obstacle` | same parse the dynamic-world runs use |
| Grid-teleport capture, four cameras, oracle column | `scripts/visibility_comparison/capture_visibility_samples.py` | produced the `L0` reference; using it is what makes the environments comparable |
| Detector scoring on saved frames | `scripts/visibility_comparison/extract_perception_targets.py` | same detector configuration path as the reference capture |
| GP fitting | `scripts/visibility_comparison/fit_belief_aware_gp.py` | the canonical GP; `gp_fields.py` calls its private helpers in a separate process, exactly as `availability_paper/gp_refit.py` does and for the same `common` name-clash reason |
| Monocular depth inference | `experiments/monocular_depth_adapter` (`MonocularDepthAdapter`) | the availability study's selected backbone, UniDepthV2 ViT-S |
| Floor anchoring, line-of-sight field | `experiments/mono_depth_visibility/ground_anchoring` | the method whose boundary (no oracle depth, no CAD) is already enforced by tests |
| Brier, log loss, AUROC, ECE | `scripts/shared/metrics.py` | an audit found 15 divergent copies of these in this repo; there is one |
| Repo root resolution | `scripts/shared/paths.repo_root` | preferred over `Path(__file__).parents[N]` in new code |
| Drivable lanes, spawn pose, camera intrinsics | `src/experiments/config/world_profiles.yaml` | the variant profiles are deep copies of the flagship entry, so they cannot drift |
| Detector weights | `logs/perception_models/warehouse_yolo_detector_4cam_v3_960` | the detector the reference capture was scored with |

## Deliberately duplicated

`common.py` carries its own copy of the two-parameter calibration link (`fit_link` /
`apply_link`) and the six spatial blocks, character-for-character identical to
`experiments/availability_paper/common.py`. Importing that module instead would pull in
its `build_apparatus()`, which loads the availability study's cached grid, its cached
monocular-depth maps and its event root — none of which this study uses, and one of
which (the cached GP fields) is known to leak. The duplication is 40 lines with a
stated reason; the alternative is an import that silently brings a different grid.

## Not reused, and why

| candidate | why not |
|---|---|
| `logs/visibility_comparison/spawn_grid_20260727/gp/*` cached GP fields | fitted on every event including held-out ones; scoring them at held-out points put the GP at Brier 0.021 against 0.218 honest. This study refits. |
| `availability_paper`'s cached monocular-depth maps | they are the `s01` dynamic-world frames at two timestamps, not this study's four environments; and their grid is 91×69 against this study's 94×72 |
| `warehouse_full_4cam_dynamic.world.sdf` | its obstacles arrive through runtime spawn events, which is the right design for a mid-run change and the wrong one for four static environments captured on one grid |
| `dynamic_world_oracle`'s scenario runner | it captures the world without a robot; this study needs detector outcomes at robot poses |
