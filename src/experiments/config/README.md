# `experiments/config`

This folder defines the experiment before anything is launched.

## Main Files

| File | Role |
| --- | --- |
| [`world_profiles.yaml`](world_profiles.yaml) | world registry, camera intrinsics, planner defaults, map bounds, and legacy packaged artifact paths |
| [`tasks.yaml`](tasks.yaml) | start/goal definitions, with each task labeled as benchmark, exploratory, sanity, or legacy |
| `../../archive/experiments/tasks_legacy.yaml` | archived aliases for older task names |

## Current Paper Surface

The compact benchmark currently reported by the paper uses:

- `warehouse_occ_light.world.sdf`
- `shadow_tradeoff_a` as the main task
- `shadow_tradeoff_b` and `sanity_open` as support tasks

`main_shadow_tradeoff` is legacy and must not be used as current paper evidence.

## Failure-Oriented Extensions

The following worlds/tasks are implemented for the next benchmark step, but they need completed runs and paper figures before they can support result claims:

- `warehouse_aws.world.sdf`: `B1_visible_goal_smoke` (first stability gate with visible final goal), `B1_clean_route_choice` (harder exploratory route-choice candidate), `B2_reacquire_after_occlusion` (stress), `B3_long_horizon_far_corner` (stress), `visible_aisle_sanity_aws`

## Important Rule

Visibility-aware paper runs must pass an explicit `visibility_artifact_path`. The profile-level `visibility_artifact` entries are compatibility metadata, not permission to silently choose an artifact for a paper run.
