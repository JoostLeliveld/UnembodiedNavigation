# `experiments/config`

This folder defines reusable experiment metadata before anything is launched:
world profiles, camera/profile defaults, and task definitions.

## Main Files

| File | Role |
| --- | --- |
| [`world_profiles.yaml`](world_profiles.yaml) | World registry, camera intrinsics, planner defaults, map bounds, and compatibility artifact paths. |
| [`tasks.yaml`](tasks.yaml) | Current paper and sanity start/goal definitions. Historical phase-coded tasks live under `archive/`. |

## Current Surface

The current benchmark is the AWS-style warehouse campaign:

- world: `warehouse_aws.world.sdf`
- campaign config: `../../../scripts/visibility_comparison/warehouse_visibility_campaign.yaml`
- detector checkpoint and its manifest: `logs/perception_models/<run>/`
- GP artifact: a fitted `.npz` (the previous curated artifact was retired 2026-08-25)
- tasks: `route_apron_to_a3_mid`, `route_apron_to_a2_mid`,
  `route_west_to_a1_upper`, and `control_west_to_a1_low`

The campaign config is the source of truth for current runs. It pins the
planner conditions, route seeds, detector checkpoint path, GP path, driveable
geometry, process/noise settings, and success/collision criteria.

## Important Rule

Visibility-aware current runs must pass an explicit `visibility_artifact_path` or
config-level GP artifact. Profile-level `visibility_artifact` entries are
compatibility metadata, not permission to silently choose a GP for a current run.
