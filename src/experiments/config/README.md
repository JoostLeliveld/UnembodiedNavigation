# `experiments/config`

This folder defines reusable experiment metadata before anything is launched:
world profiles, camera/profile defaults, and task definitions.

## Main Files

| File | Role |
| --- | --- |
| [`world_profiles.yaml`](world_profiles.yaml) | World registry, camera intrinsics, planner defaults, map bounds, and compatibility artifact paths. |
| [`tasks.yaml`](tasks.yaml) | Start/goal definitions, with tasks labeled as benchmark, exploratory, sanity, or legacy. |

## Current Paper Surface

The current paper-facing benchmark is the AWS-style warehouse campaign:

- world: `warehouse_aws.world.sdf`
- campaign config: `../../../scripts/visibility_comparison/aws_f31b1_final_config.yaml`
- detector metadata: `../../../paper_artifacts/perception/aws_yolo_simseg_v2/`
- GP artifact: `../../../paper_artifacts/gp/aws_gp_v7b/yolo_score_raw_gp.npz`
- tasks: `F31_b1_apron_a3_mid`, `b5_a4_apron_to_a2_mid`,
  `b2_a0_west_to_a1_upper`, and `b6_a0_west_to_a1_low_control`

The campaign config is the source of truth for paper-facing runs. It pins the
planner conditions, route seeds, detector checkpoint path, GP path, driveable
geometry, process/noise settings, and success/collision criteria.

## Important Rule

Visibility-aware paper runs must pass an explicit `visibility_artifact_path` or
config-level GP artifact. Profile-level `visibility_artifact` entries are
compatibility metadata, not permission to silently choose a GP for a paper run.
