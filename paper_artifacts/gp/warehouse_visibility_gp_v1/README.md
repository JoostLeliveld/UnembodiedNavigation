# Locked Current GP Artifact

Status: `LOCKED_CURRENT_CORRECT`.

This directory contains the current planner-facing empirical reliability GP:

- `yolo_score_raw_gp.npz`
- `gp_manifest.json`

The artifact matches `logs/visibility_comparison/warehouse_visibility_targets_v1/gp_targets_xy_aggregated.csv`
and is the GP referenced by the current campaign configs.

Key audit values:

| field | value |
| --- | ---: |
| train points | 139 |
| `p_train_mean` | 0.5971043165 |
| `P_conservative_plan_map` mean | 0.5727276236 |
| artifact sha256 | `ccbc058311f0e6feeac9aacf034f474af202fba712bd2141752cdfc62de192c8` |
| target table sha256 | `f7698d64316a13dd36c0c10ce3571a842e1922e2cfa45f84ed4e95906700e775` |

Known wrong/mismatched generated copy:

- `logs/visibility_comparison/archive/mismatched_warehouse_visibility_gp_v1_20260709/`

Known older superseded paper artifact:

- `paper_artifacts/gp/archive/aws_gp_v7b_superseded/`
