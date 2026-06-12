# Paper Artifacts

This folder contains the small curated artifact bundle used by the paper-facing
AWS robustness campaign. It is intentionally not a raw log archive.

## Contents

| Path | Role |
| --- | --- |
| `gp/aws_gp_v7b/` | Planner-facing GP reliability artifact and fit metadata. |
| `metrics/` | Robustness campaign table inputs and compact route/cost cache. |
| `figures/` | Paper-ready figure assets and provenance/caption sidecars. |
| `perception/aws_yolo_simseg_v2/` | YOLO training/validation metadata and one representative validation image. |

## External Local Artifacts

The YOLO checkpoint is not tracked in git. To run the Gazebo campaign locally,
place it at:

```text
local_artifacts/perception_models/aws_yolo_simseg_v2/model.pt
```

The locked campaign config points to that local path. The detector training
metadata in `perception/aws_yolo_simseg_v2/` records the dataset, split, and
validation metrics used in the paper.

## Current Campaign Summary

The locked campaign uses `warehouse_aws.world.sdf`, GP `aws_gp_v7b`, detector
`aws_yolo_simseg_v2`, and four tasks with five seeds per condition. The headline
outcome is:

- C2 visibility-aware EFE: `18/20` clean reaches, `2/20` collisions.
- C1 constant-R EFE: `12/20` clean reaches, one near-success, `7/20` collisions.

Continuous localization metrics are pooled over clean successes only.
