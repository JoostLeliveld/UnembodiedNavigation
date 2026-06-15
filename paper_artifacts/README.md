# Paper Artifacts

This folder contains the curated artifact bundle used by the paper-facing AWS
warehouse robustness campaign. It is intentionally not a raw log archive.

## Contents

| Path | Role |
| --- | --- |
| [`gp/aws_gp_v7b/`](gp/aws_gp_v7b/) | Planner-facing GP reliability artifact and fit metadata. |
| [`metrics/`](metrics/) | Robustness campaign CSV and compact route/cost cache. |
| [`figures/`](figures/) | Paper-ready figure assets and provenance/caption sidecars. |
| [`perception/aws_yolo_simseg_v2/`](perception/aws_yolo_simseg_v2/) | YOLO training/validation metadata and representative validation image. |

## Visual Catalog

| Asset | Role |
| --- | --- |
| [`figures/problem_setup_camera.png`](figures/problem_setup_camera.png) | External-camera Gazebo warehouse setup. |
| [`figures/gp_pipeline_aws.png`](figures/gp_pipeline_aws.png) | Detector score samples, conservative GP reliability, and induced covariance. |
| [`figures/localization_pathway.png`](figures/localization_pathway.png) | Runtime path from camera image to BEV state and planner correction. |
| [`figures/robustness_spread.png`](figures/robustness_spread.png) | Main C1/C2 seeded route comparison. |
| [`perception/aws_yolo_simseg_v2/val_batch0_pred.jpg`](perception/aws_yolo_simseg_v2/val_batch0_pred.jpg) | YOLO validation predictions. |
| [`perception/aws_yolo_simseg_v2/results.png`](perception/aws_yolo_simseg_v2/results.png) | YOLO training curves. |

## External Local Artifacts

The YOLO checkpoint is not tracked in git. To run the Gazebo campaign locally,
place it at:

```text
local_artifacts/perception_models/aws_yolo_simseg_v2/model.pt
```

The locked campaign config points to that local path. The detector training
metadata in `perception/aws_yolo_simseg_v2/` records the dataset, split, and
validation metrics used by the paper-facing run.

## Current Campaign Summary

The locked campaign uses `warehouse_aws.world.sdf`, GP `aws_gp_v7b`, detector
`aws_yolo_simseg_v2`, and four tasks with five seeds per condition. The current
packaged outcome is:

- C2 visibility-aware EFE: `16/20` clean reaches, `2/20` collisions,
  `1/20` near-success, and `1/20` infrastructure-invalid run.
- C1 constant-R EFE: `12/20` clean reaches and `8/20` collisions.

Continuous localization metrics are pooled over clean successes only. See
[`metrics/robustness_summary.txt`](metrics/robustness_summary.txt) and
[`metrics/robustness_metrics.csv`](metrics/robustness_metrics.csv) for the table
inputs.
