# Paper Artifacts

This folder contains curated GP, detector, figure, and historical metric
artifacts. It is intentionally not a raw log archive. The active current result
surface lives in `docs/paper_vs_current/current/`.

## Contents

| Path | Role |
| --- | --- |
| [`gp/warehouse_visibility_gp_v1/`](gp/warehouse_visibility_gp_v1/) | Planner-facing GP reliability artifact and fit metadata. |
| [`metrics/`](metrics/) | Historical generated metrics are under `metrics/archive/`; current headline metrics live in `../docs/paper_vs_current/current/`. |
| [`figures/`](figures/) | Paper-ready figure assets and provenance/caption sidecars. |
| [`perception/warehouse_yolo_detector_v1/`](perception/warehouse_yolo_detector_v1/) | YOLO training/validation metadata and representative validation image. |

## Visual Catalog

| Asset | Role |
| --- | --- |
| [`figures/problem_setup_camera.png`](figures/problem_setup_camera.png) | External-camera Gazebo warehouse setup. |
| [`figures/gp_pipeline_aws.png`](figures/gp_pipeline_aws.png) | Detector score samples, conservative GP reliability, and induced covariance. |
| [`figures/localization_pathway.png`](figures/localization_pathway.png) | Runtime path from camera image to BEV state and planner correction. |
| [`perception/warehouse_yolo_detector_v1/val_batch0_pred.jpg`](perception/warehouse_yolo_detector_v1/val_batch0_pred.jpg) | YOLO validation predictions. |
| [`perception/warehouse_yolo_detector_v1/results.png`](perception/warehouse_yolo_detector_v1/results.png) | YOLO training curves. |

## External Local Artifacts

The YOLO checkpoint is not tracked in git. To run the Gazebo campaign locally,
place it at:

```text
logs/perception_models/warehouse_yolo_detector_v1/model.pt
```

The locked campaign config points to that path. The detector training metadata
in `perception/warehouse_yolo_detector_v1/` records the dataset, split, and
validation metrics used by the current run.

## Current Campaign Summary

The current campaign uses `warehouse_aws.world.sdf`, GP
`warehouse_visibility_gp_v1`, detector `warehouse_yolo_detector_v1`, and four
tasks with five seeds per condition. Its packaged result summary is
`../docs/paper_vs_current/current/README.md`; historical submitted-paper metrics
remain archived under `metrics/archive/`.
