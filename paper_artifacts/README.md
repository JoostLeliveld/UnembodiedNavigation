# Paper Artifacts

This folder contains the curated artifact bundle used by the paper-facing AWS
warehouse robustness campaign. It is intentionally not a raw log archive.

## Contents

| Path | Role |
| --- | --- |
| [`gp/warehouse_visibility_gp_v1/`](gp/warehouse_visibility_gp_v1/) | Planner-facing GP reliability artifact and fit metadata. |
| [`metrics/`](metrics/) | Regenerated robustness metrics live here; archived generated metrics are under `metrics/archive/`. |
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
validation metrics used by the paper-facing run.

## Current Campaign Summary

The locked campaign uses `warehouse_aws.world.sdf`, GP
`warehouse_visibility_gp_v1`, detector `warehouse_yolo_detector_v1`, and four
tasks with five seeds per condition. Regenerate metrics and the robustness
spread figure from a completed canonical campaign with:

```bash
scripts/visibility_comparison/build_paper_outputs.sh \
  logs/visibility_comparison/warehouse_visibility_campaign_v1
```
