# YOLO Demo Media

[Back to YOLO module](../README.md)

This folder holds README-facing media for the detector contribution. Regenerate
the static assets from the repository root with:

```bash
python3 scripts/paper_figures/make_readme_visuals.py
```

## Available README Media

| Path | Type | Story beat |
| --- | --- | --- |
| [`images/bottom_centre_01.png`](images/bottom_centre_01.png) | PNG diagnostic | Camera frame, selected detector box, and bottom-centre localization point. |

## Existing Source Assets

| Asset | Use |
| --- | --- |
| [`../../../paper_artifacts/perception/warehouse_yolo_detector_v1/val_batch0_pred.jpg`](../../../paper_artifacts/perception/warehouse_yolo_detector_v1/val_batch0_pred.jpg) | Validation examples with selected robot boxes. |
| [`../../../paper_artifacts/perception/warehouse_yolo_detector_v1/results.png`](../../../paper_artifacts/perception/warehouse_yolo_detector_v1/results.png) | Training curves. |
| [`../../../paper_artifacts/figures/yolo_training_clarification.png`](../../../paper_artifacts/figures/yolo_training_clarification.png) | Dataset and label-generation explanation. |

## Next Media Slots

| Planned path | Type | Story beat | Source |
| --- | --- | --- | --- |
| `animations/yolo_inference.gif` | GIF preview | Raw frame to bounding box to bottom-centre point. | Short detector run. |
| `videos/warehouse_detection.mp4` | MP4 clip | Detector overlay over warehouse viewpoints. | Gazebo camera recording plus detector overlay. |
