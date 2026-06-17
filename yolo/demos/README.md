# YOLO Demo Media Plan

[Back to YOLO module](../README.md)

Planned media paths are story slots, not tracked files yet. Existing visuals are
linked from `paper_artifacts/` to avoid duplicated assets.

## Existing Assets

| Asset | Use |
| --- | --- |
| [`../../paper_artifacts/perception/warehouse_yolo_detector_v1/val_batch0_pred.jpg`](../../paper_artifacts/perception/warehouse_yolo_detector_v1/val_batch0_pred.jpg) | Validation examples with selected robot boxes. |
| [`../../paper_artifacts/perception/warehouse_yolo_detector_v1/results.png`](../../paper_artifacts/perception/warehouse_yolo_detector_v1/results.png) | Training curves. |
| [`../../paper_artifacts/figures/yolo_training_clarification.png`](../../paper_artifacts/figures/yolo_training_clarification.png) | Dataset and label-generation explanation. |

## Planned Media Slots

| Planned path | Type | Story beat | Target | Source |
| --- | --- | --- | --- | --- |
| `images/input_01.png` | PNG still | Raw external-camera RGB frame. | GitHub table cell | Capture from `warehouse_aws.world.sdf`. |
| `images/prediction_01.png` | PNG still | Selected detector box and confidence. | GitHub table cell | `yolo_robot_detector_node.py` output overlay. |
| `images/bottom_centre_01.png` | PNG diagnostic | Why bbox bottom-centre is the localization point. | 1200 px wide | Detector diagnostic overlay. |
| `animations/yolo_inference.gif` | GIF preview | Raw frame -> bounding box -> confidence -> bottom-centre point. | 10-15 s | Rendered from a short detector run. |
| `videos/warehouse_detection.mp4` | MP4 clip | Detector overlay over warehouse viewpoints. | 30-45 s | Gazebo camera recording plus detector overlay. |

## Capture Checklist

1. Launch the locked warehouse world with the external camera.
2. Record `/external_camera/image_raw`.
3. Run the detector with `warehouse_yolo_detector_v1/model.pt`.
4. Overlay selected box, score, and bottom-centre pixel.
5. Export a small GIF preview and a compressed MP4.
