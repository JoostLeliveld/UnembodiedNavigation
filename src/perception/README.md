# `perception`

This package provides the external-camera image observation used by the runtime pipeline.

It is the first visible piece of the demo: a fixed warehouse camera detects the
TurtleBot and turns the selected detection into an image-space robot
observation.

![YOLO validation predictions](../../paper_artifacts/perception/warehouse_yolo_detector_v1/val_batch0_pred.jpg)

## Active Runtime Node

- [`perception/nodes/yolo_robot_detector_node.py`](perception/nodes/yolo_robot_detector_node.py)

It publishes:

- `/perception/pixel_pose`
- `/perception/detection_diagnostics`

The node loads an Ultralytics YOLO model and detects the configured robot class.

## Detector Modes

| Mode | x, y source | theta source |
| --- | --- | --- |
| Seg/detect model | selected pixel via homography | no visual yaw; downstream uses odometry, displacement heading, or propagated heading depending on launch config |
| Pose model with `keypoint_marker_world_z > 0` | selected pixel via homography | front/rear keypoints back-projected to BEV |

The current paper-facing campaign uses the segmentation/detection path for
`x,y` and odometry-driven heading under `camera_xy_only`.

## Demonstrated Detector

The paper-facing detector is documented in
[`../../docs/perception_details.md`](../../docs/perception_details.md) and
summarized by [`../../paper_artifacts/perception/warehouse_yolo_detector_v1/manifest.json`](../../paper_artifacts/perception/warehouse_yolo_detector_v1/manifest.json).

| Item | Value |
| --- | --- |
| Base model | YOLOv11n-seg |
| Dataset | 852 simulator-labeled warehouse images, split 683 train / 169 validation |
| Runtime selected point | bounding-box bottom centre |
| Runtime masks | disabled in the locked campaign |
| Final box mAP50 | 0.938 |
| Final box mAP50-95 | 0.620 |
| Final mask mAP50 | 0.745 |

Training curves:

![YOLO training curves](../../paper_artifacts/perception/warehouse_yolo_detector_v1/results.png)

## Pose-Keypoint Support (archived)

The pose-keypoint heading path was never wired into control for the final method
(the detector is a segmentation model and heading comes from odometry under
`camera_xy_only`). It was removed from the live nodes in the 2026-07-08 cleanup;
`pose_extraction.py` / `pose_keypoints.py` and their tests are preserved under
`_archive/code/perception_pose/` in case heading estimation is revived.

## Training Scripts

Detector training and dataset generation live under `scripts/perception/`. They are support/provenance tooling, not the runtime method itself.

The trained checkpoint itself is local-only and expected at
`logs/perception_models/warehouse_yolo_detector_v1/model.pt`.
