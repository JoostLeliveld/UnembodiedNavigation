# `perception`

This package turns external-camera information into robot observations.

![Perception-to-state tutorial figure](../../docs/figures/state_pipeline_tutorial.png)

The detector is intentionally simple: it produces the image-space observation that starts the rest of the pipeline.

## Why This Folder Exists

The thesis setup depends on an external camera. This package provides the observation path from camera image or synthetic camera model to a pixel-space robot observation.

## Inputs And Outputs

- **Inputs**
  - `/external_camera/image_raw` in the primary path
  - `/odom` in the synthetic homography support path
- **Outputs**
  - `/perception/pixel_pose`
  - `/perception/detection_diagnostics`

## Central Files

| File | Role |
| --- | --- |
| [`perception/nodes/image_marker_detector_node.py`](perception/nodes/image_marker_detector_node.py) | legacy/simple image-based detector for marker-style runs |
| [`perception/nodes/yolo_robot_detector_node.py`](perception/nodes/yolo_robot_detector_node.py) | YOLO detector/segmenter backend for robot masks, mask-bottom pixels, and confidence scores |
| [`perception/core/detection_diagnostics.py`](perception/core/detection_diagnostics.py) | shared encoding for detection diagnostics |

## Support Files

| File | Role |
| --- | --- |
| `perception/nodes/homography_sim_node.py` | synthetic alternative observation path that can provide visual heading |
| `launch/tf_static.launch.py` | static transform support |

## What To Read First

1. `perception/nodes/yolo_robot_detector_node.py`
2. `perception/core/detection_diagnostics.py`
3. `perception/nodes/image_marker_detector_node.py` only if you need the legacy marker/blob support path

## Important Caveat

The main image-based detector is intentionally simple. It is a controlled simulated detector and currently provides camera-derived `x,y` only. It does **not** provide the full pose used by the planner.

## Optional YOLO Backend

The YOLO backend is enabled with `perception_backend:=yolo` and requires optional Python packages in the active ROS environment:

```bash
pip install ultralytics huggingface_hub
```

It keeps the same downstream contract as the red-blob detector. With a YOLO-seg model and `yolo_use_masks:=true`, `/perception/pixel_pose` is the bottom band of the selected robot mask; if no usable mask is available, the node falls back to the bottom-center of the selected bounding box. `/perception/detection_diagnostics` appends YOLO confidence, bbox metadata, mask metadata, and `confidence_logit = logit(yolo_score)`. The confidence is a detector-derived soft observability score, not a calibrated probability unless a later calibration step is added.

SAM and simulator-projected robot boxes are offline training helpers only. At runtime this package uses the trained YOLO model plus the existing homography/state-estimation path.
