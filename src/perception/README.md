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
| [`perception/nodes/image_marker_detector_node.py`](perception/nodes/image_marker_detector_node.py) | primary image-based detector for the current milestone |
| [`perception/core/detection_diagnostics.py`](perception/core/detection_diagnostics.py) | shared encoding for detection diagnostics |

## Support Files

| File | Role |
| --- | --- |
| `perception/nodes/homography_sim_node.py` | synthetic alternative observation path that can provide visual heading |
| `launch/tf_static.launch.py` | static transform support |

## What To Read First

1. `perception/nodes/image_marker_detector_node.py`
2. `perception/core/detection_diagnostics.py`
3. `perception/nodes/homography_sim_node.py` only if you need the synthetic support path

## Important Caveat

The main image-based detector is intentionally simple. It is a controlled simulated detector and currently provides camera-derived `x,y` only. It does **not** provide the full pose used by the planner.
