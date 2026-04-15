# `perception`

This package provides the external-camera image observation used by the runtime pipeline.

## Active Runtime Node

- [`perception/nodes/yolo_robot_detector_node.py`](perception/nodes/yolo_robot_detector_node.py)

It publishes:

- `/perception/pixel_pose`
- `/perception/detection_diagnostics`

The node outputs an image-space observation only.

## Downstream Interpretation

Downstream, the state path uses:

- `x,y` from the selected image pixel via homography
- `theta` from odometry fallback in the state estimator

## Legacy Nodes

These still exist for launch compatibility or older experiments, but they are not the active paper-facing path:

- [`perception/nodes/image_marker_detector_node.py`](perception/nodes/image_marker_detector_node.py)
- [`perception/nodes/homography_sim_node.py`](perception/nodes/homography_sim_node.py)

## Documentation

- [`../../docs/perception_training_and_inference.md`](/home/joostleliveld/Thesis/UnembodiedNavigation/docs/perception_training_and_inference.md)
- [`../../docs/perception_cleanup_audit.md`](/home/joostleliveld/Thesis/UnembodiedNavigation/docs/perception_cleanup_audit.md)
