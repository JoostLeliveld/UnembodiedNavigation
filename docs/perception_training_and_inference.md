# Perception Training And Inference

## Purpose

The perception subsystem has one narrow job:

`camera image -> robot mask or bbox -> selected bottom image point -> /perception/pixel_pose + diagnostics`

It is not the thesis contribution. It is a supporting detector that provides an image-space observation.

## Active Runtime Path

The active runtime node is:

- [`src/perception/perception/nodes/yolo_robot_detector_node.py`](/home/joostleliveld/Thesis/UnembodiedNavigation/src/perception/perception/nodes/yolo_robot_detector_node.py)

Runtime behavior:

- load one local YOLO `.pt` model
- run inference on `/external_camera/image_raw`
- choose the highest-confidence detection after optional class filtering
- use mask-bottom pixel if a usable mask exists
- otherwise use bbox-bottom pixel
- publish:
  - `/perception/pixel_pose`
  - `/perception/detection_diagnostics`

The runtime output is an image point plus diagnostics, not a full pose.

## Out-of-Box YOLO Sanity Test

Use:

- [`scripts/perception/test_yolo_out_of_box.py`](/home/joostleliveld/Thesis/UnembodiedNavigation/scripts/perception/test_yolo_out_of_box.py)

Purpose:

- quickly check whether a local YOLO segmentation model places a reasonable mask or bbox on the robot at all
- save visual overlays and a small `summary.json`

This is only a visual/runtime sanity check, not a thesis-final benchmark.

## Fine-Tuned YOLO Path If Needed

Use:

- [`scripts/perception/train_yolo_seg.py`](/home/joostleliveld/Thesis/UnembodiedNavigation/scripts/perception/train_yolo_seg.py)

If out-of-box YOLO is not good enough:

- create simple pseudo-labels
- fine-tune `YOLO11n-seg`
- save one local `model.pt`

Start with short runs:

- `5-10` epochs for a smoke test
- `20-30` epochs for the first usable model

## Optional Simulator Segmentation Labels

Use:

- [`scripts/perception/capture_simseg_dataset.py`](/home/joostleliveld/Thesis/UnembodiedNavigation/scripts/perception/capture_simseg_dataset.py)

This is the cleanest offline supervision path when manual masks are not the goal:

- fixed external RGB camera
- matching simulator semantic-segmentation camera
- robot label exported offline as a YOLO-seg dataset

This privileged supervision is offline only. Runtime still uses only the RGB camera and the trained YOLO model.

## Optional Red-Mask Pseudo-Labels

Use:

- [`scripts/perception/make_redmask_pseudolabels.py`](/home/joostleliveld/Thesis/UnembodiedNavigation/scripts/perception/make_redmask_pseudolabels.py)

This is only an offline bootstrap tool:

- projected robot bbox -> red pixels inside bbox -> segmentation pseudo-label

Red-mask is not the runtime detector and not the thesis contribution.

## What Is Not Claimed

- The runtime detector uses a local YOLO `.pt` model only.
- The runtime output is an image point, not a full pose.
- BEV `x,y` are handled downstream by the state estimator.
- Heading `theta` is not estimated by YOLO; it remains odometry-backed downstream.
- SAM is not part of the active runtime path.
- Confidence is not calibrated unless an explicit calibration step is added later.
