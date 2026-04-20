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
- run inference with a zero floor so raw candidate scores are preserved
- choose the best target-class candidate before thresholding
- use mask-bottom pixel if a usable mask exists
- otherwise use bbox-bottom pixel
- publish:
  - `/perception/pixel_pose`
  - `/perception/detection_diagnostics`

The publish decision stays threshold-gated, but the diagnostics and logs now preserve:

- `yolo_score_raw`
- `yolo_score_selected`
- `yolo_detected_after_threshold`
- `yolo_best_class_id`
- selected pixel-source metadata

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

The current dataset builders now default to stronger robustness settings:

- `--yaw-samples 8` instead of a single heading
- grouped split modes via `--split-mode {cyclic,yaw_bucket,spatial_cell,spatial_yaw_bucket}`
- deterministic split metadata in each dataset manifest

## Optional Red-Mask Pseudo-Labels

Use:

- [`scripts/perception/make_redmask_pseudolabels.py`](/home/joostleliveld/Thesis/UnembodiedNavigation/scripts/perception/make_redmask_pseudolabels.py)

This is only an offline bootstrap tool:

- projected robot bbox -> red pixels inside bbox -> segmentation pseudo-label

Red-mask is not the runtime detector and not the thesis contribution.

## Dataset Robustness Audit

Use:

- [`scripts/perception/analyze_dataset_robustness.py`](/home/joostleliveld/Thesis/UnembodiedNavigation/scripts/perception/analyze_dataset_robustness.py)

Purpose:

- count yaw coverage in capture or dataset manifests
- flag train/val leakage such as exact pose overlap or suspiciously small cross-split distances
- catch single-orientation datasets before they are used for YOLO training or calibration claims

This script is meant to be run after dataset generation and before training.

## Calibration

The active comparison pipeline now includes an explicit calibration step for YOLO raw scores:

- raw scores are preserved offline and at runtime
- temperature scaling is fitted against canonical capture labels
- reliability, Brier, ECE, PR, ROC, score histograms, and view-angle plots are generated for audit

The calibrated output is used as `yolo_score_calibrated` in the comparison backbone. It should still be treated as an empirical calibration product rather than a universal probability.

## What Is Not Claimed

- The runtime detector uses a local YOLO `.pt` model only.
- The runtime output is an image point, not a full pose.
- BEV `x,y` are handled downstream by the state estimator.
- Heading `theta` is not estimated by YOLO; it remains odometry-backed downstream.
- SAM is not part of the active runtime path.
- A single global calibration fit does not by itself prove robustness across orientation, border margin, or world changes.
