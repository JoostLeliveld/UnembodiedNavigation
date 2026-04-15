# Perception Cleanup Audit

This audit is based on source code and launch wiring, not on historical notes.

## Existing Perception Nodes

- [`src/perception/perception/nodes/yolo_robot_detector_node.py`](/home/joostleliveld/Thesis/UnembodiedNavigation/src/perception/perception/nodes/yolo_robot_detector_node.py)
  - status: `keep`
- [`src/perception/perception/nodes/image_marker_detector_node.py`](/home/joostleliveld/Thesis/UnembodiedNavigation/src/perception/perception/nodes/image_marker_detector_node.py)
  - status: `archive`
- [`src/perception/perception/nodes/homography_sim_node.py`](/home/joostleliveld/Thesis/UnembodiedNavigation/src/perception/perception/nodes/homography_sim_node.py)
  - status: `archive`

## Which Node Is Used By The Main Launch?

The main experiment launch still defaults to:

- `perception_backend:=image_markers`

Source:

- [`src/experiments/launch/warehouse_primary_comparison.launch.py`](/home/joostleliveld/Thesis/UnembodiedNavigation/src/experiments/launch/warehouse_primary_comparison.launch.py)

So the default main launch is **not** yet YOLO-first. The YOLO path is active when `perception_backend:=yolo` is selected.

## Published And Subscribed Topics

### YOLO runtime node

File:

- [`src/perception/perception/nodes/yolo_robot_detector_node.py`](/home/joostleliveld/Thesis/UnembodiedNavigation/src/perception/perception/nodes/yolo_robot_detector_node.py)

Subscribes:

- `/external_camera/image_raw`

Publishes:

- `/perception/pixel_pose`
- `/perception/detection_diagnostics`

### Downstream state estimator

File:

- [`src/state/state/nodes/pixel_to_bev_state_node.py`](/home/joostleliveld/Thesis/UnembodiedNavigation/src/state/state/nodes/pixel_to_bev_state_node.py)

Subscribes:

- `/perception/pixel_pose`
- `/perception/detection_diagnostics`
- `/odom`

Publishes:

- `/state/bev`

## Detector Paths: Active, Stale, Duplicated, Misleading

### Active

- YOLO runtime node
- local `.pt` model loading
- mask-bottom to bbox-bottom pixel selection

### Stale or misleading

- image-marker/blob detector path
- synthetic homography detector path
- older perception benchmark/report framework
- SAM-based promptmask path as a central story
- red-mask presented as more than an offline bootstrap

## Old Experiment Scripts

### Keep

- [`scripts/perception/test_yolo_out_of_box.py`](/home/joostleliveld/Thesis/UnembodiedNavigation/scripts/perception/test_yolo_out_of_box.py)
- [`scripts/perception/make_preview_grid.py`](/home/joostleliveld/Thesis/UnembodiedNavigation/scripts/perception/make_preview_grid.py)
- [`scripts/perception/train_yolo_seg.py`](/home/joostleliveld/Thesis/UnembodiedNavigation/scripts/perception/train_yolo_seg.py)
- [`scripts/perception/make_redmask_pseudolabels.py`](/home/joostleliveld/Thesis/UnembodiedNavigation/scripts/perception/make_redmask_pseudolabels.py)

### Delete candidates implemented in this cleanup

- old benchmark framework
- old report generator
- old promptmask/SAM training path
- old capture helper tied to the previous broader workflow
- internal perception helper framework
- `scripts/capture_yolo_seg_dataset.py`
- `scripts/filter_yolo_dataset_by_red_visibility.py`
- `scripts/inspect_yolo_dataset.py`

## Where Red-Mask Is Still Used

After cleanup, red-mask is only used in:

- [`scripts/perception/make_redmask_pseudolabels.py`](/home/joostleliveld/Thesis/UnembodiedNavigation/scripts/perception/make_redmask_pseudolabels.py)

It is offline only.

## Where YOLO Is Used

- runtime inference:
  - [`src/perception/perception/nodes/yolo_robot_detector_node.py`](/home/joostleliveld/Thesis/UnembodiedNavigation/src/perception/perception/nodes/yolo_robot_detector_node.py)
- out-of-box sanity test:
  - [`scripts/perception/test_yolo_out_of_box.py`](/home/joostleliveld/Thesis/UnembodiedNavigation/scripts/perception/test_yolo_out_of_box.py)
- fine-tuning:
  - [`scripts/perception/train_yolo_seg.py`](/home/joostleliveld/Thesis/UnembodiedNavigation/scripts/perception/train_yolo_seg.py)

## Where SAM Is Used

After cleanup, SAM is not part of the active perception path.

The earlier promptmask/SAM script is removed from the active surface.

## Where Homography / State Estimation Is Handled

- homography and BEV conversion:
  - [`src/state/state/nodes/pixel_to_bev_state_node.py`](/home/joostleliveld/Thesis/UnembodiedNavigation/src/state/state/nodes/pixel_to_bev_state_node.py)
- legacy synthetic homography detector:
  - [`src/perception/perception/nodes/homography_sim_node.py`](/home/joostleliveld/Thesis/UnembodiedNavigation/src/perception/perception/nodes/homography_sim_node.py)

## README / Doc Claims That Were Too Broad

The stale claims were:

- multiple competing perception stories presented as equally current
- perception described as broader than “robot bottom-pixel detector”
- SAM implied as more central than it actually is
- old benchmark/report tooling implying a more final evaluation story than the code justified

There is still one compatibility caveat:

- the main experiment launch still defaults to `perception_backend:=image_markers`
- the cleaned YOLO path is active when `perception_backend:=yolo` is selected with a valid local model path

## Minimal Runtime Path After Cleanup

`camera image -> YOLO -> best detection -> mask-bottom pixel or bbox-bottom pixel -> /perception/pixel_pose + diagnostics -> downstream homography/state estimator`

That is the active minimal perception story.
