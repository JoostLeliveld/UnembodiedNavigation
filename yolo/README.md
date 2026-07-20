# YOLO External-Camera Detection

[Back to repository overview](../README.md)

This module detects the robot in the fixed external-camera RGB image and
exports the selected bottom-centre pixel as the runtime localization point.

## Contribution At A Glance

| Question | Answer |
| --- | --- |
| Problem | The warehouse camera sees the robot from an oblique external view, so the planner needs a stable image-space localization point. |
| Contribution | A detector pipeline selects the robot box, exports the bottom-centre pixel, and logs the raw score as a reliability signal for later GP fitting. |
| Implementation | Runtime selection lives in [`../src/perception/perception/nodes/yolo_robot_detector_node.py`](../src/perception/perception/nodes/yolo_robot_detector_node.py) and [`../src/perception/perception/core/yolo_selection.py`](../src/perception/perception/core/yolo_selection.py). |

## Visual Demonstration

![Bottom-centre detector output](demos/images/bottom_centre_01.png)

The detector does not output a full pose. Runtime localization uses one selected
image point: the bounding-box bottom centre, treated as a ground-contact proxy
for the BEV projection step.

Additional media is catalogued in [`demos/`](demos/).

## Inputs And Outputs

| Input | Output |
| --- | --- |
| `/external_camera/image_raw` | `/perception/pixel_pose` |
| `logs/perception_models/warehouse_yolo_detector_v1/model.pt` | `/perception/detection_diagnostics` |
| YOLO confidence and class settings | selected pixel `(u, v)`, raw score, detection flag |

## Method

1. The Gazebo camera publishes RGB images from the locked warehouse camera pose.
2. A YOLOv11n-seg model selects the configured `robot` class.
3. The runtime uses the bounding-box bottom centre, not the mask, as the
   localization pixel.
4. The raw confidence score is recorded as an empirical reliability signal for
   the GP pipeline.

## Performance And Diagnostics

Detector metadata lives in
[`../paper_artifacts/perception/warehouse_yolo_detector_v1/manifest.json`](../paper_artifacts/perception/warehouse_yolo_detector_v1/manifest.json).
The packaged training run used 852 simulator-labeled images, split into 683
training images and 169 validation images.

![YOLO training curves](../paper_artifacts/perception/warehouse_yolo_detector_v1/results.png)

| Metric | Box | Mask |
| --- | ---: | ---: |
| Precision | 0.982 | 0.795 |
| Recall | 0.889 | 0.769 |
| mAP50 | 0.938 | 0.745 |
| mAP50-95 | 0.620 | 0.250 |

Detector accuracy and metric localization accuracy are different quantities. A
high raw YOLO score does not guarantee a small projected position residual.
The detector score is an empirical signal, not a calibrated visibility
probability.

## Reproduce

Train a detector from a prepared dataset:

```bash
python3 scripts/perception/train_yolo_seg.py \
  --data logs/perception_datasets/warehouse_yolo_dataset_v1/data.yaml \
  --base-model local_artifacts/base_models/yolo11n-seg.pt \
  --epochs 30 \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --out logs/perception_models/warehouse_yolo_detector_v1
```

Regenerate the detector-training clarification figure:

```bash
python3 scripts/paper_figures/make_yolo_training_clarification.py
```

## Relevant Implementation Files

| File | Role |
| --- | --- |
| [`../src/perception/perception/nodes/yolo_robot_detector_node.py`](../src/perception/perception/nodes/yolo_robot_detector_node.py) | Runtime ROS detector node. |
| [`../src/perception/perception/core/yolo_selection.py`](../src/perception/perception/core/yolo_selection.py) | Detection selection and class filtering. |
| [`../scripts/perception/train_yolo_seg.py`](../scripts/perception/train_yolo_seg.py) | YOLO fine-tuning wrapper. |
| [`../docs/perception_details.md`](../docs/perception_details.md) | Full detector configuration and dataset notes. |

## Limitations

- The YOLO checkpoint is local-only and not tracked in git.
- The raw detector score is uncalibrated.
- Runtime masks are disabled in the locked campaign.
- The detector supplies image-space `x,y` evidence only; heading is handled by
  the state/planning stack.

See available and planned media in [`demos/`](demos/).
