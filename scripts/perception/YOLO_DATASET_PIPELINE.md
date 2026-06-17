# YOLO Dataset Pipeline

This is the detector data path used for simulator-based YOLO training. Semantic
segmentation is used only to generate offline training labels; runtime detection
still consumes RGB images.

## Simulator Launch

Start Gazebo with the dataset-only semantic-label bridge:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch sim bringup_sim.launch.py \
  world:=warehouse_aws.world.sdf \
  show_pose_markers:=false \
  use_lidar:=false \
  bridge_scan:=false \
  bridge_contacts:=false \
  bridge_segmentation:=true \
  headless:=false
```

The normal runtime launch keeps `bridge_segmentation:=false`.

## Capture

```bash
python3 scripts/perception/capture_yolo_dataset.py \
  --world warehouse_aws.world.sdf \
  --out logs/perception_datasets/warehouse_yolo_dataset_v1 \
  --archive-existing \
  --sample-nx 16 \
  --sample-ny 14 \
  --yaw-samples 8 \
  --settle-s 0.80 \
  --image-timeout-s 8.0 \
  --sync-slop-ms 60 \
  --min-new-rgb-frames 3 \
  --min-new-label-frames 1 \
  --max-sample-attempts 4 \
  --save-masks
```

Accepted samples must pass:

- fresh RGB and semantic-label frames after teleport settle;
- RGB/label header sync within `--sync-slop-ms`;
- visible semantic robot mask with minimum area and bbox size;
- projected commanded pose close to the semantic mask;
- robot-colored pixels visible in the RGB crop;
- exact RGB frame not already accepted.

The output includes `data.yaml`, `label_diagnostics.csv`,
`dataset_manifest.json`, accepted/rejected overlays, and contact sheets under
`audit/`.

## Train

The wrapper captures, audits, archives existing outputs, and trains:

```bash
scripts/perception/train_yolo_detector.sh
```

Training should only proceed if dataset capture exits successfully and the
manifest reports acceptable duplicate rate, train/val coverage, and acceptance
fraction.
