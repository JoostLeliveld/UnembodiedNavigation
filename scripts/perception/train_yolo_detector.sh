#!/bin/bash
# YOLO detector training pipeline.
#
# It launches Gazebo with the semantic-label bridge enabled, captures a gated
# simulator-segmentation dataset, audits it, then trains YOLO on that dataset.
# Existing dataset/model folders are archived before replacement.
#
# NOTE: no 'set -e' / 'set -u' because ROS setup.bash can trip them.

cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
source /opt/ros/humble/setup.bash >/dev/null 2>&1
source install/setup.bash >/dev/null 2>&1
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-77}"

DATASET="${DATASET:-logs/perception_datasets/warehouse_yolo_dataset_v1}"
MODELOUT="${MODELOUT:-logs/perception_models/warehouse_yolo_detector_v1}"
BASE_MODEL="${BASE_MODEL:-local_artifacts/base_models/yolo11n-seg.pt}"
GZLOG=/tmp/yolo_train_gazebo.log
CAPLOG=/tmp/yolo_dataset_capture.log
TRLOG=/tmp/yolo_detector_train.log
STAMP="$(date +%Y%m%d_%H%M%S)"

archive_path() {
  local path="$1"
  if [ -e "$path" ]; then
    local parent
    parent="$(dirname "$path")"
    mkdir -p "$parent/archive"
    local target="$parent/archive/$(basename "$path")_$STAMP"
    echo "[yolo-train] archiving $path -> $target"
    mv "$path" "$target"
  fi
}

echo "[yolo-train] archiving existing dataset/model outputs"
archive_path "$DATASET"
archive_path "$MODELOUT"

echo "[yolo-train] cleaning + bringing up Gazebo with dataset-only segmentation bridge"
pkill -9 -f "ign gazebo" >/dev/null 2>&1
pkill -9 -f "gz sim" >/dev/null 2>&1
sleep 3
ros2 launch sim bringup_sim.launch.py \
  world:=warehouse_aws.world.sdf \
  show_pose_markers:=false \
  use_lidar:=false \
  bridge_scan:=false \
  bridge_contacts:=false \
  bridge_segmentation:=true \
  headless:=false > "$GZLOG" 2>&1 &
GZ=$!

echo "[yolo-train] waiting for RGB and semantic label topics (up to 120s)"
ok_rgb=0
ok_seg=0
for i in $(seq 1 40); do
  if timeout 3 ros2 topic echo --once /external_camera/image_raw >/dev/null 2>&1; then ok_rgb=1; fi
  if timeout 3 ros2 topic echo --once /external_camera/segmentation/labels_map >/dev/null 2>&1; then ok_seg=1; fi
  if [ "$ok_rgb" = "1" ] && [ "$ok_seg" = "1" ]; then break; fi
  sleep 3
done
if [ "$ok_rgb" != "1" ] || [ "$ok_seg" != "1" ]; then
  echo "[yolo-train] ABORT: required camera topics missing. topics seen:"
  ros2 topic list 2>/dev/null | grep -i camera
  echo "[yolo-train] Gazebo log tail:"
  tail -30 "$GZLOG"
  pkill -9 -f "ign gazebo" >/dev/null 2>&1
  pkill -9 -f "gz sim" >/dev/null 2>&1
  exit 1
fi

echo "[yolo-train] capturing gated semantic-segmentation dataset"
python3 scripts/perception/capture_yolo_dataset.py \
  --world warehouse_aws.world.sdf \
  --out "$DATASET" \
  --sample-nx 16 \
  --sample-ny 14 \
  --yaw-samples 8 \
  --settle-s 0.80 \
  --image-timeout-s 8.0 \
  --sync-slop-ms 60 \
  --min-new-rgb-frames 3 \
  --min-new-label-frames 1 \
  --max-sample-attempts 4 \
  --min-mask-area 80 \
  --min-mask-bbox-w 8 \
  --min-mask-bbox-h 8 \
  --max-expected-center-error-px 90 \
  --min-rgb-robot-color-fraction 0.015 \
  --max-final-duplicate-fraction 0.02 \
  --min-accepted-samples 400 \
  --min-accept-fraction 0.25 \
  --save-masks \
  --preview-count 200 \
  --rejected-preview-count 120 > "$CAPLOG" 2>&1
CAPRC=$?
echo "[yolo-train] capture exit=$CAPRC"
tail -40 "$CAPLOG"

pkill -9 -f "ign gazebo" >/dev/null 2>&1
pkill -9 -f "gz sim" >/dev/null 2>&1
sleep 3

if [ "$CAPRC" != "0" ] || [ ! -f "$DATASET/data.yaml" ]; then
  echo "[yolo-train] ABORT: dataset capture failed"
  exit 1
fi

echo "[yolo-train] training YOLO segmentation model"
python3 scripts/perception/train_yolo_seg.py \
  --data "$DATASET/data.yaml" \
  --base-model "$BASE_MODEL" \
  --device 0 \
  --out "$MODELOUT" \
  --epochs 30 \
  --imgsz 960 \
  --batch 4 > "$TRLOG" 2>&1
TRRC=$?
echo "[yolo-train] train exit=$TRRC; model files:"
find "$MODELOUT" -iname "*.pt" 2>/dev/null
tail -30 "$TRLOG"

if [ "$TRRC" != "0" ]; then
  echo "[yolo-train] ABORT: training failed"
  exit 1
fi
echo "[yolo-train] DONE"
