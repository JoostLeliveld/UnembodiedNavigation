#!/bin/bash
# Retrain YOLO on the CURRENT camera (the camera moved after the 2026-05-13 training,
# so aws_yolo_simseg_v2 is out-of-distribution). Capture a fresh seg dataset on the
# current warehouse_aws world (teleport grid + Gazebo semantic labels), then fine-tune.
# NOTE: no 'set -e' / 'set -u' (ROS setup.bash trips them).
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
source /opt/ros/humble/setup.bash >/dev/null 2>&1
source install/setup.bash >/dev/null 2>&1
export ROS_DOMAIN_ID=77
DATASET="logs/perception_datasets/aws_simseg_v3_currentcam"
MODELOUT="logs/perception_models/aws_yolo_simseg_v3"
GZLOG=/tmp/retrain_gazebo.log; CAPLOG=/tmp/retrain_capture.log; TRLOG=/tmp/retrain_train.log

echo "[retrain] cleaning + bringing up Gazebo (current camera, markers off)"
pkill -9 -f "ign gazebo" >/dev/null 2>&1; pkill -9 -f "gz sim" >/dev/null 2>&1; sleep 3
ros2 launch sim bringup_sim.launch.py world:=warehouse_aws.world.sdf show_pose_markers:=false use_lidar:=false > "$GZLOG" 2>&1 &
GZ=$!

echo "[retrain] waiting for the segmentation topic to publish (up to 90s)"
ok=0
for i in $(seq 1 30); do
  if timeout 4 ros2 topic echo --once /external_camera/segmentation/labels_map >/dev/null 2>&1; then ok=1; break; fi
  sleep 3
done
if [ "$ok" != "1" ]; then
  echo "[retrain] ABORT: /external_camera/segmentation/labels_map never published. topics seen:"; ros2 topic list 2>/dev/null | grep -i camera; echo "Gazebo log tail:"; tail -15 "$GZLOG"
  pkill -9 -f "ign gazebo" >/dev/null 2>&1; exit 1
fi
echo "[retrain] segmentation topic live -> capturing dataset (480-sample grid)"
python3 scripts/perception/capture_simseg_dataset.py --world warehouse_aws.world.sdf --out "$DATASET" > "$CAPLOG" 2>&1
CAPRC=$?
echo "[retrain] capture exit=$CAPRC; images:"; find "$DATASET" -iname "*.jpg" -o -iname "*.png" 2>/dev/null | wc -l
pkill -9 -f "ign gazebo" >/dev/null 2>&1; pkill -9 -f "gz sim" >/dev/null 2>&1; sleep 3

if [ ! -f "$DATASET/data.yaml" ]; then echo "[retrain] ABORT: no data.yaml produced"; tail -20 "$CAPLOG"; exit 1; fi
echo "[retrain] training YOLO11n-seg on GPU (30 epochs)"
python3 scripts/perception/train_yolo_seg.py --data "$DATASET/data.yaml" \
  --base-model local_artifacts/base_models/yolo11n-seg.pt --device 0 --out "$MODELOUT" --epochs 30 --imgsz 640 --batch 8 > "$TRLOG" 2>&1
echo "[retrain] train exit=$?; model:"; find "$MODELOUT" -iname "*.pt" 2>/dev/null
echo "[retrain] DONE"
