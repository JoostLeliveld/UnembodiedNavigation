#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ge 1 ]; then
  OUT_DIR="$1"
else
  TS=$(date +"%Y%m%d_%H%M%S")
  OUT_DIR="logs/rosbags/boundary_only_${TS}"
fi

mkdir -p "$(dirname "$OUT_DIR")"

echo "Recording boundary-only topics to: $OUT_DIR"

ros2 bag record -o "$OUT_DIR" \
  /external_camera/image_raw \
  /external_camera/camera_info \
  /state/bev \
  /goal_bev \
  /plan \
  /costmap \
  /tf /tf_static \
  /odom
