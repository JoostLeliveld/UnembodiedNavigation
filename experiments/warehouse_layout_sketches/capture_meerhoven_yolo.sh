#!/bin/bash
# Capture one immutable, route-held-out YOLO dataset per Meerhoven camera.
# CAMERA_FILTER=camera_D can be used for one camera; empty means A--L.

set -o pipefail

cd "$(dirname "$0")/../.." || exit 1
MEERHOVEN_ROOT="$(pwd)"
source /opt/ros/humble/setup.bash >/dev/null 2>&1
source install/setup.bash >/dev/null 2>&1

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-98}"
export ROS_LOCALHOST_ONLY=1
export IGN_IP=127.0.0.1
export GZ_IP=127.0.0.1
export IGN_PARTITION="${IGN_PARTITION:-meerhoven_yolo_v1}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/roslog_meerhoven_yolo_v1}"
export IGN_LOG_PATH="${IGN_LOG_PATH:-/tmp/ignlog_meerhoven_yolo_v1}"
mkdir -p "$ROS_LOG_DIR" "$IGN_LOG_PATH"

CAPTURE_ROOT="${CAPTURE_ROOT:-logs/perception_datasets/warehouse_meerhoven_yolo_v1}"
RUN_LOG_ROOT="${RUN_LOG_ROOT:-logs/studies/warehouse_layout_sketches/meerhoven_yolo_capture_v1}"
ROUTE_CONFIG="experiments/warehouse_layout_sketches/meerhoven_capture_routes.yaml"
CAMERA_FILTER="${CAMERA_FILTER:-}"
mkdir -p "$RUN_LOG_ROOT"

CAMERA_IDS=(camera_A camera_B camera_C camera_D camera_E camera_F camera_G camera_H camera_I camera_J camera_K camera_L)
CAMERA_MODELS=(external_camera external_camera_b external_camera_c external_camera_d external_camera_e external_camera_f external_camera_g external_camera_h external_camera_i external_camera_j external_camera_k external_camera_l)
# Accepted analytic line-of-sight AABBs, rounded outward by 0.2 m. Sampling is
# still filtered by the exact projection, SDF collisions and semantic labels.
MIN_X=(-14.6 -5.8 0.2 0.0 7.6 8.4 -14.6 -6.0 -14.6 -14.6 -2.2 -14.0)
MAX_X=(1.4 14.6 14.6 14.6 14.2 14.6 -4.0 14.6 1.8 -3.4 11.2 -0.6)
MIN_Y=(-9.0 -9.0 -6.6 -9.6 -9.6 5.2 -4.4 -9.6 -4.4 -9.0 -9.6 -5.0)
MAX_Y=(4.6 4.8 6.4 7.4 -6.4 9.6 9.0 1.4 9.0 4.4 3.8 9.0)

SIM_PID=""
stop_sim() {
  if [ -n "$SIM_PID" ] && kill -0 "$SIM_PID" >/dev/null 2>&1; then
    kill -INT "$SIM_PID" >/dev/null 2>&1
    for _shutdown_poll in $(seq 1 40); do
      if ! kill -0 "$SIM_PID" >/dev/null 2>&1; then
        break
      fi
      sleep 0.25
    done
    if kill -0 "$SIM_PID" >/dev/null 2>&1; then
      kill -TERM "$SIM_PID" >/dev/null 2>&1
      for _shutdown_poll in $(seq 1 20); do
        if ! kill -0 "$SIM_PID" >/dev/null 2>&1; then
          break
        fi
        sleep 0.25
      done
    fi
    if kill -0 "$SIM_PID" >/dev/null 2>&1; then
      # Exact PID owned by this shell; bounded final cleanup prevents a stale
      # simulator from contaminating the next camera's isolated transport.
      kill -KILL "$SIM_PID" >/dev/null 2>&1
    fi
    wait "$SIM_PID" >/dev/null 2>&1 || true
  fi
  SIM_PID=""
}
trap stop_sim EXIT INT TERM

for index in "${!CAMERA_IDS[@]}"; do
  camera_id="${CAMERA_IDS[$index]}"
  camera_model="${CAMERA_MODELS[$index]}"
  if [ -n "$CAMERA_FILTER" ] && [ "$CAMERA_FILTER" != "$camera_id" ]; then
    continue
  fi

  output="$CAPTURE_ROOT/$camera_id"
  if [ -e "$output" ]; then
    if [ -f "$output/.complete" ]; then
      echo "[meerhoven-capture] keeping completed immutable capture: $output"
      continue
    fi
    echo "[meerhoven-capture] refusing existing output: $output"
    exit 2
  fi

  suffix="${camera_id#camera_}"
  suffix="${suffix,,}"
  camera_bridge="bridge_camera_${suffix}:=true"
  if [ "$camera_id" = "camera_A" ]; then
    segmentation_bridge="bridge_segmentation:=true"
  else
    segmentation_bridge="bridge_segmentation_${suffix}:=true"
  fi
  launch_log="$RUN_LOG_ROOT/${camera_id}_gazebo.log"
  capture_log="$RUN_LOG_ROOT/${camera_id}_capture.log"

  echo "[meerhoven-capture] launching $camera_id ($camera_model)"
  ros2 launch sim bringup_sim.launch.py \
    world:=warehouse_meerhoven.world.sdf \
    headless:=true nvidia_offload:=true reset_world:=true \
    use_lidar:=false bridge_scan:=false bridge_contacts:=false \
    spawn_x:=-12.0 spawn_y:=-4.9 spawn_yaw:=0.0 \
    bridge_camera_a:=false \
    "$camera_bridge" "$segmentation_bridge" >"$launch_log" 2>&1 &
  SIM_PID=$!

  image_topic="/$camera_model/image_raw"
  labels_topic="/$camera_model/segmentation/labels_map"
  ready=0
  for _attempt in $(seq 1 60); do
    if timeout 3 ros2 topic echo --qos-reliability best_effort --once "$image_topic" >/dev/null 2>&1 \
       && timeout 3 ros2 topic echo --qos-reliability best_effort --once "$labels_topic" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 1
  done
  if [ "$ready" != "1" ]; then
    echo "[meerhoven-capture] $camera_id topics did not become ready"
    tail -80 "$launch_log"
    exit 3
  fi

  # The route-held-out 12x10x4 plans contain 244--420 projectable poses per
  # camera (checked with --plan-only). Eighty accepted positives is a fail-closed
  # floor, not the target; all accepted examples are retained in the pooled fit.
  # A fixed camera and static world have exactly one unique background once
  # the robot is geometry-certified outside the image. More candidate poses
  # are retained for the gate, but duplicate backgrounds must not be copied
  # and counted as independent training evidence.
  min_accepted=80
  echo "[meerhoven-capture] capturing $camera_id -> $output"
  python3 scripts/perception/capture_yolo_dataset.py \
    --world warehouse_meerhoven.world.sdf \
    --out "$output" \
    --camera-id "$camera_id" \
    --camera-model "$camera_model" \
    --image-topic "$image_topic" \
    --labels-topic "$labels_topic" \
    --sample-min-x "${MIN_X[$index]}" --sample-max-x "${MAX_X[$index]}" \
    --sample-min-y "${MIN_Y[$index]}" --sample-max-y "${MAX_Y[$index]}" \
    --sample-nx 12 --sample-ny 10 --yaw-samples 4 \
    --settle-s 0.55 --image-timeout-s 8.0 --sync-slop-ms 80 \
    --min-new-rgb-frames 2 --min-new-label-frames 1 --max-sample-attempts 4 \
    --min-mask-area 70 --far-range-start-m 12 --far-min-mask-area 30 \
    --min-mask-bbox-w 6 --min-mask-bbox-h 6 \
    --max-expected-center-error-px 90 \
    --min-rgb-robot-color-fraction 0.010 \
    --occlusion-policy visible-mask-positive \
    --negative-samples-per-camera 1 \
    --max-final-duplicate-fraction 0.02 \
    --min-accepted-samples "$min_accepted" --min-accept-fraction 0.20 \
    --exclude-route meerhoven_haul_lane_sanity \
    --exclude-route meerhoven_west_to_returns_direct \
    --route-exclusion-config "$ROUTE_CONFIG" --route-exclusion-buffer-m 0.80 \
    --save-masks --preview-count 80 --rejected-preview-count 60 >"$capture_log" 2>&1
  capture_rc=$?
  tail -60 "$capture_log"
  stop_sim
  if [ "$capture_rc" != "0" ]; then
    echo "[meerhoven-capture] $camera_id failed with exit $capture_rc"
    exit "$capture_rc"
  fi
done

echo "[meerhoven-capture] completed requested cameras under $CAPTURE_ROOT"
