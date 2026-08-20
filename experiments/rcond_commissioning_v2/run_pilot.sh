#!/bin/bash
# Two independent Gazebo restarts, small enough to validate end-to-end plumbing.

cd "$(dirname "$0")/../.." || exit 1
ROOT="$(pwd)"
OUT_ROOT="${OUT_ROOT:-logs/studies/rcond_commissioning_v2/pilot}"
WEIGHTS="${WEIGHTS:-logs/perception_models/yolo_pose_aws_v4/model.pt}"

for SESSION in 0 1; do
  DATASET="$OUT_ROOT/session_$SESSION/capture"
  REPORT="$OUT_ROOT/session_$SESSION/evaluation"
  if [ ! -e "$DATASET" ]; then
    WORLD=warehouse_aws.world.sdf CAMERAS=external_camera \
    DATASET="$DATASET" SAMPLE_NX=4 SAMPLE_NY=3 YAW_SAMPLES=4 REPEATS=3 \
    POSITION_JITTER_M=0.012 YAW_JITTER_RAD=0.0261799 \
    CAPTURE_SESSION_ID="session_$SESSION" PLAN_SEED="$SESSION" \
    VAL_FRACTION=0.25 SPLIT_MODE=spatial_cell SPATIAL_BLOCK_SIZE=1 \
    MIN_BBOX_PX=8.0 ROBOT_Z=0.0 \
    GZLOG="/tmp/rcond_v2_pilot_gazebo_$SESSION.log" \
    CAPLOG="/tmp/rcond_v2_pilot_capture_$SESSION.log" \
    bash scripts/perception/capture_keypoint_dataset.sh || exit $?
  fi
  if [ ! -e "$REPORT/per_sample.csv" ]; then
    python3 experiments/keypoint_measurement/evaluate_keypoint_model.py \
      --dataset "$DATASET" --weights "$WEIGHTS" --out "$REPORT" \
      --split all --imgsz 960 --conf 0.05 --batch 8 --device "${DEVICE:-cpu}" \
      --label "Rcond v2 pilot session $SESSION" || exit $?
  fi
  HONESTY="$OUT_ROOT/session_$SESSION/covariance_honesty"
  if [ ! -e "$HONESTY/per_reading.csv" ]; then
    python3 experiments/keypoint_measurement/check_covariance_is_honest.py \
      --dataset "$DATASET" --readings "$REPORT/per_sample.csv" \
      --out "$HONESTY" || exit $?
  fi
done

python3 experiments/rcond_commissioning_v2/analyse_campaign.py --root "$OUT_ROOT"
