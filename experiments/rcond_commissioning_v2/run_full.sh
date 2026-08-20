#!/bin/bash
# Full commissioning design. Intentionally separate from the pilot.

cd "$(dirname "$0")/../.." || exit 1
OUT_ROOT="${OUT_ROOT:-logs/studies/rcond_commissioning_v2/full_v1}"
WEIGHTS="${WEIGHTS:-logs/perception_models/yolo_pose_aws_v4/model.pt}"

for SESSION in 0 1 2; do
  DATASET="$OUT_ROOT/session_$SESSION/capture"
  REPORT="$OUT_ROOT/session_$SESSION/evaluation"
  if [ ! -e "$DATASET" ]; then
    WORLD=warehouse_aws.world.sdf CAMERAS=external_camera \
    DATASET="$DATASET" SAMPLE_NX=10 SAMPLE_NY=6 YAW_SAMPLES=4 REPEATS=5 \
    POSITION_JITTER_M=0.015 YAW_JITTER_RAD=0.0349066 \
    CAPTURE_SESSION_ID="session_$SESSION" PLAN_SEED="$SESSION" \
    VAL_FRACTION=0.20 SPLIT_MODE=spatial_cell SPATIAL_BLOCK_SIZE=1 \
    MIN_BBOX_PX=8.0 ROBOT_Z=0.0 \
    GZLOG="/tmp/rcond_v2_full_gazebo_$SESSION.log" \
    CAPLOG="/tmp/rcond_v2_full_capture_$SESSION.log" \
    bash scripts/perception/capture_keypoint_dataset.sh || exit $?
  fi
  if [ ! -e "$REPORT/per_sample.csv" ]; then
    python3 experiments/keypoint_measurement/evaluate_keypoint_model.py \
      --dataset "$DATASET" --weights "$WEIGHTS" --out "$REPORT" \
      --split all --imgsz 960 --conf 0.05 --batch 8 --device "${DEVICE:-0}" \
      --label "Rcond v2 full session $SESSION" || exit $?
  fi
  HONESTY="$OUT_ROOT/session_$SESSION/covariance_honesty"
  if [ ! -e "$HONESTY/per_reading.csv" ]; then
    python3 experiments/keypoint_measurement/check_covariance_is_honest.py \
      --dataset "$DATASET" --readings "$REPORT/per_sample.csv" \
      --out "$HONESTY" || exit $?
  fi
done

python3 experiments/rcond_commissioning_v2/analyse_campaign.py --root "$OUT_ROOT"
