#!/bin/bash
# Gate A of the 2026-09-23 amendment (overnight plan step 3). Runs the retrained detector
# through inference and the frozen sensor gate on the v7 working roles, compares it with
# the v5 detector on D_dev, and leaves the v7 lock pointing at whichever detector wins.
# Nothing is fitted before this decision.
#
#   bash experiments/thesis_pipeline_lock/run_v7_gate_a.sh
cd "$(dirname "$0")/../.."
export THESIS_REFERENCE_DATASET=v7_balanced
R=logs/thesis_final_pipeline_v1/recapture_v7_balanced
S=logs/thesis_final_pipeline_v1/overnight_20260923/STATUS.md
LOCK=experiments/thesis_pipeline_lock/reference_position_campaign_lock_v7_balanced.json
GATE_CFG=config/commissioning_sensor_gate_v2.yaml
CAND=$R/detector_training/imgsz960/upper_finetune/weights/best.pt
# No ROS sourcing: install/setup.bash puts the ROS `experiments` package on PYTHONPATH,
# which shadows this repository's experiments/ directory and breaks the imports.
log() { echo "- $(date '+%F %T') step 3 gate A: $*" >> "$S"; echo "$*"; }
set_detector() {  # set_detector PATH NOTE
  python3 - "$LOCK" "$1" "$2" <<'EOF'
import hashlib, json, sys
lock_path, weights, note = sys.argv[1:]
lock = json.load(open(lock_path))
lock["detector"]["checkpoint"] = weights
lock["detector"]["checkpoint_sha256"] = hashlib.sha256(open(weights, "rb").read()).hexdigest()
lock["detector"]["gate_a"] = note
json.dump(lock, open(lock_path, "w"), indent=1)
EOF
}
[ -f "$CAND" ] || { log "FAILED: candidate checkpoint missing ($CAND)"; exit 1; }
V5_DET=$(python3 -c "import json;print(json.load(open('experiments/thesis_pipeline_lock/reference_position_campaign_lock.json'))['detector']['checkpoint'])")

if [ ! -e "$R/gate_dataset_v5detector" ]; then
  log "building gate dataset for the v5 detector"
  python3 experiments/thesis_pipeline_lock/build_reference_gate_dataset.py \
    --inference "$R/detector_inference_v5detector" --gate "$GATE_CFG" \
    --output "$R/gate_dataset_v5detector" > "$R/gate_dataset_v5detector.log" 2>&1 \
    || { log "FAILED gate_dataset_v5detector"; exit 1; }
fi

set_detector "$CAND" "candidate under evaluation"
if [ ! -e "$R/detector_inference_candidate" ]; then
  log "inference with the candidate detector"
  python3 experiments/thesis_pipeline_lock/run_reference_detector_inference.py \
    --output "$R/detector_inference_candidate" --batch-size 8 \
    > "$R/detector_inference_candidate.log" 2>&1 || { log "FAILED candidate inference"; exit 1; }
fi
if [ ! -e "$R/gate_dataset_candidate" ]; then
  log "building gate dataset for the candidate detector"
  python3 experiments/thesis_pipeline_lock/build_reference_gate_dataset.py \
    --inference "$R/detector_inference_candidate" --gate "$GATE_CFG" \
    --output "$R/gate_dataset_candidate" > "$R/gate_dataset_candidate.log" 2>&1 \
    || { log "FAILED gate_dataset_candidate"; exit 1; }
fi
python3 experiments/thesis_pipeline_lock/evaluate_detector_gate_a.py \
  --baseline "$R/gate_dataset_v5detector" --candidate "$R/gate_dataset_candidate" \
  --output "$R/gate_a_report.json" || { log "FAILED gate A evaluation"; exit 1; }
PASSED=$(python3 -c "import json;print(json.load(open('$R/gate_a_report.json'))['passed'])")
if [ "$PASSED" = "True" ]; then
  set_detector "$CAND" "passed gate A: $R/gate_a_report.json"
  echo "$R/gate_dataset_candidate" > "$R/FROZEN_GATE_DATASET"
  log "PASSED, retrained detector frozen; refit uses gate_dataset_candidate"
else
  set_detector "$V5_DET" "retrained detector failed gate A: $R/gate_a_report.json"
  echo "$R/gate_dataset_v5detector" > "$R/FROZEN_GATE_DATASET"
  log "FAILED the gate, v5 detector kept; refit uses gate_dataset_v5detector"
fi
