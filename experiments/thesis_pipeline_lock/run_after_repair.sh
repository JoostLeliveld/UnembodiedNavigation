#!/bin/bash
# After the v5 repair capture: validate presence, retire fits made on broken frames,
# re-lock v8, run inference once, test the three gate variants and choose one by the
# fixed rule in evaluate_gate_variants.py, then refit with the chosen gate. Stops before
# the final audit.
cd "$(dirname "$0")/../.."
R=logs/thesis_final_pipeline_v1/recapture_v8_uniform
A=logs/thesis_final_pipeline_v1/archive_rejected_20260923/v8_fits_on_broken_frames
S=logs/thesis_final_pipeline_v1/overnight_20260923/STATUS.md
V=$R/gate_variants
log() { echo "- $(date '+%F %T') after-repair: $*" >> "$S"; echo "$*"; }
export THESIS_REFERENCE_DATASET=v8_uniform
python3 experiments/thesis_pipeline_lock/audit_v8_dataset.py > "$R/dataset_audit.log" 2>&1 \
  || { log "FAILED dataset audit, see $R/dataset_audit.log"; exit 1; }
log "dataset audit passed"
mkdir -p "$A"
for d in detector_inference gate_dataset correction covariance ddev_evaluation bayesian_covariance runtime_r012 planning_precision final_audit final_audit_fusion offline_routes campaign_configs ddev_comparison_v5_v8.json offline_routes_overview.png offline_routes_summary.json capture_positions_v8.csv; do
  [ -e "$R/$d" ] && mv "$R/$d" "$A/" ; [ -e "$R/$d.log" ] && mv "$R/$d.log" "$A/"
done
P=experiments/thesis_pipeline_lock/reference_final_audit_protocol_v8_uniform.json
[ -e "$P" ] && mv "$P" "$A/"
log "fits and audit protocol made on broken frames archived to $A"
python3 experiments/thesis_pipeline_lock/relock_v8.py || { log "FAILED relock"; exit 1; }
python3 experiments/thesis_pipeline_lock/run_reference_detector_inference.py --output "$R/detector_inference" --batch-size 8 > "$R/detector_inference.log" 2>&1 || { log "FAILED inference"; exit 1; }
log "inference done"
bash experiments/thesis_pipeline_lock/run_gate_variants.sh
