#!/bin/bash
# Gate-variant test and refit with the chosen gate: the tail of run_after_repair.sh, run on
# its own when inference already exists.
cd "$(dirname "$0")/../.."
R=logs/thesis_final_pipeline_v1/recapture_v8_uniform
S=logs/thesis_final_pipeline_v1/overnight_20260923/STATUS.md
V=$R/gate_variants
log() { echo "- $(date '+%F %T') after-repair: $*" >> "$S"; echo "$*"; }
export THESIS_REFERENCE_DATASET=v8_uniform
for g in G0 G1 G2; do
  REFIT_ROOT=$V/$g GATE_CONFIG=$V/gate_$g.yaml INFERENCE_DIR=$R/detector_inference bash experiments/thesis_pipeline_lock/run_refit.sh > $V/$g.refit.log 2>&1 || { log "FAILED refit $g"; exit 1; }
  log "gate variant $g refitted"
done
python3 experiments/thesis_pipeline_lock/evaluate_gate_variants.py > $V/evaluate.log 2>&1 || { log "FAILED gate evaluation"; exit 1; }
CH=$(python3 -c "import json;print(json.load(open('$V/gate_variant_report.json'))['chosen'])")
log "gate variant chosen by the fixed rule: $CH"
echo "$V/gate_$CH.yaml" > "$R/CHOSEN_GATE"
for d in gate_dataset correction covariance ddev_evaluation bayesian_covariance runtime_r012 planning_precision; do cp -r "$V/$CH/$d" "$R/$d"; done
log "refit with gate $CH copied into the campaign root; stopped before the final audit"
