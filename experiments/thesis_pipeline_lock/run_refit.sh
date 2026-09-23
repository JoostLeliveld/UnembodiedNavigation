#!/bin/bash
# Refit the reference chain on the v8 uniform dataset (amendment 2026-09-23): detector
# inference, sensor gate, correction, R0/R1/R2, runtime packaging and planning precision. Each stage refuses to overwrite its output, so the script skips any stage
# whose output already exists and resumes where it stopped.
#
#   bash experiments/thesis_pipeline_lock/run_refit.sh
cd "$(dirname "$0")/../.."
export THESIS_REFERENCE_DATASET=v8_uniform
# REFIT_ROOT redirects every output, for a dry run that must not touch the campaign root.
R="${REFIT_ROOT:-logs/thesis_final_pipeline_v1/recapture_v8_uniform}"
S=logs/thesis_final_pipeline_v1/recapture_v8_uniform/pipeline.log
mkdir -p "$R"
GATE="$R/gate_dataset"
# GATE_CONFIG and INFERENCE_DIR let the gate-variant test reuse one inference pass.
GATE_CONFIG="${GATE_CONFIG:-config/commissioning_sensor_gate_v3.yaml}"
INF="${INFERENCE_DIR:-$R/detector_inference}"
# No ROS sourcing: install/setup.bash puts the ROS `experiments` package on PYTHONPATH,
# which shadows this repository's experiments/ directory and breaks the imports.
log() { echo "- $(date '+%F %T') step 5: $*" >> "$S"; echo "$*"; }
run() {  # run OUTPUT_DIR command...
  local out="$1"; shift
  if [ -e "$out" ]; then log "skip $(basename "$out") (exists)"; return 0; fi
  log "start $(basename "$out")"
  "$@" > "${out}.log" 2>&1 || { log "FAILED $(basename "$out"), see ${out}.log"; exit 1; }
  log "done $(basename "$out")"
}
run "$INF" python3 experiments/thesis_pipeline_lock/run_reference_detector_inference.py \
  --output "$INF" --batch-size 8
run "$R/gate_dataset" python3 experiments/thesis_pipeline_lock/build_reference_gate_dataset.py \
  --inference "$INF" --gate "$GATE_CONFIG" --output "$GATE"
run "$R/correction" python3 experiments/thesis_pipeline_lock/fit_reference_correction.py \
  --gate-dataset "$GATE" --output "$R/correction"
run "$R/covariance" python3 experiments/thesis_pipeline_lock/fit_reference_covariance.py \
  --correction "$R/correction" --output "$R/covariance"
run "$R/ddev_evaluation" python3 experiments/thesis_pipeline_lock/evaluate_reference_ddev.py \
  --correction "$R/correction" --covariance "$R/covariance" --output "$R/ddev_evaluation"
run "$R/bayesian_covariance" python3 experiments/thesis_pipeline_lock/fit_bayesian_r012.py \
  --source "$R/covariance/corrected_residuals.npz" --output "$R/bayesian_covariance"
run "$R/runtime_r012" python3 experiments/thesis_pipeline_lock/package_canonical_r012_runtime.py \
  --correction-manifest "$R/correction/manifest.json" \
  --covariance-models "$R/bayesian_covariance/bayesian_r012_models.npz" \
  --world src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf --output "$R/runtime_r012"
run "$R/planning_precision" python3 experiments/thesis_pipeline_lock/build_reference_planning_information.py \
  --covariance-models "$R/bayesian_covariance/bayesian_r012_models.npz" \
  --output "$R/planning_precision"
log "ALL DONE"
