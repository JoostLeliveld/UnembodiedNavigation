#!/bin/bash
# Rebuild ALL paper outputs from a completed campaign log-root, in order:
#   1) metrics CSV (per-run outcome classification)   -> paper_artifacts/metrics/robustness_metrics.csv
#   2) robustness spread figure (trajectories + counts)-> paper_artifacts/figures/robustness_spread.png
#   3) solve diagnostics (offline convergence/timing)  -> logs/paper_figures/solve_diagnostics.png
#   4) offline plan sanity (route check vs runtime)    -> logs/paper_figures/offline_plan_sanity.png
# Then prints the headline C1-vs-C2 counts that feed the abstract / results table.
#
# Usage:  ./build_paper_outputs.sh logs/visibility_comparison/robustness_campaign_v2
set -eu
cd "$(dirname "$0")/../.."
LOG_ROOT="${1:?usage: build_paper_outputs.sh <campaign-log-root>}"
GP="paper_artifacts/gp/aws_gp_v7b/yolo_score_raw_gp.npz"
METRICS="paper_artifacts/metrics/robustness_metrics.csv"

source /opt/ros/humble/setup.bash >/dev/null 2>&1 || true
source install/setup.bash >/dev/null 2>&1 || true

echo ">>> [1/4] metrics from ${LOG_ROOT}/campaign_log.json"
python3 scripts/visibility_comparison/compute_paper_metrics.py \
  --campaign-log "${LOG_ROOT}/campaign_log.json" --gp-artifact "${GP}" --out "${METRICS}"

echo ">>> [2/4] robustness spread figure"
python3 scripts/paper_figures/make_robustness_spread.py --campaign-root "${LOG_ROOT}" --metrics "${METRICS}"

echo ">>> [3/4] solve diagnostics (offline convergence/timing/stability)"
python3 scripts/paper_figures/plot_solve_diagnostics.py

echo ">>> [4/4] offline plan sanity (route check)"
python3 scripts/paper_figures/plot_offline_plan_sanity.py

echo ">>> headline counts (feed abstract + tab:results):"
python3 scripts/visibility_comparison/monitor_campaign.py "${LOG_ROOT}"
echo ">>> DONE. Figures in paper_artifacts/figures/ and logs/paper_figures/."
