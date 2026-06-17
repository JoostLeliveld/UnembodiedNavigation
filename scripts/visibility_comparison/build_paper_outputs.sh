#!/bin/bash
# Rebuild maintained paper outputs from a completed campaign log-root:
#   1) metrics CSV (per-run outcome classification)    -> paper_artifacts/metrics/robustness_metrics.csv
#   2) robustness spread figure (trajectories + counts) -> paper_artifacts/figures/robustness_spread.png
# Then print the headline C1-vs-C2 counts that feed the abstract / results table.
#
# Usage:  ./build_paper_outputs.sh logs/visibility_comparison/robustness_campaign_v2
set -eu
cd "$(dirname "$0")/../.."
LOG_ROOT="${1:?usage: build_paper_outputs.sh <campaign-log-root>}"
GP="paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz"
METRICS="paper_artifacts/metrics/robustness_metrics.csv"

source /opt/ros/humble/setup.bash >/dev/null 2>&1 || true
source install/setup.bash >/dev/null 2>&1 || true

echo ">>> [1/2] metrics from ${LOG_ROOT}/campaign_log.json"
python3 scripts/visibility_comparison/compute_paper_metrics.py \
  --campaign-log "${LOG_ROOT}/campaign_log.json" --gp-artifact "${GP}" --out "${METRICS}"

echo ">>> [2/2] robustness spread figure"
python3 scripts/paper_figures/make_robustness_spread.py --campaign-root "${LOG_ROOT}" --metrics "${METRICS}"

echo ">>> headline counts (feed abstract + tab:results):"
python3 scripts/visibility_comparison/monitor_campaign.py "${LOG_ROOT}"
echo ">>> DONE. Figures in paper_artifacts/figures/ and logs/paper_figures/."
