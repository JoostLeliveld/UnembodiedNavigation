#!/bin/bash
# v11 part 2, run by hand after the D_dev check: the test set (once), routes, campaign and
# analysis. Own systemd unit and a sleep/lid inhibitor, as in the v10 night.
#   systemd-run --user --unit=v11-part2 --collect bash pipeline/ops/run_v11_part2.sh
cd "$(dirname "$0")/../.."
V=logs/thesis/captures/v11
R=logs/thesis
S=$R/superseded
st() { echo "[$(date '+%F %T')] $*" | tee -a "$V/STATUS"; }
run() { st "start $1"; shift; "$@" || { st "FAILED: $*"; exit 1; }; }
mkdir -p "$S"
for x in final_audit final_audit_fusion final_audit_protocol.json final_audit_corrected_xy.npz routes campaign_configs analysis analysis_taskC_rerun.out campaign_seed91500.log campaign_seed91501.log campaign_seed91502.log; do
  [ -e "$R/$x" ] && mv "$R/$x" "$S/v10_$x"
done
mkdir -p "$R/campaign_configs"
systemd-inhibit --what=sleep:idle:handle-lid-switch --who=thesis --why=v11-campaign bash -c '
  set -e
  cd "'"$PWD"'"
  python3 pipeline/audit_protocol.py
  python3 pipeline/final_audit.py --protocol logs/thesis/final_audit_protocol.json --output logs/thesis/final_audit > logs/thesis/final_audit.log 2>&1
  python3 pipeline/final_audit_fusion.py --audit logs/thesis/final_audit --runtime-root logs/thesis/fits/runtime_r012 \
    --rproj-ddev logs/thesis/fits/ddev_evaluation/manifest.json --output logs/thesis/final_audit_fusion > logs/thesis/final_audit_fusion.log 2>&1
  echo "[$(date "+%F %T")] test set done" >> '"$V"'/STATUS
  python3 pipeline/campaign_configs.py > logs/thesis/campaign_configs.log 2>&1
  bash pipeline/routes.sh
  echo "[$(date "+%F %T")] routes done" >> '"$V"'/STATUS
  bash pipeline/campaign.sh
  echo "[$(date "+%F %T")] campaign done" >> '"$V"'/STATUS
  python3 pipeline/analyze_campaign.py > logs/thesis/analysis.log 2>&1
' || { st "part 2 FAILED, see pipeline.log and the step logs"; exit 1; }
st "part 2 done: test set, routes, campaign and analysis"
