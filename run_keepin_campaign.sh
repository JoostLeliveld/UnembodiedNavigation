#!/bin/bash
# Canonical launcher for the robustness campaign (keep_in no-go, lane-graph seeds).
# Uses the runner's built-in timeouts (--first-cmd-timeout 270s, --run-timeout 420s)
# which are sized so the contended global solve (~120-220s wall) completes with room
# to execute. Do NOT pass a shorter --run-timeout: the old 240s value guillotined slow
# solves mid-optimization and was the dominant past failure mode.
#
# Best results: reboot first and keep only VS Code open (idle) — the global solve is
# single-threaded and gets starved by background CPU load, which stretches it 6-10x.
#
# Usage:   ./run_keepin_campaign.sh [LOG_ROOT_NAME]
set -u
cd "$(dirname "$0")"
LOG_NAME="${1:-robustness_campaign_v2}"
LOG_ROOT="logs/visibility_comparison/${LOG_NAME}"

# clean any stragglers from a previous attempt (the runner also does this per-run)
pkill -9 -f "ign gazebo" >/dev/null 2>&1 || true
pkill -9 -f "run_visibility_campaign" >/dev/null 2>&1 || true
sleep 3

source /opt/ros/humble/setup.bash >/dev/null 2>&1
source install/setup.bash >/dev/null 2>&1

# --resume is safe: completed runs are skipped, so re-running after an interruption
# only fills in the missing runs.
exec python3 scripts/visibility_comparison/run_visibility_campaign.py \
  --config scripts/visibility_comparison/aws_f31b1_final_config.yaml \
  --log-root "${LOG_ROOT}" \
  --resume
