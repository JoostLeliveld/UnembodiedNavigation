#!/bin/bash
# Run the v10 night in its own systemd user unit (not inside the Claude app, whose group
# systemd-oomd killed at 00:23 together with the capture): memory guard, lane-grid capture
# (resumes committed rows), then the overnight chain.
#   systemd-run --user --unit=v10-night --collect bash pipeline/ops/v10_night.sh
cd "$(dirname "$0")/../.."
V=logs/thesis/captures/v10
echo "[$(date '+%F %T')] night driver started in its own unit (after the oomd kill at 00:23)" >> "$V/STATUS"
bash pipeline/ops/memory_guard.sh 3.0 "$V/lane/capture.log" capture_positions >> "$V/memory_guard.log" 2>&1 &
guard=$!
systemd-inhibit --what=sleep:idle:handle-lid-switch --who=thesis --why=v10-night \
  bash -c 'bash pipeline/capture/run_v10.sh && bash pipeline/ops/v10_overnight.sh'
rc=$?
kill "$guard" 2>/dev/null
echo "[$(date '+%F %T')] night driver ended, exit $rc" >> "$V/STATUS"
