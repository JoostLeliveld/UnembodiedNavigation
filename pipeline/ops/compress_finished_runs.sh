#!/bin/bash
# Losslessly compress runtime_event_deliveries.jsonl of FINISHED campaign runs (a run is
# finished when run_summary.json exists and the jsonl has not changed for 2 minutes).
# Nothing in the pipeline reads this file; zstd -d restores it exactly. Loops until the
# campaign unit ends, so the disk does not fill during the night (v11: ~66 MB per run raw).
#   systemd-run --user --unit=v11-compress bash pipeline/ops/compress_finished_runs.sh
cd "$(dirname "$0")/../.."
once() {
  find logs/thesis/campaign -name runtime_event_deliveries.jsonl -mmin +2 | while read -r f; do
    [ -f "$(dirname "$f")/run_summary.json" ] || continue
    zstd -q -3 -T2 -f "$f" -o "$f.zst" && zstd -q -t "$f.zst" && rm -f "$f"
  done
}
while systemctl --user is-active -q v11-part2; do once; sleep 180; done
once
