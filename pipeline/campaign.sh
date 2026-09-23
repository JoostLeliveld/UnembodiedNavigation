#!/bin/bash
# Execute the v8 five-task dropout campaign seed by seed (amendment 2026-09-23): 30 runs per
# seed, seeds 91500, 91501, 91502 in that order, one simulator at a time. --resume skips runs
# already recorded in a seed's campaign_log.json, so the script can be restarted safely.
#
#   bash pipeline/campaign.sh [SEED ...]
cd "$(dirname "$0")/.."
R=logs/thesis
S=logs/thesis/pipeline.log
source /opt/ros/humble/setup.bash >/dev/null 2>&1
source install/setup.bash >/dev/null 2>&1
log() { echo "- $(date '+%F %T') campaign: $*" >> "$S"; echo "$*"; }
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(91500 91501 91502)
for seed in "${SEEDS[@]}"; do
  cfg="$R/campaign_configs/campaign_seed${seed}.yaml"
  [ -f "$cfg" ] || { log "FAILED: missing $cfg"; exit 1; }
  log "seed $seed started"
  python3 pipeline/campaign_runner.py \
    --config "$cfg" --log-root "$R/campaign/seed${seed}" --resume \
    >> "$R/campaign_seed${seed}.log" 2>&1
  rc=$?
  log "seed $seed finished, exit $rc"
  [ "$rc" = "0" ] || exit "$rc"
done
log "ALL SEEDS DONE"
