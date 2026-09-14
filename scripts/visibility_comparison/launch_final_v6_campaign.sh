#!/usr/bin/env bash
# Launch the final four-arm navigation campaign (80 runs).
#
# Everything this needs is already verified as of 2026-09-15:
#   - the four arms differ in exactly one argument (the planner field)
#   - every artifact exists and hashes as declared
#   - the world is byte-identical to the one commissioning was captured in
#   - optimizer_control_block_steps = 1, so the route seeds survive into the
#     solve and the hard terminal-goal gate can admit them
#   - all four tasks solve to the goal offline
#
# Timeouts come from run_visibility_campaign.py's defaults, which were sized
# from measured solve times (see docs/PLANNER_LOCK.md). Do not lower them
# without re-measuring.
# No -u: the ROS setup scripts read unset variables by design.
set -eo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

CONFIG=experiments/reference_controlled_commissioning_v1/final_navigation_campaign_v6_locked.yaml
# A fresh root per launch. The shared default root already holds unrelated
# studies and a failed pilot, and resume counts every attempt it finds there.
LOG_ROOT=logs/studies/reference_controlled_commissioning_v1/final_navigation_v6_$(date +%Y%m%d_%H%M%S)

# Never launch on top of a running simulator: two campaigns sharing the GPU
# starve each other's solves past the first-command deadline.
# Match on the executable name, not the full command line: -f also matches this
# script and the shell that launched it, so the guard would always fire.
if pgrep -x "ign gazebo" >/dev/null 2>&1 \
   || pgrep -x "gzserver" >/dev/null 2>&1 \
   || pgrep -x "ruby" >/dev/null 2>&1; then
    echo "REFUSING: a simulator is already running." >&2
    echo "Two campaigns sharing the GPU starve each other's solves." >&2
    pgrep -a -x "ign gazebo" >&2 || true
    exit 1
fi

source /opt/ros/humble/setup.bash
source install/setup.bash

echo "config   : $CONFIG"
echo "log root : $LOG_ROOT"
echo "runs     : 80 (4 tasks x 4 arms x 5 seeds)"
echo

exec python3 scripts/visibility_comparison/run_visibility_campaign.py \
    --config "$CONFIG" \
    --log-root "$LOG_ROOT" \
    "$@"
