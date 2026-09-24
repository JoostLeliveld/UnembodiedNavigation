#!/bin/bash
# v10 lane-grid capture (METHOD amendment v10, validation-1 decision). Status: logs/thesis/captures/v10/STATUS.
cd "$(dirname "$0")/../.."
V=logs/thesis/captures/v10
st() { echo "[$(date '+%F %T')] $*" | tee -a "$V/STATUS"; }
st "lane-grid capture started ($(python3 -c "import json;print(len(json.load(open('$V/lane/capture_poses_v10_lane.json'))))") poses)"
bash pipeline/capture/recapture.sh "$V/lane/capture_poses_v10_lane.json" "$V/lane/capture" || { st "lane capture exited non-zero"; exit 5; }
st "lane-grid capture finished: $(($(wc -l < $V/lane/capture/capture_index.csv) - 1)) rows"
