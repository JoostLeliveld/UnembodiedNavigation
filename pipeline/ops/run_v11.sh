#!/bin/bash
# v11 (METHOD amendment 2026-09-25 afternoon), part 1: after the fill capture, rebuild the
# split and the lock, audit the dataset, set the v10 fits aside and refit. Part 2 (the D_dev
# check, the test set, routes and campaign) is run by hand after the check.
# Status: logs/thesis/captures/v11/STATUS.
cd "$(dirname "$0")/../.."
V=logs/thesis/captures/v11
st() { echo "[$(date '+%F %T')] $*" | tee -a "$V/STATUS"; }
until grep -q "fill capture exited" "$V/STATUS"; do sleep 30; done
grep -q "fill capture exited 0" "$V/STATUS" || { st "part 1 not started: fill capture failed"; exit 5; }
st "part 1 started: $(($(wc -l < $V/fill/capture/capture_index.csv) - 1)) fill rows"
python3 pipeline/capture/partition_v11.py > "$V/partition_v11.log" 2>&1 || { st "partition FAILED"; exit 1; }
python3 pipeline/write_dataset_lock.py > "$V/lock_v11.log" 2>&1 || { st "lock FAILED"; exit 1; }
python3 pipeline/audit_dataset.py > "$V/audit_dataset_v11.log" 2>&1 || { st "dataset audit FAILED"; exit 1; }
grep -q '"passed": true' "$V/audit_dataset_v11.log" || { st "dataset audit did not pass"; exit 1; }
st "split, lock and dataset audit done"
S=logs/thesis/superseded
mkdir -p "$S"
[ -e logs/thesis/fits ] && mv logs/thesis/fits "$S/v10_fits"
[ -e logs/thesis/fits_v11 ] && mv logs/thesis/fits_v11 "$S/v11_first_resplit_fits"
bash pipeline/refit.sh > "$V/refit_v11.log" 2>&1 || { st "refit FAILED"; exit 1; }
st "part 1 done: refit complete in logs/thesis/fits"
