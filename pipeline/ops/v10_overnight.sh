#!/bin/bash
# v10 overnight chain (author's go, 2026-09-25 01:00): after the lane-grid capture, switch the
# dataset to v10, refit, open the sealed audit once, solve and check routes, pilot, campaign,
# analysis. Every step logs to logs/thesis/captures/v10/STATUS and stops the chain on failure.
cd "$(dirname "$0")/../.."
V=logs/thesis/captures/v10
R=logs/thesis
st() { echo "[$(date '+%F %T')] $*" | tee -a "$V/STATUS"; }
die() { st "STOPPED: $*"; exit 1; }
source /opt/ros/humble/setup.bash >/dev/null 2>&1; source install/setup.bash >/dev/null 2>&1
export PYTHONWARNINGS=ignore

st "chain waiting for the lane-grid capture"
until grep -q "lane-grid capture finished\|lane capture exited non-zero" "$V/STATUS"; do sleep 60; done
grep -q "lane capture exited non-zero" "$V/STATUS" && die "lane capture exited non-zero"
sleep 20

# 1. capture complete and the capture-only camera models restored
python3 - <<'EOF' || die "capture incomplete"
import csv, json
rows = list(csv.DictReader(open('logs/thesis/captures/v10/lane/capture/capture_index.csv')))
m = json.load(open('logs/thesis/captures/v10/lane/capture/capture_manifest.json'))
poses = len({r['pose_id'] for r in rows})
ok = all(r['capture_status'] == 'ok' for r in rows)
print('rows', len(rows), 'poses', poses, 'planned', m['plan']['pose_count'], 'all ok', ok)
assert ok and len(rows) == 5 * m['plan']['pose_count'] == 5 * poses
EOF
git diff --quiet -- src/sim/models || die "camera models not restored after the capture"
st "1 capture complete, camera models restored"

# 2. roles for the lane positions (reads the v9 loader, so before the switch)
python3 pipeline/capture/partition_v10.py >> "$V/STATUS" 2>&1 || die "partition_v10 failed"
# 3. switch the loader to v10, audit the dataset, commit the code
git apply "$V/patches/dataset.patch" "$V/patches/audit_dataset.patch" "$V/patches/write_dataset_lock.patch" || die "patches do not apply"
python3 pipeline/dataset.py >> "$V/STATUS" 2>&1 || die "v10 loader failed"
python3 pipeline/audit_dataset.py > "$V/dataset_audit.out" 2>&1 || die "dataset audit failed (see $V/dataset_audit.out)"
git add pipeline/dataset.py pipeline/audit_dataset.py pipeline/write_dataset_lock.py && \
  git commit -q -m "Dataset v10: lane-grid capture as a source, roles from partition_v10, audit accepts its identity

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || die "commit of the v10 loader failed"
st "3 dataset v10 loaded, audited and committed"
# 4. the dataset lock, as its own action and commit
python3 pipeline/write_dataset_lock.py >> "$V/STATUS" 2>&1 || die "dataset lock failed"
git add pipeline/dataset_lock.json && git commit -q -m "Dataset lock v10

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || die "commit of the lock failed"
st "4 dataset lock v10 written and committed"

# 5. refit (retrains the correction), the v9 fits kept as superseded evidence
mkdir -p "$R/superseded" && mv "$R/fits" "$R/superseded/v9_fits" || die "cannot move the v9 fits"
bash pipeline/refit.sh >> "$V/STATUS" 2>&1 || die "refit failed (see $R/pipeline.log)"
# 6. sanity of the refit (declared bands, not tuning): corrected D_dev median <= 3 cm and
#    p95 <= 16 cm; every R0/R1/R2 95 % coverage on D_dev in [0.90, 0.98]
python3 - >> "$V/STATUS" 2>&1 <<'EOF' || die "refit outside its sanity bands"
import json
c = json.load(open('logs/thesis/fits/correction/manifest.json'))['D_dev_metrics']['structured_plus_visibility']
d = json.load(open('logs/thesis/fits/ddev_evaluation/manifest.json'))['D_dev_covariance_metrics']
v9c = json.load(open('logs/thesis/superseded/v9_fits/correction/manifest.json'))['D_dev_metrics']['structured_plus_visibility']
print('correction D_dev median %.2f p95 %.2f cm (v9 %.2f / %.2f)' % (100*c['pooled_median_m'], 100*c['pooled_p95_m'], 100*v9c['pooled_median_m'], 100*v9c['pooled_p95_m']))
cov = {k: v['equal_position_95pct_coverage'] for k, v in d.items()}
print('D_dev 95% coverage', {k: round(v, 3) for k, v in cov.items()})
assert c['pooled_median_m'] <= 0.03 and c['pooled_p95_m'] <= 0.16
assert all(0.90 <= v <= 0.98 for k, v in cov.items() if k.startswith('R'))
EOF
st "6 refit done and inside its sanity bands"

# 7. sealed audit, opened once
python3 pipeline/audit_protocol.py >> "$V/STATUS" 2>&1 || die "audit protocol failed"
python3 pipeline/final_audit.py --protocol "$R/final_audit_protocol.json" --output "$R/final_audit" > "$V/final_audit.out" 2>&1 || die "final audit failed"
python3 pipeline/final_audit_fusion.py --audit "$R/final_audit" --runtime-root "$R/fits/runtime_r012" \
  --rproj-ddev "$R/fits/ddev_evaluation/manifest.json" --output "$R/final_audit_fusion" > "$V/final_audit_fusion.out" 2>&1 || die "fusion audit failed"
st "7 sealed audit and fusion audit done"

# 8. configs, routes, follower replay check
python3 pipeline/campaign_configs.py >> "$V/STATUS" 2>&1 || die "campaign configs failed"
bash pipeline/routes.sh >> "$V/STATUS" 2>&1 || die "route solving failed"
python3 pipeline/replay_routes.py > "$V/replay_routes.out" 2>&1 || die "a route fails the ff_fb replay (see $V/replay_routes.out)"
st "8 routes solved, bound and replay-checked"

# 9. pilot: the former defect cell, before anything is locked
if pgrep -f "[i]gn gazebo" >/dev/null; then die "a simulator is already running"; fi
python3 pipeline/campaign_runner.py --config "$R/campaign_configs/campaign_seed91500.yaml" \
  --log-root "$R/qualification/v10_pilot" --only-task thesis10_camera_c_inner_warehouse_detour \
  --only-condition global_removal > "$V/pilot.out" 2>&1 || die "pilot run failed (see $V/pilot.out)"
python3 - >> "$V/STATUS" 2>&1 <<'EOF' || die "pilot run does not look right"
import glob, json, csv
runs = glob.glob('logs/thesis/qualification/v10_pilot/**/run_summary.json', recursive=True)
assert runs, 'no pilot run summary'
d = runs[0].rsplit('/', 1)[0]
a = list(csv.DictReader(open(d + '/correction_assimilations.csv')))
statuses = sorted({r['status'] for r in a})
import yaml
m = json.load(open(d + '/run_manifest.json'))
cfg = yaml.safe_load(open('logs/thesis/campaign_configs/campaign_seed91500.yaml'))
ms = json.loads(m.get('manager_settings_json', '{}'))
print('pilot statuses', statuses)
assert 'accepted_bootstrap' not in statuses, 'camera bootstrap still used'
assert cfg.get('initial_belief_from_task_start') is True, 'initial prior not in the campaign config'
assert float(m['encoder_noise_linear_slip_mean']) == 0.0, 'encoder not calibrated in the run'
assert float(m['manager_fusion_disagreement_gate_m']) == 0.0 and ms.get('manager_assume_initial_belief_anchor') is True, 'gate or anchor wrong'
assert any(r['status'] == 'accepted' for r in a), 'no camera update accepted'
log = json.load(open(glob.glob('logs/thesis/qualification/v10_pilot/campaign_log.json')[0]))
print('pilot outcome', json.dumps(log)[:300])
assert 'infra_invalid' not in json.dumps(log), 'pilot infra_invalid'
EOF
st "9 pilot passed"

# 10. lock the campaign inputs (clean tree), then run the campaign and analyse
[ -z "$(git status --porcelain)" ] || die "working tree not clean before the campaign manifest"
avail=$(df --output=avail -B1G . | tail -1); [ "$avail" -ge 4 ] || die "less than 4 GB free"
python3 pipeline/campaign_manifest.py >> "$V/STATUS" 2>&1 || die "campaign manifest failed"
st "10 campaign manifest written; campaign started (no commits until it ends)"
bash pipeline/campaign.sh >> "$V/STATUS" 2>&1 || die "campaign failed (see $R/pipeline.log)"
python3 pipeline/campaign_manifest.py >> "$V/STATUS" 2>&1 || die "campaign inputs drifted"
python3 pipeline/analyze_campaign.py >> "$V/STATUS" 2>&1 || die "analysis failed"
st "DONE: campaign analysed -> $R/analysis/"
