#!/bin/bash
# v10 continuation: steps 9-10 of pipeline/ops/v10_overnight.sh (pilot, campaign lock, campaign,
# analysis), after steps 1-8 passed on 2026-09-25 (dataset v10, lock, refit, sealed audit, routes).
cd "$(dirname "$0")/../.."
V=logs/thesis/captures/v10
R=logs/thesis
st() { echo "[$(date '+%F %T')] $*" | tee -a "$V/STATUS"; }
die() { st "STOPPED: $*"; exit 1; }
source /opt/ros/humble/setup.bash >/dev/null 2>&1; source install/setup.bash >/dev/null 2>&1
export PYTHONWARNINGS=ignore

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
assert float(m['manager_fusion_disagreement_gate_m']) == 0.0 and m.get('manager_assume_initial_belief_anchor') is True, 'gate or anchor wrong'
assert m.get('git_sha'), 'run manifest has no git_sha'
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
