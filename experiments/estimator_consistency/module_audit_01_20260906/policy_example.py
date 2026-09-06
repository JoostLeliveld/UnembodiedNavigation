"""Inspect the completed, explicitly selected P2 pilot's longest acceptance gap.

Only terminal event accounting is read through the required loader. No localization
accuracy is scored and no counterfactual camera acceptance is assumed.
"""
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'experiments/fusion_on_fixed_routes'))
import aligned

selection_path = ROOT / 'logs/studies/icra_commissioning_20260905/network_navigation_runtime_evidence/selection.json'
selection = json.loads(selection_path.read_text())
selected = [r for r in selection['runs'] if r['key'] == 'fusion_network_traverse__P2__seed210']
assert len(selected) == 1
record = selected[0]
run = ROOT / record['run']
ledger_path = run / 'correction_assimilations.csv'
assert hashlib.sha256(ledger_path.read_bytes()).hexdigest() == record['files']['correction_assimilations.csv']
rows = sorted(aligned.assimilations(run), key=lambda r: r['apply_stamp'])
accepted = [r for r in rows if r['accepted'] and r['status'] in ('accepted','accepted_bootstrap','reanchored')]
before, after = max(zip(accepted, accepted[1:]), key=lambda pair: pair[1]['apply_stamp']-pair[0]['apply_stamp'])
window = [r for r in rows if before['apply_stamp'] < r['apply_stamp'] <= after['apply_stamp']]
for row in window:
    if not math.isfinite(row['nis']):
        row['nis'] = None
result = dict(scope='One-run terminal-event diagnostic, not a policy-effect estimate',
              selection_path=str(selection_path.relative_to(ROOT)),
              selection_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest(),
              run=record['run'], ledger_sha256=record['files']['correction_assimilations.csv'],
              selection_rule='Longest apply-time gap between accepted terminal events in selected P2 seed 210',
              before=before, after=after,
              acceptance_gap_s=after['apply_stamp']-before['apply_stamp'],
              events_after_previous_acceptance=window,
              limit='Actual motion-support snapshots and alternative gate outcomes are not logged')
print(json.dumps(result,indent=2,allow_nan=False))
