#!/usr/bin/env python3
"""Read exactly two registered frozen selections; inventory, never rescore results.

Sample counts are data-accounting counts, not independent replicates. No accuracy,
NEES, coverage or navigation-improvement estimate is emitted by this script.
"""
import hashlib
import json
import math
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'experiments/fusion_on_fixed_routes'))
import aligned as A

OUT=REPO/'experiments/runtime_integrity/alignment_reporting_20260906'
OUT.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def f(row,key):
    try:return float(row[key])
    except (ValueError,TypeError,KeyError):return float('nan')

registry_path=REPO/'docs/localization_metrics_registry.json'
registry=json.loads(registry_path.read_text())
inventory=[]
required=set(registry['required_runtime_evidence'])
for key in ('thesis_commissioning_pilot','network_navigation_runtime_pilot'):
    boundary=registry[key]
    path=REPO/boundary['selection']
    frozen=json.loads(path.read_text())
    seen=set();runs=[]
    for entry in frozen['runs']:
        run=REPO/entry['run']
        if str(run.resolve()) in seen:raise ValueError('repeated selected run')
        seen.add(str(run.resolve()))
        if not required<=entry['files'].keys():raise ValueError('missing required hashes')
        for filename,expected in entry['files'].items():
            if sha(run/filename)!=expected:raise ValueError(f'changed frozen bytes: {run/filename}')
        m=json.loads((run/'run_manifest.json').read_text())
        s=json.loads((run/'run_summary.json').read_text())
        if (m['task'],m['seed'])!=(entry['task'],entry['seed']):raise ValueError('identity mismatch')
        table=A.rows(run);obs=A.observations(run);ass=A.assimilations(run)
        bids={o['source_batch_id'] for o in obs}
        if bids!={a['source_batch_id'] for a in ass}:raise ValueError('unaccounted correction')
        if any(a['status'] not in registry['run_validity']['valid_statuses'] or
               (a['status'] in ('rejected','dropped') and not a['reason']) for a in ass):
            raise ValueError('unclassifiable terminal outcome')
        start,stop=float(s['first_cmd_stamp']),float(s['stop_stamp'])
        stamps=[f(r,'planner_belief_stamp') for r in table]
        unique={t for t in stamps if math.isfinite(t) and start<=t<=stop}
        truth=A.truth_series(run,table)
        manager_candidates={(o['source_batch_id'],o['camera']) for o in obs}
        runs.append(dict(key=entry['key'],run=entry['run'],schema=A.schema_version(run),
            task=entry['task'],seed=entry['seed'],selection_file_hashes_verified=len(entry['files']),
            manifest_source_and_artifact_hashes={k:v for k,v in m.items() if k.endswith('_sha256') or k=='git_sha'},
            manager_observation_model=m.get('manager_observation_model'),
            manager_covariance_profile=m.get('manager_covariance_profile'),
            learned_correction_path=m.get('manager_learned_correction_path'),
            gt_stamp_source=s.get('gt_stamp_source'),loader_truth_source=truth.source,
            logger_rows=len(table),unique_in_drive_belief_stamps=len(unique),
            manager_candidate_identities=len(manager_candidates),fused_batch_identities=len(bids),
            assimilation_rows=len(ass),
            accepted_flag_conflicts=sum(a['accepted']!=(a['status'] in ('accepted','accepted_bootstrap','reanchored')) for a in ass),
            summary_assimilation_count=s.get('correction_assimilation_count'),
            negative_logger_time_steps=sum(f(b,'stamp')<f(a,'stamp') for a,b in zip(table,table[1:])),
            negative_belief_time_steps=sum(b<a for a,b in zip(stamps,stamps[1:]) if math.isfinite(a) and math.isfinite(b)),
            summary_collision_contact=s.get('collision_contact'),contact_messages_seen=s.get('contact_messages_seen')))
    group=dict(registry_key=key,selection=boundary['selection'],selection_sha256=sha(path),
        status=boundary['status'],permitted_use=boundary['permitted_use'],
        replay_timing=boundary.get('replay_timing','logged live beliefs; no arrival-time counterfactual replay'),
        runs=runs)
    if boundary.get('protocol'):
        protocol_path=REPO/boundary['protocol']
        protocol=json.loads(protocol_path.read_text())
        group['protocol_sha256']=sha(protocol_path)
        group['frozen_protocol_sources']=protocol.get('sources')
    inventory.append(group)

# Exact model/input provenance, separate from the executed source provenance in each run.
model_paths=[
    'logs/perception_models/warehouse_v2_yolo_detect_halfopen_20260825_r1/model.pt',
    'logs/perception_models/box_feature_bias_correction_20260831/models.joblib',
    'logs/studies/icra_commissioning_20260905/models.joblib',
    'logs/studies/icra_commissioning_20260905/field_study/field.joblib',
    'logs/studies/icra_commissioning_20260905/network_planner/reference_calibration.json',
]
result=dict(kind='registered_data_inventory_not_performance_results',registry_sha256=sha(registry_path),
    loader_sha256=sha(REPO/'experiments/fusion_on_fixed_routes/aligned.py'),
    current_named_model_sha256={p:sha(REPO/p) for p in model_paths},selections=inventory)
(OUT/'registered_inventory.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
for group in inventory:
    print(group['registry_key'],group['selection_sha256'])
    for run in group['runs']:
        print(run['key'],run['schema'],run['unique_in_drive_belief_stamps'],run['manager_candidate_identities'],run['assimilation_rows'],run['gt_stamp_source'])
