#!/usr/bin/env python3
"""Explicit, append-only thesis evidence selection; never choose runs by glob.

The two recovered executions are named in advance from their run manifests.
Remaining executions must be named by the resumed campaign's terminal ledger.
No incomplete campaign is promoted to replicated paper evidence.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import json,sys,argparse
from pathlib import Path
import joblib,numpy as np
from field_driving import load_run,analyze,REQUIRED,CAMERAS
from field_study import FIELD_OUT,OUT,REPO,REQ,digest,load_data,grouped_probability_score
from study import load,score,KINDS

DEST=OUT/'thesis_evidence'
INITIAL=[('fusion_overlap_rich',110,'experiment_20260905_181145'),
         ('fusion_overlap_rich',111,'experiment_20260905_182525')]
EXPECTED=[(t,s) for t in ['fusion_overlap_rich','fusion_network_traverse'] for s in [110,111,112]]

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');temporary.replace(path)

def selection():
    candidates={(t,s):(OUT/'field_pilot'/t/'N1'/f'seed{s}'/r,'recovered_explicit_execution') for t,s,r in INITIAL}
    ledger=OUT/'field_pilot_remaining/campaign_log.json'
    if ledger.exists():
        entries=json.loads(ledger.read_text())
        for (t,s) in EXPECTED:
            e=entries.get(f'{t}__N1__seed{s}',{})
            if e.get('finished_at') and e.get('run_dir'):
                candidates[t,s]=(Path(e['run_dir']),'remaining_campaign_terminal_ledger')
    selected=[];pending=[];invalid=[]
    for t,s in EXPECTED:
        if (t,s) not in candidates:pending.append(dict(task=t,seed=s));continue
        run,source=candidates[t,s]
        try:
            manifest=json.loads((run/'run_manifest.json').read_text())
            summary=json.loads((run/'run_summary.json').read_text())
            if (manifest['task'],manifest['seed'])!=(t,s):raise ValueError('identity mismatch')
            if manifest['logging_schema_version']<7:raise ValueError('opportunity contract unavailable')
            if not summary['completed']:raise ValueError('incomplete execution')
            if not summary.get('valid_run'):raise ValueError('runtime invalid run')
            entry=dict(key=f'{t}__N1__seed{s}',run=str(run.relative_to(REPO)),task=t,seed=s,
                files={name:digest(run/name) for name in REQUIRED},status='completed',
                field_sha256=digest(FIELD_OUT/'field.joblib'),requirements_sha256=digest(REQ))
            destination=DEST/'driving'/entry['key']/'manifest.json'
            if destination.exists() and json.loads(destination.read_text())!=entry:raise ValueError('frozen execution changed')
            load_run(entry)  # hashes, fixed Q, measured motion, aligned.py, full opportunities
            save(destination,entry)
            selected.append(dict(**entry,selection_source=source))
        except Exception as error:invalid.append(dict(task=t,seed=s,run=str(run),reason=str(error)))
    result=dict(status='complete_pilot_diagnostic' if not pending and not invalid else 'partial_pilot_diagnostic',
                expected=[dict(task=t,seed=s) for t,s in EXPECTED],runs=selected,pending=pending,invalid=invalid,
                provenance_note='Two original executions recovered by explicit identity; four remaining predeclared executions use a new campaign ledger. Original empty ledger preserved.',
                no_navigation_comparison=True)
    save(DEST/'selection.json',result);return result

def static():
    # Both loaders verify the original frozen inputs and reproduce the NN means.
    data,counts=load(OUT);models=joblib.load(OUT/'models.joblib')
    rows=[r for r in data if r['role']=='evaluation'];scores={}
    for kind in KINDS:
        e=[];R=[]
        for r in rows:
            z,C=models[r['camera'],kind].predict([r]);e.append(z[0]-r['truth']);R.append(C[0])
        scores[kind]=score(np.asarray(e),np.asarray(R),[r['group'] for r in rows])
    fields,geometry,C=load_data(FIELD_OUT);field=joblib.load(FIELD_OUT/'field.joblib')
    ev=fields['role']=='evaluation';availability={}
    for kind,model in field.availability.items():
        p=model.predict(fields['pose'][ev]);availability[kind]=grouped_probability_score(fields['hits'][ev],p,fields['group'][ev])
    result=dict(evidence='previously_inspected_grouped_development',counts=counts,covariance=scores,availability=availability,
        frozen_inputs=digest(OUT/'manifest.json'),field_inputs=digest(FIELD_OUT/'manifest.json'),
        accepted=len(rows),opportunities=int(ev.sum()*5),groups=len(set(fields['group'][ev])))
    save(DEST/'static.json',result)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--static',action='store_true');parser.add_argument('--select-only',action='store_true');args=parser.parse_args()
    DEST.mkdir(parents=True,exist_ok=True)
    if args.static:static()
    selected=selection();print(json.dumps({k:v for k,v in selected.items() if k!='runs'}),flush=True)
    if not args.select_only:
        for entry in selected['runs']:
            print('Analyzing',entry['key'],flush=True)
            analyze(entry,DEST/'driving')
    save(DEST/'analysis_sources.json',{str(p.relative_to(REPO)):digest(p) for p in [Path(__file__),
        REPO/'experiments/icra_commissioning/field_driving.py',REPO/'experiments/icra_commissioning/replay.py',
        REPO/'experiments/icra_commissioning/model.py',REPO/'experiments/fusion_on_fixed_routes/aligned.py']})
