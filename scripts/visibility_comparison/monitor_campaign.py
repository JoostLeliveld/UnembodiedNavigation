#!/usr/bin/env python3
"""Triage from an explicit campaign ledger; this output is not scoring evidence.

--config adds the exact planned task/condition/seed denominator. Without it, only
ledger entries are inventoried, so an absent planned trial cannot be inferred.
Logger summary means are tick-weighted diagnostics, not independent observations.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OFFLINE = ROOT / 'logs/paper_figures/offline_plan_sanity.json'


def _num(x, nd=2):
    try:
        value=float(x)
        return round(value,nd) if math.isfinite(value) else None
    except (TypeError,ValueError):
        return None


def _offline_routes():
    if not OFFLINE.is_file():return {}
    data=json.loads(OFFLINE.read_text())
    return {(task,condition):record['route'] for task,groups in data.items()
            for condition,record in groups.items() if isinstance(record,dict) and 'route' in record}


def expected_trials(config):
    import yaml
    cfg=yaml.safe_load(Path(config).read_text())
    expected={}
    for task,group in cfg['tasks'].items():
        for condition in group['conditions']:
            for seed in group['seeds']:
                key=f'{task}__{condition}__seed{seed}'
                if key in expected:raise ValueError('duplicate configured trial identity')
                expected[key]=(task,condition,str(seed))
    return expected


def collect(log_root: Path, config=None):
    path=Path(log_root)/'campaign_log.json'
    if not path.is_file():raise ValueError(f'{path}: explicit campaign ledger required')
    ledger=json.loads(path.read_text())
    expected=expected_trials(config) if config else {}
    off=_offline_routes();rows=[]
    for key in sorted(set(ledger)|set(expected)):
        parts=key.rsplit('__',2)
        if len(parts)!=3 or not parts[2].startswith('seed'):
            raise ValueError(f'malformed campaign key: {key}')
        task,cond,seed=parts[0],parts[1],parts[2][4:]
        event=ledger.get(key,{})
        history=event.get('attempts',[])
        attempts=[(a,False) for a in history]+[(event,True)]
        for index,(attempt,current) in enumerate(attempts):
            explicit=(attempt.get('task',task),attempt.get('condition',cond),str(attempt.get('seed',seed)))
            if explicit!=(task,cond,seed):raise ValueError(f'{key}: attempt identity disagrees with campaign key')
            run=Path(attempt['run_dir']) if attempt.get('run_dir') else None
            if run is not None and not run.is_absolute():run=ROOT/run
            summary_path=run/'run_summary.json' if run else None
            rs=json.loads(summary_path.read_text()) if summary_path and summary_path.is_file() else {}
            meta_path=run/'global_plan_meta.json' if run else None
            meta=json.loads(meta_path.read_text()) if meta_path and meta_path.is_file() else {}
            identity_error=None
            manifest_path=run/'run_manifest.json' if run else None
            if manifest_path and manifest_path.is_file():
                manifest=json.loads(manifest_path.read_text())
                if (manifest.get('task'),str(manifest.get('seed')))!=(task,seed):
                    identity_error='manifest task/seed mismatch'
            completion=rs.get('completion_reason') or attempt.get('completion_reason')
            if not rs:completion='missing_summary' if run else ('pending' if not attempt else 'no_run_dir')
            rows.append(dict(task=task,cond=cond,seed=seed,key=key,current=current,
                attempt=attempt.get('attempt_id',str(index)),run=str(run) if run else None,
                planned=(key in expected) if config else None,identity_error=identity_error,
                has_plan=bool(meta),route=str(meta.get('selected_source','')).removeprefix('solver:route:').removeprefix('solver:'),
                off=off.get((task,cond),'?'),fcmd=_num(rs.get('first_cmd_stamp'),1),
                completion=completion,collision=rs.get('collision_any'),valid=rs.get('valid_run'),
                invalid=rs.get('invalid_reason'),gt_err=_num(rs.get('mean_belief_error_gt_after_first_cmd_m'),3),
                drop_frac=_num(rs.get('correction_dropped_fraction'),3),
                rejected=rs.get('correction_rejected_count'),blind_s=_num(rs.get('longest_correction_gap_s'),1)))
    return rows,off


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log_root',type=Path)
    parser.add_argument('--config',type=Path,help='exact campaign YAML to show missing/extra trials')
    parser.add_argument('--expect',type=int,help='diagnostic count hint only; --config supplies exact seeds')
    args=parser.parse_args(argv)
    root=args.log_root if args.log_root.is_absolute() else ROOT/args.log_root
    rows,_=collect(root,args.config)
    print('TRIAGE ONLY: logger-tick mean belief error versus GT at belief time; metres. '
          'dropped is not all refusals; blind gap follows logger summary support.')
    if args.config is None:print('Planned denominator unavailable: supply --config; only ledger entries are shown.')
    for r in rows:
        suffix=' current' if r['current'] else ' prior attempt'
        extra=' EXTRA' if r['planned'] is False else ''
        print(f"{r['task']}/{r['cond']}/seed{r['seed']} attempt={r['attempt']}{suffix}{extra} "
              f"outcome={r['completion']} valid={r['valid']} plan={r['has_plan']} "
              f"mean_belief_gt_m={r['gt_err']} dropped_fraction={r['drop_frac']} "
              f"longest_gap_s={r['blind_s']} identity_error={r['identity_error']}")
    current=[r for r in rows if r['current']]
    for task,cond in sorted({(r['task'],r['cond']) for r in rows}):
        cells=[r for r in current if (r['task'],r['cond'])==(task,cond)]
        attempts=sum((r['task'],r['cond'])==(task,cond) and r['run'] is not None for r in rows)
        print(f'{task}/{cond}: {len({r["seed"] for r in cells})} unique seed cells; {attempts} recorded attempts; '
              f'current outcomes {dict(Counter(r["completion"] for r in cells))}')
    return 0


if __name__=='__main__':raise SystemExit(main())
