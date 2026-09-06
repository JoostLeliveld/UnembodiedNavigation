#!/usr/bin/env python3
"""Score only the explicit completed baseline, via the repository alignment contract."""
import json,sys
from pathlib import Path
import numpy as np
from study import OUT,REPO,digest,writejson
sys.path.insert(0,str(REPO/'experiments/fusion_on_fixed_routes'))
import aligned as A

RUN='logs/studies/icra_commissioning_20260905/baseline_cpu_complete/fusion_network_traverse/N1/seed10/experiment_20260905_122406'

def main():
    p=REPO/RUN;files=['run_manifest.json','run_summary.json','experiment.csv','fusion_observations.csv','correction_assimilations.csv']
    selection=OUT/'baseline_manifest.json'
    if not selection.exists():writejson(selection,dict(status='new_runtime_reproduction_not_replicated_paper_result',run=RUN,files={f:digest(p/f) for f in files}))
    frozen=json.loads(selection.read_text())
    for f,h in frozen['files'].items():
        if digest(p/f)!=h:raise ValueError('baseline artifact changed')
    manifest=json.loads((p/'run_manifest.json').read_text());summary=json.loads((p/'run_summary.json').read_text())
    ass=A.assimilations(p);obs=A.observations(p)
    assert {o['source_batch_id'] for o in obs}=={a['source_batch_id'] for a in ass}
    assert all(a['status'] in ['accepted','accepted_bootstrap','reanchored','rejected','dropped'] and
       (a['status'] not in ['dropped','rejected'] or a['reason']) for a in ass)
    b=A.aligned_error_cm(p,'belief');keep=A.landed_mask(b['stamp'])&np.isfinite(b['aligned_cm'])
    # Clip to first command through terminal event; omit pre-drive bootstrap from drive accuracy.
    first=summary['first_cmd_stamp'];stop=summary['stop_stamp'];keep&=(b['stamp']>=first)&(b['stamp']<=stop)
    e=b['aligned_cm'][keep];accepted=[a['belief_stamp_after'] for a in ass if a['accepted']]
    times=sorted([first,stop]+[t for t in accepted if first<=t<=stop])
    dropped=sum(a['status']=='dropped' for a in ass)
    result=dict(run=RUN,run_id=manifest['run_id'],outcome=summary['completion_reason'],
       reference='simulator gt at planner_belief_stamp',quantity='planner belief position error',n=int(keep.sum()),
       median_cm=float(np.median(e)),p95_cm=float(np.quantile(e,.95)),rmse_cm=float(np.sqrt(np.mean(e*e))),
       dropped_fraction=dropped/len(ass),longest_gap_s=float(np.max(np.diff(times))),
       accounting=f'{len(ass)} batches, each accounted for',dropped_count=dropped,
       collision_reason=summary.get('collision_reason'),elapsed_s=summary['elapsed_after_first_cmd_s'],
       trajectory={k:b[k][keep].tolist() for k in ['x','y','gt_x','gt_y','stamp']})
    writejson(OUT/'baseline_report.json',result);print({k:v for k,v in result.items() if k!='trajectory'})
if __name__=='__main__':main()
