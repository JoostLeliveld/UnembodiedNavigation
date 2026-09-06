#!/usr/bin/env python3
"""Trace the longest accepted-correction gap in an explicitly selected pilot arm."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

import network_navigation_analysis as nav


def main(selection, arm, out):
    entry=next(e for e in json.loads(selection.read_text())['runs'] if e['arm']==arm)
    run=nav.REPO/entry['run']
    for name,expected in entry['files'].items():
        if nav.digest(run/name)!=expected:raise ValueError('Changed selected input')
    summary=json.loads((run/'run_summary.json').read_text())
    start,stop=summary['first_cmd_stamp'],summary['stop_stamp']
    ass=nav.aligned.assimilations(run)
    times=[start]+[r['apply_stamp'] for r in ass if r['accepted'] and start<=r['apply_stamp']<=stop]+[stop]
    index=int(np.argmax(np.diff(times)));lo,hi=times[index:index+2]
    events=[r for r in ass if lo<r['apply_stamp']<=hi]
    opportunities=[]
    for line in (run/'camera_opportunities.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row['valid_contract'] and not row['duplicate'] and lo<row['observation']['timestamp_s']<=hi:
            opportunities.append(row['observation'])
    cameras=sorted({r['camera_id'] for r in opportunities})
    counts={c:dict(opportunities=sum(r['camera_id']==c for r in opportunities),
                   detection_valid=sum(r['camera_id']==c and r['detection_valid'] for r in opportunities)) for c in cameras}
    table=nav.aligned.rows(run);belief=nav.aligned.aligned_error_cm(run,'belief',table)
    mask=nav.aligned.landed_mask(belief['stamp'])&(belief['stamp']>=lo)&(belief['stamp']<=hi)
    events=[{k:(None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in e.items()} for e in events]
    result=dict(kind='single_run_correction_opportunity_gap_diagnostic',arm=arm,run=entry['run'],
        gap_apply_start_s=lo,gap_apply_end_s=hi,gap_s=hi-lo,
        physical_opportunity_batches=len({r['source_batch_id'] for r in opportunities}),
        per_camera_capture_counts=counts,terminal_fused_events=events,
        status_counts=dict(Counter(e['status'] for e in events)),
        interpretation='Detections do not establish usable localization measurements. '
            'The log does not retain a terminal per-camera manager admission reason for every opportunity, '
            'so the disappearance between detections and published fusion cannot be assigned to one gate from these logs. '
            'Camera-gap refusals happen before NIS. No counterfactual acceptance or navigation improvement is asserted.',
        sources={str(p.relative_to(nav.REPO)):nav.digest(p) for p in [selection,Path(__file__),Path(nav.__file__)]})
    out.mkdir(parents=True,exist_ok=True);nav.writejson(out/'gap_diagnostic.json',result)
    nav.style();fig,axes=nav.plt.subplots(2,1,figsize=(9,5.8),sharex=True,layout='constrained')
    for i,c in enumerate(cameras):
        t=[r['timestamp_s']-lo for r in opportunities if r['camera_id']==c and r['detection_valid']]
        axes[0].scatter(t,np.full(len(t),i),s=5,color='#536f86')
    for e in events:
        axes[0].scatter(e['correction_stamp']-lo,-1,marker='o' if e['accepted'] else 'x',
                        s=55,color='#207b70' if e['accepted'] else '#bb5540',zorder=5)
    axes[0].set(yticks=[-1,*range(len(cameras))],
        yticklabels=['Fused candidate',*[f"Camera {c[-1]}" for c in cameras]],ylim=(-1.6,len(cameras)-.5))
    axes[0].scatter([],[],s=8,color='#536f86',label='Detection-valid camera frame (capture time)')
    axes[0].scatter([],[],s=35,marker='x',color='#bb5540',label='Fused candidate refused')
    axes[0].scatter([],[],s=30,color='#207b70',label='Fused candidate accepted')
    axes[0].legend(loc='upper right',fontsize=8,framealpha=.95)
    axes[1].plot(belief['stamp'][mask]-lo,belief['aligned_cm'][mask],color='#207b70',label='Aligned online-belief XY error')
    for e in events:
        if e['reason']=='replay_gap_too_large':
            x=e['apply_stamp']-lo
            axes[1].axvline(x,color='#bb5540',ls=':',lw=1.)
    axes[1].set(xlabel='Time since last accepted correction [s, simulation]',
                 ylabel='Belief position error [cm]',xlim=(-1,hi-lo+1))
    axes[1].legend(fontsize=8,loc='upper left')
    for ax in axes:ax.grid(alpha=.15)
    fig.suptitle(f"{hi-lo:.1f} seconds without an accepted update, with detector sightings present\n"
                 'Refused isolated candidates and missing admission accounting need separate diagnosis',fontsize=11)
    nav.savefig(fig,out/'gap_diagnostic')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--arm',default='P2')
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    main(a.selection.resolve(),a.arm,a.out.resolve())
