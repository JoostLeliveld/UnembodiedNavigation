"""Offline-only join of private observer phases to nearest native world pose."""
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
d=json.loads((HERE/'watchdog_results.json').read_text())
epochs=[[]]
for line in (HERE/'watchdog.ground_truth.jsonl').read_text().splitlines():
    if not line.startswith('{'):continue
    r=json.loads(line);s=r.get('header',{}).get('stamp',{})
    t=float(s.get('sec',0))+float(s.get('nsec',0))*1e-9
    if epochs[-1] and t<epochs[-1][-1]['stamp_s']:epochs.append([])
    robot=next((p for p in r.get('pose',[]) if p.get('name')=='turtlebot3'),None)
    if robot:epochs[-1].append({'stamp_s':t,'entity':robot})
joined=[];epoch=0
for phase in d['phases']:
    if phase['label']=='after_reset_no_new_command':epoch=1
    target=phase['last']['odom']['stamp_s']
    sample=min(epochs[epoch],key=lambda p:abs(p['stamp_s']-target))
    joined.append({'label':phase['label'],'epoch':epoch,'odom_stamp_s':target,
                   'native_reference':sample,'reference_offset_s':sample['stamp_s']-target})
by={r['label']:r for r in joined}
deltas={}
for a,b in [('before_clock_bridge_loss','clock_bridge_lost_physics_running'),
            ('before_adapter_death','fresh_adapter_without_new_input')]:
    start=by[a]['native_reference'];end=by[b]['native_reference']
    p=start['entity']['position'];q=end['entity']['position']
    deltas[a]={'from':a,'to':b,'sim_interval_s':end['stamp_s']-start['stamp_s'],
                'dx_m':q.get('x',0)-p.get('x',0),'dy_m':q.get('y',0)-p.get('y',0),
                'distance_m':math.hypot(q.get('x',0)-p.get('x',0),q.get('y',0)-p.get('y',0))}
result={'scope':'Offline native reference only; nearest source-stamped samples, no interpolation, no online use.',
         'partition':d['partition'],'epoch_sample_counts':[len(e) for e in epochs],
         'phases':joined,'physical_displacements':deltas}
(HERE/'watchdog_native_summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(deltas,indent=2))
