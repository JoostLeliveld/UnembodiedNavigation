"""Deterministic admission/recovery audit, no ROS graph or runtime edits.

Assertions reproduce the reviewed baseline; a passing probe is not policy approval.
Run: source install/setup.bash; python3 docs/module_audits/06_policy_probe.py
"""
from pathlib import Path
import sys, json, math, hashlib, itertools
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tests/planning'))
from test_planner_node_state_correction import make_state_node
from test_planner_node_per_camera_correction import observation
from test_planner_node_correction_wiring import stamp
from test_outage_motion_replay import MotionModel
from planning.core.motion_history import covers_interval
from planning.core.belief_correction import CorrectionGates
from reliability.nodes.camera_manager_node import _largest_agreeing_group, _gated_fusion, _nearest_state_pose
from std_msgs.msg import String

def node(**kw):
    n=make_state_node(pixel_timeout_s=.5, **kw)
    n.state_reanchor_m=0.
    n.heading_update_mode='coupled'
    n.max_predict_speed_mps=.22
    n._seen_map_observation_stamps={}
    n._seen_state_source_batch_ids=set()
    n._resolve_plan_frame_id=lambda: 'map_bev'
    return n

def event(n,t,z,identity):
    n._clock.seconds=t+.01
    out=n._apply_metric_correction(stamp(t),np.array(z),np.eye(2)*.01,source_batch_id=identity)
    row=json.loads(n.correction_assimilation_pub.published[-1])
    row['nis']=row['nis'] if math.isfinite(row['nis']) else None
    row['m']=n.belief_m.tolist(); row['Pdiag']=np.diag(n.belief_S).tolist()
    return out,row

results={}
for complete in (True,False):
    n=node(belief_stamp_s=0.,now_s=10.01)
    n.use_odom_for_predict=True; n.planner=MotionModel()
    history=[(float(t),.2 if t<2 or t>=4 else 0.,math.pi/4 if 2<=t<4 else 0.) for t in np.arange(0.,10.21,.1)]
    n._odom_log=history if complete else [e for e in history if e[0]>=9.]
    _,first=event(n,10.,[.4,1.2],'return')
    _,second=event(n,10.2,[.4,1.24],'next')
    assert first['status']=='dropped' and first['reason']=='replay_gap_too_large'
    assert first['belief_stamp_after']==10.
    if complete:
        assert abs(first['m'][2]-math.pi/2)<1e-8
        assert second['status']=='accepted'
    results['blind_turn_complete' if complete else 'blind_turn_incomplete']={'supported':covers_interval(n._odom_log,0.,10.,1.5),'first':first,'next':second}

n=node(belief_cov=.01,belief_stamp_s=0.)
rows=[]
for k in range(1,31):
    out,row=event(n,.2*k,[2.,0.],f'outlier-{k}')
    rows.append(row)
    if out is not None and out.accepted: break
assert rows[0]['status']=='rejected' and rows[-1]['status']=='accepted'
results['persistent_outlier']={'first_acceptance_event':len(rows),'events':rows}
n=node(belief_cov=.01,belief_stamp_s=0.)
rows=[event(n,.2*k,[2.,0.],f'reject-{k}')[1] for k in range(1,5)]
rows.append(event(n,1.,[0.,0.],'valid-recovery')[1])
assert rows[-1]['status']=='accepted' and abs(n.belief_m[0])<1e-12
results['rejection_then_valid']=rows

# A bootstrap returns None, so the batch's accepted_any flag misses it.
n=node(); n.belief_m=n.belief_S=n.belief_stamp=None
n._apply_map_observations([observation('A',1,0,seconds=9.95,var=.01)])
assert np.isclose(n.belief_S[0,0],.06)
results['per_camera_bootstrap_inflation']={'Pxx':float(n.belief_S[0,0]),'Rxx':.01}

# Two successive frames of ONE camera meet the quorum's length test.
n=node(); n.state_reanchor_m=2.
n._apply_map_observations([observation('A',3,0,seconds=9.94),observation('A',3,0,seconds=9.95)])
assert np.isclose(n.belief_m[0],3.)
results['one_camera_quorum']={'mean':n.belief_m.tolist(),'seen':n._seen_map_observation_stamps}

# Fixed batch permutation is deterministic; renaming which camera sorts first is not.
ordered=[]
for xs in ((.6,-.6),(-.6,.6)):
    n=node(belief_cov=.05)
    batch=[observation('A',xs[0],0,seconds=9.95,var=.01),observation('B',xs[1],0,seconds=9.95,var=.01)]
    n._apply_map_observations(batch)
    before=n.belief_S.copy(); n._apply_map_observations(batch)
    assert np.array_equal(before,n.belief_S)
    ordered.append(n.belief_m.tolist())
assert ordered[0][0]>0 and ordered[1][0]<0
results['same_time_order']={'camera_id_order_results':ordered,'exact_duplicate_no_covariance_change':True}
permuted=[]
batch=[observation('A',.6,0,seconds=9.95,var=.01),observation('B',-.6,0,seconds=9.95,var=.01)]
for permutation in itertools.permutations(batch):
    n=node(); n._apply_map_observations(permutation); permuted.append(n.belief_m.tolist())
assert np.allclose(permuted[0],permuted[1])
results['same_batch_arrival_permutation_invariant']=True

# Long-gap handling ignores the caller's inflate_on_reject=False, then outer batch inflates.
n=node(belief_stamp_s=0.); before=n.belief_S.copy()
n._apply_map_observations([observation('A',0,0,seconds=9.95)])
direct_variance=float(n.belief_S[0,0])
f=node(belief_stamp_s=0.)
f._apply_metric_correction(stamp(9.95),np.array([0.,0.]),np.eye(2)*.03)
assert np.isclose(direct_variance-f.belief_S[0,0],.05)
results['direct_gap_extra_batch_inflation']=float(direct_variance-f.belief_S[0,0])

# Stale new frames inflate once per batch, even with no statistical test.
n=node(); before=n.belief_S.copy()
n._apply_map_observations([observation('A',0,0,seconds=8.)])
assert np.isclose(n.belief_S[0,0]-before[0,0],.05)
results['stale_batch_inflation']=float(n.belief_S[0,0]-before[0,0])

# Wrong-frame direct transport is parsed, but the receiver ignores its frame.
n=node(); n.state_correction_mode='per_camera'
from reliability.fusion import map_observations_to_json
msg=String(); msg.data=map_observations_to_json([observation('A',.05,0,seconds=9.95)],frame_id='foreign')
n._map_observations_cb(msg)
assert n.belief_m[0]>0
results['foreign_frame_direct_accepted']=n.belief_m.tolist()

# Non-finite timestamp is not checked by MapObservation; it poisons the dedup watermark.
n=node()
# Callback conversion would fatal after the watermark mutation; record both effects.
try: n._apply_map_observations([observation('A',0,0,seconds=math.inf)])
except (OverflowError,ValueError): pass
assert n._seen_map_observation_stamps['A']==math.inf
before=n.belief_S.copy(); n._apply_map_observations([observation('A',0,0,seconds=9.95)])
assert np.array_equal(n.belief_S,before)
results['nonfinite_direct_stamp_claimed_before_failure']=True

# Future envelope can advance the anchor and make the next present event old.
n=node(belief_stamp_s=9.9,now_s=10.)
n._apply_metric_correction(stamp(10.4),np.array([0.,0.]),np.eye(2)*.01,source_batch_id='future')
_,row=event(n,10.1,[0.,0.],'present')
assert row['reason']=='not_newer_than_belief'
results['future_anchor']=row
results['nearest_pose_uses_later_state']=_nearest_state_pose([(9.8,(0.,0.,0.)),(10.1,(1.,0.,1.))],10.,max_delta_s=.35)
assert results['nearest_pose_uses_later_state']==(1.,0.,1.)

# Manager's actual eligibility is unique-by-camera; its pure helper requires >=2.
assert not _largest_agreeing_group([observation('A',0,0,seconds=1.)],.91)
results['bootstrap_min_one_helper_empty']=True
fusion=[]
for distance in (.9,1.3):
    batch=[observation('A',0,0,seconds=1.),observation('B',distance,0,seconds=1.)]
    r=_gated_fusion(batch,disagreement_gate_m=.6,rule='joint_network')
    fusion.append({'separation_m':distance,'used':r.accepted_camera_ids,'mean':r.mean_xy})
assert len(fusion[0]['used'])==2 and not fusion[1]['used']
results['two_camera_disagreement']=fusion
assert CorrectionGates(pixel_timeout_s=.5,dt_nominal_s=.25,max_predict_dt_s=1.5).max_dt_s==1.5
results['configured_prediction_limit']=1.5

# Every ordinary identified branch emits exactly one terminal reasoned row.
ledger={}
for kind in ['accepted','bootstrap','stale','old','gap','nis','reanchor']:
    n=node(belief_cov=.01)
    t=9.95; z=[0.,0.]
    if kind=='bootstrap': n.belief_m=n.belief_S=n.belief_stamp=None
    if kind=='stale': t=9.
    if kind=='old': t=9.8
    if kind=='gap': n.belief_stamp=stamp(0.)
    if kind in ('nis','reanchor'): z=[3.,0.]
    if kind=='reanchor': n.state_reanchor_m=2.
    n._apply_metric_correction(stamp(t),np.array(z),np.eye(2)*.01,source_batch_id=kind)
    assert len(n.correction_assimilation_pub.published)==1
    row=json.loads(n.correction_assimilation_pub.published[0]); assert row['reason']
    ledger[kind]={'status':row['status'],'reason':row['reason']}
results['terminal_branch_accounting']=ledger

# Actual hull gate, with a synthetic perfect box and a deliberately wrong prior.
from reliability.projection import camera_model_from_world
from reliability.silhouette_observation import plausibility_reasons
from unav_common.robot_hull import VISUAL_HULL, silhouette_box
manifest=json.loads((ROOT/'logs/perception_datasets/warehouse_v2_bbox_characterization_20260831/capture_manifest.json').read_text())
geometry=[]
for item in manifest['cameras']:
    cam=camera_model_from_world(ROOT/'src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf',include_name=item['camera_model'])
    for x,y in itertools.product(range(-8,9,2),range(-8,9,2)):
        box=silhouette_box(cam,float(x),float(y),0.,VISUAL_HULL)
        if box is None or plausibility_reasons(box,cam,x,y,0.): continue
        position=plausibility_reasons(box,cam,x+2.,y+2.,0.)
        heading=plausibility_reasons(box,cam,x,y,math.pi/2)
        if position and heading:
            geometry.append({'camera':item['camera_id'],'pose':[x,y,0.],'perfect_box':list(box),'correct_prior_reasons':[],'wrong_position_reasons':position,'wrong_heading_reasons':heading})
            break
    if geometry: break
assert geometry
results['useful_geometry_refused_by_wrong_prior']=geometry

# Recheck registered frozen ledger files through the required loader; no accuracy metrics.
import csv
sys.path.insert(0,str(ROOT/'experiments/fusion_on_fixed_routes'))
import aligned
selection_path=ROOT/'logs/studies/icra_commissioning_20260905/network_navigation_runtime_evidence/selection.json'
selection=json.loads(selection_path.read_text())
accounting=[]
for selected in selection['runs']:
    run=ROOT/selected['run']
    for filename in ('run_manifest.json','fusion_observations.csv','correction_assimilations.csv'):
        assert hashlib.sha256((run/filename).read_bytes()).hexdigest()==selected['files'][filename]
    observations=aligned.observations(run); terminal=aligned.assimilations(run)
    published={r['source_batch_id'] for r in observations}
    ids=[r['source_batch_id'] for r in terminal]
    assert len(ids)==len(set(ids)) and set(ids)==published and published
    assert all(r['status'] in ('accepted','accepted_bootstrap','reanchored','rejected','dropped') for r in terminal)
    assert all(r['reason'] for r in terminal if r['status'] in ('rejected','dropped'))
    accounting.append({'arm':selected['arm'],'run':selected['run'],'published_unique_batches':len(published),'terminal_rows':len(terminal),'status_counts':{s:sum(r['status']==s for r in terminal) for s in sorted({r['status'] for r in terminal})},'reasoned_refusals_valid':True})
results['registered_ledger_accounting']={'selection':str(selection_path.relative_to(ROOT)),'selection_sha256':hashlib.sha256(selection_path.read_bytes()).hexdigest(),'runs':accounting}
paths=['src/reliability/reliability/observation_gates.py','src/reliability/reliability/observation_opportunity.py','src/reliability/reliability/camera_manager.py','src/reliability/reliability/nodes/camera_manager_node.py','src/reliability/reliability/silhouette_observation.py','src/planning/planning/core/belief_correction.py','src/planning/planning/nodes/unicycle_planner_node.py','experiments/icra_commissioning/network_navigation_runtime_pilot.yaml']
results['source_sha256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
import planning.nodes.unicycle_planner_node as planner_module
import planning.core.belief_correction as correction_module
import reliability.nodes.camera_manager_node as manager_module
results['resolved_imports']={m.__name__:str(Path(m.__file__).resolve()) for m in (planner_module,correction_module,manager_module)}
print(json.dumps(results,indent=2,allow_nan=False))
