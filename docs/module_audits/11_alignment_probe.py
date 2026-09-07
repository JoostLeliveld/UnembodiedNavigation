#!/usr/bin/env python3
"""Audit 11: reproducible synthetic evidence, never a drive-performance analysis.

Runs real consumer functions. Assertions distinguish valid controls from reproduced
defects; a successful probe run does NOT mean the production contract is satisfied.
All generated logs/output belong to the dedicated audit directory below.
"""
from __future__ import annotations

import contextlib
import copy
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/audit11_mpl")
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "experiments/runtime_integrity/alignment_reporting_20260906"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "experiments/icra_commissioning"))
import field_driving as FD
import replay as CR
import aligned as A
import network_navigation_analysis as NAV
import network_replay as NR
import thesis_evidence as TE
from study import score as distribution_score


def module(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


S = module("score", "experiments/fusion_on_fixed_routes/score.py")
FR = module("audit11_old_replay", "experiments/fusion_on_fixed_routes/replay.py")
MON = module("audit11_monitor", "scripts/visibility_comparison/monitor_campaign.py")
M = module("audit11_metrics", "scripts/shared/metrics.py")
sys.path.insert(0, str(REPO / "experiments/fusion_on_fixed_routes/story"))
STORY = module("audit11_fusion_examples", "experiments/fusion_on_fixed_routes/story/fusion_examples.py")
CMP = module("audit11_compare", "experiments/fusion_on_fixed_routes/compare.py")

SOURCES = [
    "experiments/fusion_on_fixed_routes/aligned.py",
    "experiments/fusion_on_fixed_routes/replay.py",
    "experiments/fusion_on_fixed_routes/score.py",
    "experiments/fusion_on_fixed_routes/compare.py",
    "experiments/fusion_on_fixed_routes/story/fusion_examples.py",
    "experiments/icra_commissioning/replay.py",
    "experiments/icra_commissioning/field_driving.py",
    "experiments/icra_commissioning/network_replay.py",
    "experiments/icra_commissioning/network_navigation_analysis.py",
    "experiments/icra_commissioning/network_runtime_report.py",
    "experiments/icra_commissioning/thesis_evidence.py",
    "experiments/icra_commissioning/study.py",
    "experiments/icra_commissioning/model.py",
    "scripts/shared/metrics.py",
    "scripts/visibility_comparison/monitor_campaign.py",
    "tests/experiments/test_fusion_study_alignment.py",
    "tests/test_network_replay.py",
    "src/experiments/experiments/nodes/experiment_logger.py",
    "src/planning/planning/core/dynamics.py",
    "docs/localization_metrics.md",
    "docs/localization_metrics_registry.json",
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


before = {p: digest(REPO / p) for p in SOURCES}
snapshots = {}
for rel, expected in before.items():
    data = (REPO / rel).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise RuntimeError(f"Source changed before snapshot: {rel}")
    snapshot = OUT / "source_snapshot" / expected / Path(rel).name
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if snapshot.exists() and snapshot.read_bytes() != data:
        raise RuntimeError(f"Conflicting audit source snapshot: {snapshot}")
    snapshot.write_bytes(data)
    snapshots[rel] = str(snapshot.relative_to(REPO))
cases = []


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def record(name, expected, observed, classification="confirmed defect"):
    cases.append(dict(case=name, classification=classification, expected=expected, observed=clean(observed)))


def call(fn):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = fn()
        return dict(returned=True, result=result)
    except (Exception, SystemExit) as exc:
        return dict(returned=False, exception=type(exc).__name__, message=str(exc))


def writecsv(path, data, columns=None):
    columns = columns or list(dict.fromkeys(k for row in data for k in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(data)


def tick(t, error=0., *, belief_stamp=None, truth_x=None):
    bs = t if belief_stamp is None else belief_stamp
    ss = max([x for x in (.4, .8, 1.2) if x <= t], default=0.)
    return dict(stamp=t, gt_available=1, gt_stamp=t, gt_x=t if truth_x is None else truth_x,
        gt_y=0, gt_yaw=0, planner_belief_stamp=bs, planner_belief_x=bs+error,
        planner_belief_y=0, planner_belief_yaw=0, planner_cov_x=.01,
        planner_cov_xy=0, planner_cov_y=.01, planner_belief_cov_x_theta=0,
        planner_belief_cov_y_theta=0, planner_belief_cov_theta_theta=.01,
        state_available=int(t >= .4), state_stamp=ss, state_x=ss+.01, state_y=0,
        odom_map_x=t, odom_map_y=0, odom_map_yaw=0, odom_v=1,
        odom_noisy_stamp=t, odom_noisy_v=1, odom_noisy_w=0, odom_map_gt_drift_m=0,
        truth_x=t+10, truth_y=0)


def observation(batch, cap, camera="A", *, error=.02, fused_stamp=None):
    fs = cap if fused_stamp is None else fused_stamp
    return dict(stamp=cap+.15, source_batch_id=batch, camera=camera, used=1,
        obs_stamp=cap, obs_x=cap+error, obs_y=0, obs_cov_xx=.01, obs_cov_xy=0, obs_cov_yy=.01,
        fused_stamp=fs, fused_x=fs+.01, fused_y=0, fused_cov_xx=.005,
        fused_cov_xy=0, fused_cov_yy=.005, n_candidates=2, n_used=2,
        common_capture_stamp=cap, conf=.9, bbox_h_px=80, bbox_w_px=100, range_m=6,
        obs_repeat=0, raw_obs_x=cap, raw_obs_y=0, gt_x=cap+.15, gt_y=0,
        gt_x_at_obs=cap, gt_y_at_obs=0)


def assimilation(batch, cap, *, status="accepted", reason="accepted", apply=None, accepted=None):
    return dict(source_batch_id=batch, correction_stamp=cap, apply_stamp=cap+.16 if apply is None else apply,
        belief_stamp_after=cap, status=status, reason=reason,
        accepted=int(status in {"accepted", "accepted_bootstrap", "reanchored"}) if accepted is None else accepted,
        nis=1)


def make_run(name, *, table=None, obs=None, ass=None, schema=7, meta=None, summary=None):
    run = OUT / "fixtures" / name
    run.mkdir(parents=True, exist_ok=True)
    table = [tick(round(i/10, 6)) for i in range(21)] if table is None else table
    obs = [observation(f"b{i}", cap, c, error=err) for i, cap in enumerate((.4,.8,1.2))
           for c,err in (("A",.02),("B",-.02))] if obs is None else obs
    ass = [assimilation(f"b{i}", cap) for i,cap in enumerate((.4,.8,1.2))] if ass is None else ass
    manifest = dict(logging_schema_version=schema, task=S.TASKS[0], seed=0, run_id=name,
        manager_fusion_rule="joint_network", manager_observation_model="hull",
        goal_termination_reference="planner_belief", process_noise_xy=.01, process_noise_theta=.02,
        use_odom_for_predict=True, odom_topic="/odom_noisy", task_start_pose=dict(x=0,y=0,yaw=0))
    manifest.update(meta or {})
    summ = dict(completed=True, valid_run=True, completion_reason="stuck", first_cmd_stamp=0,
        stop_stamp=2, elapsed_after_first_cmd_s=2, final_goal_distance=1,
        minimum_goal_distance=1, final_goal_distance_reference="ground_truth")
    summ.update(summary or {})
    save(run / "run_manifest.json", manifest)
    save(run / "run_summary.json", summ)
    writecsv(run / "experiment.csv", table)
    writecsv(run / "fusion_observations.csv", obs, list(observation("",0)))
    writecsv(run / "correction_assimilations.csv", ass, list(assimilation("",0)))
    return run


def selected(run, arm="F4", task=None):
    task = task or S.TASKS[0]
    path = OUT / "synthetic_frozen_runs.json"
    save(path, dict(schema_version=1,seeds=[0],runs={task:{arm:[str(run)]}}))
    with patch.object(S,"FROZEN_RUNS",path):
        return S._selected_runs(arm,task)


# Independent oracle: x_GT(t)=t metres, camera residual +/-2 cm, fused residual 1 cm.
base = make_run("base")
assert np.allclose([r["error_cm"] for r in A.readings(base)], 2.)
assert np.allclose([r["error_cm"] for r in A.fused_answers(base)], 1.)
assert np.allclose(A.aligned_error_cm(base,"belief")["aligned_cm"], 0.)
record("separate_own_time_errors", "6 camera readings at 2 cm; 3 fused events at 1 cm; 21 beliefs at 0 cm, unrelated wheel x+10 ignored",
       dict(camera_n=6,camera_error_cm=2,fused_n=3,fused_error_cm=1,belief_n=21,belief_error_cm=0), "verified control")

truth = A.truth_series(base)
outside = truth.at([-1,3,float("nan")])[0]
assert np.isnan(outside).all()
record("out_of_range", "NaN references outside truth support", outside, "verified fail-closed reference")

# Accounting is independent of whether a truth-paired score can be made.
for name, mutate in [
    ("missing_assimilation", lambda rows: rows[:-1]),
    ("extra_assimilation", lambda rows: rows+[assimilation("extra",1.4)]),
    ("unclassifiable_status", lambda rows: [dict(rows[0],status="teleported"),*rows[1:]]),
    ("reasonless_rejection", lambda rows: [dict(rows[0],status="rejected",accepted=0,reason=""),*rows[1:]]),
    ("accepted_flag_conflict", lambda rows: [dict(rows[0],accepted=0),*rows[1:]]),
    ("stamp_mismatch", lambda rows: [dict(rows[0],correction_stamp=1.9,belief_stamp_after=1.9),*rows[1:]]),
]:
    data = [assimilation(f"b{i}",cap) for i,cap in enumerate((.4,.8,1.2))]
    run = make_run(name, ass=mutate(data))
    a = call(lambda: A.belief_at_fusion_events(run))
    s = call(lambda: selected(run))
    assert a["returned"]
    record(name, "Reject inconsistent evidence before returning event scores",
           dict(aligned_event_ids=[r["source_batch_id"] for r in a["result"]],fixed_route_selection=s))

duplicate = make_run("duplicate_assimilation",ass=[assimilation("b0",.4),assimilation("b0",.4)])
missingid = make_run("missing_assimilation_id",ass=[assimilation("",.4)])
for name,run in [("duplicate_assimilation_id_refused",duplicate),("blank_assimilation_id_refused",missingid)]:
    r = call(lambda: A.assimilations(run))
    assert not r["returned"]
    record(name,"Reject",r,"verified fail-closed control")

reasoned = make_run("reasoned_refusals",ass=[assimilation("b0",.4),
    assimilation("b1",.8,status="rejected",reason="nis"),
    assimilation("b2",1.2,status="dropped",reason="replay_gap")])
assert len(selected(reasoned)) == 1
assert [r["source_batch_id"] for r in A.belief_at_fusion_events(reasoned)] == ["b0"]
record("reasoned_refusals", "Run remains valid; only accepted b0 enters event-belief population",["b0"],"verified repair")

old = make_run("schema3_no_assimilation",schema=3,ass=[])
r = A.belief_at_fusion_events(old)
assert len(r)==3 and {x["assimilation_status"] for x in r}=={"legacy_timestamp_inference"}
assert not call(lambda:selected(old))["returned"]
record("schema3_boundary","No confirmed post-update population without explicit outcomes",
       dict(aligned_inferred_events=len(r),label=r[0]["assimilation_status"],fixed_route_selection="refused"),
       "legacy compatibility limit with guarded fixed-route entry")
empty4=make_run("schema4_no_assimilation",schema=4,ass=[])
assert not call(lambda:A.belief_at_fusion_events(empty4))["returned"]
record("schema4_empty_ledger", "Reject event-belief inference",call(lambda:A.belief_at_fusion_events(empty4)),"verified fail-closed control")

# Same observation identity must not change with order or cross-event timestamp reuse.
o = observation("physical1",.4)
changed = dict(o,obs_x=1.4,fused_x=1.4)
for name,oo in [("conflicting_duplicate_first",[o,changed]),("conflicting_duplicate_reverse",[changed,o])]:
    run=make_run(name,obs=oo,ass=[assimilation("physical1",.4)])
    value = A.readings(run)[0]["error_cm"]
    record(name,"Reject conflicting payloads for same (batch,camera)",dict(selected_error_cm=value,fused_error_cm=A.fused_answers(run)[0]["error_cm"]))
run=make_run("distinct_batches_equal_capture",obs=[o,dict(o,source_batch_id="physical2",obs_x=.44)],
             ass=[assimilation("physical1",.4),assimilation("physical2",.4)])
assert len(A.readings(run))==1 and len(A.fused_answers(run))==2
record("distinct_batches_equal_capture","Preserve two declared identities, or refuse ambiguous physical identity; never silently choose one",
       dict(reading_ids=[r["source_batch_id"] for r in A.readings(run)],fused_ids=[r["source_batch_id"] for r in A.fused_answers(run)]))
run=make_run("same_batch_changed_capture",obs=[o,dict(o,obs_stamp=.45,obs_x=.47)],ass=[assimilation("physical1",.4)])
assert len(A.readings(run))==2
record("same_batch_changed_capture","Reject changed timestamp under same (batch,camera)",dict(reading_count=2,fused_count=len(A.fused_answers(run))))
run=make_run("missing_source_ids",obs=[observation("",.4),observation("",.8)],ass=[])
assert len(A.fused_answers(run))==2
record("missing_observation_ids","Schema 7 correction events require nonempty identities",dict(fused_ids=[r["source_batch_id"] for r in A.fused_answers(run)]))
for name,ts,expected in [("reordered_landed",[1,2,1,2],[True,True,False,False]),
                         ("initial_nan_landed",[float("nan"),1,1,2],[False,True,False,True])]:
    actual=A.landed_mask(ts).tolist()
    assert actual!=expected
    record(name,dict(first_unique_mask=expected),dict(stamps=ts,actual_mask=actual))

# A post-update claim requires an event posterior/revision, not capture-time proximity.
tbl=[tick(t,error=e) for t,e in [(0,0),(.9,0),(1.1,.1),(1.2,.2),(1.4,.4),(2,0)]]
run=make_run("before_apply_is_post_correction",table=tbl,obs=[observation("late",1)],
             ass=[assimilation("late",1,apply=1.3)])
ev=A.belief_at_fusion_events(run)[0]
assert np.isclose(ev["error_cm"],10)
record("before_apply_is_post_correction","No exact event posterior is logged. First public candidate after apply is at 1.4 (40 cm), not the 1.1 sample (10 cm) before apply=1.3",
       ev)
run=make_run("multiple_updates_between_ticks",table=[tick(0),tick(1.1,.3),tick(2)],
    obs=[observation("e1",1),observation("e2",1.05)],
    ass=[assimilation("e1",1,apply=1.01),assimilation("e2",1.05,apply=1.06)])
ev=A.belief_at_fusion_events(run)
assert len(ev)==2 and np.allclose([r["error_cm"] for r in ev],30)
record("multiple_updates_between_ticks","Cannot recover two distinct posteriors from one public belief; explicit unscoreable event count required",ev)
run=make_run("shutdown_before_final_belief",table=[tick(0),tick(.9),tick(1)],
    obs=[observation("final",1.1)],ass=[assimilation("final",1.1,apply=1.15)],summary=dict(stop_stamp=1.2))
ev=A.belief_at_fusion_events(run)
failure=call(lambda:selected(run))
assert ev==[] and not failure["returned"] and "extra" in failure["message"]
record("shutdown_before_final_belief","ID sets match: valid accounted event, unscoreable truth/posterior tail. Report incompleteness separately",
       dict(raw_obs_ids=[o["source_batch_id"] for o in A.observations(run)],assimilation_ids=[a["source_batch_id"] for a in A.assimilations(run)],event_scores=ev,selection=failure))

# Truth time must not silently erase an epoch or conflicting sample.
for name,table,query in [
    ("truth_tie",[tick(0),tick(1),tick(1,truth_x=9),tick(2)],1.),
    ("truth_clock_reset",[tick(0),tick(1),tick(0,truth_x=100),tick(1,truth_x=101)],.5),
    ("truth_large_gap",[tick(0),tick(10)],5.),
]:
    run=make_run(name,table=table)
    tt=A.truth_series(run)
    record(name,"Reject conflicting time/epoch; declare interpolation support limit for gap",dict(input_t=[r["gt_stamp"] for r in table],sorted_t=tt.t,query=query,returned_x=tt.at([query])[0]))
tt=A.TruthSeries([0,1,2,3],[0,1,2,3],[0]*4,[0,float("nan"),.1,.2],"synthetic")
assert np.isnan(tt.yaw_at([2,3])).all()
record("missing_heading_poisons_following","Known yaw at t=2,3 is .1,.2 rad; handle finite segments independently",tt.yaw_at([2,3]))
table=[tick(0),tick(1,error=.5,belief_stamp=.5,truth_x=0)]
table[1].update(gt_x_at_belief_stamp=1,gt_y_at_belief_stamp=0,belief_error_gt_m=0)
run=make_run("buffered_truth_columns_ignored",table=table)
got=A.aligned_error_cm(run,"belief")["aligned_cm"][1]
assert np.isclose(got,100)
record("buffered_truth_columns_ignored","Synthetic triangular truth x(0)=0,x(.5)=1,x(1)=0: logged buffered reference at belief .5 establishes zero error, and docstring promises preference",
       dict(buffered_error_cm=0,loader_error_cm=got,coarse_truth_endpoint_x=[0,0]))
tt=A.TruthSeries([0,1],[0,0],[0,0],np.deg2rad([179,-179]),"synthetic")
assert np.isclose(np.rad2deg(tt.yaw_at([.5])[0]),180)
record("heading_wrap_control","179 to -179 deg crosses 180, not zero",dict(midpoint_deg=180),"verified control")

# Gap windows and logger weighting have independently calculable answers.
run=make_run("blind_boundaries",table=[tick(t) for t in [0,4,5,10]],
    obs=[observation("one",4),observation("two",5)],ass=[assimilation("one",4),assimilation("two",5)],summary=dict(stop_stamp=10))
rows=A.rows(run)
for row,st in zip(rows,[float("nan"),4,5,5]):
    row.update(state_stamp=str(st),state_available=str(int(math.isfinite(st))))
counts=A.corrections(run,rows)
assert counts["longest_gap_s"]==1
record("blind_boundaries","Gaps on [0,10] with accepted updates 4,5 are [4,1,5]; longest=5 s",counts)
quiet=make_run("belief_weight_quiet",table=[tick(0),tick(1,1)])
busy=make_run("belief_weight_held",table=[tick(0)]+[tick(t,1,belief_stamp=1) for t in [1,1.1,1.2,1.3,1.4]])
qa=S._score_one(quiet,"F4");ba=S._score_one(busy,"F4")
assert qa["belief_error_cm"]["median"]==50 and ba["belief_error_cm"]["median"]==100
record("held_belief_weights_score","Two unique beliefs [0,100] cm => median=50 cm regardless of repetitions",
       dict(quiet=qa["belief_error_cm"],held=ba["belief_error_cm"]))

# SPD is stronger than determinant > 0; solve and slogdet alone do not validate it.
bad=-np.eye(2)
n=A.nees([[1,0]],[bad])[0]
assert n==-1 and n<A.CHI2_95_2D
other=A.nees([[1,0]],[[[1,0],[10,1]]])[0]
record("negative_covariance_nees","Reject negative-definite covariance, never negative NEES/inside ellipse",
       dict(nees=n,covered=bool(n<=A.CHI2_95_2D),asymmetric_nees=other))
sc=distribution_score(np.array([[1.,0.]]),np.array([bad]),["one_run"])
assert sc["coverage"]["0.95"]==1.
record("commissioning_covariance_validation","Reject invalid covariance before score",sc)

# Fixed-route replay uses row time, rounds captures onto ticks and drops distinct views.
oo=[dict(camera="A",cap=t,xy=np.array([t,0.]),cov=np.eye(2),used=True,range_m=6) for t in [.96,1.04]]
steps=[dict(t=t,odom=np.array([t,0.]),yaw=0.,v=1.,gt=np.array([t,0.])) for t in [0.,1.,2.]]
attached=FR.bind(steps,oo,{"A"})
assert len(attached[1])==1
record("legacy_replay_tick_collision","Two unique camera captures .96,1.04 must remain two updates at their own times, or replay must refuse this resolution",
       dict(selected_captures=[r["cap"] for r in attached[1]]))
oldscore=FR.summarise([dict(t=t,gt=np.array([t,0.])) for t in [0,4,5,10]],
    dict(corrected_at=np.array([4,5]),errors=np.zeros(4),nees=np.array([1,2,3,4]),claims=np.ones(4)))
assert oldscore["worst_blind_s"]==1
record("legacy_replay_gap_and_nees_label","Longest gap 5 s; median raw NEES 2.5, mean raw NEES 2.5",
       dict(longest_gap=oldscore["worst_blind_s"],reported_nees=oldscore["nees"]))
run=make_run("story_wrong_fused_time",obs=[dict(observation("b",.4,fused_stamp=.6),fused_x=.6)],ass=[assimilation("b",.6)])
with patch.object(STORY,"showcase_run",lambda *args:run):
    _,moments=STORY.load("F4")
assert np.isclose(moments[0]["error_cm"],20) and np.isclose(A.fused_answers(run)[0]["error_cm"],0)
record("story_uses_capture_truth_for_fused","Perfect fused x=.6 at fused_stamp=.6 => 0 cm; camera stamp .4 cannot be its reference",
       dict(story_fused_error_cm=moments[0]["error_cm"],aligned_fused_error_cm=0))

# Capture-time replay is deliberately different from recorded arrival-time gating.
truth=A.TruthSeries([0,1],[0,1],[0,0],[0,0],"synthetic")
m=dict(task_start_pose=dict(x=0,y=0,yaw=0))
reading=dict(camera="camera_A",t=.5,original_z=np.array([.5,0]),original_R=np.eye(2)*.01,batch="later")
_,rr,ii=CR.run_filter(m,truth,{0.:np.array([1.,0.]),1.:np.array([1.,0.])},[reading],{},"recorded",["camera_A"])
assert ii[0]["t"]==.5
record("capture_time_policy","Capture-time replay applies at .5 independent of unavailable arrival/status; this is intentional, not live performance",dict(update_times=[e["t"] for e in ii]),"intentional policy limit")
# At t=0 the two scalar information contributions are independently checkable:
# precision=1/.0025+1/.01+1/.01=600; information mean=(.2+.4)/.01=60.
same=[dict(camera=c,t=0.,original_z=np.array([z,0.]),original_R=np.eye(2)*.01,batch="simultaneous")
      for c,z in [("camera_A",.2),("camera_B",.4)]]
answers=[]
for observations in [same,list(reversed(same))]:
    ss,rr,ii=CR.run_filter(m,truth,{0.:np.zeros(2),1.:np.zeros(2)},observations,{},"recorded",["camera_A","camera_B"])
    assert np.isclose(rr[0]["state"][0],.1) and np.allclose(rr[0]["P"][:2,:2],np.eye(2)/600)
    answers.append(dict(x=rr[0]["state"][0],Pxy=rr[0]["P"][:2,:2],updates=ss["updates"]))
record("simultaneous_reordered_camera_updates","Both camera identities counted once; x=.1 m, Pxy=I/600 m² regardless of order",answers,"verified control")
duplicate_result=call(lambda:CR.run_filter(m,truth,{0.:np.zeros(2),1.:np.zeros(2)},[same[0],same[0]],{},"recorded",["camera_A"]))
assert not duplicate_result["returned"]
record("duplicate_replay_update_refused","Reject duplicate camera/time update before applying it again",duplicate_result,"verified fail-closed control")
# The no-camera arm must not change simply because prediction-only grid nodes differ.
paths=[]
for grid in [None,[.25,.75]]:
    ss,rr,_=CR.run_filter(m,truth,{0.:np.array([1.,1.]),1.:np.array([1.,1.])},[],{},"recorded",[],prediction_times=grid)
    paths.append(dict(final_P=rr[-1]["P"],score=ss))
assert not np.allclose(paths[0]["final_P"],paths[1]["final_P"])
record("prediction_grid_changes_covariance","Within an arm comparison use one identical propagation grid; old callers omit it",paths,
       "confirmed caller-control gap; network_replay repaired")
_,rr,_=CR.run_filter(m,A.TruthSeries([0,10],[0,10],[0,0],[0,0],"synthetic"),
    {0.:np.array([1.,0.]),10.:np.array([1.,0.])},[],{},"recorded",[])
assert np.isclose(rr[-1]["state"][0],10)
record("unbounded_missing_motion","Explicitly mark 10 s interval unsupported under declared motion-history limit; no bound is configured in replay",
       dict(final_x=rr[-1]["state"][0],interval_s=10),"unbounded ZOH policy limit")

# Selection/provenance and aggregate metadata.
assert len(selected(base))==1
record("missing_provenance_accepted","Required source/artifact hashes must be present, not all None",dict(fixed_route_selected=True,manifest_has_no_provenance_hashes=True))
f1=make_run("different_arm_source",meta=dict(manager_fusion_rule="best_single",git_sha="source_B",seed=0))
assert len(selected(f1,"F1"))==1 and len(selected(base,"F4"))==1
record("cross_arm_source_identity","Matched arms must share controlled source/config/model identities",dict(F1="source_B",F4=None,both_selected=True))
first=S._score_one(base,"F4");second=copy.deepcopy(first)
second.update(completion="collision",duration_s=100)
second["driving"]["ground_truth_path_m"]=100
second["corrections"]["longest_gap_s"]=99
second["run"]="synthetic-second-run"
with patch.object(S,"_selected_runs",lambda *args:["one","two"]),patch.object(S,"_score_one",side_effect=[first,second]):
    aggregate=S.score("F4")
assert aggregate["duration_s"]==2 and aggregate["completion"]=="stuck"
record("first_run_metadata_in_aggregate","Aggregate two-run duration/path/gaps/outcomes or identify those as one named run; disclose per-metric valid n",
       dict(n_runs=aggregate["n_runs"],completion=aggregate["completion"],completion_counts=aggregate["completion_counts"],duration=aggregate["duration_s"],path=aggregate["driving"]["ground_truth_path_m"],gaps=aggregate["corrections"],aggregation=aggregate["aggregation"]))
second["belief_error_cm"]["median"]=None
with patch.object(S,"_selected_runs",lambda *args:["one","two"]),patch.object(S,"_score_one",side_effect=[first,second]):
    aggregate=S.score("F4")
record("unscoreable_metric_denominator","Keep two attempts and disclose median based on only one run",dict(n_runs=aggregate["n_runs"],median=aggregate["belief_error_cm"]["median"],per_run=aggregate["belief_error_cm"]["per_run"],aggregation=aggregate["aggregation"]))

# Exercise actual CLI parsing with computation replaced so no study artifacts are written.
seen=[]
def fail_score(arm,task):
    seen.append([arm,task]);raise SystemExit("synthetic: no score written")
with patch.object(S,"score",fail_score),patch.object(sys,"argv",["score.py","--taks=fusion_overlap_rich","F4"]):
    cli=call(S.main)
assert cli["returned"] and len(seen)==4
record("score_unknown_cli_option","Unknown --taks must exit nonzero before selecting any route",dict(calls=seen,result=cli))
record("compare_cli_typo_control","Unknown options refused before selection",
       (lambda:None)() or {})
with patch.object(sys,"argv",["compare.py","--taks=fusion_overlap_rich"]):
    cli=call(CMP.main)
assert not cli["returned"]
cases[-1].update(observed=cli,classification="verified repair")
parsed=[]
def reached_task(task):parsed.append(task);raise RuntimeError("stop after argument parsing")
for argv in [["compare.py","--task=fusion_overlap_rich"],["compare.py","--task","fusion_overlap_rich"]]:
    with patch.multiple(CMP,STORY_ROOT=OUT/"compare_cli",per_arm=reached_task),patch.object(sys,"argv",argv):
        call(CMP.main)
assert parsed==["fusion_overlap_rich"]*2
record("compare_task_spellings","Both documented task spellings select the same route",parsed,"verified repair")

# Campaign monitor: exact duplicate attempts and absent summaries, no real campaign.
root=OUT/"monitor_fixture"
for task,seed,rid in [("fusion_overlap_rich",110,"attempt1"),("fusion_overlap_rich",110,"attempt2"),
                       ("fusion_network_traverse",110,"attempt1")]:
    run=root/task/"N1"/f"seed{seed}"/rid
    save(run/"run_summary.json",dict(completed=True,completion_reason="stuck",valid_run=True,
        mean_belief_error_gt_after_first_cmd_m=.032,mean_belief_error_odom_after_first_cmd_m=.297,
        correction_dropped_fraction=.1,longest_correction_gap_s=2))
(root/"fusion_overlap_rich"/"N1"/"seed111"/"no_summary").mkdir(parents=True,exist_ok=True)
with patch.object(MON,"_offline_routes",lambda:{}):
    rows,_=MON.collect(root)
assert len(rows)==3 and {r["task"] for r in rows}=={"fusion"}
assert all(r["gt_err"]==.032 for r in rows)
record("monitor_task_and_attempt_identity","2 tasks, 2 copies of seed110 distinguished from independent seeds, seed111 missing summary visible",
       dict(task_ids=[r["task"] for r in rows],seeds=[r["seed"] for r in rows],rows=len(rows),summaryless_visible=False))
record("monitor_ground_truth_field","Use .032 m GT summary, never .297 m wheel disagreement",dict(gt_err_m=.032),"verified repair")

# Actual freeze accepts an extra campaign key and does not bind manifest identity.
navroot=OUT/"navigation_fixture";navroot.mkdir(exist_ok=True)
config=navroot/"config.yaml"
config.write_text("tasks:\n  requested_route:\n    conditions: [P0]\n    seeds: [210]\n")
save(navroot/"protocol.json",{})
ev=dict(finished_at="synthetic",run_dir=str(base),outcome="stuck",completion_reason="stuck")
save(navroot/"campaign_log.json",{"requested_route__P0__seed210":ev,"unexpected__P2__seed999":ev})
with patch.multiple(NAV,CONFIG=config,CAMPAIGN=navroot):
    entries=NAV.freeze(navroot)
assert len(entries)==1 and entries[0]["task"]=="requested_route"
record("navigation_freeze_extra_and_wrong_identity","Fail extra ledger key and manifest task/seed != declared entry",
       dict(entries=len(entries),declared_task=entries[0]["task"],actual_task=S.TASKS[0],declared_seed=210,actual_seed=0,extra_key_retained=False))

# Isolate consumer metadata validation from actual camera projection/model fitting.
fieldrun=make_run("schema3_opportunity_consumer",schema=3)
opportunities=[dict(valid_contract=True,duplicate=False,observation=dict(timestamp_s=t,
    camera_id=c,source_batch_id=f"b{i}",detection_valid=False))
    for i,t in enumerate((.4,.8,1.2)) for c in FD.CAMERAS]
(fieldrun/"camera_opportunities.jsonl").write_text("".join(json.dumps(r)+"\n" for r in opportunities))
save(navroot/"capture_manifest.json",{})
badentry=dict(run=str(fieldrun),files={},task="wrong_route",seed=999,field_sha256="wrong_model")
with patch.multiple(FD,CAPTURE=navroot,LearnedBoxCorrection=lambda p:object(),camera_models=lambda m:{}):
    loaded=FD.load_run(badentry)
assert loaded[0]["logging_schema_version"]==3
record("field_loader_unverified_entry","Require schema >=7, all required file hashes, exact task/seed and frozen model identity",
       dict(schema=3,declared_task="wrong_route",actual_task=loaded[0]["task"],hashes_checked=0,loaded_batches=len(loaded[5])),
       "confirmed consumer guard gap; upstream thesis selector rejects schema3")

# analyze_run itself still accepts mismatched manifest task/seed after its other guards.
navrun=make_run("navigation_wrong_task")
save(navroot/"network_planner/reference_calibration.json",{})
save(navroot/"field.npz",{})
config.write_text("tasks:\n  requested_route:\n    conditions: [P0]\n    seeds: [210]\nconditions:\n  P0:\n    camera_network_artifact_path: "+str(navroot/"field.npz")+"\n")
manifest=json.loads((navrun/"run_manifest.json").read_text())
manifest.update(campaign_config_sha256=digest(config),manager_covariance_profile="commissioned_reference_r",
    manager_commissioned_world_covariance_sha256=digest(navroot/"network_planner/reference_calibration.json"),
    camera_network_artifact_sha256=digest(navroot/"field.npz"))
save(navrun/"run_manifest.json",manifest)
(navrun/"camera_opportunities.jsonl").write_text("")
save(navrun/"global_plan_meta.json",dict(selected_source="synthetic"))
naventry=dict(arm="P0",task="requested_route",seed=210,run=str(navrun),event=dict(outcome="stuck"),
    files={name:digest(navrun/name) for name in ["run_manifest.json","run_summary.json","experiment.csv",
           "fusion_observations.csv","correction_assimilations.csv","camera_opportunities.jsonl","global_plan_meta.json"]})
with patch.multiple(NAV,CONFIG=config,OUT=navroot,audit_live_camera_model=lambda *a:{"synthetic_stub":True},sensor_diagnostics=lambda *a:{}):
    analyzed,_=NAV.analyze_run(naventry)
assert analyzed["status"]=="stuck"
record("navigation_analyze_wrong_identity","Reject task/seed mismatch before computing scores; camera-model calculation stubbed only",
       dict(declared_task=naventry["task"],manifest_task=manifest["task"],declared_seed=210,manifest_seed=0,status=analyzed["status"]))
customconfig=navroot/"custom_config.yaml"
customconfig.write_text("tasks:\n  fusion_overlap_rich:\n    conditions: [P0]\n    seeds: [999]\n")
protocolout=OUT/"custom_protocol"
# Protocol files deliberately forbid overwrite; preserve repeatability by invoking
# only the JSON writer on subsequent runs and comparing identical generated payloads.
captured=[]
original_exists=Path.exists
with patch.multiple(NAV,CONFIG=customconfig,OUT=navroot,ARMS=("P0",),writejson=lambda p,obj:captured.append(obj)), \
     patch.object(Path,"exists",lambda p:False if p==protocolout/"protocol.json" else original_exists(p)):
    NAV.protocol(protocolout)
assert captured[0]["task"]=="fusion_network_traverse" and captured[0]["seeds"]==[210]
save(protocolout/"generated_protocol.json",captured[0])
record("navigation_protocol_hardcoded_task_seed","--config requesting fusion_overlap_rich/seed999 must freeze that identity or reject unsupported task",
       dict(config_task="fusion_overlap_rich",config_seed=999,protocol_task=captured[0]["task"],protocol_seeds=captured[0]["seeds"]))

# Network entry checks list length, not a unique expected-key product. Stop before any
# numerical replay; capture the just-constructed protocol on an isolated output root.
selection=OUT/"duplicate_network_selection.json"
save(selection,dict(status="complete_pilot_diagnostic",pending=[],invalid=[],runs=[badentry]*6))
registry=json.loads((REPO/"docs/localization_metrics_registry.json").read_text())
registry["thesis_commissioning_pilot"]["selection"]=str(selection.relative_to(REPO))
read_text=Path.read_text
class ReachedLoad(Exception):pass
def fake_read(path,*args,**kwargs):
    if path==REPO/"docs/localization_metrics_registry.json":return json.dumps(registry)
    return read_text(path,*args,**kwargs)
def reached_load(entry):raise ReachedLoad("duplicate selection reached numerical loader")
with patch.object(Path,"read_text",fake_read),patch.object(NR.joblib,"load",lambda p:{}),patch.object(FD,"load_run",reached_load):
    duplicate_result=call(lambda:NR.main(selection,OUT/"duplicate_network_output"))
assert duplicate_result.get("exception")=="ReachedLoad"
record("network_duplicate_selected_runs","Reject six repeated entries as an invalid 2-route x 3-seed selection before writing protocol/loading",duplicate_result,
       "confirmed guard gap; registry lookup mocked to bind synthetic fixture")

# A cache must be checked against the requested entry/model before returning.
cache=OUT/"stale_cache";save(cache/"same_key/results.json",dict(run="old_run",replay="stale",field_sha256="old"))
cached=FD.analyze(dict(key="same_key",run="missing_changed_run",files={"experiment.csv":"changed"}),cache)
assert cached["run"]=="old_run"
record("cached_results_skip_validation","Validate requested frozen inputs and model/source hashes before reusing results",cached)

# Utility metric controls and nonfinite input behavior; no external datasets.
assert M.auroc([1,0],[.5,.5])==.5 and M.auprc([1,0],[.5,.5])==.5
record("tied_binary_metrics","Two tied scores with one positive => ROC AUC and AP both .5",dict(auc=.5,ap=.5),"verified control")
code="import sys;sys.path.insert(0,'scripts/shared');import metrics;print('ready',flush=True);metrics.auprc([1,0],[.5,float('nan')])"
try:
    child=subprocess.run([sys.executable,"-c",code],cwd=REPO,capture_output=True,text=True,timeout=4)
    outcome=dict(returncode=child.returncode,stdout=child.stdout,stderr=child.stderr)
except subprocess.TimeoutExpired as exc:
    outcome=dict(timeout=True,stdout=(exc.stdout or b"").decode() if isinstance(exc.stdout,bytes) else exc.stdout)
assert outcome.get("timeout") and "ready" in outcome["stdout"]
record("auprc_nan_does_not_advance","Reject/nonfinite-mask explicitly, terminate with finite population count",outcome,"confirmed optional utility bug; no current caller found")
record("ece_invalid_predictions_look_calibrated","Reject invalid probabilities / report excluded n",dict(ece=M.ece([1,0],[float("nan"),2.])))

after={p:digest(REPO/p) for p in SOURCES}
payload=dict(kind="synthetic_software_audit_not_drive_results",source_sha256=before,
    source_snapshot=snapshots,
    resolved_modules={k:str(getattr(v,"__file__","")) for k,v in dict(aligned=A,commissioning_replay=CR,
        fixed_route_replay=FR,field_driving=FD,score=S,compare=CMP).items()},
    changed_sources_during_probe={p:dict(before=before[p],after=after[p]) for p in SOURCES if before[p]!=after[p]},
    cases=cases)
save(OUT/"results.json",payload)
print(f"{len(cases)} synthetic cases recorded; source changes during probe: {len(payload['changed_sources_during_probe'])}")
print(OUT/"results.json")
