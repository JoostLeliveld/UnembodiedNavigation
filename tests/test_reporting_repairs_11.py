"""Audit 11 regression checks for selected-run reporting and idealized replay."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'experiments/icra_commissioning'))
import field_driving as FD
import network_replay as NR
import network_navigation_analysis as NAV
import study
R=sys.modules[FD.run_filter.__module__]
A=R.aligned


def module(name, path):
    sys.path.insert(0,str((REPO/path).parent))
    spec=importlib.util.spec_from_file_location(name,REPO/path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value

S=module('audit11_score','experiments/fusion_on_fixed_routes/score.py')
M=module('audit11_metrics','scripts/shared/metrics.py')
MON=module('audit11_monitor','scripts/visibility_comparison/monitor_campaign.py')
F=module('audit11_fixture','tests/experiments/test_alignment_repairs_11.py')


def test_aggregation_retains_failures_and_unscoreable_denominators():
    def report(run,error,duration):
        return dict(arm='F4',task='route',logging_schema_version=7,truth_clock='synthetic',
            reference_max_gap_s=None,reference_method='exact_logged_gt',run=run,
            completion='collision' if run=='a' else 'timeout',duration_s=duration,
            belief_error_cm=dict(median=error,n_samples=2),honesty={},correction_error_cm={},
            corrections=dict(longest_gap_s=duration),accounting={},driving=dict(ground_truth_path_m=duration),
            odometry={},fusion=dict(logged=False))
    result=S.aggregate_reports([report('a',2,10),report('b',8,20)])
    assert result['belief_error_cm']['median']==5
    assert result['duration_s']==15 and result['driving']['ground_truth_path_m']==15
    assert result['completion_counts']=={'collision':1,'timeout':1}
    result=S.aggregate_reports([report('a',2,10),report('b',None,20)])
    assert result['belief_error_cm']['median'] is None
    assert result['belief_error_cm']['median_n_runs']==1 and result['n_runs']==2


def test_score_cli_unknown_option_cannot_select_default_task(monkeypatch):
    called=[];monkeypatch.setattr(S,'score',lambda *a,**kw:called.append(a))
    with pytest.raises(SystemExit) as exc:S.main(['--taks=wrong'])
    assert exc.value.code==2 and not called


@pytest.mark.parametrize('spelling',[['--task','fusion_overlap_sparse'],['--task=fusion_overlap_sparse']])
def test_score_cli_route_spellings_are_equivalent(tmp_path,monkeypatch,spelling):
    calls=[]
    def score(arm,task,**kwargs):
        calls.append((arm,task));return dict(completion_counts={'timeout':1},n_runs=1)
    monkeypatch.setattr(S,'score',score)
    assert S.main(['F4',*spelling,'--out',str(tmp_path)])==0
    assert calls==[('F4','fusion_overlap_sparse')]
    assert (tmp_path/'fusion_overlap_sparse'/S.FOLDER['F4']/'numbers.json').is_file()


def test_score_cli_failure_exits_nonzero(tmp_path,monkeypatch):
    def fail(*a,**kw):raise SystemExit('missing frozen selection')
    monkeypatch.setattr(S,'score',fail)
    assert S.main(['F4','--task=fusion_overlap_sparse','--out',str(tmp_path)])==1


def test_selected_runs_validate_raw_tail_not_truth_filtered_events(tmp_path,monkeypatch):
    run,_,_,_=F.fixture(tmp_path/'run',stamps=(.2,.4,1.1))
    manifest=json.loads((run/'run_manifest.json').read_text())
    manifest.update(task='fusion_overlap_sparse',seed=4,manager_fusion_rule='joint_network',
        manager_observation_model='hull',goal_termination_reference='planner_belief',
        campaign_config_sha256='a'*64,git_sha='a'*40,git_diff_sha256='b'*64,
        git_untracked_content_sha256='c'*64,yolo_model_sha256='d'*64,
        visibility_geometry_sha256='e'*64,collision_geometry_sha256='f'*64,
        process_noise_xy=.01,process_noise_theta=.02,use_odom_for_predict=True,odom_topic='/odom_noisy')
    (run/'run_manifest.json').write_text(json.dumps(manifest))
    summary=json.loads((run/'run_summary.json').read_text());summary.update(completed=True,valid_run=True)
    (run/'run_summary.json').write_text(json.dumps(summary))
    selection=tmp_path/'selection.json'
    files={name:study.digest(run/name) for name in ['run_manifest.json','run_summary.json','experiment.csv','fusion_observations.csv','correction_assimilations.csv']}
    selection.write_text(json.dumps(dict(schema_version=1,seeds=[4],runs={'fusion_overlap_sparse':{'F4':[dict(run=str(run),files=files)]}})))
    monkeypatch.setattr(S,'FROZEN_RUNS',selection)
    assert S._selected_runs('F4','fusion_overlap_sparse')==[run.resolve()]


def test_frozen_loader_requires_hashes_schema_and_manifest_identity(tmp_path):
    run,_,_,_=F.fixture(tmp_path/'run')
    manifest=json.loads((run/'run_manifest.json').read_text());manifest.update(task='route',seed=1)
    (run/'run_manifest.json').write_text(json.dumps(manifest))
    required=['run_manifest.json','run_summary.json','experiment.csv']
    entry=dict(run=str(run),task='route',seed=1,files={p:study.digest(run/p) for p in required})
    A.verify_frozen_entry(entry,required,minimum_schema=7)
    with pytest.raises(ValueError,match='hashes missing'):A.verify_frozen_entry(dict(entry,files={}),required)
    with pytest.raises(ValueError,match='identity mismatch'):A.verify_frozen_entry(dict(entry,seed=2),required)
    manifest['logging_schema_version']=3;(run/'run_manifest.json').write_text(json.dumps(manifest))
    entry['files']['run_manifest.json']=study.digest(run/'run_manifest.json')
    with pytest.raises(ValueError,match='schema 7'):A.verify_frozen_entry(entry,required,minimum_schema=7)


@pytest.mark.parametrize('metric',[M.brier,M.logloss,M.auroc,M.auprc,M.ece])
@pytest.mark.parametrize('bad',[[.2,float('nan')],[.2,float('inf')],[.2,1.2]])
def test_probability_scores_refuse_invalid_values(metric,bad):
    with pytest.raises(ValueError):metric([0,1],bad)


def test_probability_shape_and_tie_handling():
    with pytest.raises(ValueError):M.ece([0,1],[[.5],[.5]])
    assert M.auprc([1,0],[.5,.5])==pytest.approx(.5)
    assert M.auroc([1,0],[.5,.5])==pytest.approx(.5)
    assert np.isnan(M.ece([],[]))


def test_monitor_keeps_tasks_attempts_missing_summaries_and_exact_seeds(tmp_path):
    log=tmp_path/'campaign';log.mkdir()
    def attempt(name,task,seed):
        run=tmp_path/name;run.mkdir()
        (run/'run_manifest.json').write_text(json.dumps(dict(task=task,seed=seed)))
        return dict(run_dir=str(run),task=task,condition='P0',seed=seed,attempt_id=name)
    first=attempt('first','fusion_a',1);current=attempt('retry','fusion_a',1)
    (Path(first['run_dir'])/'run_summary.json').write_text(json.dumps(dict(completion_reason='collision',valid_run=True)))
    current['attempts']=[first]
    ledger={'fusion_a__P0__seed1':current,'fusion_b__P0__seed1':attempt('b','fusion_b',1)}
    (log/'campaign_log.json').write_text(json.dumps(ledger))
    cfg=tmp_path/'config.yaml';cfg.write_text('tasks:\n  fusion_a:\n    conditions: [P0]\n    seeds: [1, 2]\n  fusion_b:\n    conditions: [P0]\n    seeds: [1, 2]\n')
    rows,_=MON.collect(log,cfg)
    assert len(rows)==5 and sum(r['current'] for r in rows)==4
    assert {r['task'] for r in rows}=={'fusion_a','fusion_b'}
    assert sum(r['completion']=='pending' for r in rows)==2
    assert sum(r['completion']=='missing_summary' for r in rows)==2
    assert rows[0]['completion']=='collision'
    assert all(r['planned'] for r in rows)


def test_cached_result_is_not_returned_before_run_validation(tmp_path,monkeypatch):
    directory=tmp_path/'trial';directory.mkdir();(directory/'results.json').write_text('{"old":true}')
    def invalid(*a,**kw):raise ValueError('current selected run is invalid')
    monkeypatch.setattr(FD,'load_run',invalid)
    with pytest.raises(ValueError,match='current selected run'):FD.analyze(dict(key='trial'),tmp_path)


def test_every_replay_arm_uses_all_capture_times_by_default(monkeypatch):
    truth=A.TruthSeries([0,1],[0,.2],[0,0],[0,0],'synthetic')
    observations=[dict(camera=c,t=t,batch=b,original_z=np.array([.2*t,0]),original_R=np.eye(2)*.01)
                  for b,c,t in [('a','camera_A',.25),('b','camera_B',.75)]]
    odom={0.:np.array([.2,0]),1.:np.array([.2,0])};m={'task_start_pose':dict(x=0,y=0,yaw=0)}
    steps=[];original=R.unicycle_step
    def track(x,u,dt):steps.append(dt);return original(x,u,dt)
    monkeypatch.setattr(R,'unicycle_step',track)
    for cameras in [['camera_A'],['camera_A','camera_B']]:
        steps.clear();score,records,_=R.run_filter(m,truth,odom,observations,{},'recorded',cameras)
        assert steps==[.25,.5,.25]
        assert [r['t'] for r in records]==[0,1]
        assert score['group_bootstrap_nll_ci95'] is None


def test_measured_odometry_ties_and_missing_causal_prefix():
    table=[dict(odom_noisy_stamp=0,odom_noisy_v=1,odom_noisy_w=0)]
    with pytest.raises(ValueError,match='conflicting'):A.measured_odometry(table+[dict(table[0],odom_noisy_v=2)])
    with pytest.raises(ValueError,match='causal'):A.measured_odometry(table,start=-1,stop=1)


def selection():
    entries=[dict(task=t,seed=s,key=f'{t}__N1__seed{s}',run=f'logs/{t}/{s}') for t in ['a','b'] for s in [1,2,3]]
    return dict(status='complete_pilot_diagnostic',pending=[],invalid=[],runs=entries,
                expected=[dict(task=e['task'],seed=e['seed']) for e in entries])


@pytest.mark.parametrize('mutation',['duplicate','missing','extra','path','key'])
def test_network_selection_requires_exact_trial_product(mutation):
    data=selection();reg=dict(runs=6,routes=2,seeds_per_route=3)
    NR.validate_selection(data,reg)
    if mutation=='duplicate':data['runs'][-1]=copy.deepcopy(data['runs'][0])
    elif mutation=='missing':data['runs'].pop()
    elif mutation=='extra':data['runs'].append(dict(data['runs'][0],seed=4))
    elif mutation=='path':data['runs'][-1]['run']=data['runs'][0]['run']
    elif mutation=='key':data['runs'][0]['key']='wrong'
    with pytest.raises(ValueError):NR.validate_selection(data,reg)


def test_navigation_protocol_uses_selected_task_and_seed(tmp_path,monkeypatch):
    config=tmp_path/'config.yaml';config.write_text('tasks:\n  custom_route:\n    conditions: [P0]\n    seeds: [999]\n')
    monkeypatch.setattr(NAV,'CONFIG',config);monkeypatch.setattr(NAV,'REPO',tmp_path)
    monkeypatch.setattr(NAV,'OUT',tmp_path/'study');monkeypatch.setattr(NAV,'__file__',str(tmp_path/'analyzer.py'))
    monkeypatch.setattr(NAV,'digest',lambda p:'a'*64)
    NAV.protocol(tmp_path/'output',max_reference_gap_s=.125)
    protocol=json.loads((tmp_path/'output/protocol.json').read_text())
    assert protocol['task']=='custom_route' and protocol['seeds']==[999] and protocol['arms']==['P0']
    assert protocol['reference_policy']==dict(max_reference_gap_s=.125)


def test_story_uses_own_time_residuals_and_batch_identity(tmp_path,monkeypatch):
    story=module('audit11_story','experiments/fusion_on_fixed_routes/story/fusion_examples.py')
    run,_,obs,ass=F.fixture(tmp_path/'run')
    for o in obs[:2]:o.update(fused_stamp=.4,fused_x=.41)
    ass[0].update(correction_stamp=.4,apply_stamp=.41)
    F.write_csv(run/'fusion_observations.csv',obs)
    F.write_csv(run/'correction_assimilations.csv',ass)
    monkeypatch.setattr(story,'showcase_run',lambda *a:run)
    _,moments=story.load('F4','synthetic')
    first=next(m for m in moments if m['source_batch_id']=='b0')
    assert first['error_cm']==pytest.approx(1)
    assert [np.linalg.norm(o['xy']-first['gt'])*100 for o in first['obs']]==pytest.approx([2,2])


def test_retired_legacy_replay_cannot_glob_evidence(monkeypatch):
    legacy=module('audit11_legacy_replay','experiments/fusion_on_fixed_routes/replay.py')
    def forbidden(*a):raise AssertionError('glob reached')
    monkeypatch.setattr(Path,'glob',forbidden)
    with pytest.raises(SystemExit,match='retired'):legacy.main()


def test_compare_does_not_swallow_invalid_declared_arm(tmp_path,monkeypatch):
    comparison=module('audit11_compare','experiments/fusion_on_fixed_routes/compare.py')
    selected=tmp_path/'selection.json';selected.write_text(json.dumps(dict(runs={'route':{'F1':['bad']}})))
    monkeypatch.setattr(comparison,'FROZEN_RUNS',selected)
    def invalid(*a):raise SystemExit('invalid raw ledger')
    monkeypatch.setattr(comparison,'_selected_runs',invalid)
    with pytest.raises(SystemExit,match='invalid raw ledger'):comparison.per_arm('route',partial=True)


def test_score_counts_unique_mission_beliefs_and_full_event_corrections(tmp_path,monkeypatch):
    run,table,_,_=F.fixture(tmp_path/'run')
    for i,row in enumerate(table):
        row.update(planner_belief_stamp=.2 if i<10 else .4,
                   planner_belief_x=.2 if i<10 else 1.4,planner_belief_y=0,
                   planner_cov_x=1,planner_cov_xy=0,planner_cov_y=1)
    F.write_csv(run/'experiment.csv',table)
    summary=json.loads((run/'run_summary.json').read_text())
    summary.update(completion_reason='timeout',elapsed_after_first_cmd_s=1)
    (run/'run_summary.json').write_text(json.dumps(summary))
    monkeypatch.setattr(S,'REPO',tmp_path)
    monkeypatch.setattr(S,'_route_polyline',lambda task:np.array([[0.,0.],[1.,0.]]))
    result=S._score_one(run,'F4','synthetic')
    assert result['belief_error_cm']['n_samples']==2
    assert result['belief_error_cm']['median']==pytest.approx(50)
    assert result['honesty']['n_samples']==2
    assert result['correction_error_cm']['n']==3  # All published fused events, independent of outcome.
    assert result['accounting']['accepted_updates']==1


def test_analysis_protocol_never_overwrites_unidentified_or_changed_outputs(tmp_path):
    (tmp_path/'results.json').write_text('{"historical":true}')
    with pytest.raises(ValueError,match='lack matching'):A.freeze_analysis_protocol(tmp_path,'protocol.json',{'task':'a'},owned_outputs=['results.json'])
    assert json.loads((tmp_path/'results.json').read_text())=={'historical':True}
    out=tmp_path/'fresh';A.freeze_analysis_protocol(out,'protocol.json',{'task':'a'})
    with pytest.raises(ValueError,match='differ'):A.freeze_analysis_protocol(out,'protocol.json',{'task':'b'})
    assert json.loads((out/'protocol.json').read_text())=={'task':'a'}
