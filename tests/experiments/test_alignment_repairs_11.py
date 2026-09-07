"""Independent synthetic evidence for audit 11 correctness repairs (no drive metrics)."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('audit11_aligned', REPO/'experiments/fusion_on_fixed_routes/aligned.py')
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fixture(tmp_path, *, outcomes=('accepted', 'rejected', 'dropped'), stamps=(.2, .4, .6)):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path/'run_manifest.json').write_text(json.dumps(dict(logging_schema_version=7)))
    (tmp_path/'run_summary.json').write_text(json.dumps(dict(first_cmd_stamp=0., stop_stamp=1., gt_stamp_source='receipt_sim_clock')))
    table = [dict(stamp=t, gt_stamp=t, gt_available=1, gt_x=t, gt_y=0, gt_yaw=0,
                  planner_belief_stamp=t, planner_belief_x=t+.03, planner_belief_y=0)
             for t in np.round(np.arange(0, 1.01, .1), 8)]
    obs, ass = [], []
    for i, (status, t) in enumerate(zip(outcomes, stamps)):
        for cam, err in [('A', .02), ('B', -.02)]:
            obs.append(dict(source_batch_id=f'b{i}', camera=cam, used=1, stamp=t+.01,
                obs_stamp=t, obs_x=t+err, obs_y=0, obs_cov_xx=.01, obs_cov_xy=0, obs_cov_yy=.01,
                fused_stamp=t, fused_x=t+.01, fused_y=0, fused_cov_xx=.005,
                fused_cov_xy=0, fused_cov_yy=.005, n_candidates=2, n_used=2))
        ass.append(dict(source_batch_id=f'b{i}', correction_stamp=t, apply_stamp=t+.01,
            belief_stamp_after=t, status=status, accepted=int(status in ('accepted', 'accepted_bootstrap', 'reanchored')),
            reason='gate_decision', nis=1))
    write_csv(tmp_path/'experiment.csv', table)
    write_csv(tmp_path/'fusion_observations.csv', obs)
    write_csv(tmp_path/'correction_assimilations.csv', ass)
    return tmp_path, table, obs, ass


def test_three_quantities_use_own_times(tmp_path):
    run, table, obs, ass = fixture(tmp_path)
    assert [o['error_cm'] for o in A.readings(run)] == pytest.approx([2]*6)
    assert [o['error_cm'] for o in A.fused_answers(run)] == pytest.approx([1]*3)
    assert A.aligned_error_cm(run, 'belief')['aligned_cm'] == pytest.approx([3]*11)
    with pytest.raises(A.PosteriorUnavailable):
        A.belief_at_fusion_events(run)


@pytest.mark.parametrize('mutation', ['missing','extra','duplicate','blank','status','reason','flag','stamp','nan'])
def test_raw_ledger_fails_closed(tmp_path, mutation):
    run, table, obs, ass = fixture(tmp_path)
    if mutation == 'missing': ass.pop()
    elif mutation == 'extra': ass.append(dict(ass[-1], source_batch_id='extra'))
    elif mutation == 'duplicate': ass.append(dict(ass[-1]))
    elif mutation == 'blank': ass[-1]['source_batch_id'] = ''
    elif mutation == 'status': ass[-1]['status'] = 'mystery'
    elif mutation == 'reason': ass[-1]['reason'] = ''
    elif mutation == 'flag': ass[-1]['accepted'] = 1
    elif mutation == 'stamp': ass[-1]['correction_stamp'] = .9
    elif mutation == 'nan': obs[0]['fused_x'] = float('nan')
    write_csv(run/'fusion_observations.csv', obs)
    write_csv(run/'correction_assimilations.csv', ass)
    with pytest.raises(ValueError): A.validate_run_ledger(run)
    with pytest.raises(ValueError): A.fused_answers(run)


def test_shutdown_tail_is_unscoreable_but_accounted(tmp_path):
    run, _, _, _ = fixture(tmp_path, stamps=(.2, .4, 1.1))
    ledger = A.validate_run_ledger(run)
    assert set(ledger.by_batch) == {'b0','b1','b2'}
    assert [o['source_batch_id'] for o in A.fused_answers(run)] == ['b0','b1']


def test_identity_survives_simultaneous_reordered_batches(tmp_path):
    run, _, obs, ass = fixture(tmp_path, stamps=(.2, .2, .4))
    write_csv(run/'fusion_observations.csv', list(reversed(obs)) + [obs[0]])
    write_csv(run/'correction_assimilations.csv', list(reversed(ass)))
    assert [(o['source_batch_id'],o['camera']) for o in A.readings(run)] == [
        ('b0','A'),('b0','B'),('b1','A'),('b1','B'),('b2','A'),('b2','B')]
    assert len(A.fused_answers(run)) == 3


@pytest.mark.parametrize('field,value', [('fused_x',99),('obs_stamp',.3),('used',0)])
def test_conflicting_copies_do_not_select_by_file_order(tmp_path, field, value):
    run, _, obs, _ = fixture(tmp_path)
    obs.append(dict(obs[0], **{field:value}))
    write_csv(run/'fusion_observations.csv', obs)
    with pytest.raises(ValueError): A.readings(run)


def test_reference_support_ties_resets_and_heading():
    with pytest.raises(ValueError, match='conflicting'):
        A.TruthSeries([0,0],[0,1],[0,0],[0,0],'synthetic')
    with pytest.raises(ValueError, match='clock reset'):
        A.TruthSeries([0,1,0],[0,1,2],[0,0,0],[0,0,0],'synthetic')
    truth=A.TruthSeries([0,0,1,2],[0,0,1,2],[0]*4,[0,0,np.nan,1],'synthetic')
    assert truth.at([0,1,2])[0] == pytest.approx([0,1,2])
    assert np.isnan(truth.at([.5,1.5,-1,3])[0]).all()
    assert truth.yaw_at([2])[0] == 1
    bounded=A.TruthSeries([0,.1,1],[0,.1,1],[0]*3,[0]*3,'synthetic',max_reference_gap_s=.11)
    assert bounded.at([.05])[0][0] == pytest.approx(.05)
    assert np.isnan(bounded.at([.5])[0][0])


def test_logger_clock_reset_is_not_sorted_away(tmp_path):
    run, table, _, _ = fixture(tmp_path)
    table[-1]['stamp'] = 0
    write_csv(run/'experiment.csv', table)
    with pytest.raises(ValueError, match='clock reset'): A.rows(run)


def test_holds_and_out_of_order_repeats_are_not_replicates():
    assert A.landed_mask([np.nan,1,2,1,2,3]).tolist() == [False,True,True,False,False,True]


def test_blind_gap_endpoints_and_refusals(tmp_path):
    run, _, _, _ = fixture(tmp_path, stamps=(.4,.5,.6))
    counts=A.correction_accounting(run)
    assert counts['accepted_updates'] == 1
    assert counts['correction_dropped_fraction'] == pytest.approx(1/3)
    assert counts['longest_correction_gap_s'] == pytest.approx(.59)


@pytest.mark.parametrize('cov',[np.diag([-1.,-1.]), [[1,0],[.5,1]], [[1,1],[1,1]], [[1,np.nan],[np.nan,1]]])
def test_full_covariance_must_be_spd(cov):
    with pytest.raises(ValueError): A.nees([[1,0]],[cov])


def test_nees_has_independent_analytic_answer():
    # inverse([[4,1],[1,2]]) = [[2,-1],[-1,4]]/7.
    assert A.nees([[2,3]], [[[4,1],[1,2]]])[0] == pytest.approx(32/7)


def test_multiple_commits_between_logger_ticks_require_distinct_records(tmp_path):
    run, _, _, ass = fixture(tmp_path, outcomes=('accepted','accepted','dropped'), stamps=(.2,.2,.6))
    for i,a in enumerate(ass[:2]):
        a.update(schema_version=2, epoch='epoch-a', revision_before=i, revision_after=i+1,
            frame_id='map_bev', state_stamp_ns=200000000, posterior_mean=json.dumps([.2+(i+1)*.01,0,0]),
            posterior_covariance=json.dumps(np.diag([.01,.01,.1]).tolist()), valid=1,motion_supported=1)
    write_csv(run/'correction_assimilations.csv', ass)
    events=A.belief_at_fusion_events(run,reference_frame='map_bev')
    assert [e['error_cm'] for e in events] == pytest.approx([1,2])
    assert [e['revision'] for e in events] == [1,2]
    with pytest.raises(ValueError,match='frame mismatch'):
        A.belief_at_fusion_events(run,reference_frame='odom')


def test_schema3_never_infers_assimilation_from_sidecars(tmp_path):
    run, _, _, _ = fixture(tmp_path)
    (run/'run_manifest.json').write_text(json.dumps(dict(logging_schema_version=3)))
    with pytest.raises(ValueError,match='schema 4'): A.validate_run_ledger(run)
    with pytest.raises(ValueError,match='schema 4'): A.belief_at_fusion_events(run)


def v2_fixture(tmp_path):
    run,table,obs,ass=fixture(tmp_path)
    (run/'run_manifest.json').write_text(json.dumps(dict(logging_schema_version=8)))
    publications=[]
    for a in ass:
        stamp_ns=round(a['correction_stamp']*1e9)
        a.update(schema_version=2,epoch='belief-epoch',source_epoch='source-epoch',
                 correction_stamp_ns=stamp_ns,apply_stamp_ns=round(a['apply_stamp']*1e9))
        publications.append(dict(source_batch_id=a['source_batch_id'],correction_stamp=a['correction_stamp'],
            correction_stamp_ns=stamp_ns,epoch='source-epoch',frame_id='map_bev',schema_version=2,
            accepted_camera_ids=json.dumps(['A','B']),member_ids=json.dumps(['A','B']),payload_sha256='f'*64))
    write_csv(run/'correction_assimilations.csv',ass)
    write_csv(run/'correction_publications.csv',publications)
    return run,publications,ass


def test_schema8_uses_publication_ledger_and_separate_source_epoch(tmp_path):
    run,pubs,ass=v2_fixture(tmp_path)
    ledger=A.validate_run_ledger(run)
    assert ledger.by_batch['b0']['epoch']=='source-epoch'
    assert type(ledger.by_batch['b0']['correction_stamp_ns']) is int
    assert A.assimilations(run)[0]['epoch']=='belief-epoch'
    # Missing diagnostic camera rows do not fabricate a missing publication.
    write_csv(run/'fusion_observations.csv',[])
    assert len(A.validate_run_ledger(run).by_batch)==3


@pytest.mark.parametrize('mutation',['nanoseconds','missing_ns','duplicate_publication','source_epoch'])
def test_v2_publication_lineage_is_exact(tmp_path,mutation):
    run,pubs,ass=v2_fixture(tmp_path)
    if mutation=='nanoseconds':pubs[0]['correction_stamp_ns']+=1
    elif mutation=='missing_ns':pubs[0]['correction_stamp_ns']=''
    elif mutation=='duplicate_publication':pubs.append(dict(pubs[0]))
    elif mutation=='source_epoch':ass[0]['source_epoch']='wrong'
    write_csv(run/'correction_assimilations.csv',ass)
    write_csv(run/'correction_publications.csv',pubs)
    with pytest.raises(ValueError):A.validate_run_ledger(run)


def test_ambiguous_same_time_beliefs_do_not_choose_first_payload(tmp_path):
    run,table,_,_=fixture(tmp_path)
    table.insert(2,dict(table[1],planner_belief_x=9))
    write_csv(run/'experiment.csv',table)
    with pytest.raises(ValueError,match='ambiguous belief'):A.aligned_error_cm(run,'belief')


def test_camera_retransmission_requires_agreement_before_reference_selection(tmp_path):
    run,_,_,_=fixture(tmp_path)
    o=dict(source_batch_id='beyond-truth',camera_id='camera_A',timestamp_s=20.,detection_valid=False)
    original=dict(valid_contract=True,duplicate=False,observation=o)
    duplicate=dict(valid_contract=True,duplicate=True,observation=o)
    path=run/'camera_opportunities.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in [duplicate,original]))
    rows,count=A.camera_opportunities(run)
    assert len(rows)==1 and count==1 and rows[0]['timestamp_s']==20
    duplicate['observation']=dict(o,timestamp_s=21)
    path.write_text('\n'.join(json.dumps(r) for r in [duplicate,original]))
    with pytest.raises(ValueError,match='conflicting camera opportunity'):A.camera_opportunities(run)
