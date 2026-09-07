"""Controls needed for a defensible camera-subset comparison."""
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/icra_commissioning'))
# Use the same entry point as the CLI before any older script named replay can
# shadow it. Several historical studies use that generic module filename.
from field_driving import run_filter
replay = sys.modules[run_filter.__module__]
import aligned
from network_replay import camera_subsets, CAMERAS


def test_exhaustive_subsets_are_fixed_without_performance_selection():
    subsets = camera_subsets()
    assert len(subsets)==16 and len(set(subsets))==16
    assert [len(c) for c in subsets]==[1]*5+[2]*10+[5]
    assert subsets[-1]==tuple(CAMERAS)


def test_mask_changes_evidence_but_not_motion_or_scoring_grid(monkeypatch):
    truth = aligned.TruthSeries([0., 1.], [0., .2], [0., 0.], [0., 0.], 'fixture')
    rows = [dict(camera=cam, t=t, original_z=np.array([.2*t, 0.]),
                 original_R=np.eye(2)*.01, batch=f'frame-{i}')
            for i, (cam,t) in enumerate([('camera_A', .25), ('camera_B', .75)])]
    steps = []
    original = replay.unicycle_step
    def track(state, u, dt):
        steps.append(dt)
        return original(state, u, dt)
    monkeypatch.setattr(replay, 'unicycle_step', track)
    results = []
    for mask in [['camera_A'], ['camera_A', 'camera_B']]:
        steps.clear()
        score, records, events = replay.run_filter(
            {'task_start_pose':{'x':0., 'y':0., 'yaw':0.}}, truth,
            {0.:np.array([.2, 0.]), 1.:np.array([.2, 0.])}, rows, {},
            'recorded', mask, prediction_times=[.25, .75])
        results.append((list(steps), score['updates'], [r['t'] for r in records], len(events)))
    assert results[0]==([.25, .5, .25], 1, [0., 1.], 1)
    assert results[1]==([.25, .5, .25], 2, [0., 1.], 2)


def test_nonfinite_prediction_grid_is_refused():
    with pytest.raises(ValueError, match='nonfinite prediction event'):
        replay.run_filter({}, None, {0.:np.zeros(2)}, [], {}, 'recorded', [],
            prediction_times=[float('nan')])
