from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
for rel in ('scripts/visibility_comparison',):
    path = str((REPO_ROOT / rel).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)

from common import accepted_completed_run, run_has_usable_logs, write_manifest
from make_visibility_comparison_report import _latest_run_dir
from run_planner_method_sweep import _compute_run_config_hash


def _make_run_dir(tmp_path: Path, *, method: str = 'oracle_visibility', ambiguity_weight: float = 1.0) -> Path:
    run_dir = tmp_path / f'run_{ambiguity_weight}'
    run_dir.mkdir()
    artifact = run_dir / f'{method}_gp.npz'
    np.savez(
        artifact,
        xs=np.array([0.0, 1.0], dtype=float),
        ys=np.array([0.0, 1.0], dtype=float),
        P_conservative_plan_map=np.array([[0.2, 0.4], [0.6, 0.8]], dtype=float),
        P_mean_map=np.array([[0.2, 0.4], [0.6, 0.8]], dtype=float),
        F_mean_map=np.zeros((2, 2), dtype=float),
        F_std_map=np.ones((2, 2), dtype=float) * 0.1,
        camera_pos=np.array([-3.0, -3.0, 6.0], dtype=float),
        camera_pose=np.array([-3.0, -3.0, 6.0, 0.0, 0.0, 0.0], dtype=float),
        look_at=np.array([1.0, 1.0, 0.0], dtype=float),
        img_width=np.array([1280], dtype=np.int32),
        img_height=np.array([720], dtype=np.int32),
        fov_h_rad=np.array([1.5708], dtype=float),
        geometry_json=np.array(['{}']),
        geometry_sha256=np.array(['abc']),
        artifact_schema_version=np.array([2], dtype=np.int32),
        world=np.array(['warehouse_occ_light.world.sdf']),
        world_name=np.array(['warehouse_occ_light.world.sdf']),
        world_path=np.array(['/tmp/world']),
    )
    write_manifest(run_dir / 'run_manifest.json', {
        'method': method,
        'planner': 'efe1',
        'ambiguity_weight': ambiguity_weight,
        'visibility_artifact_path': str(artifact),
        'git_commit': 'deadbeef',
    })
    (run_dir / 'experiment.csv').write_text('stamp\n0.0\n', encoding='utf-8')
    return run_dir


def test_run_config_hash_changes_when_manifest_changes(tmp_path: Path) -> None:
    run_a = _make_run_dir(tmp_path, ambiguity_weight=1.0)
    run_b = _make_run_dir(tmp_path, ambiguity_weight=2.0)

    assert _compute_run_config_hash(run_a) != _compute_run_config_hash(run_b)


def test_accepted_completed_run_and_log_filtering(tmp_path: Path) -> None:
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    (run_dir / 'experiment.csv').write_text('stamp\n0.0\n', encoding='utf-8')

    accepted = {'completed': True, 'completion_reason': 'timeout_after_first_cmd'}
    interrupted = {'completed': False, 'completion_reason': 'interrupted'}

    assert accepted_completed_run(accepted) is True
    assert accepted_completed_run(interrupted) is False
    assert run_has_usable_logs(run_dir) is True


def test_report_latest_run_dir_excludes_newer_interrupted_run(tmp_path: Path) -> None:
    method_root = tmp_path / 'oracle_visibility'
    accepted_dir = method_root / 'accepted'
    interrupted_dir = method_root / 'interrupted'
    accepted_dir.mkdir(parents=True)
    interrupted_dir.mkdir(parents=True)

    write_manifest(accepted_dir / 'run_summary.json', {
        'completed': True,
        'completion_reason': 'goal_reached',
    })
    write_manifest(interrupted_dir / 'run_summary.json', {
        'completed': False,
        'completion_reason': 'interrupted',
    })
    (accepted_dir / 'experiment.csv').write_text('stamp\n0.0\n', encoding='utf-8')
    (interrupted_dir / 'experiment.csv').write_text('stamp\n0.0\n', encoding='utf-8')

    accepted_dir.touch()
    interrupted_dir.touch()

    assert _latest_run_dir(tmp_path, 'oracle_visibility') == accepted_dir
