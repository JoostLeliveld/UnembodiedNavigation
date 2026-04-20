from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = REPO_ROOT / 'logs' / 'visibility_comparison' / 'test_outputs'


def test_temperature_scaling_script_writes_artifact_and_plots(tmp_path: Path) -> None:
    perception_targets = tmp_path / 'perception_targets.csv'
    capture_manifest = tmp_path / 'capture_manifest.json'
    out_dir = LOG_ROOT / f'calibration_{tmp_path.name}'
    artifact_out = LOG_ROOT / f'yolo_score_calibration_{tmp_path.name}.json'

    perception_targets.write_text(
        '\n'.join([
            'sample_id,x,y,theta,yolo_score_raw,yolo_detected_after_threshold,oracle_visible',
            'a,0.0,0.0,0.0,0.10,0,0',
            'b,1.0,0.0,0.0,0.20,0,0',
            'c,2.0,0.0,0.0,0.75,1,1',
            'd,3.0,0.0,0.0,0.90,1,1',
            'e,4.0,0.0,0.0,0.60,1,1',
            'f,5.0,0.0,0.0,0.35,0,0',
        ]),
        encoding='utf-8',
    )
    capture_manifest.write_text(json.dumps({'camera_pos': [-3.0, -3.0, 6.0]}), encoding='utf-8')
    out_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / 'scripts' / 'visibility_comparison' / 'plot_yolo_calibration.py'),
            '--perception-targets',
            str(perception_targets),
            '--capture-manifest',
            str(capture_manifest),
            '--out-dir',
            str(out_dir),
            '--artifact-out',
            str(artifact_out),
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )

    artifact = json.loads(artifact_out.read_text(encoding='utf-8'))
    assert 0.0 < float(artifact['temperature'])
    assert artifact['sample_count'] == 6
    for key in ('raw_brier', 'calibrated_brier', 'raw_ece', 'calibrated_ece'):
        assert float(artifact[key]) >= 0.0
    for rel in (
        out_dir / 'yolo_reliability.png',
        out_dir / 'yolo_pr_roc.png',
        out_dir / 'yolo_score_histograms.png',
        out_dir / 'yolo_view_angle_bias.png',
    ):
        assert rel.is_file()
        assert rel.stat().st_size > 0
