#!/usr/bin/env python3
"""Export the frozen NN-residual mean/R for the runtime, without refitting."""
import argparse
import json
import io
from pathlib import Path
import sys
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / p) for p in ('src/reliability', 'src/unav_common')]
from study import OUT, digest, writejson
from reliability.reference_calibration import ReferenceCalibration
from unav_common.capture_integrity import atomic_json, checked_bytes


def export(out):
    if out.exists():
        raise RuntimeError('Calibration exists; choose a new output file')
    manifest_path = OUT / 'manifest.json'
    source = json.loads(manifest_path.read_text())
    for relative, expected in source['files'].items():
        checked_bytes(REPO/relative, expected)
    mean = REPO / source['mean_artifact']
    if digest(mean) != source['files'][source['mean_artifact']]:
        raise ValueError('Mean model changed since residual calibration')
    network_root = OUT / 'network_planner'
    network_manifest = json.loads((network_root / 'manifest.json').read_text())
    for relative, expected in network_manifest['sources'].items():
        if Path(relative).suffix != '.py':
            checked_bytes(REPO/relative, expected)
    cameras = None
    for arm in ('uniform', 'geometry', 'gp'):
        path = network_root / f'{arm}.npz'
        encoded = checked_bytes(path, network_manifest['artifacts'][arm]['sha256'])
        with np.load(io.BytesIO(encoded), allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata_json'].item()))
            current = {str(c): dict(bias_m=metadata['required_runtime_mean_offset_m'][str(c)],
                                   R_m2=data['R_cond_m2'][i].tolist())
                       for i, c in enumerate(data['camera_ids'])}
        if cameras is not None and current != cameras:
            raise ValueError('Planner arms must have identical mean/R for this comparison')
        cameras = current
    payload = dict(schema='camera_reference_calibration.v1', frame='map_bev',
                   reference='robot_ground_reference_xy', covariance_units='m2',
                   mean_order='bbox_feature_nn_then_subtract_bias',
                   mean_checkpoint_sha256=digest(mean), cameras=cameras,
                   fitted_population='frozen covariance_fit for residual mean/centered covariance; selection for global scale and isotropic shrinkage',
                   fitting_history=dict(mean_excluded_from_NN_training=True,
                       residual_fit_role='covariance_fit', covariance_regularization_role='selection',
                       transformations=['subtract fitted per-camera residual mean',
                                        'apply selection-fitted isotropic shrinkage and global scale']),
                   status='static development calibration; sequential consistency not established',
                   source_hashes={str(p.relative_to(REPO)): digest(p) for p in (
                       manifest_path, network_root / 'manifest.json', OUT / 'models.joblib', Path(__file__))})
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(out, payload)
    ReferenceCalibration(out, mean, tuple(cameras))
    print(out, digest(out))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT / 'network_planner/reference_calibration.json')
    export(parser.parse_args().out.resolve())
