#!/usr/bin/env python3
"""Export the frozen NN-residual mean/R for the runtime, without refitting."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src/reliability'))
from study import OUT, digest, writejson
from reliability.reference_calibration import ReferenceCalibration


def export(out):
    if out.exists():
        raise RuntimeError('Calibration exists; choose a new output file')
    manifest_path = OUT / 'manifest.json'
    source = json.loads(manifest_path.read_text())
    mean = REPO / source['mean_artifact']
    if digest(mean) != source['files'][source['mean_artifact']]:
        raise ValueError('Mean model changed since residual calibration')
    network_root = OUT / 'network_planner'
    network_manifest = json.loads((network_root / 'manifest.json').read_text())
    cameras = None
    for arm in ('uniform', 'geometry', 'gp'):
        path = network_root / f'{arm}.npz'
        if digest(path) != network_manifest['artifacts'][arm]['sha256']:
            raise ValueError(f'Changed {arm} field artifact')
        with np.load(path, allow_pickle=False) as data:
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
                   fitted_population='frozen grouped covariance_fit; accepted finite NN reference readings',
                   status='static development calibration; sequential consistency not established',
                   source_hashes={str(p.relative_to(REPO)): digest(p) for p in (
                       manifest_path, network_root / 'manifest.json', OUT / 'models.joblib', Path(__file__))})
    out.parent.mkdir(parents=True, exist_ok=True)
    writejson(out, payload)
    ReferenceCalibration(out, mean, tuple(cameras))
    print(out, digest(out))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT / 'network_planner/reference_calibration.json')
    export(parser.parse_args().out.resolve())
