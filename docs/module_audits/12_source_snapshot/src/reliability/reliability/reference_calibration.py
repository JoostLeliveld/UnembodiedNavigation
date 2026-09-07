"""Frozen residual calibration AFTER the bbox-feature reference-position NN.

This defines a measurement of reference XY in map_bev, metres, h(x)=[x,y].
It is a conditional residual covariance, not a fusion posterior or score field.
"""
from pathlib import Path
import hashlib
import json
import numpy as np


class ReferenceCalibration:
    def __init__(self, path, mean_checkpoint, camera_ids):
        self.path = Path(path).resolve()
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        data = json.loads(self.path.read_text())
        required = dict(schema='camera_reference_calibration.v1',
                        frame='map_bev', reference='robot_ground_reference_xy',
                        covariance_units='m2', mean_order='bbox_feature_nn_then_subtract_bias')
        for key, value in required.items():
            if data.get(key) != value:
                raise ValueError(f'reference calibration {key} must be {value}')
        mean_hash = hashlib.sha256(Path(mean_checkpoint).read_bytes()).hexdigest()
        if data.get('mean_checkpoint_sha256') != mean_hash:
            raise ValueError('reference calibration mean checkpoint hash differs')
        self.bias, self.covariance = {}, {}
        for camera in camera_ids:
            if camera not in data['cameras']:
                raise ValueError(f'reference calibration missing {camera}')
            entry = data['cameras'][camera]
            bias, R = np.asarray(entry['bias_m'], float), np.asarray(entry['R_m2'], float)
            if bias.shape != (2,) or R.shape != (2, 2) or not np.isfinite(bias).all() or not np.isfinite(R).all():
                raise ValueError(f'nonfinite or wrong-shaped calibration for {camera}')
            if not np.allclose(R, R.T, atol=1e-12, rtol=0):
                raise ValueError(f'non-symmetric covariance for {camera}')
            try:
                np.linalg.cholesky(R)
            except np.linalg.LinAlgError as exc:
                raise ValueError(f'non-positive-definite covariance for {camera}') from exc
            self.bias[camera], self.covariance[camera] = bias, R

    def apply(self, camera_id, nn_xy):
        """Call once on a fresh NN reference reading, before any robot update."""
        z = np.asarray(nn_xy, float)
        if z.shape != (2,) or not np.isfinite(z).all():
            raise ValueError('reference reading must be finite XY metres')
        z = z - self.bias[camera_id]
        return tuple(z), tuple(tuple(row) for row in self.covariance[camera_id])
