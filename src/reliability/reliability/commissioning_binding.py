"""Bind unchanged learned/reference artifacts to their commissioned projection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np

REPO = Path(__file__).resolve().parents[3]


def resolve_commissioning_binding(reference_path, *, reference_sha256=None, repo=REPO):
    """An adjacent new-bundle sidecar, or an explicitly hash-keyed legacy sidecar."""
    reference_path = Path(reference_path).resolve()
    reference_sha256 = reference_sha256 or hashlib.sha256(reference_path.read_bytes()).hexdigest()
    if not re.fullmatch('[0-9a-f]{64}', reference_sha256):
        raise ValueError('invalid reference hash for binding lookup')
    adjacent = reference_path.with_suffix('.binding.json')
    registered = Path(repo) / 'config/commissioned_bindings' / (reference_sha256 + '.json')
    for path in (adjacent, registered):
        if path.is_file():
            return path
    raise FileNotFoundError(f'no commissioned projection binding for {reference_sha256}')


class CommissioningBinding:
    def __init__(self, path, *, mean_sha256, reference_sha256, detector_sha256,
                 expected_sha256=None):
        self.path = Path(path).resolve()
        encoded = self.path.read_bytes()
        self.sha256 = hashlib.sha256(encoded).hexdigest()
        if expected_sha256 is not None and self.sha256 != expected_sha256:
            raise ValueError('commissioning binding bytes differ from expected identity')
        data = json.loads(encoded)
        required = dict(schema='commissioned_projection_binding.v1', frame='map_bev',
                        reference='robot_ground_reference_xy', position_units='m',
                        pixel_convention='original_image_half_open_bbox_bottom_centre',
                        ground_plane_z_m=0.0)
        for key, value in required.items():
            if data.get(key) != value:
                raise ValueError(f'incompatible commissioning binding {key}')
        for key, expected in (('mean_checkpoint_sha256', mean_sha256),
                              ('reference_calibration_sha256', reference_sha256),
                              ('detector_sha256', detector_sha256)):
            if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected) or data.get(key) != expected:
                raise ValueError(f'commissioning binding {key} differs from loaded/configured artifact')
        for key in ('capture_manifest_sha256', 'world_sha256', 'world_profiles_sha256'):
            if not isinstance(data.get(key), str) or not re.fullmatch('[0-9a-f]{64}', data[key]):
                raise ValueError(f'missing commissioning lineage {key}')
        cameras = data.get('cameras')
        if not isinstance(cameras, dict) or not cameras:
            raise ValueError('commissioning binding requires a camera registry')
        inference = data.get('detector_inference', {})
        if inference.get('class_name') != 'robot' or inference.get('selected_point') != 'bbox_bottom_centre':
            raise ValueError('commissioning binding requires robot-class bbox-bottom inference')
        for camera, entry in cameras.items():
            for key, shape in (('cam_pos', (3,)), ('K', (3,3)), ('R', (3,3))):
                value = np.asarray(entry.get(key), dtype=float)
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f'invalid commissioned {camera} {key}')
            if any(type(entry.get(k)) is not int or entry[k] <= 0 for k in ('image_width','image_height')):
                raise ValueError(f'invalid commissioned image dimensions for {camera}')
        self.metadata = data

    def validate_detector(self, *, image_size, confidence_threshold, iou_threshold, use_masks=False):
        expected = self.metadata['detector_inference']
        if use_masks:
            raise ValueError('commissioned NN expects bbox bottom, not a mask-derived pixel')
        for key, value in (('image_size', image_size), ('confidence_threshold', confidence_threshold),
                           ('iou_threshold', iou_threshold)):
            if value != expected.get(key):
                raise ValueError(f'commissioned detector {key} differs')

    def validate_cameras(self, cameras):
        if set(cameras) != set(self.metadata['cameras']):
            raise ValueError('runtime and commissioned camera registries differ')
        for camera, model in cameras.items():
            expected = self.metadata['cameras'][camera]
            if (model.img_width, model.img_height) != (expected['image_width'], expected['image_height']):
                raise ValueError(f'commissioned image dimensions differ for {camera}')
            for key in ('cam_pos', 'K', 'R'):
                actual = np.asarray(getattr(model, key), dtype=float)
                if actual.shape != np.asarray(expected[key]).shape or not np.isfinite(actual).all() or not np.allclose(actual, expected[key], rtol=0, atol=1e-10):
                    raise ValueError(f'commissioned projection {key} differs for {camera}')
