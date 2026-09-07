#!/usr/bin/env python3
"""Derive a projection sidecar from exact frozen capture/model bytes; never refit."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/p) for p in ('src/reliability','src/unav_common',
                                    'experiments/camera_observation_characterization')]
from derive_interpretations import camera_models
from reliability.commissioning_binding import CommissioningBinding
from reliability.learned_box_correction import LearnedBoxCorrection
from reliability.reference_calibration import ReferenceCalibration
from unav_common.capture_integrity import atomic_json, checked_bytes, digest


def export(capture: Path, mean: Path, reference: Path, output: Path):
    if output.exists():
        raise FileExistsError('binding exists; choose a new output')
    summary = json.loads((mean.parent/'summary.json').read_bytes())
    mean_bytes = checked_bytes(mean, summary['artifact_sha256'])
    capture_bytes = checked_bytes(capture/'capture_manifest.json', summary['source_hashes']['capture_manifest'])
    for name, file in [('bias_update_interpretations','bias_update_interpretations.csv'),
                       ('bias_update_manifest','bias_update_interpretations_manifest.json')]:
        checked_bytes(capture/file, summary['source_hashes'][name])
    metadata = json.loads(capture_bytes)
    detector = json.loads((capture/'bbox_detector_manifest.json').read_bytes())
    if detector['capture_manifest_sha256'] != digest(capture_bytes):
        raise ValueError('detector and mean did not use the same capture manifest')
    checked_bytes(capture/'capture_index.csv', detector['capture_index_sha256'])
    checked_bytes(capture/'bbox_observations.csv', detector['bbox_observations_sha256'])
    checked_bytes(Path(detector['weights']), detector['weights_sha256'])
    wrapper = LearnedBoxCorrection(mean, expected_sha256=digest(mean_bytes))
    calibration = ReferenceCalibration(reference, wrapper, wrapper.camera_ids)
    cameras = camera_models(metadata, capture_root=capture)
    if set(cameras) != set(wrapper.camera_ids):
        raise ValueError('mean and capture camera registries differ')
    for camera in metadata['cameras']:
        entry = wrapper._geometry[camera['camera_id']]
        if (list(entry['xy']) != camera['pose_xyz_rpy'][:2] or entry['yaw'] != camera['pose_xyz_rpy'][5]
                or entry['width'] != camera['image_width'] or entry['height'] != camera['image_height']):
            raise ValueError('mean artifact geometry differs from its declared capture')
    payload = dict(schema='commissioned_projection_binding.v1', frame='map_bev',
        reference='robot_ground_reference_xy', position_units='m', ground_plane_z_m=0.0,
        pixel_convention='original_image_half_open_bbox_bottom_centre',
        mean_checkpoint_sha256=wrapper.sha256, reference_calibration_sha256=calibration.sha256,
        detector_sha256=detector['weights_sha256'], capture_manifest_sha256=digest(capture_bytes),
        world_sha256=metadata['world_sha256'], world_profiles_sha256=metadata['world_profiles_sha256'],
        detector_inference=detector['runtime'],
        cameras={c:dict(cam_pos=m.cam_pos.tolist(), K=m.K.tolist(), R=m.R.tolist(),
                        image_width=m.img_width, image_height=m.img_height) for c,m in cameras.items()},
        historical_limits=['commanded simulator model origin; actual settled pose not verified',
                           'capture did not record expanded robot-description identity',
                           'static development artifact; this sidecar does not establish empirical calibration'],
        derivation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, payload)
    binding = CommissioningBinding(output, mean_sha256=wrapper.sha256,
        reference_sha256=calibration.sha256, detector_sha256=detector['weights_sha256'])
    binding.validate_cameras(cameras)
    return binding.sha256


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--mean', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args=parser.parse_args()
    print(export(args.capture.resolve(),args.mean.resolve(),args.reference.resolve(),args.out.resolve()))
