"""Ideal reference geometry and identity checks on the installation actually used."""
import json,sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/p) for p in ['experiments/camera_observation_characterization','src/unav_common','src/experiments']]
from derive_interpretations import camera_models


def test_ideal_ground_reference_round_trip_all_cameras_and_headings():
    cameras=camera_models(json.loads((REPO/'logs/perception_datasets/warehouse_v2_bbox_characterization_20260831/capture_manifest.json').read_text()))
    checked=0
    for camera in cameras.values():
        for x in np.linspace(-10,10,9):
            for y in np.linspace(-8,8,7):
                for yaw in np.arange(8)*np.pi/4:
                    u,v,visible=camera.world_to_pixel(x,y,0)
                    if not visible:continue
                    # A ground-reference position observation is heading invariant.
                    np.testing.assert_allclose(camera.g_uv([x,y,yaw]),[u,v],atol=1e-10)
                    np.testing.assert_allclose(camera.pixel_to_world(u,v),[x,y],atol=1e-10)
                    checked+=1
    assert checked>1000
