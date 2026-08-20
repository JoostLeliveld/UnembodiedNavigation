"""The marker-pair reading: does it invert its own observation model, and is its
covariance the geometry rather than a fitted table?
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from state.core.marker_keypoint_reading import (
    marker_world_position,
    project_marker_pair,
    read_marker_keypoints,
)
from unav_common.camera_model import ObliqueCameraModel

# The single camera in the method-development warehouse.
CAM_POS = (0.0, -5.5, 4.8)
LOOK_AT = (0.0, -1.845, 0.0)
FRONT_X, REAR_X = 0.040, -0.100
PLANE_Z = 0.210


def camera() -> ObliqueCameraModel:
    return ObliqueCameraModel(
        cam_pos=CAM_POS, look_at=LOOK_AT,
        img_width=1280, img_height=720, fov_h_rad=1.5708,
    )


def read_at(x: float, y: float, heading: float, pixel_sigma: float = 1.8):
    cam = camera()
    pixels = project_marker_pair(
        cam, x=x, y=y, heading_rad=heading,
        front_offset_x=FRONT_X, rear_offset_x=REAR_X, plane_z=PLANE_Z,
    )
    assert pixels is not None
    return read_marker_keypoints(
        cam, pixels[:2], pixels[2:],
        front_offset_x=FRONT_X, rear_offset_x=REAR_X,
        plane_z=PLANE_Z, pixel_sigma=pixel_sigma,
    )


@pytest.mark.parametrize('x,y,heading', [
    (0.0, -2.0, 0.0),
    (-3.0, 0.0, math.pi / 2),
    (2.5, 1.5, -2.0),
    (0.0, 3.0, math.pi),
])
def test_a_perfect_detection_recovers_the_pose_exactly(x, y, heading) -> None:
    """With no detector error the reading must return the pose it started from —
    otherwise the observation model and its inverse disagree, and every residual
    downstream inherits that disagreement as a bias."""
    reading = read_at(x, y, heading)
    assert reading is not None
    assert reading.xy_m[0] == pytest.approx(x, abs=1e-6)
    assert reading.xy_m[1] == pytest.approx(y, abs=1e-6)
    wrapped = (reading.heading_rad - heading + math.pi) % (2 * math.pi) - math.pi
    assert wrapped == pytest.approx(0.0, abs=1e-6)


def test_the_offset_markers_do_not_bias_the_position() -> None:
    """Both markers sit forward of base_link on average, so a reading that just
    averaged them would sit 3 cm ahead of the robot at every heading."""
    for heading in (0.0, math.pi / 3, math.pi, -math.pi / 2):
        reading = read_at(1.0, 0.5, heading)
        assert reading is not None
        assert reading.xy_m[0] == pytest.approx(1.0, abs=1e-6)
        assert reading.xy_m[1] == pytest.approx(0.5, abs=1e-6)


def test_uncertainty_grows_with_range() -> None:
    near = read_at(0.0, -3.5, 0.0)
    far = read_at(0.0, 4.0, 0.0)
    assert near is not None and far is not None
    near_sigma = math.sqrt(np.trace(np.array(near.covariance_xy_m2)) / 2)
    far_sigma = math.sqrt(np.trace(np.array(far.covariance_xy_m2)) / 2)
    assert far_sigma > near_sigma


def test_heading_degrades_faster_than_position() -> None:
    """Heading precision scales with how far apart the markers appear, so it must
    fall off faster with range than position does. This is why the reading's
    heading needs its own health signal."""
    near = read_at(0.0, -3.5, 0.0)
    far = read_at(0.0, 4.0, 0.0)
    assert near is not None and far is not None
    position_ratio = (
        math.sqrt(np.trace(np.array(far.covariance_xy_m2)))
        / math.sqrt(np.trace(np.array(near.covariance_xy_m2)))
    )
    heading_ratio = math.sqrt(far.heading_variance_rad2 / near.heading_variance_rad2)
    assert heading_ratio > position_ratio


def test_stated_uncertainty_scales_with_the_pixel_noise_it_is_given() -> None:
    """Doubling the detector's pixel spread must double the stated sigma, not
    something else — the covariance is J R J', linear in R."""
    one = read_at(0.0, 0.0, 0.0, pixel_sigma=1.0)
    two = read_at(0.0, 0.0, 0.0, pixel_sigma=2.0)
    assert one is not None and two is not None
    ratio = (
        math.sqrt(np.trace(np.array(two.covariance_xy_m2)))
        / math.sqrt(np.trace(np.array(one.covariance_xy_m2)))
    )
    assert ratio == pytest.approx(2.0, rel=1e-3)


def test_covariance_is_a_valid_covariance() -> None:
    reading = read_at(-2.0, 1.0, 0.7)
    assert reading is not None
    cov = np.array(reading.covariance_xyh)
    assert np.allclose(cov, cov.T, atol=1e-12), 'must be symmetric'
    assert np.all(np.linalg.eigvalsh(cov) > 0.0), 'must be positive definite'


def test_marker_separation_is_reported_and_shrinks_with_range() -> None:
    near = read_at(0.0, -3.5, 0.0)
    far = read_at(0.0, 4.0, 0.0)
    assert near is not None and far is not None
    assert near.marker_separation_px > far.marker_separation_px > 0.0


def test_a_marker_behind_the_camera_is_not_a_reading() -> None:
    cam = camera()
    behind = project_marker_pair(
        cam, x=0.0, y=-20.0, heading_rad=0.0,
        front_offset_x=FRONT_X, rear_offset_x=REAR_X, plane_z=PLANE_Z,
    )
    assert behind is None


def test_marker_world_position_places_the_disks_on_their_plane() -> None:
    front = marker_world_position(x=1.0, y=2.0, heading_rad=0.0,
                                  offset_x=FRONT_X, plane_z=PLANE_Z)
    assert front[0] == pytest.approx(1.0 + FRONT_X)
    assert front[1] == pytest.approx(2.0)
    assert front[2] == pytest.approx(PLANE_Z)
    turned = marker_world_position(x=0.0, y=0.0, heading_rad=math.pi / 2,
                                  offset_x=FRONT_X, plane_z=PLANE_Z)
    assert turned[0] == pytest.approx(0.0, abs=1e-9)
    assert turned[1] == pytest.approx(FRONT_X)


def test_reading_the_floor_plane_instead_of_the_marker_plane_biases_the_answer() -> None:
    """Guard against the mistake the deployed IPM makes with elevated points:
    back-projecting a marker at z=0.21 onto z=0 puts the robot metres away."""
    cam = camera()
    pixels = project_marker_pair(
        cam, x=0.0, y=2.0, heading_rad=0.0,
        front_offset_x=FRONT_X, rear_offset_x=REAR_X, plane_z=PLANE_Z,
    )
    assert pixels is not None
    on_floor = cam.pixel_to_world(pixels[0], pixels[1])
    assert on_floor is not None
    assert math.dist(on_floor, (0.0 + FRONT_X, 2.0)) > 0.2
