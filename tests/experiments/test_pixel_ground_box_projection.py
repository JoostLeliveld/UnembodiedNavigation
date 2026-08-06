"""The candidate box->ground path: experiments/pixel_ground_path.

Guards its opt-in/default boundary, geometry, frame convention and fail-closed behavior.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

_PIXEL_GROUND = Path(__file__).resolve().parents[2] / "experiments" / "pixel_ground_path"
if str(_PIXEL_GROUND) not in sys.path:
    sys.path.insert(0, str(_PIXEL_GROUND))

from box_projection import (  # noqa: E402
    BOX_STATISTIC_ALPHA,
    BOX_STATISTIC_PLANE_Z_M,
    BOX_STATISTIC_REFERENCE_MOUNT,
    BOX_STATISTIC_SIGMA_UV_PX,
    BOX_STATISTIC_SIGMA_YAW_M,
    box_statistic_mount_deviation,
    box_statistic_pixel,
    project_box_to_world,
    project_box_to_world_with_covariance,
)
from unav_common.camera_model import ObliqueCameraModel

_REPO_ROOT = Path(__file__).resolve().parents[2]
_E4_SUMMARY = (
    _REPO_ROOT / "logs/studies/pixel_ground_path/e4_covariance_calibration/summary.json"
)


class _AffineCamera:
    """Exact local map used to test covariance rotation without camera-model coupling."""

    cam_pos = np.asarray((0.0, 0.0, 6.1), dtype=float)

    @staticmethod
    def pixel_to_world_at_z(u, v, _plane):
        return float(u), float(v)


def _camera():
    """camera_A of warehouse_full_4cam: (-6, -10, 6.10), pitch 0.92 rad, 1280x720, 90 deg."""
    pitch, yaw = 0.92, math.pi / 2
    x, y, z = -6.0, -10.0, 6.10
    forward = (math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), -math.sin(pitch))
    scale = -z / forward[2]
    return ObliqueCameraModel(
        cam_pos=(x, y, z),
        look_at=(x + scale * forward[0], y + scale * forward[1], 0.0),
        img_width=1280,
        img_height=720,
        fov_h_rad=1.5708,
    )


def test_alpha_zero_reproduces_the_bottom_centre_pixel_exactly():
    """The two paths must differ in one number, not in kind."""
    box = (600.0, 300.0, 620.0, 318.0)
    assert box_statistic_pixel(box, alpha=0.0) == (610.0, 318.0)


def test_default_statistic_is_the_box_centre():
    box = (600.0, 300.0, 620.0, 320.0)
    assert BOX_STATISTIC_ALPHA == 0.5
    assert box_statistic_pixel(box) == (610.0, 310.0)


@pytest.mark.parametrize(
    "box",
    [
        (math.nan, 0.0, 1.0, 1.0),
        (0.0, 0.0, math.inf, 1.0),
        (-1.0, 0.0, 1.0, 1.0),
        (0.0, -1.0, 1.0, 1.0),
        (1.0, 0.0, 1.0, 1.0),
        (2.0, 0.0, 1.0, 1.0),
        (0.0, 2.0, 1.0, 1.0),
    ],
)
def test_box_statistic_rejects_nonfinite_and_nonpositive_geometry(box):
    with pytest.raises(ValueError):
        box_statistic_pixel(box)


@pytest.mark.parametrize("alpha", [-0.01, 1.01, math.nan, math.inf])
def test_box_statistic_rejects_alpha_outside_the_box(alpha):
    with pytest.raises(ValueError):
        box_statistic_pixel((0.0, 0.0, 2.0, 2.0), alpha=alpha)


def test_plane_is_not_the_floor_and_not_a_contact_height():
    """0.085 m is derived from the robot's shape; a regression to 0.0 or 0.05 is a bug."""
    assert BOX_STATISTIC_PLANE_Z_M == pytest.approx(0.085)


@pytest.mark.parametrize("plane", [-0.01, math.nan, math.inf, 6.1, 7.0])
def test_projection_rejects_nonphysical_planes(plane):
    with pytest.raises(ValueError):
        project_box_to_world((0.0, 0.0, 2.0, 2.0), _AffineCamera(), plane_z_m=plane)


def test_projection_lands_on_the_floor_side_of_the_camera():
    camera = _camera()
    box = (600.0, 300.0, 620.0, 320.0)
    point = project_box_to_world(box, camera)
    assert point is not None
    # inside the warehouse footprint, and further from the camera than nothing
    assert -12.0 < point[0] < 12.0 and -9.0 < point[1] < 9.0


def test_centre_statistic_lands_further_out_than_the_bottom_edge():
    """The bottom edge images the near side, so it must read SHORT of the centre."""
    camera = _camera()
    box = (600.0, 300.0, 620.0, 320.0)
    centre = project_box_to_world(box, camera)
    bottom = project_box_to_world(box, camera, alpha=0.0, plane_z_m=0.0)
    assert centre is not None and bottom is not None
    cx, cy = float(camera.cam_pos[0]), float(camera.cam_pos[1])
    d_centre = math.hypot(centre[0] - cx, centre[1] - cy)
    d_bottom = math.hypot(bottom[0] - cx, bottom[1] - cy)
    assert d_bottom < d_centre


def test_covariance_is_spd_and_anisotropic_and_grows_with_range():
    camera = _camera()
    near = (600.0, 420.0, 622.0, 444.0)
    far = (600.0, 250.0, 613.0, 264.0)
    results = []
    for box in (near, far):
        out = project_box_to_world_with_covariance(box, camera)
        assert out is not None
        point, cov = out
        (xx, xy), (yx, yy) = cov
        assert xy == pytest.approx(yx)
        # symmetric positive definite
        assert xx > 0.0 and yy > 0.0
        assert xx * yy - xy * xy > 0.0
        cam_xy = (float(camera.cam_pos[0]), float(camera.cam_pos[1]))
        results.append((math.hypot(point[0] - cam_xy[0], point[1] - cam_xy[1]), xx + yy))
    (d_near, trace_near), (d_far, trace_far) = results
    assert d_far > d_near
    assert trace_far > trace_near, "covariance must grow with range"


def test_yaw_term_can_be_dropped_and_only_shrinks_the_covariance():
    """Conditioning on heading removes Sigma_yaw; it must never inflate."""
    camera = _camera()
    box = (600.0, 300.0, 620.0, 320.0)
    with_yaw = project_box_to_world_with_covariance(box, camera)
    without = project_box_to_world_with_covariance(box, camera, sigma_yaw_m=None)
    assert with_yaw is not None and without is not None
    assert with_yaw[0] == without[0], "the point estimate must not depend on the covariance"
    trace_with = with_yaw[1][0][0] + with_yaw[1][1][1]
    trace_without = without[1][0][0] + without[1][1][1]
    assert trace_without < trace_with
    # the yaw term is the dominant one at these ranges
    assert trace_without < 0.5 * trace_with


@pytest.mark.parametrize(
    ("camera_xy", "off_diagonal"),
    [
        ((0.0, 0.0), 1.5),
        ((2.0, 0.0), -1.5),
        ((2.0, 2.0), 1.5),
        ((0.0, 2.0), -1.5),
    ],
)
def test_yaw_covariance_rotates_in_every_bearing_quadrant(camera_xy, off_diagonal):
    camera = _AffineCamera()
    camera.cam_pos = np.asarray((*camera_xy, 6.1), dtype=float)
    out = project_box_to_world_with_covariance(
        (0.0, 0.0, 2.0, 2.0),
        camera,
        sigma_uv_px=(1.0, 1.0),
        sigma_yaw_m=(2.0, 1.0),
    )
    assert out is not None
    point, covariance = out
    assert point == (1.0, 1.0)
    assert np.asarray(covariance) == pytest.approx(
        np.asarray(((3.5, off_diagonal), (off_diagonal, 3.5)))
    )


class _NadirCamera(_AffineCamera):
    @staticmethod
    def pixel_to_world_at_z(u, v, _plane):
        return float(u) - 1.0, float(v) - 1.0


def test_undefined_camera_bearing_fails_closed_when_yaw_term_is_requested():
    box = (0.0, 0.0, 2.0, 2.0)
    assert project_box_to_world_with_covariance(box, _NadirCamera()) is None
    assert project_box_to_world_with_covariance(box, _NadirCamera(), sigma_yaw_m=None) is not None


class _RankDeficientCamera(_AffineCamera):
    @staticmethod
    def pixel_to_world_at_z(u, _v, _plane):
        return float(u), 1.0


def test_degenerate_projection_jacobian_fails_closed():
    assert project_box_to_world_with_covariance(
        (0.0, 0.0, 2.0, 2.0), _RankDeficientCamera()
    ) is None


@pytest.mark.parametrize(
    "sigma_uv",
    [(-1.0, 1.0), (0.0, 1.0), (math.nan, 1.0), (1.0, math.inf)],
)
def test_covariance_rejects_invalid_pixel_noise(sigma_uv):
    with pytest.raises(ValueError):
        project_box_to_world_with_covariance(
            (0.0, 0.0, 2.0, 2.0), _AffineCamera(), sigma_uv_px=sigma_uv,
        )


@pytest.mark.parametrize("sigma_yaw", [(-1.0, 1.0), (math.nan, 1.0), (1.0, math.inf)])
def test_covariance_rejects_invalid_yaw_noise(sigma_yaw):
    with pytest.raises(ValueError):
        project_box_to_world_with_covariance(
            (0.0, 0.0, 2.0, 2.0), _AffineCamera(), sigma_yaw_m=sigma_yaw,
        )


class _NonFiniteIntersectionCamera(_AffineCamera):
    @staticmethod
    def pixel_to_world_at_z(_u, _v, _plane):
        return math.nan, 0.0


def test_nonfinite_plane_intersection_fails_closed():
    box = (0.0, 0.0, 2.0, 2.0)
    assert project_box_to_world(box, _NonFiniteIntersectionCamera()) is None
    assert project_box_to_world_with_covariance(box, _NonFiniteIntersectionCamera()) is None


def test_frozen_constants_carry_their_provenance_values():
    """The values are frozen, although Sigma_uv is truth-backed commissioning evidence."""
    assert BOX_STATISTIC_SIGMA_UV_PX == (1.15, 0.77)
    assert BOX_STATISTIC_SIGMA_YAW_M == (0.0303, 0.0222)


def test_frozen_constants_still_match_the_locked_e4_evidence():
    """A constant that drifts from the summary it was read out of is a silent fork."""
    if not _E4_SUMMARY.exists():
        pytest.skip("e4 evidence not present in this checkout")
    summary = json.loads(_E4_SUMMARY.read_text(encoding="utf-8"))
    assert summary["alpha"] == pytest.approx(BOX_STATISTIC_ALPHA)
    assert summary["z_star_m"] == pytest.approx(BOX_STATISTIC_PLANE_Z_M)
    assert summary["sigma_uv_px"]["u"] == pytest.approx(BOX_STATISTIC_SIGMA_UV_PX[0], abs=5e-3)
    assert summary["sigma_uv_px"]["v"] == pytest.approx(BOX_STATISTIC_SIGMA_UV_PX[1], abs=5e-3)
    assert summary["sigma_yaw"]["radial_m"] == pytest.approx(
        BOX_STATISTIC_SIGMA_YAW_M[0], abs=5e-5
    )
    assert summary["sigma_yaw"]["lateral_m"] == pytest.approx(
        BOX_STATISTIC_SIGMA_YAW_M[1], abs=5e-5
    )


def test_covariance_is_the_e4_bearing_frame_matrix_expressed_in_map_axes():
    """The candidate module must be a frame change of the evidence construction.

    e4 builds ``R_bearing = rot (J Sigma_uv J^T) rot^T + diag(sigma_r^2, sigma_l^2)`` and
    scores NEES there; this module builds the same thing in map axes.  If the two ever stop
    agreeing, one of them is no longer the path the NEES numbers were measured on.
    """
    camera = _camera()
    sd_u, sd_v = BOX_STATISTIC_SIGMA_UV_PX
    sigma_r, sigma_l = BOX_STATISTIC_SIGMA_YAW_M
    rng = np.random.default_rng(20260806)
    checked = 0
    for _ in range(200):
        u0, v0 = rng.uniform(0.0, 1150.0), rng.uniform(0.0, 620.0)
        box = (u0, v0, u0 + rng.uniform(10.0, 80.0), v0 + rng.uniform(10.0, 60.0))
        result = project_box_to_world_with_covariance(box, camera)
        if result is None:
            continue
        (px, py), cov = result

        # the e4 construction, verbatim in shape
        u, v = box_statistic_pixel(box)
        step, jac = 0.5, np.zeros((2, 2))
        for axis in (0, 1):
            du, dv = (step, 0.0) if axis == 0 else (0.0, step)
            plus = camera.pixel_to_world_at_z(u + du, v + dv, BOX_STATISTIC_PLANE_Z_M)
            minus = camera.pixel_to_world_at_z(u - du, v - dv, BOX_STATISTIC_PLANE_Z_M)
            jac[0, axis] = (plus[0] - minus[0]) / (2.0 * step)
            jac[1, axis] = (plus[1] - minus[1]) / (2.0 * step)
        bearing_x = px - float(camera.cam_pos[0])
        bearing_y = py - float(camera.cam_pos[1])
        norm = math.hypot(bearing_x, bearing_y)
        ux, uy = bearing_x / norm, bearing_y / norm
        rot = np.asarray(((ux, uy), (-uy, ux)))
        r_bearing = rot @ (jac @ np.diag((sd_u**2, sd_v**2)) @ jac.T) @ rot.T
        r_bearing[0, 0] += sigma_r**2
        r_bearing[1, 1] += sigma_l**2

        assert np.asarray(cov) == pytest.approx(rot.T @ r_bearing @ rot, abs=1e-15)
        checked += 1
    assert checked > 100, "the sampled boxes must actually reach the floor"


@pytest.mark.parametrize("step", [0.005, 0.05, 2.0])
def test_propagated_covariance_does_not_depend_on_the_jacobian_step(step):
    """Central differencing is a numerical device, so it must not carry a modelling choice."""
    camera = _camera()
    box = (600.0, 300.0, 620.0, 320.0)
    base = project_box_to_world_with_covariance(box, camera)
    alt = project_box_to_world_with_covariance(box, camera, jacobian_step_px=step)
    assert base is not None and alt is not None
    base_trace = base[1][0][0] + base[1][1][1]
    alt_trace = alt[1][0][0] + alt[1][1][1]
    assert alt_trace == pytest.approx(base_trace, rel=1e-3)


def test_reference_mount_is_reported_and_the_evidence_cameras_match_it():
    camera = _camera()
    height_error, pitch_error = box_statistic_mount_deviation(camera)
    assert height_error == pytest.approx(0.0, abs=1e-9)
    assert pitch_error == pytest.approx(0.0, abs=1e-9)
    assert BOX_STATISTIC_REFERENCE_MOUNT == (6.10, 0.92)


def test_mount_deviation_is_measured_not_gated():
    """A different mount must be reportable; the constants' validity is the caller's call."""
    lower = ObliqueCameraModel(
        cam_pos=(0.0, 0.0, 3.0), look_at=(3.0, 0.0, 0.0),
        img_width=1280, img_height=720, fov_h_rad=1.5708,
    )
    height_error, pitch_error = box_statistic_mount_deviation(lower)
    assert height_error == pytest.approx(-3.10)
    assert pitch_error == pytest.approx(math.atan2(3.0, 3.0) - 0.92)
    # and it still projects, because this is a measurement and not a gate
    assert project_box_to_world((600.0, 300.0, 620.0, 320.0), lower) is not None
